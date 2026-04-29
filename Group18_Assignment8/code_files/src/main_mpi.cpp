/* ================================================================
 *  main_mpi.cpp — Assignment 08 Hybrid MPI + OpenMP Pipeline
 *
 *  Particle decomposition across MPI ranks:
 *    local particles -> local mesh (OpenMP)
 *    local meshes    -> global mesh (MPI_Allreduce)
 *    global mesh     -> local particle mover (OpenMP)
 *
 *  Usage:
 *    mpirun -np <ranks> ./pipeline_mpi <input.bin> <omp_threads> <variant>
 *
 *  variant: serial | atomic | private_reduction | all
 * ================================================================ */
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <mpi.h>
#include <omp.h>

#include "init.h"
#include "utils.h"
#include "mover.h"

struct InputHeader {
    int n_particles;
    int Nx;
    int Ny;
    int Maxiter;
};

static void die_all(const char *msg, int code)
{
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) fprintf(stderr, "%s\n", msg);
    MPI_Abort(MPI_COMM_WORLD, code);
}

static void local_range(int n_global, int nranks, int rank,
                        int &start, int &count)
{
    int base = n_global / nranks;
    int rem = n_global % nranks;
    count = base + (rank < rem ? 1 : 0);
    start = rank * base + std::min(rank, rem);
}

static void alloc_particles(Particles &p, int n)
{
    p.count = n;
    p.n_active = n;
    p.x = (double *)malloc((size_t)std::max(n, 1) * sizeof(double));
    p.y = (double *)malloc((size_t)std::max(n, 1) * sizeof(double));
    p.f = (double *)malloc((size_t)std::max(n, 1) * sizeof(double));
    p.active = (uint8_t *)malloc((size_t)std::max(n, 1) * sizeof(uint8_t));
    if (!p.x || !p.y || !p.f || !p.active)
        die_all("[ERROR] Failed to allocate local particle arrays", 2);
}

static void load_particles_mpi(const char *filename, Particles &p,
                               InputHeader &hdr, int rank, int nranks)
{
    MPI_File fh;
    int rc = MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY,
                           MPI_INFO_NULL, &fh);
    if (rc != MPI_SUCCESS)
        die_all("[ERROR] MPI_File_open failed for input file", 3);

    int raw_header[4] = {0, 0, 0, 0};
    MPI_Status st;
    rc = MPI_File_read_at_all(fh, 0, raw_header, 4, MPI_INT, &st);
    if (rc != MPI_SUCCESS)
        die_all("[ERROR] Failed to read binary input header", 4);

    hdr.n_particles = raw_header[0];
    hdr.Nx = raw_header[1];
    hdr.Ny = raw_header[2];
    hdr.Maxiter = raw_header[3];

    int start = 0, local_n = 0;
    local_range(hdr.n_particles, nranks, rank, start, local_n);
    alloc_particles(p, local_n);

    double *raw = (double *)malloc((size_t)std::max(local_n, 1) * 3 * sizeof(double));
    if (!raw) die_all("[ERROR] Failed to allocate local raw input buffer", 5);

    MPI_Offset offset = (MPI_Offset)4 * sizeof(int) +
                        (MPI_Offset)start * 3 * sizeof(double);
    rc = MPI_File_read_at_all(fh, offset, raw, local_n * 3,
                              MPI_DOUBLE, &st);
    if (rc != MPI_SUCCESS)
        die_all("[ERROR] Failed to read local particle chunk", 6);

    MPI_File_close(&fh);

    for (int i = 0; i < local_n; i++) {
        p.x[i] = raw[3 * i];
        p.y[i] = raw[3 * i + 1];
        p.f[i] = raw[3 * i + 2];
        p.active[i] = 1;
    }
    free(raw);

    fprintf(stderr,
            "[rank %d/%d] loaded particles [%d, %d) count=%d Nx=%d Ny=%d Maxiter=%d\n",
            rank, nranks, start, start + local_n, local_n,
            hdr.Nx, hdr.Ny, hdr.Maxiter);
}

static int count_active_parallel(const Particles &p)
{
    int cnt = 0;
    #pragma omp parallel for reduction(+:cnt) schedule(static)
    for (int i = 0; i < p.count; i++)
        cnt += p.active[i] ? 1 : 0;
    return cnt;
}

static void reset_particles(Particles &p,
                            const double *x0, const double *y0,
                            const uint8_t *active0)
{
    memcpy(p.x, x0, (size_t)p.count * sizeof(double));
    memcpy(p.y, y0, (size_t)p.count * sizeof(double));
    memcpy(p.active, active0, (size_t)p.count * sizeof(uint8_t));
    p.n_active = p.count;
}

static void run_pipeline_mpi(const char *variant_name,
                             bool use_serial_interp,
                             bool use_atomic,
                             bool use_private_reduction,
                             Particles &p,
                             const InputHeader &hdr,
                             int omp_threads,
                             int rank,
                             int nranks)
{
    const int Nx = hdr.Nx;
    const int Ny = hdr.Ny;
    const double dx = 1.0 / Nx;
    const double dy = 1.0 / Ny;
    const int grid_size = (Nx + 1) * (Ny + 1);

    double *local_mesh = (double *)calloc((size_t)grid_size, sizeof(double));
    double *global_mesh = (double *)calloc((size_t)grid_size, sizeof(double));
    if (!local_mesh || !global_mesh)
        die_all("[ERROR] Failed to allocate mesh buffers", 7);

    double *priv_buf = nullptr;
    if (use_private_reduction) {
        priv_buf = (double *)calloc((size_t)omp_threads * grid_size,
                                    sizeof(double));
        if (!priv_buf)
            die_all("[ERROR] Failed to allocate private reduction buffers", 8);

        #pragma omp parallel num_threads(omp_threads)
        {
            int tid = omp_get_thread_num();
            memset(priv_buf + (size_t)tid * grid_size, 0,
                   (size_t)grid_size * sizeof(double));
        }
    }

    double *x0 = (double *)malloc((size_t)p.count * sizeof(double));
    double *y0 = (double *)malloc((size_t)p.count * sizeof(double));
    uint8_t *active0 = (uint8_t *)malloc((size_t)p.count * sizeof(uint8_t));
    if (!x0 || !y0 || !active0)
        die_all("[ERROR] Failed to allocate particle backup", 9);

    memcpy(x0, p.x, (size_t)p.count * sizeof(double));
    memcpy(y0, p.y, (size_t)p.count * sizeof(double));
    memcpy(active0, p.active, (size_t)p.count * sizeof(uint8_t));
    reset_particles(p, x0, y0, active0);

    for (int iter = 0; iter < hdr.Maxiter; iter++) {
        MPI_Barrier(MPI_COMM_WORLD);

        double t0 = MPI_Wtime();
        if (use_serial_interp) {
            interpolation_serial(local_mesh, p, Nx, Ny, dx, dy);
        } else if (use_atomic) {
            interpolation_atomic(local_mesh, p, Nx, Ny, dx, dy);
        } else if (use_private_reduction) {
            interpolation_private_reduction(local_mesh, p, Nx, Ny, dx, dy,
                                            priv_buf, omp_threads);
        }
        double t_interp_done = MPI_Wtime();

        MPI_Allreduce(local_mesh, global_mesh, grid_size,
                      MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double t_comm_done = MPI_Wtime();

        double local_interp = t_interp_done - t0;
        double local_comm = t_comm_done - t_interp_done;
        double interp_time = 0.0;
        double comm_time = 0.0;
        MPI_Reduce(&local_interp, &interp_time, 1, MPI_DOUBLE, MPI_MAX,
                   0, MPI_COMM_WORLD);
        MPI_Reduce(&local_comm, &comm_time, 1, MPI_DOUBLE, MPI_MAX,
                   0, MPI_COMM_WORLD);

        double saved_min = 0.0, saved_max = 0.0;
        mesh_normalize(global_mesh, grid_size, saved_min, saved_max);
        double t_norm_done = MPI_Wtime();

        double local_norm = t_norm_done - t_comm_done;
        double norm_time = 0.0;
        MPI_Reduce(&local_norm, &norm_time, 1, MPI_DOUBLE, MPI_MAX,
                   0, MPI_COMM_WORLD);

        double active_frac = p.count > 0
            ? (double)p.n_active / (double)p.count
            : 1.0;
        bool use_dynamic = active_frac < 0.90;

        MPI_Barrier(MPI_COMM_WORLD);
        double t2 = MPI_Wtime();
        if (use_serial_interp) {
            mover_serial(p, global_mesh, Nx, Ny, dx, dy);
        } else {
            mover_parallel(p, global_mesh, Nx, Ny, dx, dy, use_dynamic);
        }
        double t3 = MPI_Wtime();

        double local_mover = t3 - t2;
        double mover_time = 0.0;
        MPI_Reduce(&local_mover, &mover_time, 1, MPI_DOUBLE, MPI_MAX,
                   0, MPI_COMM_WORLD);

        double t4 = MPI_Wtime();
        mesh_denormalize(global_mesh, grid_size, saved_min, saved_max);
        double t5 = MPI_Wtime();

        double local_denorm = t5 - t4;
        double denorm_time = 0.0;
        MPI_Reduce(&local_denorm, &denorm_time, 1, MPI_DOUBLE, MPI_MAX,
                   0, MPI_COMM_WORLD);

        p.n_active = count_active_parallel(p);
        int global_active = 0;
        MPI_Reduce(&p.n_active, &global_active, 1, MPI_INT, MPI_SUM,
                   0, MPI_COMM_WORLD);

        if (rank == 0) {
            int total_cores = nranks * omp_threads;
            double total_time = interp_time + comm_time + norm_time +
                                mover_time + denorm_time;
            fprintf(stdout,
                    "%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%d\n",
                    variant_name, nranks, omp_threads, total_cores,
                    Nx, Ny, hdr.n_particles, p.count, hdr.Maxiter, iter,
                    interp_time, comm_time, norm_time, mover_time,
                    denorm_time, total_time,
                    global_active);
        }
    }

    reset_particles(p, x0, y0, active0);
    free(x0);
    free(y0);
    free(active0);
    free(local_mesh);
    free(global_mesh);
    free(priv_buf);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    if (argc < 4) {
        if (rank == 0) {
            fprintf(stderr,
                    "Usage: %s <input.bin> <omp_threads> <variant>\n"
                    "  variant: serial|atomic|private_reduction|all\n",
                    argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    const char *input_file = argv[1];
    int omp_threads = atoi(argv[2]);
    const char *variant = argv[3];
    if (omp_threads < 1) omp_threads = 1;
    omp_set_num_threads(omp_threads);

    Particles p;
    memset(&p, 0, sizeof(p));
    InputHeader hdr;
    load_particles_mpi(input_file, p, hdr, rank, nranks);

    if (rank == 0) {
        fprintf(stdout,
                "variant,mpi_ranks,omp_threads,total_cores,Nx,Ny,"
                "N_particles,local_particles,Maxiter,iteration,"
                "t_interp_s,t_comm_s,t_norm_s,t_mover_s,t_denorm_s,"
                "t_total_s,n_active\n");
    }

    if (strcmp(variant, "all") == 0) {
        run_pipeline_mpi("serial", true, false, false,
                         p, hdr, omp_threads, rank, nranks);
        run_pipeline_mpi("atomic", false, true, false,
                         p, hdr, omp_threads, rank, nranks);
        run_pipeline_mpi("private_reduction", false, false, true,
                         p, hdr, omp_threads, rank, nranks);
    } else if (strcmp(variant, "serial") == 0) {
        run_pipeline_mpi("serial", true, false, false,
                         p, hdr, omp_threads, rank, nranks);
    } else if (strcmp(variant, "atomic") == 0) {
        run_pipeline_mpi("atomic", false, true, false,
                         p, hdr, omp_threads, rank, nranks);
    } else if (strcmp(variant, "private_reduction") == 0) {
        run_pipeline_mpi("private_reduction", false, false, true,
                         p, hdr, omp_threads, rank, nranks);
    } else {
        if (rank == 0) fprintf(stderr, "[ERROR] Unknown variant '%s'\n", variant);
        free_particles(p);
        MPI_Finalize();
        return 1;
    }

    free_particles(p);
    MPI_Finalize();
    return 0;
}
