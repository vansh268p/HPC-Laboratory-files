/* ================================================================
 *  main_openmp_reference.cpp — copied Assignment 7 OpenMP-only reference
 *  Full pipeline: interpolation → normalise → mover → denormalise
 *
 *  Usage:
 *    ./pipeline <input.bin> <num_threads> <variant>
 *
 *  variant: serial | atomic | private_reduction | all
 * ================================================================ */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <omp.h>

#include "init.h"
#include "utils.h"
#include "mover.h"

/* ── Run full pipeline for one variant, emitting per-iteration CSV ── */
static void run_pipeline(const char *variant_name,
                         bool use_serial_interp,
                         bool use_atomic,
                         bool use_private_reduction,
                         Particles &p,
                         int Nx, int Ny, double dx, double dy,
                         int Maxiter, int nthreads,
                         FILE *csv_out)
{
    int gx = Nx + 1;
    int gy = Ny + 1;
    int grid_size = gx * gy;

    double *mesh = (double *)calloc(grid_size, sizeof(double));

    /* Pre-allocate private buffers for private_reduction */
    double *priv_buf = nullptr;
    if (use_private_reduction) {
        priv_buf = (double *)calloc((size_t)nthreads * grid_size, sizeof(double));
        /* First-touch initialisation */
        #pragma omp parallel num_threads(nthreads)
        {
            int tid = omp_get_thread_num();
            memset(priv_buf + (size_t)tid * grid_size, 0,
                   grid_size * sizeof(double));
        }
    }

    /* Reset all particles to active */
    for (int i = 0; i < p.count; i++) p.active[i] = 1;
    p.n_active = p.count;

    /* Backup original positions (for re-running other variants) */
    double *x_bak = (double *)malloc(p.count * sizeof(double));
    double *y_bak = (double *)malloc(p.count * sizeof(double));
    memcpy(x_bak, p.x, p.count * sizeof(double));
    memcpy(y_bak, p.y, p.count * sizeof(double));

    for (int iter = 0; iter < Maxiter; iter++) {

        /* ── Phase 1: Forward interpolation (particle → mesh) ──── */
        double t_interp_start = omp_get_wtime();

        if (use_serial_interp) {
            interpolation_serial(mesh, p, Nx, Ny, dx, dy);
        } else if (use_atomic) {
            interpolation_atomic(mesh, p, Nx, Ny, dx, dy);
        } else if (use_private_reduction) {
            interpolation_private_reduction(mesh, p, Nx, Ny, dx, dy,
                                            priv_buf, nthreads);
        }

        double t_interp_end = omp_get_wtime();
        double t_interp = t_interp_end - t_interp_start;

        /* ── Phase 2: Normalise grid to [-1, 1] ───────────────── */
        double saved_min, saved_max;
        mesh_normalize(mesh, grid_size, saved_min, saved_max);

        /* ── Phase 3: Mover (reverse interpolation + update) ──── */
        double active_frac = (double)p.n_active / p.count;
        bool use_dynamic = (active_frac < 0.90);

        double t_mover_start = omp_get_wtime();

        if (use_serial_interp) {
            mover_serial(p, mesh, Nx, Ny, dx, dy);
        } else {
            mover_parallel(p, mesh, Nx, Ny, dx, dy, use_dynamic);
        }

        double t_mover_end = omp_get_wtime();
        double t_mover = t_mover_end - t_mover_start;

        /* ── Phase 4: Denormalise grid ─────────────────────────── */
        mesh_denormalize(mesh, grid_size, saved_min, saved_max);

        /* ── Update active count ───────────────────────────────── */
        p.n_active = count_active(p);

        double t_total = t_interp + t_mover;

        /* ── Emit CSV row ──────────────────────────────────────── */
        fprintf(csv_out,
                "%s,%d,%d,%d,%d,%d,%d,%.9f,%.9f,%.9f,%d\n",
                variant_name, nthreads, Nx, Ny, p.count, Maxiter,
                iter, t_interp, t_mover, t_total, p.n_active);
    }

    /* Restore original positions for next variant */
    memcpy(p.x, x_bak, p.count * sizeof(double));
    memcpy(p.y, y_bak, p.count * sizeof(double));
    free(x_bak);
    free(y_bak);

    free(mesh);
    if (priv_buf) free(priv_buf);
}

int main(int argc, char *argv[])
{
    if (argc < 4) {
        fprintf(stderr,
            "Usage: %s <input.bin> <num_threads> <variant>\n"
            "  variant: serial|atomic|private_reduction|all\n", argv[0]);
        return 1;
    }

    const char *input_file = argv[1];
    int nthreads = atoi(argv[2]);
    const char *variant   = argv[3];

    omp_set_num_threads(nthreads);

    /* ── Load particles ──────────────────────────────────────── */
    Particles p;
    int Nx, Ny, Maxiter;
    if (load_particles(input_file, p, Nx, Ny, Maxiter) != 0)
        return 1;

    double dx = 1.0 / Nx;
    double dy = 1.0 / Ny;

    /* ── CSV header ──────────────────────────────────────────── */
    fprintf(stdout,
            "variant,threads,Nx,Ny,N_particles,Maxiter,"
            "iteration,t_interp_s,t_mover_s,t_total_s,n_active\n");

    /* ── Dispatch ────────────────────────────────────────────── */
    if (strcmp(variant, "all") == 0) {
        run_pipeline("serial",            true,  false, false,
                     p, Nx, Ny, dx, dy, Maxiter, nthreads, stdout);
        run_pipeline("atomic",            false, true,  false,
                     p, Nx, Ny, dx, dy, Maxiter, nthreads, stdout);
        run_pipeline("private_reduction", false, false, true,
                     p, Nx, Ny, dx, dy, Maxiter, nthreads, stdout);
    } else if (strcmp(variant, "serial") == 0) {
        run_pipeline("serial",            true,  false, false,
                     p, Nx, Ny, dx, dy, Maxiter, nthreads, stdout);
    } else if (strcmp(variant, "atomic") == 0) {
        run_pipeline("atomic",            false, true,  false,
                     p, Nx, Ny, dx, dy, Maxiter, nthreads, stdout);
    } else if (strcmp(variant, "private_reduction") == 0) {
        run_pipeline("private_reduction", false, false, true,
                     p, Nx, Ny, dx, dy, Maxiter, nthreads, stdout);
    } else {
        fprintf(stderr, "[ERROR] Unknown variant '%s'\n", variant);
        free_particles(p);
        return 1;
    }

    free_particles(p);
    return 0;
}
