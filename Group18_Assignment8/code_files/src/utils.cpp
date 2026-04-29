#include "utils.h"
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <omp.h>

/* ================================================================
 *  HELPER: inline scatter of one particle onto four grid corners.
 *  Standard bilinear weights (unnormalized).
 * ================================================================ */
static inline void scatter_one(double *mesh, double xi, double yi, double fi,
                               int Nx, int Ny, double dx, double dy)
{
    double xp = xi / dx;
    double yp = yi / dy;
    int ci = (int)xp;
    int cj = (int)yp;

    if (ci >= Nx) ci = Nx - 1;
    if (ci < 0)   ci = 0;
    if (cj >= Ny) cj = Ny - 1;
    if (cj < 0)   cj = 0;

    double lx = xi - ci * dx;
    double ly = yi - cj * dy;

    double w00 = (dx - lx) * (dy - ly);
    double w10 = lx         * (dy - ly);
    double w01 = (dx - lx)  * ly;
    double w11 = lx         * ly;

    int gx = Nx + 1;
    mesh[cj       * gx + ci    ] += w00 * fi;
    mesh[cj       * gx + ci + 1] += w10 * fi;
    mesh[(cj + 1) * gx + ci    ] += w01 * fi;
    mesh[(cj + 1) * gx + ci + 1] += w11 * fi;
}

/* ================================================================
 *  1. SERIAL BASELINE  (skips inactive particles)
 * ================================================================ */
void interpolation_serial(double *mesh, const Particles &p,
                          int Nx, int Ny, double dx, double dy)
{
    int grid_size = (Nx + 1) * (Ny + 1);
    memset(mesh, 0, grid_size * sizeof(double));

    for (int k = 0; k < p.count; k++) {
        if (!p.active[k]) continue;
        scatter_one(mesh, p.x[k], p.y[k], p.f[k], Nx, Ny, dx, dy);
    }
}

/* ================================================================
 *  2. PARALLEL — ATOMIC
 * ================================================================ */
void interpolation_atomic(double *mesh, const Particles &p,
                          int Nx, int Ny, double dx, double dy)
{
    int grid_size = (Nx + 1) * (Ny + 1);
    memset(mesh, 0, grid_size * sizeof(double));

    int gx = Nx + 1;
    #pragma omp parallel for schedule(static)
    for (int k = 0; k < p.count; k++) {
        if (!p.active[k]) continue;
        double xi = p.x[k], yi = p.y[k], fi = p.f[k];

        double xp = xi / dx;
        double yp = yi / dy;
        int ci = (int)xp;
        int cj = (int)yp;
        if (ci >= Nx) ci = Nx - 1;
        if (ci < 0)   ci = 0;
        if (cj >= Ny) cj = Ny - 1;
        if (cj < 0)   cj = 0;

        double lx = xi - ci * dx;
        double ly = yi - cj * dy;

        double w00 = (dx - lx) * (dy - ly) * fi;
        double w10 = lx        * (dy - ly) * fi;
        double w01 = (dx - lx) * ly        * fi;
        double w11 = lx        * ly        * fi;

        #pragma omp atomic
        mesh[cj       * gx + ci    ] += w00;
        #pragma omp atomic
        mesh[cj       * gx + ci + 1] += w10;
        #pragma omp atomic
        mesh[(cj + 1) * gx + ci    ] += w01;
        #pragma omp atomic
        mesh[(cj + 1) * gx + ci + 1] += w11;
    }
}

/* ================================================================
 *  3. PARALLEL — CRITICAL
 * ================================================================ */
void interpolation_critical(double *mesh, const Particles &p,
                            int Nx, int Ny, double dx, double dy)
{
    int grid_size = (Nx + 1) * (Ny + 1);
    memset(mesh, 0, grid_size * sizeof(double));

    int gx = Nx + 1;
    #pragma omp parallel for schedule(static)
    for (int k = 0; k < p.count; k++) {
        if (!p.active[k]) continue;
        double xi = p.x[k], yi = p.y[k], fi = p.f[k];

        double xp = xi / dx;
        double yp = yi / dy;
        int ci = (int)xp;
        int cj = (int)yp;
        if (ci >= Nx) ci = Nx - 1;
        if (ci < 0)   ci = 0;
        if (cj >= Ny) cj = Ny - 1;
        if (cj < 0)   cj = 0;

        double lx = xi - ci * dx;
        double ly = yi - cj * dy;

        double w00 = (dx - lx) * (dy - ly) * fi;
        double w10 = lx        * (dy - ly) * fi;
        double w01 = (dx - lx) * ly        * fi;
        double w11 = lx        * ly        * fi;

        #pragma omp critical
        {
            mesh[cj       * gx + ci    ] += w00;
            mesh[cj       * gx + ci + 1] += w10;
            mesh[(cj + 1) * gx + ci    ] += w01;
            mesh[(cj + 1) * gx + ci + 1] += w11;
        }
    }
}

/* ================================================================
 *  4. PARALLEL — PRIVATE MESH REDUCTION  ★ LEADERBOARD WINNER ★
 * ================================================================ */
void interpolation_private_reduction(double *mesh, const Particles &p,
                                     int Nx, int Ny, double dx, double dy,
                                     double *priv_buf, int nthreads)
{
    int gx = Nx + 1;
    int grid_size = gx * (Ny + 1);

    memset(mesh, 0, grid_size * sizeof(double));

    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        double *my = priv_buf + (size_t)tid * grid_size;

        memset(my, 0, grid_size * sizeof(double));

        #pragma omp for schedule(static) nowait
        for (int k = 0; k < p.count; k++) {
            if (!p.active[k]) continue;
            double xi = p.x[k], yi = p.y[k], fi = p.f[k];

            double xp = xi / dx;
            double yp = yi / dy;
            int ci = (int)xp;
            int cj = (int)yp;
            if (ci >= Nx) ci = Nx - 1;
            if (ci < 0)   ci = 0;
            if (cj >= Ny) cj = Ny - 1;
            if (cj < 0)   cj = 0;

            double lx = xi - ci * dx;
            double ly = yi - cj * dy;

            double w00 = (dx - lx) * (dy - ly) * fi;
            double w10 = lx        * (dy - ly) * fi;
            double w01 = (dx - lx) * ly        * fi;
            double w11 = lx        * ly        * fi;

            my[cj       * gx + ci    ] += w00;
            my[cj       * gx + ci + 1] += w10;
            my[(cj + 1) * gx + ci    ] += w01;
            my[(cj + 1) * gx + ci + 1] += w11;
        }

        #pragma omp barrier

        #pragma omp for schedule(static)
        for (int g = 0; g < grid_size; g++) {
            double sum = 0.0;
            for (int t = 0; t < nthreads; t++)
                sum += priv_buf[(size_t)t * grid_size + g];
            mesh[g] = sum;
        }
    }
}

/* ================================================================
 *  NORMALISE mesh to [-1, 1]
 * ================================================================ */
void mesh_normalize(double *mesh, int grid_size,
                    double &out_min, double &out_max)
{
    double vmin = DBL_MAX, vmax = -DBL_MAX;

    #pragma omp parallel for reduction(min:vmin) reduction(max:vmax) schedule(static)
    for (int i = 0; i < grid_size; i++) {
        if (mesh[i] < vmin) vmin = mesh[i];
        if (mesh[i] > vmax) vmax = mesh[i];
    }

    out_min = vmin;
    out_max = vmax;

    double range = vmax - vmin;
    if (range < 1e-30) range = 1.0;  /* Avoid division by zero */

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < grid_size; i++) {
        mesh[i] = 2.0 * (mesh[i] - vmin) / range - 1.0;
    }
}

/* ================================================================
 *  DENORMALISE mesh back from [-1,1] to original range.
 * ================================================================ */
void mesh_denormalize(double *mesh, int grid_size,
                      double saved_min, double saved_max)
{
    double range = saved_max - saved_min;
    if (range < 1e-30) range = 1.0;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < grid_size; i++) {
        mesh[i] = (mesh[i] + 1.0) * 0.5 * range + saved_min;
    }
}

/* ================================================================
 *  L2 norm of difference.
 * ================================================================ */
double mesh_l2_diff(const double *a, const double *b, int size)
{
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        double d = a[i] - b[i];
        sum += d * d;
    }
    return sqrt(sum);
}
