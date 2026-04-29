#include "mover.h"
#include <omp.h>
#include <cstring>

/* ================================================================
 *  Reverse interpolation (gather: mesh → particle) + position update.
 *
 *  For each active particle i:
 *    1. Find cell (ci, cj) and compute bilinear weights
 *    2. Fi = w00·F(ci,cj) + w10·F(ci+1,cj) +
 *            w01·F(ci,cj+1) + w11·F(ci+1,cj+1)
 *    3. x_new = x_i + Fi · dx
 *       y_new = y_i + Fi · dy
 *    4. If outside [0,1]² → mark inactive
 *
 *  The mover is embarrassingly parallel: the mesh is read-only,
 *  and each particle writes only to its own slots.
 * ================================================================ */

static inline void move_one(Particles &p, int k,
                            const double *mesh,
                            int Nx, int Ny, double dx, double dy)
{
    double xi = p.x[k], yi = p.y[k];

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

    /* Same bilinear weights as forward interpolation */
    double w00 = (dx - lx) * (dy - ly);
    double w10 = lx         * (dy - ly);
    double w01 = (dx - lx)  * ly;
    double w11 = lx         * ly;

    int gx = Nx + 1;
    double F00 = mesh[cj       * gx + ci    ];
    double F10 = mesh[cj       * gx + ci + 1];
    double F01 = mesh[(cj + 1) * gx + ci    ];
    double F11 = mesh[(cj + 1) * gx + ci + 1];

    /* Reverse interpolation: field value at particle position */
    double Fi = w00 * F00 + w10 * F10 + w01 * F01 + w11 * F11;

    /* Position update */
    double x_new = xi + Fi * dx;
    double y_new = yi + Fi * dy;

    /* Boundary check */
    if (x_new < 0.0 || x_new > 1.0 || y_new < 0.0 || y_new > 1.0) {
        p.active[k] = 0;
    } else {
        p.x[k] = x_new;
        p.y[k] = y_new;
    }
}

/* ================================================================
 *  SERIAL MOVER
 * ================================================================ */
void mover_serial(Particles &p, const double *mesh,
                  int Nx, int Ny, double dx, double dy)
{
    for (int k = 0; k < p.count; k++) {
        if (!p.active[k]) continue;
        move_one(p, k, mesh, Nx, Ny, dx, dy);
    }
}

/* ================================================================
 *  PARALLEL MOVER
 *
 *  use_dynamic: when active fraction < 90%, switch to
 *  schedule(dynamic, 1024) to handle load imbalance from
 *  clustered inactive particles.
 * ================================================================ */
void mover_parallel(Particles &p, const double *mesh,
                    int Nx, int Ny, double dx, double dy,
                    bool use_dynamic)
{
    if (use_dynamic) {
        #pragma omp parallel for schedule(dynamic, 1024)
        for (int k = 0; k < p.count; k++) {
            if (!p.active[k]) continue;
            move_one(p, k, mesh, Nx, Ny, dx, dy);
        }
    } else {
        #pragma omp parallel for schedule(static)
        for (int k = 0; k < p.count; k++) {
            if (!p.active[k]) continue;
            move_one(p, k, mesh, Nx, Ny, dx, dy);
        }
    }
}
