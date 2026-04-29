#ifndef MOVER_H
#define MOVER_H

#include "init.h"

/* ──────────────────────────────────────────────────────────────
 * Reverse interpolation (gather: mesh → particle) + position update.
 *
 * For each active particle i:
 *   1. Compute bilinear weights (same as forward scatter)
 *   2. Fi = weighted sum of grid values at four corners
 *   3. x_new = x_i + Fi * dx,  y_new = y_i + Fi * dy
 *   4. If (x_new, y_new) outside [0,1]^2 → mark inactive
 *
 * The mover is embarrassingly parallel: read-only on mesh,
 * independent writes to each particle's own data.
 * ────────────────────────────────────────────────────────────── */

/* Serial mover baseline */
void mover_serial(Particles &p, const double *mesh,
                  int Nx, int Ny, double dx, double dy);

/* Parallel mover */
void mover_parallel(Particles &p, const double *mesh,
                    int Nx, int Ny, double dx, double dy,
                    bool use_dynamic);

#endif /* MOVER_H */
