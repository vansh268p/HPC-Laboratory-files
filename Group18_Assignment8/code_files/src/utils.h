#ifndef UTILS_H
#define UTILS_H

#include "init.h"

/* ──────────────────────────────────────────────────────────────
 * Interpolation kernels — scatter active particle values onto mesh.
 *
 * All functions expect a pre-allocated mesh of size (Nx+1)*(Ny+1).
 * Inactive particles (active[k]==0) are skipped.
 * ────────────────────────────────────────────────────────────── */

/* Serial baseline */
void interpolation_serial(double *mesh, const Particles &p,
                          int Nx, int Ny, double dx, double dy);

/* Parallel: atomic */
void interpolation_atomic(double *mesh, const Particles &p,
                          int Nx, int Ny, double dx, double dy);

/* Parallel: critical */
void interpolation_critical(double *mesh, const Particles &p,
                            int Nx, int Ny, double dx, double dy);

/* Parallel: per-thread private mesh + reduction (★ leaderboard) */
void interpolation_private_reduction(double *mesh, const Particles &p,
                                     int Nx, int Ny, double dx, double dy,
                                     double *priv_buf, int nthreads);

/* ── Grid normalisation / denormalisation ─────────────────────── */

/* Normalise mesh values to [-1, 1].  Returns (min_val, max_val). */
void mesh_normalize(double *mesh, int grid_size,
                    double &out_min, double &out_max);

/* Denormalise mesh values back from [-1,1] using saved min/max.   */
void mesh_denormalize(double *mesh, int grid_size,
                      double saved_min, double saved_max);

/* ── Utility ──────────────────────────────────────────────────── */
double mesh_l2_diff(const double *a, const double *b, int size);

#endif /* UTILS_H */
