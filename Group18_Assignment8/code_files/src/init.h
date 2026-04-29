#ifndef INIT_H
#define INIT_H

#include <cstdint>
#include <cstdio>

/* ────────────────────────────────────────────────────────────────
 * Structure-of-Arrays (SoA) particle storage with active flags.
 * ──────────────────────────────────────────────────────────────── */
struct Particles {
    double   *x;        /* x-coordinates */
    double   *y;        /* y-coordinates */
    double   *f;        /* function values (fi = 1) */
    uint8_t  *active;   /* 1 = active, 0 = inactive (left domain) */
    int       count;    /* total allocated (includes inactive) */
    int       n_active; /* current active count */
};

/* Load particles from binary file.
 * Binary format:
 *   [int32 N][int32 Nx][int32 Ny][int32 Maxiter]
 *   N × { double x, double y, double f }                         */
int load_particles(const char *filename, Particles &p,
                   int &Nx, int &Ny, int &Maxiter);

/* Write the (Nx+1)×(Ny+1) mesh to a human-readable text file.    */
void write_mesh(const char *filename, const double *mesh,
                int Nx, int Ny);

/* Free SoA particle arrays.                                       */
void free_particles(Particles &p);

/* Count currently active particles.                               */
int count_active(const Particles &p);

#endif /* INIT_H */
