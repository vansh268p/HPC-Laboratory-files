#include "init.h"
#include <cstdlib>
#include <cstring>

/* ────────────────────────────────────────────────────────────────
 * Load particles from binary file into SoA layout.
 * ──────────────────────────────────────────────────────────────── */
int load_particles(const char *filename, Particles &p,
                   int &Nx, int &Ny, int &Maxiter)
{
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "[ERROR] Cannot open '%s'\n", filename);
        return -1;
    }

    int N;
    if (fread(&N,       sizeof(int), 1, fp) != 1 ||
        fread(&Nx,      sizeof(int), 1, fp) != 1 ||
        fread(&Ny,      sizeof(int), 1, fp) != 1 ||
        fread(&Maxiter, sizeof(int), 1, fp) != 1) {
        fprintf(stderr, "[ERROR] Failed to read header from '%s'\n", filename);
        fclose(fp);
        return -1;
    }

    p.count    = N;
    p.n_active = N;
    p.x      = (double  *)malloc((size_t)N * sizeof(double));
    p.y      = (double  *)malloc((size_t)N * sizeof(double));
    p.f      = (double  *)malloc((size_t)N * sizeof(double));
    p.active = (uint8_t *)malloc((size_t)N * sizeof(uint8_t));

    if (!p.x || !p.y || !p.f || !p.active) {
        fprintf(stderr, "[ERROR] malloc failed for %d particles\n", N);
        fclose(fp);
        return -1;
    }

    /* Bulk read AoS, then deinterleave to SoA */
    double *raw = (double *)malloc(3ULL * N * sizeof(double));
    if (!raw) {
        fprintf(stderr, "[ERROR] malloc failed for raw buffer\n");
        fclose(fp);
        return -1;
    }

    size_t got = fread(raw, sizeof(double), 3ULL * N, fp);
    if ((int)(got / 3) != N) {
        fprintf(stderr, "[WARN] Expected %d particles, read %zu triples\n",
                N, got / 3);
    }
    fclose(fp);

    for (int i = 0; i < N; i++) {
        p.x[i]      = raw[3 * i];
        p.y[i]      = raw[3 * i + 1];
        p.f[i]      = raw[3 * i + 2];
        p.active[i] = 1;
    }
    free(raw);

    fprintf(stderr, "[INFO] Loaded %d particles, Nx=%d Ny=%d Maxiter=%d\n",
            N, Nx, Ny, Maxiter);
    return 0;
}

/* ────────────────────────────────────────────────────────────────
 * Write mesh to text file.
 * ──────────────────────────────────────────────────────────────── */
void write_mesh(const char *filename, const double *mesh,
                int Nx, int Ny)
{
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "[ERROR] Cannot open '%s' for writing\n", filename);
        return;
    }
    int gx = Nx + 1, gy = Ny + 1;
    for (int j = 0; j < gy; j++) {
        for (int i = 0; i < gx; i++) {
            fprintf(fp, "%.15e", mesh[j * gx + i]);
            if (i < gx - 1) fputc(' ', fp);
        }
        fputc('\n', fp);
    }
    fclose(fp);
}

/* ────────────────────────────────────────────────────────────────
 * Free particle arrays.
 * ──────────────────────────────────────────────────────────────── */
void free_particles(Particles &p)
{
    free(p.x);      p.x      = nullptr;
    free(p.y);      p.y      = nullptr;
    free(p.f);      p.f      = nullptr;
    free(p.active); p.active = nullptr;
    p.count = p.n_active = 0;
}

/* ────────────────────────────────────────────────────────────────
 * Count active particles.
 * ──────────────────────────────────────────────────────────────── */
int count_active(const Particles &p)
{
    int cnt = 0;
    for (int i = 0; i < p.count; i++)
        cnt += p.active[i];
    return cnt;
}
