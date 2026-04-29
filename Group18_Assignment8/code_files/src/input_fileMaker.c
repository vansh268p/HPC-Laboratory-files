/* ================================================================
 *  input_fileMaker.c — Generate binary input for Assignments 6 & 7
 *
 *  Usage:
 *    ./input_fileMaker <Nx> <Ny> <N_particles> <Maxiter> <output_file>
 *
 *  Binary format written:
 *    int32  N_particles
 *    int32  Nx
 *    int32  Ny
 *    int32  Maxiter
 *    N_particles × { double x, double y, double f }
 *
 *  All particle positions are uniform random in [0, 1] × [0, 1].
 *  f_i = 1.0 for every particle.
 * ================================================================ */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char *argv[])
{
    if (argc < 6) {
        fprintf(stderr,
            "Usage: %s <Nx> <Ny> <N_particles> <Maxiter> <output_file>\n",
            argv[0]);
        return 1;
    }

    int Nx       = atoi(argv[1]);
    int Ny       = atoi(argv[2]);
    int N        = atoi(argv[3]);
    int Maxiter  = atoi(argv[4]);
    const char *ofile = argv[5];

    FILE *fp = fopen(ofile, "wb");
    if (!fp) {
        fprintf(stderr, "Error: cannot open '%s' for writing\n", ofile);
        return 1;
    }

    /* Header */
    fwrite(&N,       sizeof(int), 1, fp);
    fwrite(&Nx,      sizeof(int), 1, fp);
    fwrite(&Ny,      sizeof(int), 1, fp);
    fwrite(&Maxiter, sizeof(int), 1, fp);

    /* Seed the RNG deterministically for reproducibility */
    srand(42);

    /* Write particles in 64 KB chunks to keep memory usage low */
    const int CHUNK = 8192;
    double buf[CHUNK * 3];

    int remaining = N;
    while (remaining > 0) {
        int batch = (remaining < CHUNK) ? remaining : CHUNK;
        for (int i = 0; i < batch; i++) {
            buf[3 * i    ] = (double)rand() / RAND_MAX;   /* x in [0,1] */
            buf[3 * i + 1] = (double)rand() / RAND_MAX;   /* y in [0,1] */
            buf[3 * i + 2] = 1.0;                         /* f = 1      */
        }
        fwrite(buf, sizeof(double), 3 * batch, fp);
        remaining -= batch;
    }

    fclose(fp);
    fprintf(stderr, "[input_fileMaker] Wrote %d particles "
            "(Nx=%d, Ny=%d, Maxiter=%d) to '%s'\n",
            N, Nx, Ny, Maxiter, ofile);
    return 0;
}
