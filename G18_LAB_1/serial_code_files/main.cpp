#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "init.h"
#include "utils.h"

#define CLK CLOCK_MONOTONIC

int main() {

    // structs to measure timings
    struct timespec start_e2e, end_e2e, start_alg, end_alg;

    int minProbSize = 1 << 3;  // pow(2, 3)
    int maxProbSize = 1 << 29; // pow(2, 29)
    int totalParticles   = maxProbSize;

    double *x, *y, *v, *S;

    // print the output order
    printf("Np-ProblemSize, RUNS, TotalParticles, E2ETime, AlgoTime\n");

    for (int Np = minProbSize; Np <= maxProbSize; Np *= 2) {

        clock_gettime(CLK, &start_e2e);

        // Initialize Vectors
        init_vectors(Np, &x, &y, &v, &S);

        int RUNS = totalParticles / Np;

        clock_gettime(CLK, &start_alg);

        for (int run = 0; run < RUNS; run++) {
            vector_triad_operation(x, y, v, S, Np);
        }

        clock_gettime(CLK, &end_alg);
        clock_gettime(CLK, &end_e2e);

        double e2e_time = (end_e2e.tv_sec - start_e2e.tv_sec) + (end_e2e.tv_nsec - start_e2e.tv_nsec) * 1e-9;
        double alg_time = (end_alg.tv_sec - start_alg.tv_sec) + (end_alg.tv_nsec - start_alg.tv_nsec) * 1e-9;

        printf("%d, %d, %d, %.9lf, %.9lf\n", Np, RUNS, Np * RUNS, e2e_time, alg_time);

        free(x);
        free(y);
        free(v);
        free(S);
    }

    return 0;
}
