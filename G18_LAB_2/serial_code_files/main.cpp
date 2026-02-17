#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "init.h"
#include "utils.h"

#define CLK CLOCK_MONOTONIC

int main() {

    // Structs to measure timings
    struct timespec start_e2e, end_e2e, start_alg, end_alg;

    // Define range: 2^2 to 2^12 
    int minProbSize = 1 << 2;  
    int maxProbSize = 1 << 12; 

    double** m1;
    double** m2;
    double** mt; // For transpose
    double** result;

    printf("ProblemSize, E2ETime, AlgoTime\n");

    for (int Np = minProbSize; Np <= maxProbSize; Np *= 2) {

        clock_gettime(CLK, &start_e2e);

        // 1. Initialize Matrices
        init_matrices(Np, &m1, &m2, &result);

        clock_gettime(CLK, &start_alg);
        
        // SELECT ALGORITHM VIA MACRO
        #if defined(ALG_IJK)
            matrix_multiplication_ijk(m1, m2, result, Np);
        #elif defined(ALG_IKJ)
            matrix_multiplication_ikj(m1, m2, result, Np);
        #elif defined(ALG_JIK)
            matrix_multiplication_jik(m1, m2, result, Np);
        #elif defined(ALG_JKI)
            matrix_multiplication_jki(m1, m2, result, Np);
        #elif defined(ALG_KIJ)
            matrix_multiplication_kij(m1, m2, result, Np);
        #elif defined(ALG_KJI)
            matrix_multiplication_kji(m1, m2, result, Np);

        #elif defined(ALG_TRANSPOSE)
            // Allocate mt specifically for this step
            mt = (double**)malloc(Np * sizeof(double*));
            for(int k=0; k<Np; k++) mt[k] = (double*)malloc(Np * sizeof(double));
            
            transpose(m2, mt, Np);
            transposed_matrix_multiplication(m1, mt, result, Np);
            
            // Free mt locally
            for(int k=0; k<Np; k++) free(mt[k]);
            free(mt);

        #elif defined(ALG_BLOCK)
            int B = (Np >= 32 ? 32 : Np); // Set Block Size
            block_matrix_multiplication(m1, m2, result, B, Np);
        #else
            #error "No algorithm defined! Compile with -DALG_XXX (e.g., -DALG_IJK)"
        #endif


        clock_gettime(CLK, &end_alg);

        // End End-to-End timing
        clock_gettime(CLK, &end_e2e);

        double e2e_time = (end_e2e.tv_sec - start_e2e.tv_sec) + (end_e2e.tv_nsec - start_e2e.tv_nsec) * 1e-9;
        double alg_time = (end_alg.tv_sec - start_alg.tv_sec) + (end_alg.tv_nsec - start_alg.tv_nsec) * 1e-9;

        printf("%d, %.9lf, %.9lf\n", Np, e2e_time, alg_time);
        fflush(stdout); // Ensure data is printed immediately

        // Free standard matrices
        free_matrices(Np, m1, m2, result);
    }

    return 0;
}
