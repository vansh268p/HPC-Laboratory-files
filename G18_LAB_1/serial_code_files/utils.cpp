#include <math.h>
#include "utils.h"

void vector_triad_operation(double *x, double *y, double *v, double *S, int Np) {

    for (int p = 0; p < Np; p++) {
        // S[p] = 0.5 * 2 * (pow(v[p],2)); this is the energy kernel
        // x[p]=y[p];  this is the copy eqaution
        // x[p]=v[p]+y[p]; this is the scale operation
        // s[p]=x[p]+y[p]; this is the add operation
        // s[p]=x[p]+v[p]*y[p]; this is the triad operation
        // Prevent compiler from optimizing away the loop
        
        if (((double)p) == 333.333)
            dummy(p);

    }
}

void dummy(int x) {
    x = 10 * sin(x / 10.0);
}
