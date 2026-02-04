#ifndef INIT_H
#define INIT_H

// Allocates and initializes vectors of length `Np`
// x, y ∈ [0,1], v is random, S initialized to 0
void init_vectors(int Np, double **x, double **y, double **v, double **S);

#endif
