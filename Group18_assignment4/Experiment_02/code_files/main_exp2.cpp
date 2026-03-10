#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "init.h"
#include "utils.h"

int GRID_X, GRID_Y, NX, NY;
int NUM_Points, Maxiter;
double dx, dy;

int main() {
    int nx_vals[] = {250, 500, 1000};
    int ny_vals[] = {100, 200, 400};
    Maxiter = 10;
    NUM_Points = 100000000; // Fixed at 10^8

    // CSV Header
    printf("Config,NX,NY,Total_Interp_Time\n");

    for (int c = 0; c < 3; c++) {
        NX = nx_vals[c];
        NY = ny_vals[c];
        GRID_X = NX + 1;
        GRID_Y = NY + 1;
        dx = 1.0 / NX;
        dy = 1.0 / NY;

        double *mesh_value = (double *) calloc(GRID_X * GRID_Y, sizeof(double));
        Points *points = (Points *) calloc(NUM_Points, sizeof(Points));

        initializepoints(points); // Initialize ONCE outside the loop

        double total_time = 0.0;
        for (int iter = 0; iter < Maxiter; iter++) {
            clock_t start = clock();
            interpolation(mesh_value, points);
            clock_t end = clock();
            
            total_time += (double)(end - start) / CLOCKS_PER_SEC;
        }
        
        printf("%d,%d,%d,%lf\n", c+1, NX, NY, total_time);

        free(mesh_value);
        free(points);
    }
    return 0;
}
