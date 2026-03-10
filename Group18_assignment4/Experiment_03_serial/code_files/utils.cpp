#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "utils.h"

// Interpolation (Serial Code)
void interpolation(double *mesh_value, Points *points) {
    memset(mesh_value, 0, GRID_X * GRID_Y * sizeof(double));

    for (int p = 0; p < NUM_Points; p++) {
        double xp = points[p].x / dx;
        double yp = points[p].y / dy;
        
        int i = (int)xp;
        int j = (int)yp;
        
        if (i >= NX) i = NX - 1;
        if (j >= NY) j = NY - 1;
        
        double wx = xp - i;
        double wy = yp - j;
        
        mesh_value[j * GRID_X + i]           += (1.0 - wx) * (1.0 - wy);
        mesh_value[j * GRID_X + i + 1]       += wx * (1.0 - wy);
        mesh_value[(j + 1) * GRID_X + i]     += (1.0 - wx) * wy;
        mesh_value[(j + 1) * GRID_X + i + 1] += wx * wy;
    }
}

// Stochastic Mover (Serial Code) - FIXED for fair benchmarking
void mover_serial(Points *points, double deltaX, double deltaY) {
    unsigned int serial_seed = 9999; // Give it a single fixed seed
    
    for (int p = 0; p < NUM_Points; p++) {
        double new_x, new_y;
        do {
            // Using rand_r() here ensures we are comparing the exact same math 
            // as the parallel version, isolating the hardware thread performance.
            double rx = ((double)rand_r(&serial_seed) / RAND_MAX) * 2.0 - 1.0;
            double ry = ((double)rand_r(&serial_seed) / RAND_MAX) * 2.0 - 1.0;
            
            new_x = points[p].x + rx * deltaX;
            new_y = points[p].y + ry * deltaY;
            
        } while (new_x < 0.0 || new_x > 1.0 || new_y < 0.0 || new_y > 1.0); 
        
        points[p].x = new_x;
        points[p].y = new_y;
    }
}
// Stochastic Mover (Parallel Code) 
void mover_parallel(Points *points, double deltaX, double deltaY) {
    #pragma omp parallel
    {
        unsigned int local_seed = 12345 + omp_get_thread_num();
        
        #pragma omp for
        for (int p = 0; p < NUM_Points; p++) {
            double new_x, new_y;
            do {
                double rx = ((double)rand_r(&local_seed) / RAND_MAX) * 2.0 - 1.0;
                double ry = ((double)rand_r(&local_seed) / RAND_MAX) * 2.0 - 1.0;
                new_x = points[p].x + rx * deltaX;
                new_y = points[p].y + ry * deltaY;
            } while (new_x < 0.0 || new_x > 1.0 || new_y < 0.0 || new_y > 1.0);
            
            points[p].x = new_x;
            points[p].y = new_y;
        }
    }
}

// Write mesh to file (Provided)
void save_mesh(double *mesh_value) {
    FILE *fd = fopen("Mesh.out", "w");
    if (!fd) {
        printf("Error creating Mesh.out\n");
        exit(1);
    }
    for (int i = 0; i < GRID_Y; i++) {
        for (int j = 0; j < GRID_X; j++) {
            fprintf(fd, "%lf ", mesh_value[i * GRID_X + j]);
        }
        fprintf(fd, "\n");
    }
    fclose(fd);
}
