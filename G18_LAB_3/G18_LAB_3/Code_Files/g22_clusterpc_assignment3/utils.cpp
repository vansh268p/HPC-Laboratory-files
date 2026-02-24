#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "utils.h"

// Serial interpolation 
void interpolation(double *mesh_value, Points *points) {
	for (int i = 0; i < GRID_X * GRID_Y; i++) {
        	mesh_value[i] = 0.0;
    	}
    
    for (int p = 0; p < NUM_Points; p++) {
        double x = points[p].x;
        double y = points[p].y;
        
        int i = (int)(x / dx);
        int j = (int)(y / dy);
        
        if (i >= NX) i = NX - 1;
        if (j >= NY) j = NY - 1;
        
        double X_i = i * dx;
        double Y_j = j * dy;
        
        double dist_x = (x - X_i) / dx;
        double dist_y = (y - Y_j) / dy;
        
        double w_bl = (1.0 - dist_x) * (1.0 - dist_y);  
        double w_br = dist_x * (1.0 - dist_y);          
        double w_tl = (1.0 - dist_x) * dist_y;          
        double w_tr = dist_x * dist_y;                  
        
        mesh_value[j * GRID_X + i] += w_bl;
        mesh_value[j * GRID_X + (i + 1)] += w_br;
        mesh_value[(j + 1) * GRID_X + i] += w_tl;
        mesh_value[(j + 1) * GRID_X + (i + 1)] += w_tr;
    }
}

// Write mesh to file
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
