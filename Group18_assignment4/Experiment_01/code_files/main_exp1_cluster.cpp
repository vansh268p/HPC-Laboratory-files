#include <stdio.h>
#include <stdlib.h>
#include <time.h> // Keep this for clock_gettime
#include "init.h"
#include "utils.h"

int GRID_X, GRID_Y, NX, NY;
int NUM_Points, Maxiter;
double dx, dy;

int main() {
    int nx_vals[] = {250, 500, 1000};
    int ny_vals[] = {100, 200, 400};
    
    // NOTE: 1,000,000,000 (1e9) is commented out for your Lab PC to prevent swap-crashing. 
    // Uncomment it ONLY when running on the HPC cluster!
    int pts_vals[] = {100, 10000, 1000000, 100000000, 1000000000}; 
    
    // Calculate the number of test cases based on the array size
    int num_pts_tests = sizeof(pts_vals) / sizeof(pts_vals[0]); 
    Maxiter = 10;

    // Standard Output (This goes to the CSV file)
    printf("Config,NX,NY,Points,Total_Interp_Time\n");

    for (int c = 0; c < 3; c++) {
        NX = nx_vals[c];
        NY = ny_vals[c];
        GRID_X = NX + 1;
        GRID_Y = NY + 1;
        dx = 1.0 / NX;
        dy = 1.0 / NY;

        // Standard Error (This prints LIVE to your screen)
        fprintf(stderr, "\n=== Starting Config %d: %dx%d ===\n", c+1, NX, NY);

        for (int p = 0; p < num_pts_tests; p++) {
            NUM_Points = pts_vals[p];
            
            fprintf(stderr, " -> Allocating memory for %d points... ", NUM_Points);
            double *mesh_value = (double *) calloc(GRID_X * GRID_Y, sizeof(double));
            Points *points = (Points *) calloc(NUM_Points, sizeof(Points));

            if (points == NULL || mesh_value == NULL) {
                fprintf(stderr, "FAILED! Out of memory.\n");
                continue; 
            }
            fprintf(stderr, "Success.\n");

            double total_time = 0.0;
            double dummy_check = 0.0; // Added to stop the compiler from skipping the math

            for (int iter = 0; iter < Maxiter; iter++) {
                fprintf(stderr, "    [Iter %d/%d] Initializing & Interpolating... ", iter+1, Maxiter);
                
                initializepoints(points); 
                
                // Use a high-resolution POSIX timer instead of clock()
                struct timespec start, end;
                clock_gettime(CLOCK_MONOTONIC, &start);
                
                interpolation(mesh_value, points);
                
                clock_gettime(CLOCK_MONOTONIC, &end);
                
                // Calculate time in seconds
                double iter_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000000000.0;
                total_time += iter_time;
                
                // Add a value to the dummy check so the compiler knows the data is used
                dummy_check += mesh_value[0]; 

                fprintf(stderr, "Done (Interpolation took: %.6fs)\n", iter_time);
            }
            
            // Standard Output (Prints to CSV with 6 decimal places to show small numbers)
            printf("%d,%d,%d,%d,%.6lf\n", c+1, NX, NY, NUM_Points, total_time);

            // Print the dummy check to standard error so it does not mess up your CSV
            if (dummy_check < 0.0) fprintf(stderr, "Check: %f\n", dummy_check); 

            free(mesh_value);
            free(points);
        }
    }
    
    fprintf(stderr, "\nExperiment 1 Complete!\n");
    return 0;
}