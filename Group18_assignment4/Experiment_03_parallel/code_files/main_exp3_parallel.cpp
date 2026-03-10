#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <random>
#include <omp.h>
#include "init.h"
#include "utils.h"

// Parallel mover using OpenMP
void mover_parallel(double* points_x, double* points_y, int num_points, double dx, double dy) {
    omp_set_num_threads(4); // Fixed to 4 threads as per assignment

    #pragma omp parallel
    {
        // Thread-local RNG to prevent locking bottlenecks
        std::mt19937 rng(omp_get_thread_num() + 12345); 
        std::uniform_real_distribution<double> dist_x(-dx, dx);
        std::uniform_real_distribution<double> dist_y(-dy, dy);

        #pragma omp for
        for (int i = 0; i < num_points; ++i) {
            bool valid_move = false;
            while (!valid_move) {
                double new_x = points_x[i] + dist_x(rng);
                double new_y = points_y[i] + dist_y(rng);

                // Check boundaries
                if (new_x >= 0.0 && new_x <= 1.0 && new_y >= 0.0 && new_y <= 1.0) {
                    points_x[i] = new_x;
                    points_y[i] = new_y;
                    valid_move = true;
                }
            }
        }
    }
}

int main() {
    int Nx = 1000, Ny = 400, maxiter = 10;
    int num_particles = 14000000; 
    double dx = 1.0 / Nx;
    double dy = 1.0 / Ny;

    // Allocate and initialize arrays using your init/utils structure
    double* points_x = new double[num_particles];
    double* points_y = new double[num_particles];
    double* points_val = new double[num_particles];
    double* mesh = new double[(Nx + 1) * (Ny + 1)];

    initialize_points(points_x, points_y, points_val, num_particles);

    std::ofstream outfile("../data_cluster/exp3_results_Clusters.csv");
    outfile << "Iteration,InterpolationTime,MoverTime,TotalTime\n";

    for (int iter = 1; iter <= maxiter; ++iter) {
        reset_mesh(mesh, Nx, Ny);

        auto t1 = std::chrono::high_resolution_clock::now();
        interpolate(points_x, points_y, points_val, mesh, num_particles, Nx, Ny);
        auto t2 = std::chrono::high_resolution_clock::now();
        
        mover_parallel(points_x, points_y, num_particles, dx, dy);
        
        auto t3 = std::chrono::high_resolution_clock::now();

        double interp_time = std::chrono::duration<double>(t2 - t1).count();
        double mover_time = std::chrono::duration<double>(t3 - t2).count();
        double total_time = interp_time + mover_time;

        outfile << iter << "," << interp_time << "," << mover_time << "," << total_time << "\n";
        std::cout << "Parallel Iter " << iter << " | Mover: " << mover_time << "s\n";
    }

    outfile.close();
    delete[] points_x; delete[] points_y; delete[] points_val; delete[] mesh;
    return 0;
}