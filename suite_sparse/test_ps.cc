#include <cstdlib>
#include <fstream>
#include <functional>
#include <mpi.h>
#include <iostream>
#include <random>

#include "particle_swarm_t.h"

int main(int argc, char** argv) {
    
    using std::cout;    
    using std::endl;    

    // Initialize MPI
    MPI_Init(&argc, &argv);

    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // ps hyperparams
    const int n_dims = 2;
    const int n_particles = 10;
    const int max_steps = 50;
    const double omega = 0.3;
    const double phi_g = 0.3;
    const double phi_p = 0.3;
    
    // Indentifier for test particle
    std::string id = "test";

    // Random number generators
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> d1(0.0, 1.0);
    std::uniform_real_distribution<double> d2(-1.0, 1.0);

    // Create a reward function that is a capturing lambda function, so that we
    // can test the functionality of using std::function
    double value = 4.0;
    std::function<double(Particle<n_dims>&)> reward_fn = [&](Particle<n_dims>& p) {
        double result = 0.0;
        for (int i = 0; i < n_dims; i++) {
            result -= p.pos[i] * p.pos[i]; 
        }
        result += value;
        return result;
    };

    // Stream to write history
    std::ofstream ofs("ofs.csv", std::ios::out | std::ios::trunc);

    if (rank != 0) {
        ofs.close();
    }        

    // Particle Swarm object
    ParticleSwarm<n_dims, n_particles> ps(
            max_steps,
            omega,
            phi_g,
            phi_p,
            reward_fn,
            ofs,
            gen,
            d1,
            d2
    );

    ps.swarm();
    
    if (rank == 0) {
        ofs.close();
    }

    MPI_Finalize();
            
}
