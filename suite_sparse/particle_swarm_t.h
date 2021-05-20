#pragma once

#include <array>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <string>

template<int n_dims>
struct Particle {

    // Indentifier for this particle
    std::string id;

    // Position, velocity, and reward of this particle
    std::array<double, n_dims> pos;
    std::array<double, n_dims> vel;
    std::array<double, n_dims> ran;
    double reward;

    // Best position and reward of this particle
    std::array<double, n_dims> best_pos;
    double best_reward;

    // Random number generators
    std::mt19937* gen;
    std::uniform_real_distribution<double>* pos_dis;
    std::uniform_real_distribution<double>* vel_dis;
   
    // Reward function for particle swarm
    std::function<double(Particle<n_dims>&)>* reward_fn;
    
    // Default constuctor. Does not initialize any of the values. This should
    // only be called when creating an std::array of particles, or some other
    // object which calls the default constructor during its construction
    Particle() {}

    // Real constructor. Initializes the particle's position and velocity with
    // random values generated from passed random number generators
    Particle(std::string id, 
             std::mt19937* gen,
             std::uniform_real_distribution<double>* pos_dis,
             std::uniform_real_distribution<double>* vel_dis,
             std::function<double(Particle<n_dims>&)>* reward_fn) :
             id(id),
             gen(gen),
             pos_dis(pos_dis),
             vel_dis(vel_dis),
             reward_fn(reward_fn) {
        
        // Initialize position and velocity with random values
        for (int i = 0; i < n_dims; i++) {
            pos[i] = (*pos_dis)(*gen);
            vel[i] = (*vel_dis)(*gen);
            ran[i] = (*vel_dis)(*gen);
            best_pos[i] = pos[i];
        }
        
        // Initialize rewards 
        reward = (*reward_fn)(*this);
        best_reward = reward;
    }

    // Get a string representation of the particle
    std::string to_string() {
        std::string result = "Particle: " + id + "\n";

        result += "\tPosition: ";
        for (int i = 0; i < n_dims; i++) {
            result += std::to_string(pos[i]);
            if (i < n_dims - 1) result += ", ";
        }
        result += "\n";

        result += "\tVelocity: ";
        for (int i = 0; i < n_dims; i++) {
            result += std::to_string(vel[i]);
            if (i < n_dims - 1) result += ", ";
        }
        result += "\n";

        result += "\tReward: " + std::to_string(reward) + "\n";
        
        result += "\tBest Position: ";
        for (int i = 0; i < n_dims; i++) {
            result += std::to_string(best_pos[i]);
            if (i < n_dims - 1) result += ", ";
        }
        result += "\n";

        result += "\tBest Reward: " + std::to_string(best_reward);

        return result;
    }

    // Write the particle's current information (pos, vel, reward) to a file
    // stream at the given step
    void write_to_stream(std::ofstream& ofs, int step) {
        
        ofs << step << ", " << id;
        for (int i = 0; i < n_dims; i++) {
            ofs << ", " << pos[i];
        }
        for (int i = 0; i < n_dims; i++) {
            ofs << ", " << vel[i];
        }
        ofs << ", " << reward << "\n";

    }

    // Update the particle's position using particle swarm hyperparameters
    // along with the current global best position
  void update_pos(double omega, double phi_g, double phi_p, double r,
                    std::array<double, n_dims>& global_best_pos) {
        
        // Main loop
        for (int i = 0; i < n_dims; i++) {
            
            // Random factors in update
            double r_g = (*pos_dis)(*gen);
            double r_p = (*pos_dis)(*gen);

            // Update velocity
            vel[i] = omega * vel[i] +
                     r_g * phi_g * (global_best_pos[i] - pos[i]) +
                     r_p * phi_p * (best_pos[i] - pos[i]) +
                     r * (*vel_dis)(*gen);

            // Update position
            pos[i] += vel[i];

            if (pos[i] >= 1.0) pos[i] = 0.999999;
            if (pos[i] <  0.0) pos[i] = 0.0;
        }
    }
};

template <int n_dims, int n_particles>
struct ParticleSwarm {

    // Algorithm hyperparameters
    const int max_steps;
    const double omega;
    const double phi_g;
    const double phi_p;
  const double r;
    // Reward function
    std::function<double(Particle<n_dims>&)>& reward_fn;
    
    // File stream to write history results to
    std::ofstream& ofs;

    // Random number generators
    std::mt19937& gen;
    std::uniform_real_distribution<double>& pos_dis;
    std::uniform_real_distribution<double>& vel_dis;

    // MPI Info
    MPI_Comm comm;
    int rank;
    int size;

    // Array of particles
    std::array<Particle<n_dims>, n_particles> particles;

    // Constructor. Sets all hyperparameters, writes header to history file
    // stream, and then initializes all particles in the particles array
    ParticleSwarm(const int max_steps,
                  const double omega,
                  const double phi_g,
                  const double phi_p,
                  const double r,
                  std::function<double(Particle<n_dims>&)>& reward_fn,
                  std::ofstream& ofs,
                  std::mt19937& gen,
                  std::uniform_real_distribution<double>& pos_dis,
                  std::uniform_real_distribution<double>& vel_dis,
                  MPI_Comm comm=MPI_COMM_WORLD):
                  max_steps(max_steps),
                  omega(omega),
                  phi_g(phi_g),
                  phi_p(phi_p),
                  r(r),
                  reward_fn(reward_fn),
                  ofs(ofs),
                  gen(gen),
                  pos_dis(pos_dis),
                  vel_dis(vel_dis),
                  comm(comm) {
        
        // Initialize MPI
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_rank(comm, &size);

        if (rank == 0) {
            // Write header to history csv
            ofs << "step, id";
            for (int i = 0; i < n_dims; i++) {
                ofs << ", pos" + std::to_string(i);
            }
            for (int i = 0; i < n_dims; i++) {
                ofs << ", vel" + std::to_string(i);
            }
            ofs << ", reward\n";

        }

        // Initialize particles
        for (int i = 0; i < n_particles; i++) {

            // Call normal constructor
            particles[i] = Particle<n_dims>(std::to_string(i), &gen, &pos_dis, &vel_dis, &reward_fn);

            // Write 0th step to history file
            if (rank == 0) particles[i].write_to_stream(ofs, 0);
        }
    }

    void swarm() {
        
        // Main loop        
        for (int i = 0; i < max_steps; i++) {

            // Only do calculations on rank 0, we then send the relevant data to
            // particles on other ranks after the calculations have completed. This
            // is not the most efficient way to do this, but does deal with needing
            // all ranks to call the reward function (in our case because the
            // underlying linear operator is distributed), and is probably the
            // simplest way to implement 
            if (rank == 0) {
                
                // Update the global best position
                std::array<double, n_dims>& global_best_pos = particles[0].best_pos;
                update_global_best(global_best_pos);
                
                // Update the particles' positions using the found global best 
                for (int j = 0; j < n_particles; j++) {
		  particles[j].update_pos(omega, phi_g, phi_p, r, global_best_pos);
                }
            }

            // Calculate Reward function 
            for (int j = 0; j < n_particles; j++) {
                particles[j].reward = reward_fn(particles[j]);
                
                // Update best reward and position if necessary
                if (particles[j].reward > particles[j].best_reward) {
                    particles[j].best_reward = particles[j].reward;
                    for (int k = 0; k < n_dims; k++) {
                        particles[j].best_pos[k] = particles[j].pos[k];
                    }
                }
                if (rank == 0) particles[j].write_to_stream(ofs, i + 1);
            }
        }
    }

    // Update the global best position and reward from all particles in the
    // array
    void update_global_best(std::array<double, n_dims>& global_best_pos) {

        // Best found reward
        double global_best_reward = particles[0].best_reward;
        
        // Loop over all particles until best is found
        for (int i = 0; i < n_particles; i++) {
            if (particles[i].best_reward > global_best_reward) {
                global_best_reward = particles[i].best_reward;
                global_best_pos    = particles[i].best_pos;
            }
        }
    }

};
