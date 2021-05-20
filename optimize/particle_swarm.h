#pragma once

#include <array>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <fstream>
#include <functional>
#include <limits>
#include <string>
#include <vector>
#include <random>
#include <iomanip>

/**
*
* Templated particle struct. 
*
* int dims : The dimensions this particle exists in (how many params to optimize)
*
*/
template < int dims >
    struct Particle {

        // Position, velocity, and reward
        std::array < double, dims > pos;
        std::array < double, dims > vel;
        double reward;

        // Best position and reward found for this particle
        std::array < double, dims > best_pos;
        double best_reward;
        int best_CG_steps;

        // Initializes the positions to a random number between 0 and 1
        // the velocities to a random number between -1 and 1
        // and the reward to smallest double
        Particle() {
            best_CG_steps = 1001;
            std::random_device r;
            std::mt19937 gen(r());
            std::uniform_real_distribution < > cdist(0.0, 1.0);
            std::uniform_real_distribution < > sdist(-1.0, 1.0);
            for (int i = 0; i < dims; i++) {
                pos[i] = cdist(gen);
                vel[i] = sdist(gen);
                best_pos[i] = pos[i];
                reward = std::numeric_limits < double > ::lowest();
                best_reward = std::numeric_limits < double > ::lowest();
            }
        }

        // Function for updating velocity according to particle swarm algorithm
        void update_pos(double omega, double r_p, double r_g, double phi_p, double phi_g, std::array < double, dims > global_best_pos) {
            std::random_device r;
            std::mt19937 gen(r());
            std::uniform_real_distribution < > cdist(0.0, 1.0);
            std::uniform_real_distribution < > sdist(-1.0, 1.0);
            for (int i = 0; i < dims; i++) {
                double r_p = cdist(gen);
                double r_g = cdist(gen);
                vel[i] = omega * vel[i] + phi_p * r_p * (best_pos[i] - pos[i]) + phi_g * r_g * (global_best_pos[i] - pos[i]);
            }

            for (int i = 0; i < dims; i++) {
                pos[i] = pos[i] + vel[i];
                if (pos[i] > 1.0) {
                    pos[i] = 1.0;
                    vel[i] = -vel[i];
                };
                if (pos[i] < 0.0) {
                    pos[i] = 0.0;
                    vel[i] = -vel[i];
                };
            }
        }

        // To string function for Particle object
        std::string to_string() {

            // Result string
            std::string result = "---- Particle ----";

            // Strings to hold members
            std::string pos_string = "Position: ";
            std::string vel_string = "Velocity: ";
            std::string reward_string = "Reward: ";
            std::string best_pos_string = "Best Found Position: ";
            std::string best_reward_string = "Best Found Reward: ";

            // Fill Strings
            for (int i = 0; i < dims; i++) {
                pos_string += std::to_string(pos[i]) + " ";
                vel_string += std::to_string(vel[i]) + " ";
                best_pos_string += std::to_string(best_pos[i]) + " ";
            }
            reward_string += std::to_string(reward);
            best_reward_string += std::to_string(best_reward);

            // Make result
            result += "\n";
            result += pos_string;
            result += "\n";
            result += vel_string;
            result += "\n";
            result += reward_string;
            result += "\n";
            result += best_pos_string;
            result += "\n";
            result += best_reward_string;
            result += "\n";
            return result;
        }
    };

/**
*
* Templated particle swarm struct. Contains logic for
* updated particles and optimizing an objective function
*
* int num_particles : The total number of particles in this swarm
* int dims          : What dimension problem to run particle swarm on
* class T           : Hacky way to include a templated function as a
*                     member variable in a struct
*
*/
template < int num_particles, int dims, class T >
    struct ParticleSwarm {

        // Current particles
        std::array < Particle < dims > , num_particles > particles;

        // History of particles and global best rewards
        std::vector < std::array < Particle < dims > , num_particles >> history;
        std::vector < double > rewards;

        //all stored function values and coords for 1 swarm
        std::vector < double > f_vals;
        std::vector < std::array < double, dims >> swarm_pos;

        std::array < double, dims > global_best_pos;
        Particle < dims > global_best_particle;
        double global_best_reward;

        // Algorithm Parameters
        int max_steps;
        double omega;
        double phi_p;
        double phi_g;
        double(T:: * reward_fn)(Particle < dims > & );
        T & obj;

        // Constructor
        ParticleSwarm(int max_steps, double omega, double phi_p, double phi_g, double(T:: * reward_fn)(Particle < dims > & ), T & obj):
            max_steps(max_steps), omega(omega), phi_p(phi_p), phi_g(phi_g), reward_fn(reward_fn), obj(obj) {
                // Put fill-ins in constructor so multiple swarms can run
                global_best_pos.fill(0);
                global_best_reward = std::numeric_limits < double > ::lowest();
                global_best_particle = particles[0];
            }

        // Main algorithm
        void swarm() {
            std::random_device r;
            std::mt19937 gen(r());
            std::uniform_real_distribution < > cdist(0.0, 1.0);
            std::uniform_real_distribution < > sdist(-1.0, 1.0);
            // NOTE: Figuring out global bests is handled by MPI communicators now

            // Push particles to history
            history.push_back(particles);
            rewards.push_back(global_best_reward);

            //reset info from last swarm
            f_vals.clear();
            swarm_pos.clear();

            // Main loop
            for (int i = 0; i < max_steps; i++) {
                //            std::cout << "<--- Particle Swarm Iteration " << i << " --->" << std::endl;
                //                        std::cout << "<";
                //            for(int z = 0; z<n-2; z++)
                //            std::cout << "-";
                //                std::cout << ">" << std::endl;
                for (int j = 0; j < num_particles; j++) {
                    //std::cout << "Particle " << j << std::endl;

                    // Update current particle's velocity and position
                    double r_p = cdist(gen);
                    double r_g = cdist(gen);
                    particles[j].update_pos(omega, r_p, r_g, phi_p, phi_g, global_best_pos);
                    //push updated position to history
                    swarm_pos.push_back(particles[j].pos);
                    // Update local max if found
                    double f = (obj.*reward_fn)(particles[j]);
                    particles[j].reward = f;

                    //store the f value somewhere
                    f_vals.push_back(f);

                    if (f > particles[j].best_reward) {
                        particles[j].best_reward = f;
                        particles[j].best_pos = particles[j].pos;
                    }
                    // Update global max if found
                    if (f > global_best_reward) {
                        global_best_reward = f;
                        global_best_pos = particles[j].pos;
                        global_best_particle = particles[j];

                    }
		    //                    std::cout << particles[j].to_string() << std::endl;
                    //               std::cout << "." << std::flush;

                }
                //            std::cout << std::endl;

                // Update history
                history.push_back(particles);
                rewards.push_back(global_best_reward);
            }
        }

        //helper function to write coords to file
        void save_history(std::string name) {

            //convert fname to char *
            const char * fname = name.c_str();

            std::ofstream hist_writer;
            hist_writer.open(fname);

            //go through all history for the last swarm
                            //save particle array reference
                std::array < double, dims > temp;
                temp = global_best_particle.pos;

                //get reward for this position
                double f = global_best_reward;

                if (f == std::numeric_limits < double > ::lowest() || f != f) {
                    hist_writer << -1000.0 << ", ";
                } else {

                    //write reward value to file
                    hist_writer << std::fixed << std::setprecision(6) << f;
                    hist_writer << ", ";

                }

                //iterate through all dimensions
                for (int j = 0; j < temp.size(); j++) {
                    hist_writer << std::fixed << std::setprecision(6) << temp[j];
                    if (j != temp.size() - 1) {
                        hist_writer << ",";
                    }
                }

                //write newline
                hist_writer << "\n";
		hist_writer.close();

        }
    };

/**
*
* Templated particle swarm MCMC struct. Instead of having particles
* swarm towards maxima/minima the particles randomly explore space
* with no call to update position. 
*
* int num_particles : The total number of particles in this swarm
* int dims          : What dimension problem to run particle swarm on
* class T           : Hacky way to include a templated function as a
*                     member variable in a struct
*
*/
template < int n, int dims, class T >
    struct ParticleMCMC {

        // Current particles
        std::array < Particle < dims > , n > particles;

        // History of particles and global best rewards
        std::vector < std::array < Particle < dims > , n >> history;
        std::vector < double > rewards;

        //all stored function values and coords for 1 swarm
        std::vector < double > f_vals;
        std::vector < std::array < double, dims >> swarm_pos;

        std::array < double, dims > global_best_pos;
        Particle < dims > global_best_particle;
        double global_best_reward;

        // Algorithm Parameters
        int max_steps;
        double omega;
        double phi_p;
        double phi_g;
        double(T:: * reward_fn)(Particle < dims > & );
        T & obj;

        // Constructor
        ParticleMCMC(int max_steps, double omega, double phi_p, double phi_g, double(T:: * reward_fn)(Particle < dims > & ), T & obj):
            max_steps(max_steps), omega(omega), phi_p(phi_p), phi_g(phi_g), reward_fn(reward_fn), obj(obj) {
                // Put fill-ins in constructor so multiple swarms can run
                global_best_pos.fill(0);
                global_best_reward = std::numeric_limits < double > ::lowest();
                global_best_particle = particles[0];
            }

        // Main algorithm
        void swarm() {
            double lr = 0.1;

            std::random_device r;
            std::mt19937 gen(r());
            std::uniform_real_distribution < > cdist(0.0, 1.0);
            std::uniform_real_distribution < > sdist(-1.0, 1.0);
            // NOTE: Figuring out global bests is handled by MPI communicators now

            // Push particles to history
            history.push_back(particles);
            rewards.push_back(global_best_reward);

            //reset info from last swarm
            f_vals.clear();
            swarm_pos.clear();

            // Main loop
            for (int i = 0; i < max_steps; i++) {
                for (int j = 0; j < n; j++) {
                    for (int k = 0; k < dims; k++) {
                        particles[j].pos[k] += lr * sdist(gen);
                        if (particles[j].pos[k] >= 1.0) particles[j].pos[k] = 0.999;
                        if (particles[j].pos[k] < 0.0) particles[j].pos[k] = 0.0;
                    }

                    swarm_pos.push_back(particles[j].pos);
                    double f = (obj.*reward_fn)(particles[j]);
                    particles[j].reward = f;
                    f_vals.push_back(f);

                }
                //            std::cout << std::endl;

                // Update history
                history.push_back(particles);
                rewards.push_back(global_best_reward);
            }
        }

        //helper function to write coords to file
        void save_history(std::string rank) {

            //append rank to file name
            std::string name = "coords_" + rank + ".csv";

            //convert fname to char *
            const char * fname = name.c_str();

            std::ofstream hist_writer;
            hist_writer.open(fname);

            //go through all history for the last swarm
            for (int i = 0; i < swarm_pos.size(); i++) {
                //save particle array reference
                std::array < double, dims > temp;
                temp = swarm_pos[i];

                //get reward for this position
                double f = f_vals[i];

                if (f == std::numeric_limits < double > ::lowest() || f != f) {
                    hist_writer << -1000.0 << ", ";
                } else {

                    //write reward value to file
                    hist_writer << std::fixed << std::setprecision(6) << f;
                    hist_writer << ", ";

                }

                //iterate through all dimensions
                for (int j = 0; j < temp.size(); j++) {
                    hist_writer << std::fixed << std::setprecision(6) << temp[j];
                    if (j != temp.size() - 1) {
                        hist_writer << ",";
                    }
                }

                //write newline
                hist_writer << "\n";

            }

            hist_writer.close();

        }
    };
