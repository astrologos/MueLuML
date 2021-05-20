// Clean up compilation
#pragma GCC diagnostic ignored "-Wunused-parameter"

// ---- Deal.II Includes ----
#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/utilities.h>
#include <deal.II/lac/trilinos_epetra_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/base/geometry_info.h>
#include <deal.II/base/timer.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/grid/grid_refinement.h>
# define protected public // Just a hack -- only works in deal.ii > 9.1.1?
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/base/utilities.h>

// ---- Trilinos Includes ----
#include <Teuchos_StandardCatchMacros.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_RCPStdSharedPtrConversions.hpp>
#include <EpetraExt_CrsMatrixIn.h>

// ---- STL Includes ----
#include <fstream>
#include <cstdlib>
#include <unistd.h>
#include <iostream>
#include <memory>
#include <random>
#include <vector>
#include <string>
#include <limits>
#include <math.h>

// ---- MueLu Includes ----
#include <MueLu_CreateEpetraPreconditioner.hpp>

// ---- User defined includes ----
#include "particle_swarm.h"

//declare the namespace
using namespace dealii;

std::random_device rd; //Will be used to obtain a seed for the random number engine
std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()

// --------- Normal Distribution Objects ----------
std::normal_distribution < > d1 {
    0.0,
    1.0e-3
};
std::normal_distribution < > d2 {
    0.0,
    1.0e-2
};
std::normal_distribution < > d3 {
    0.0,
    1.0e-1
};
std::normal_distribution < > d4 {
    0.0,
    1.0
};
std::normal_distribution < > d5 {
    0.0,
    1.0e1
};
// -----------------------------------------------


/**

Struct to store the CG results after a call to PoissonProblem::solve_system().

*/
struct CGResult {

    size_t iterations;
    std::vector < double > residuals;

};


/**

Templated struct for a Poisson Problem. Contains the logic for creating and runnning
a Poisson PDE.

int dim : Dimension to solve the poisson problem in.

The functions include:

    void run(int swarmsteps, int swarmiter, double alpha)
        - Runner for the problem. Instantiates a swarm and sets up problem
    void make_grid()
        - Creates mesh/grid through continuous refinements. Uses grid generator
          class to apply perturbations at each refinement step.
    void setup_system()
        - Distributes the dofs using trilinos and deal.ii logic
    void assemble_system()
        - Assembles lhs and rhs. Wolfgang Bangerth was a huge help for this.
    bool solve_system(double * k, double * rho, int & steps, int cg_steps);
        - Use SolverControl to initialize neccesary Trilinos and Deal.ii objects.
          Use the conjugate gradient solver built into deal.ii
    void create_preconditioner()
        - Write MueLu information to file and call the MueLu create_preconditioner
          function.

*/
template < int dim >
    struct PoissonProblem {

        // Number of MueLu parameters being used
        static  const int n = 5;

        // Constructor
  PoissonProblem(int myrank, std::string MTXfilename);
        int rank;
  std::string mtx_filename;
        // Functions
        void run(int swarmsteps, int swarmiter, double alpha);
        void make_grid();
        void setup_system();
        void assemble_system();
        double get_preconditioner_density();
        bool solve_system(double * k, double * rho, int & steps, int cg_steps);
        double reward_fn(Particle < n > & particle);
        void create_preconditioner(TrilinosWrappers::PreconditionAMGMueLu & preconditioner, Teuchos::ParameterList paramList);

        // Get condition number
        static void get_cond_num(double input, double * k) {
            * k = input;
        }

        // Deal II system info
        Triangulation < dim > triangulation;
        FE_Q < dim > fe;
        DoFHandler < dim > dof_handler;
        SparsityPattern sparsity_pattern;
        SparseMatrix < double > system_matrix;
        Vector < double > solution;
        Vector < double > system_rhs;
        SolverControl solver_control;
        // TrilinosWrappers
        TrilinosWrappers::SparseMatrix trilinos_system_matrix;
        TrilinosWrappers::PreconditionAMGMueLu preconditioner;
        // ParameterList
        Teuchos::ParameterList lastparams;
    };

// ---------------- Classes required to set up PDE system ----------

template < int dim >
    class RightHandSide: public Function < dim > {
        public: RightHandSide(): Function < dim > () {}
        virtual double value(const Point < dim > & p,
            const unsigned int component = 0) const override;
    };

template < int dim >
    class BoundaryValues: public Function < dim > {
        public: BoundaryValues(): Function < dim > () {}
        virtual double value(const Point < dim > & p,
            const unsigned int component = 0) const override;
    };

template < int dim >
    double RightHandSide < dim > ::value(const Point < dim > & p,
        const unsigned int /*component*/ ) const {
        return 1.0 + d1(gen) + d2(gen) + d3(gen) + d4(gen) + d5(gen);
    }

template < int dim >
    double BoundaryValues < dim > ::value(const Point < dim > & p,
        const unsigned int /*component*/ ) const {
        return 0.0 + d1(gen) + d2(gen) + d3(gen) + d4(gen) + d5(gen);
    }

// ----------------- END Classes required to set up PDE system ---------

template < int dim >
PoissonProblem < dim > ::PoissonProblem(int myrank, std::string MTXfilename): fe(1), dof_handler(triangulation),trilinos_system_matrix() {
  mtx_filename = MTXfilename;
        rank = myrank;
    }

/*

Function definition contained in the PoissonProblem struct

*/
template < int dim >
    void PoissonProblem < dim > ::make_grid() {
        // Refine until unknowns > 100,000
        GridGenerator::hyper_L(triangulation);
        if (rank == 0)
            std::cout << "   Refining: " << triangulation.n_vertices() << " dofs" << std::endl;
        while ((int) triangulation.n_vertices() < 20000) {
            if (rank == 0)
                std::cout << "   Refining: " << std::flush;

            triangulation.refine_global();
            GridTools::distort_random(0.2, triangulation);
            if (rank == 0)
                std::cout << triangulation.n_vertices() << " dofs" << std::endl;
        }
        if (rank == 0) {
            std::cout << "   Refining: done!" << std::endl;
            std::cout << "------------------------------";
        }
        int number = (int) triangulation.n_vertices();
        while (number != 0) {
            number /= 10;
            if (rank == 0)
                std::cout << "-";
        }
        // Formatting output
        if (rank == 0) {
            std::cout << std::endl;
            std::cout << "   Number of active cells: " << triangulation.n_active_cells() <<
                std::endl <<
                "   Total number of cells: " << triangulation.n_cells() <<
                std::endl;
            std::cout << "   Degrees of freedom: " << triangulation.n_vertices() <<
                std::endl;
            std::cout << "------------------------------";
            number = (int) triangulation.n_vertices();
            while (number != 0) {
                number /= 10;
                if (rank == 0)
                    std::cout << "-";
            }
            std::cout << std::endl;
            std::ofstream out("grid.vtu");
            GridOut grid_out;
            grid_out.write_vtu(triangulation, out);
        }
    }

/*

Function definition contained in the PoissonProblem struct

*/
template < int dim >
    void PoissonProblem < dim > ::setup_system() {
        if (rank == 0)
            std::cout << "Setting up system ";
        dof_handler.distribute_dofs(fe);
        DynamicSparsityPattern dsp(dof_handler.n_dofs());
        DoFTools::make_sparsity_pattern(dof_handler, dsp);
        sparsity_pattern.copy_from(dsp);
        if (rank == 0)
            std::cout << ". " << std::flush;
        system_matrix.reinit(sparsity_pattern);

        if (rank == 0)
            std::cout << ". " << std::flush;
        solution.reinit(dof_handler.n_dofs());
        if (rank == 0)
            std::cout << ". " << std::endl << std::flush;
        system_rhs.reinit(dof_handler.n_dofs());
    }

/*

Function definition contained in the PoissonProblem struct

*/
template < int dim >
    void PoissonProblem < dim > ::assemble_system() {
        if (rank == 0)
            std::cout << "Assembling system ";
        QGauss < dim > quadrature_formula(fe.degree + 1);
        const RightHandSide < dim > right_hand_side;
        FEValues < dim > fe_values(fe,
            quadrature_formula,
            update_values | update_gradients |
            update_quadrature_points | update_JxW_values);
        const unsigned int dofs_per_cell = fe.dofs_per_cell;
        const unsigned int n_q_points = quadrature_formula.size();
        FullMatrix < double > cell_matrix(dofs_per_cell, dofs_per_cell);
        Vector < double > cell_rhs(dofs_per_cell);
        std::vector < types::global_dof_index > local_dof_indices(dofs_per_cell);
        if (rank == 0)
            std::cout << ". " << std::flush;
        for (const auto & cell: dof_handler.active_cell_iterators()) {
            fe_values.reinit(cell);
            cell_matrix = 0;
            cell_rhs = 0;
            for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
                for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                        cell_matrix(i, j) +=
                        (fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
                            fe_values.shape_grad(j, q_index) * // grad phi_j(x_q)
                            fe_values.JxW(q_index)); // dx
                    const auto x_q = fe_values.quadrature_point(q_index);
                    cell_rhs(i) += (fe_values.shape_value(i, q_index) * // phi_i(x_q)
                        right_hand_side.value(x_q) * // f(x_q)
                        fe_values.JxW(q_index)); // dx
                }
            cell -> get_dof_indices(local_dof_indices);
            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    system_matrix.add(local_dof_indices[i],
                        local_dof_indices[j],
                        cell_matrix(i, j));

                system_rhs(local_dof_indices[i]) += cell_rhs(i);
            }
        }
        if (rank == 0)
            std::cout << ". " << std::flush;

        std::map < types::global_dof_index, double > boundary_values;
        VectorTools::interpolate_boundary_values(dof_handler,
            0,
            BoundaryValues < dim > (),
            boundary_values);
        MatrixTools::apply_boundary_values(boundary_values,
            system_matrix,
            solution,
            system_rhs);
        trilinos_system_matrix.reinit(system_matrix);
        if (rank == 0)
            std::cout << ". " << std::endl << std::flush;

        // Apply A*b, trivially save as x
        system_matrix.vmult(solution, system_rhs);
        // Swap so that b=A*b, x=b
        // (enforce b in R(A) so that Krylov iterations work)
        system_rhs.swap(solution);
        // Reinit x=0
        solution.reinit(dof_handler.n_dofs());
    }

/*

Function definition contained in the PoissonProblem struct

*/
template < int dim >
    void PoissonProblem < dim > ::create_preconditioner(TrilinosWrappers::PreconditionAMGMueLu & preconditioner, Teuchos::ParameterList paramList) {
        const auto teuchos_wrapped_matrix =
            Teuchos::rcp(const_cast < Epetra_CrsMatrix * > ( & trilinos_system_matrix.trilinos_matrix()), false);

       Teuchos::RCP < MueLu::EpetraOperator > tptr = MueLu::CreateEpetraPreconditioner(teuchos_wrapped_matrix,
            paramList);

        preconditioner.preconditioner = tptr;
    }


/*

Function definition contained in the PoissonProblem struct

*/
template < int dim >
    bool PoissonProblem < dim > ::solve_system(double * k, double * rho, int & steps, int cg_steps) {

        // Set up a CG solver with n max iterations, and tolerance 1e-10
        // We DEFINITELY don't want our optimal to take 10000 steps
        // so stop if we get there (hopefully speeds up reward function)
        solver_control = SolverControl(cg_steps, 1e-10, true, true);

        // Enable the output of history data
        solver_control.enable_history_data();

        // Set up a CG solver object
        SolverCG < > cg(solver_control);

        // Get the condition number from solver using std::bind
        cg.connect_condition_number_slot(
            std::bind(get_cond_num, std::placeholders::_1, k));
        //   if (rank==0)
        //  std::cout << "Condition Number Estimate: " << *k << std::endl;

        // Reinit solution so we don't solve in 0 iterations
        solution.reinit(trilinos_system_matrix.n());

        try {
            // Solve the system using preconditioner and cg solver
            cg.solve(trilinos_system_matrix, solution, system_rhs, preconditioner);
        } catch (SolverControl::NoConvergence & ) {
            ;
        }
        // Get residual history
        //std::vector < double > residuals = solver_control.get_history_data();

        // Write residual history to csv file
	//        std::ofstream cg_residual_file("cg_residuals.csv");
	//        cg_residual_file << "step,residual\n";
	//        for (int i = 0; i < (int) residuals.size(); i++) {
            //   if (rank==0)
            //std::cout << "Residual at step " << i << ": " << residuals.at(i) << std::endl;
	//            cg_residual_file << i << "," << residuals.at(i) << "\n";
	//        }
        //cg_residual_file.close();

        // Ensure success
        const SolverControl::State state = solver_control.last_check();
        steps = (int) solver_control.last_step();
        return (state == SolverControl::success);

    }

/*

When observed, a particle's position is dicretized into it's appropriate bin.

*/
template < int bins >
    std::string quantum_jump(double normalized_position, std::array < std::string, bins > string_map) {
        if (normalized_position >= 1.0) {
            normalized_position = 0.999;
        }

        int bin = int(normalized_position * bins);
        return string_map[bin];
    }


/*

Reward function for particle swarm. This function calls the PoissonProblem::solve_system()
After solving the appropriate award is returned based on time (as of now)

*/
template < int dim >
    double PoissonProblem < dim > ::reward_fn(Particle < n > & particle) {

        // Take normalized params to MueLu parameters
        double drop_tolerance     = std::exp(10.00*particle.pos[0]-10.00);
        double relaxation_damping = 1.00/8.00*tan(2.8*particle.pos[1]-1.4)+1.00;
        double sa_damping         = 1.00/8.00*tan(2.8*particle.pos[2]-1.4)+1.00;
        int coarse_max_size       = (int) (32*std::exp(8.20*particle.pos[3])+500.00);
        int max_levels            = 20;
        int relaxation_sweeps     = (int) (std::exp(8.2*particle.pos[4]-3.5)+1.0);
	
        Teuchos::ParameterList paramList;
        Teuchos::ParameterList smoothList;
	
        std::string multigrid_algorithm  = "sa";
        std::string smoother_type        = "RELAXATION";
	std::string verbosity            = "none";
        paramList.get("aggregation: drop tol", drop_tolerance);
        paramList.get("sa: damping factor", sa_damping);
        paramList.get("coarse: max size", coarse_max_size);
        paramList.get("max levels", max_levels);
	paramList.get("verbosity", verbosity);
	
	smoothList.get("relaxation: type", "Symmetric Gauss-Seidel");
        smoothList.get("relaxation: damping factor", relaxation_damping);
        smoothList.get("relaxation: sweeps", relaxation_sweeps);
        paramList.get("smoother: params", smoothList);

        lastparams = paramList;
        paramList.print();

        // Do some timing
        dealii::Timer timer0;
        timer0.start();

        // Create the preconditioner using calculated param list
        create_preconditioner(preconditioner, paramList);
        //   if (rank==0)
        //std::cout << "Created preconditioner successfully in reward function" << std::endl;

        // Solve the system to get the condition number, and density of preconditioner operator
        double k;
        double rho;
        int steps;
        bool success = false;
        // Set the max number of CG steps to be 1000 or 10*global best CG steps,
        // whichever comes first
        // (ensure code halts before the heat death of the universe)
        int cg_steps = 1000;
        int threshold = (3 * particle.best_CG_steps + (std::exp(-particle.best_CG_steps / 4) + 2));
        if (threshold < cg_steps)
            cg_steps = threshold;
        success = solve_system( & k, & rho, steps, cg_steps);
        double elapsed = timer0.stop();

        timer0.reset();
        if (success) {
            if (steps < particle.best_CG_steps)
                particle.best_CG_steps = steps;
            return -elapsed;
        }         return std::numeric_limits < double > ::lowest();
    }

/*

Function definition contained in the PoissonProblem struct

*/
template < int dim >
    void PoissonProblem < dim > ::run(int swarmsteps, int swarmiter, double alpha) {
        int nprocs;
        MPI_Comm_size(MPI_COMM_WORLD, & nprocs);
        MPI_Barrier(MPI_COMM_WORLD);

        // Output problem info
	
	// Insert matrix import here
	Epetra_CrsMatrix *A=NULL;
      	std::cout << mtx_filename << std::endl << std::flush;
	try { 	  int error = EpetraExt::MatrixMarketFileToCrsMatrix(mtx_filename.c_str(), Utilities::Trilinos::comm_self(), A, false, true);
	  std::cout << error << std::endl << std::flush;
	  std::remove(mtx_filename.c_str());
	  	}
	  catch (const std::exception&)  {
	  std::cout<< "\nException: could not read file." << std::flush;
	  std::exit(-1);
	}

	
	//	const auto mtx_wrapped = Teuchos::rcp(const_cast < Epetra_CrsMatrix * > (mtx), false);

	trilinos_system_matrix.reinit(*A);
	solution.reinit(trilinos_system_matrix.n());
	system_rhs.reinit(trilinos_system_matrix.n());
	solution.add(1);
	trilinos_system_matrix.vmult(system_rhs,solution);
	solution.reinit(trilinos_system_matrix.n());
	

        // Make particle swarm object
        const int num_particles = 10;

        ParticleSwarm < num_particles, n, PoissonProblem < dim >> ps(swarmiter, alpha*0.3, alpha*0.3, alpha*0.3, & PoissonProblem < dim > ::reward_fn, * this);

	//        if (rank == 0)
	//   std::cout << "Initializing parallel swarm..." << std::endl;

        MPI_Barrier(MPI_COMM_WORLD);
        for (int i = 0; i < swarmsteps; i++) {
            ps.swarm();

            //after swarm save all coords and f vals
	    //  if(rank == 0) {
            //    std::cout << "Swarm Done! Saving history..." << std::endl << std::flush;
            //}

            std::array < double, n > local_best_pos;
            double local_best_reward;
            // MPI
            // Send global bests to master rank
            if (rank != 0) {
                local_best_reward = ps.global_best_reward;
                local_best_pos = ps.global_best_pos;
                MPI_Send(&local_best_pos, n, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
                // Master rank receives best <- tempbest
                MPI_Send(&local_best_reward, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            }
            // Receive local bests from other ranks
            else {
                for (int j = 1; j < nprocs; j++) {
                    // Master rank receives best position <- tempbest_pos
                    MPI_Recv(&local_best_pos, n, MPI_DOUBLE, j, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    // Master rank receives best <- tempbest
                    MPI_Recv(&local_best_reward, 1, MPI_DOUBLE, j, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    if (local_best_reward > ps.global_best_reward) {
                        ps.global_best_pos = local_best_pos;
                        ps.global_best_reward = local_best_reward;
                    }
                }
            }
            // Sync
            MPI_Barrier(MPI_COMM_WORLD);

            // Broadcast best pos from master rank
            MPI_Bcast( & ps.global_best_pos, n, MPI_DOUBLE, 0,
                MPI_COMM_WORLD);
            // Sync
            MPI_Barrier(MPI_COMM_WORLD);
            // Broadcast best reward from master rank
            MPI_Bcast( & ps.global_best_reward, 1, MPI_DOUBLE, 0,
                MPI_COMM_WORLD);
            // Sync
            MPI_Barrier(MPI_COMM_WORLD);
        }
    

        if (rank == 0) {
            dealii::Timer timer;
            timer.start();
            reward_fn(ps.global_best_particle);
            double totelapsed = timer.stop();
            timer.reset();
	    //            std::cout << "Seconds elapsed during best precondition & solve: " << totelapsed << std::endl;
	    // std::cout << "CG steps for best precondition & solve: " << solver_control.last_step() << std::endl;
	    // std::cout << "Optimal MueLu parameters: " << std::endl;
            lastparams.print();
            timer.reset();
        }
	if (rank==0)
	  ps.save_history(mtx_filename.substr(0,24)+"../csv"+mtx_filename.substr(23,mtx_filename.length()-27) + ".csv");
    }


/*

Main function to instantiate a PoissonProblem. Also handles setting up
MPI communication for parallel processing.

*/
int main(int argc, char ** argv) {

    int rank;
    int size;
    int swarmsteps = 20;
    int swarmiter = 1;
    double alpha = 0.2;

    MPI_Init( & argc, & argv);
    MPI_Comm_rank(MPI_COMM_WORLD, & rank);
    MPI_Comm_size(MPI_COMM_WORLD, & size);

    std::string mtx_filename = argv[1];
    
    // Create Poisson Problem
    //    if (rank == 0)
      //        std::cout << "Running optimization with " << size << " processes" << std::endl;
    PoissonProblem < 3 > poisson(rank, mtx_filename);

    // Learning rate
    // Run Problem
    poisson.run(swarmsteps, swarmiter, alpha);

    MPI_Finalize();

    return 0;
}
