// Trilinos header files


// Kokkos
#include <Kokkos_DefaultNode.hpp>

// MueLu
#include <MueLu_CreateTpetraPreconditioner.hpp>
#include <MueLu_TpetraOperator_fwd.hpp>

// Teuchos
#include <Teuchos_Comm.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCPDecl.hpp>

// Tpetra
#include <MatrixMarket_Tpetra.hpp>
#include <Tpetra_Core.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Teuchos_DefaultMpiComm.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_MultiVector.hpp>

// Belos
#include <BelosBlockCGSolMgr.hpp>
#include <BelosIteration.hpp>
#include <BelosMultiVec.hpp>
#include <BelosMultiVecTraits.hpp>
#include <BelosOperatorTraits.hpp>
#include <Belos_TpetraAdapter_MP_Vector.hpp>

// STL header files
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <functional>
#include <string>

// Other header files
#include <mpi.h>

// User-defined header files
#include "particle_swarm_t.h"

int main(int argc, char** argv) {
    
    // Avoid excessive typing
    using std::cout;
    using std::endl;

    // Check that the correct number of arguments have been passed
    if (argc < 2) {
        cout << "Usage ./mtx_optimal <matrix market file name>" << endl;
        return EXIT_FAILURE;
    }
    
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    
    {
        // ps hyperparams
        const int n_dims = 2;
        const int n_particles = 20;
        const int max_steps = 20;
        const double omega = 0.25;
        const double phi_g = 0.4;
        const double phi_p = 0.25;
        const double r = 0.1;

        // Define scalar, local ordinal, global ordinal, and node type 
        typedef double ST; 
        typedef int    LO; 
        typedef int    GO; 
        typedef KokkosClassic::DefaultNode::DefaultNodeType NT; 
        
        // Define matrix, multivector, row map, column map, and muelu operator
        // types from the previously defined scalar, local ordinal, global ordinal,
        // and node type
        typedef Tpetra::CrsMatrix<ST, LO, GO, NT> mtx_t;
        typedef Tpetra::Operator<ST, LO, GO, NT> tpetra_op_t;
        typedef Tpetra::MultiVector<ST, LO, GO, NT> mv_t;
        typedef Tpetra::Map<LO, GO, NT> row_map_t;
        typedef Tpetra::Map<LO, GO, NT> col_map_t;
        typedef MueLu::TpetraOperator<ST, LO, GO, NT> muelu_op_t;

        // Build communicator
        Teuchos::RCP<const Teuchos::Comm<int>> t_comm(new Teuchos::MpiComm<int> (comm));
        
        // Get the matrix marker file name from passed args
        std::string mtx_filename = argv[1];
        
        if (t_comm->getRank() == 0) {
            cout << "Finding optimal parameters for matrix market file: " << mtx_filename.substr(0,mtx_filename.length()-4) << endl;
        }
        

        // Read the matrix market file to a Tpetra matrix
        Teuchos::RCP<mtx_t> A;
        try {
            A = Tpetra::MatrixMarket::Reader<mtx_t>::readSparseFile(mtx_filename, t_comm);
        } catch (...) {
            cout << "Error reading file: " << mtx_filename << endl;
        }

        // Define reward function as a lambda
        std::function<double(Particle<n_dims>&)> reward_fn = [&](Particle<n_dims>& p) {
            
            // Determine the muelu parameters

      
            double drop_tol = std::exp((p.pos[0] * 5.00) - 5.05);
            double sa_damping = std::exp((p.pos[1] * 5.00) - 4.33);
            double t_elapsed;
            double t_start = MPI_Wtime();
            
            // Set param list options
            Teuchos::ParameterList param_list;

            // Defaults
            param_list.get("aggregation: type", "uncoupled");
            param_list.get("verbosity", "high");
            param_list.get("max levels", 10);
            param_list.get("smoother: pre or post", "both");

            // ps params
            param_list.get("aggregation: drop tol", drop_tol);
            param_list.get("multigrid algorithm", "sa");
            param_list.get("sa: damping factor", sa_damping);

            // Create a MueLu operator that will represent the preconditioning
            // operator from the given parameters
            try {
            Teuchos::RCP<muelu_op_t> M = MueLu::CreateTpetraPreconditioner(A, param_list);

            // Get the domain map of the read matrix
            Teuchos::RCP<const row_map_t> map = A->getDomainMap();

            // Create a RHS and an initial guess
            int n_rhs = 1;
            
            // Create multivectors representing solution and initial guess
            Teuchos::RCP<mv_t> b;
            Teuchos::RCP<mv_t> x;
            x = Teuchos::rcp(new mv_t(map, n_rhs));
            b = Teuchos::rcp(new mv_t(map, n_rhs));
            Belos::MultiVecTraits<ST, mv_t>::MvRandom(*x);
            Belos::OperatorTraits<ST, mv_t, tpetra_op_t>::Apply(*A, *x, *b);
            Belos::MultiVecTraits<ST, mv_t>::MvInit(*x, 0.0);

            // Create a linear problem
            Teuchos::RCP<Belos::LinearProblem<ST, mv_t, tpetra_op_t>> problem;
            problem = Teuchos::rcp(new Belos::LinearProblem<ST, mv_t, tpetra_op_t>(A, x, b));
            
            // Set the linear problem to the left preconditioner
            problem->setLeftPrec(M);
            bool set = problem->setProblem();

            // Parameters for Belos solver
            Teuchos::ParameterList belos_list_raw;
            belos_list_raw.set("Block Size", t_comm->getSize() );
            belos_list_raw.set("Maximum Iterations", 100);
            belos_list_raw.set("Convergence Tolerance", 1e-10);
            belos_list_raw.set("Output Frequency", 1 );
            belos_list_raw.set("Verbosity", Belos::TimingDetails + Belos::FinalSummary);
            Teuchos::RCP<Teuchos::ParameterList> belos_list = Teuchos::rcp(&belos_list_raw, false);

            // Create Belos solver
            Belos::BlockCGSolMgr<ST, mv_t, tpetra_op_t, true> solver(problem, belos_list);

            // Solve system
            Belos::ReturnType ret = solver.solve();
            bool success;
            if (ret != Belos::Converged) {
                success = false;
                std::cout << std::endl << "ERROR:  Belos did not converge!" << std::endl;
            } else {
                success = true;
                std::cout << std::endl << "SUCCESS:  Belos converged!" << std::endl;
            }

            double t_end = MPI_Wtime();
            
            // Return the total elapsed time to precondition and solve
            t_elapsed = t_start - t_end; }
            
            catch (const std::exception &exc) {std::cerr << exc.what();return -1000.00;}
            
            return t_elapsed;
        };
        // Random number generators
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> d1(0.0, 1.0);
        std::uniform_real_distribution<double> d2(-1.0, 1.0);

        // Define a file stream to write particle swarm output to
        std::ofstream ofs;

        ofs = std::ofstream("data/ps/" + 
                            mtx_filename.substr(0,mtx_filename.length()-4) + 
                            "_ps.csv" + 
                            std::to_string(t_comm->getRank()), std::ios::out | std::ios::trunc);

        // Particle Swarm object
        ParticleSwarm<n_dims, n_particles> ps(
                max_steps,
                omega,
                phi_g,
                phi_p,
                r,
                reward_fn,
                ofs,
                gen,
                d1,
                d2,
                comm
        );
        
        // Run particle swarm to determine optimal parameters
        ps.swarm();

        // Close the file stream to which the particle history was being written
        ofs.close();

    }
        
    MPI_Finalize();

    return EXIT_SUCCESS;
}
