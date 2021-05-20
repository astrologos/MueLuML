// Clean up compilation
#pragma GCC diagnostic ignored "-Wunused-parameter"

// ---- Deal.II Includes ----
# define protected public // Just a hack -- only works in deal.ii > 9.1.1?
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
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/numerics/derivative_approximation.h>

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
#define _USE_MATH_DEFINES

// ---- Trilinos Includes ----
#include <Epetra_CrsMatrix.h>
#include <EpetraExt_RowMatrixOut.h>


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

std::uniform_int_distribution <unsigned int> i1{
    0,
    2147483647
};
std::uniform_real_distribution < > r1{
    0.0,
    1.0
};

template < int dim >
class Stretch {
    public:
        Stretch(const double factor):factor(factor) {}
    Point < dim > p;
    Point < dim > operator() (const Point<dim> p) const {
        Point <dim> q=p;
        q[0]+= std::abs(factor*sin(q[1]+factor*M_PI) + factor*sin(q[2]*factor+M_PI));
        return q;
    }
    private:
        const double factor;
};

template < int dim >
class Waves {
    public:
        Waves (const double minwl, Triangulation<dim,dim> &triangulation, double alpha, double beta, double gamma, double delta, double epsilon, double xi, double xscale, double yscale, double zscale):
       
     minwl(minwl),triangulation(triangulation),
     alpha(alpha),beta(beta),gamma(gamma),delta(delta),epsilon(epsilon),xi(xi),
     xscale(xscale),yscale(yscale),zscale(zscale) {}
    Point < dim > p;
    Point < dim > operator() (const Point<dim> p) const {
    
        double maxfreq=0.001/minwl;
        double maxmag=0.01/minwl;
        Point <dim> q=p;
        q[0] += yscale*sin(beta*maxfreq*q[1] + epsilon*M_PI) + 
                zscale*sin(gamma*maxfreq*q[2] + xi*M_PI);
        q[1] += xscale*sin(alpha*maxfreq*q[0] + delta*M_PI) + 
                zscale*sin(gamma*maxfreq*q[2] + xi*M_PI);

        return q;
    }
    private:
        Triangulation<dim,dim> &triangulation;
        const double minwl;
        double alpha;
        double beta;
        double gamma;
        double delta;
        double epsilon;
        double xi;
        double xscale;
        double yscale;
        double zscale;
};


template < int dim >
    struct PoissonProblem {

        // Functions
        PoissonProblem(int p_deg);
        void run();
        void make_grid();
        void setup_system();
        void assemble_system();
        Point < dim > stretch(const Point < dim > &p);

        // Deal II system info
        int p_deg;
        Triangulation < dim, dim > triangulation;
        FE_Q < dim > fe;
        DoFHandler < dim > dof_handler;
        SparsityPattern sparsity_pattern;
        SparseMatrix < double > system_matrix;
        Vector < double > solution;
        Vector < double > system_rhs;

        // TrilinosWrappers
        TrilinosWrappers::SparseMatrix trilinos_system_matrix;
    };

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
    PoissonProblem < dim > ::PoissonProblem(int p_deg): fe(p_deg), dof_handler(triangulation) {}


template < int dim >
    void PoissonProblem < dim > ::make_grid() {
        // Select grid here
        switch (i1(gen)%6)
        {
            case 0: GridGenerator::hyper_cube(triangulation);
                break;
            case 1: GridGenerator::cheese(triangulation, std::vector<unsigned int> {1+i1(gen)%9,1+i1(gen)%9,1+i1(gen)%9});
                break;
            case 2: GridGenerator::plate_with_a_hole(triangulation);
                break;
            case 3: GridGenerator::channel_with_cylinder(triangulation);
                break;
            case 4: GridGenerator::enclosed_hyper_cube(triangulation, (0.5+r1(gen))/2);
                break;
            case 5: GridGenerator::hyper_ball(triangulation);
                break;
            case 6: GridGenerator::quarter_hyper_ball(triangulation);
                break;
            case 7: GridGenerator::half_hyper_ball(triangulation);
                break;
            case 8: GridGenerator::cylinder(triangulation);
                break;
            case 9: GridGenerator::truncated_cone(triangulation, (0.5+r1(gen))/2);
                break;
            case 10: GridGenerator::hyper_L(triangulation);
                break;
            case 11: GridGenerator::hyper_cube_slit(triangulation);
                break;
            case 12: GridGenerator::hyper_shell(triangulation, Point<3>(), (r1(gen)+0.3)/2, (r1(gen)+1.5)/2);
                break;
            case 13: GridGenerator::half_hyper_shell(triangulation, Point<3>(), (r1(gen)+0.5)/2, (r1(gen)+2)/2);
                break;
            case 14: GridGenerator::quarter_hyper_shell(triangulation, Point<3>(), (r1(gen)+0.5)/2, (r1(gen)+2)/2);
                break;
            case 15: GridGenerator::cylinder_shell(triangulation, 1.5, (r1(gen)+0.5)/2, (r1(gen)+2)/2);
                break;
            case 16: GridGenerator::torus(triangulation, r1(gen)+1.2, r1(gen)+0.5);
                break;
            case 17: GridGenerator::concentric_hyper_shells(triangulation, Point<3>());
                break;
            default: GridGenerator::moebius(triangulation,20,i1(gen)%6, r1(gen)+1.2, r1(gen)+0.5);
                break;
         }   
         
        // Rotate Grid Here
        GridTools::rotate(M_PI*r1(gen), i1(gen)%3, triangulation);
        GridTools::rotate(M_PI*r1(gen), i1(gen)%3, triangulation);
        while (triangulation.n_vertices() < 40000)
            triangulation.refine_global();                
        for (int j = 3; j < 5 + i1(gen)%3; j++) {  // this line to be changed
	  //            if(GridTools::diameter(triangulation) > 10)
          //      GridTools::scale(0.5, triangulation);
            if(triangulation.n_vertices() > 1000000) break;
            
            double alpha = r1(gen);
            double beta = r1(gen);
            double gamma = r1(gen);
            double delta = r1(gen);
            double epsilon = r1(gen);
            double xi = r1(gen);
            double mcd = GridTools::minimal_cell_diameter(triangulation);
            double maxmag=(1/j)*0.00001/mcd;
            double xscale=maxmag*r1(gen);
            double yscale=maxmag*r1(gen);
            double zscale=maxmag*r1(gen);
                // Jiggle &\ twist grid here
	    //                GridTools::transform (Waves < dim > (mcd, triangulation, alpha, beta, gamma, delta, epsilon, xi, xscale, yscale, zscale), triangulation);
                
                // Stretch grid here
                GridTools::transform (Stretch < dim > ((1/j)*3+r1(gen)), triangulation);
                triangulation.refine_global();
            }
            
        std::cout << triangulation.n_vertices() << std::endl;
	std::cout << "\nDone with mesh configuration!" << std::endl;
    
}


template < int dim >
    void PoissonProblem < dim > ::setup_system() {
        dof_handler.distribute_dofs(fe);
        DynamicSparsityPattern dsp(dof_handler.n_dofs());
        DoFTools::make_sparsity_pattern(dof_handler, dsp);
        sparsity_pattern.copy_from(dsp);
        system_matrix.reinit(sparsity_pattern);
        solution.reinit(dof_handler.n_dofs());
        system_rhs.reinit(dof_handler.n_dofs());
    }

template < int dim >
    void PoissonProblem < dim > ::assemble_system() {
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
        
        
    }


template < int dim >
    void PoissonProblem < dim > ::run() {
  std::cout << "Making grid...\n" << std::flush;
  make_grid();

  std::cout << "Setting up system...\n" << std::flush;
        setup_system();
	std::cout << "Done setting up system!\n" << std::flush;
	std::cout << "Assembling system...\n" << std::flush;
        assemble_system();
	std::cout << "Done assembling system!\n" << std::flush;

        std::string num = std::to_string(i1(gen));
        std:: string uuids = "./matrices/" + num + ".mtx";
        const char* uuidcc = uuids.c_str();
	std::cout << "Writing system...\n" << std::flush;
        int err = EpetraExt::RowMatrixToMatrixMarketFile( uuidcc,
                trilinos_system_matrix.trilinos_matrix());
	std::cout << err << std::endl;
	
	//        std::ofstream mesh_out("../meshgen/" + num + ".vtu");
	//        GridOut().write_vtu(triangulation, mesh_out);
	//        mesh_out.close();

        
}



int main(int argc, char ** argv) {
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    int p_deg = (i1(gen)%3)+1;
    PoissonProblem < 3 > poisson(p_deg);
    poisson.run();
    return 0;
}
