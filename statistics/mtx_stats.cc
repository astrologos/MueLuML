//trilinos includes

// Kokkos
#include <Kokkos_DefaultNode.hpp>

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
#include <TpetraExt_MatrixMatrix_def.hpp>

// STL header files
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <functional>
#include <string>
#include <stdio.h>
#include <ctime>

//other
#include <mpi.h>



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
		//declare constant sampling rows
		const int num_samples = 1601;

	    // Define scalar, local ordinal, global ordinal, and node type
	    typedef double ST;
	    typedef int    LO;
	    typedef int    GO;
	    typedef KokkosClassic::DefaultNode::DefaultNodeType NT;


	    // Define matrix, multivector, row map, column map, and muelu operator
	    // types from the previously defined scalar, local ordinal, global ordinal,
	    // and node type
	    typedef Tpetra::CrsMatrix<ST, LO, GO, NT> mtx_t;
		typedef Tpetra::Vector<ST, LO, GO, NT> vec_t;
	    typedef Tpetra::Operator<ST, LO, GO, NT> tpetra_op_t;
	    //typedef Tpetra::MultiVector<ST, LO, GO, NT> mv_t;
	    typedef Tpetra::Map<LO, GO, NT> row_map_t;
	    typedef Tpetra::Map<LO, GO, NT> col_map_t;
	    //typedef MueLu::TpetraOperator<ST, LO, GO, NT> muelu_op_t;

	    // Build communicator
	    Teuchos::RCP<const Teuchos::Comm<int>> t_comm(new Teuchos::MpiComm<int> (comm));

		//set rank
		int rank = t_comm->getRank();

	    // Get the matrix marker file name from passed args
	    std::string mtx_filename = argv[1];

	    if (rank == 0) {
	        cout << "Reading data from mtx file: " << mtx_filename << endl;
	    }


	    // Read the matrix market file to a Tpetra matrix
	    Teuchos::RCP<mtx_t> A;
	    try {
	        A = Tpetra::MatrixMarket::Reader<mtx_t>::readSparseFile(mtx_filename, t_comm);
	    } catch (...) {
	        cout << "Error reading file: " << mtx_filename << endl;
	    }

	    // Get the domain map of the read matrix
	    Teuchos::RCP<const row_map_t> map = A->getDomainMap();

	    //get number of elements (rows)
	    const size_t num_rows = map->getGlobalNumElements();

	    if (rank == 0) {
	        cout <<  "Generating sampling matrix with " << num_rows << " rows and " << num_samples  << " columns" << endl;
	    }

		//save the seed
		std::time_t seed = std::time(nullptr);

	    // Rank 0 is the only rank that will write sample matrix
	    if (rank == 0) {

			// Define a file stream to write sampling matrix to
			std::ofstream of;

	        of = std::ofstream("temp_sampling_mat.mtx", std::ios::out | std::ios::trunc);

			//write info for mtx parsing
			of << "%%MatrixMarket matrix coordinate real general\n";

			//write the initial header for the mtx file
			of << num_rows << " " << num_samples << " " << num_samples << "\n";

			//set the random seed
			std::srand(seed);

			for(int i = 0; i < num_samples; i++) {

				//generate a random number to sample a row
				int rand_col = std::rand() % num_rows + 1;

				//each line is the following format:
				//row col entry
				of << rand_col << " " << i+1 << " " << "1.0\n";

			}

			//close the file
			of.close();
	    }

		//load the file we just made and perform matrix multiplication
		Teuchos::RCP<mtx_t> B;

		if(rank == 0) {
			cout << "Reading in temp matrix file" << endl;
		}

		//try catch to load the matrix file we just generated
		try {
	        B = Tpetra::MatrixMarket::Reader<mtx_t>::readSparseFile("temp_sampling_mat.mtx", t_comm);
	    } catch (...) {
	        cout << "Error reading file: " << mtx_filename << endl;
	    }

		//declare C and set it to have the same map as A (required for Multiply)
		Teuchos::RCP<mtx_t> C;
		C = Teuchos::rcp(new mtx_t(B->getDomainMap(), num_rows));

		if(rank == 0) {
			cout << "Performing matrix multiplication B^T * A = C..." << endl;
		}

		//Use MatrixMatrix to fill C with sampled cols
		Tpetra::MatrixMatrix::Multiply(*B, true, *A, false, *C);

		//getting some basic stats to check if it worked
		Teuchos::RCP<const row_map_t> col_map_c = C->getRangeMap();
		Teuchos::RCP<const row_map_t> row_map_c = C->getDomainMap();

	    //get number of elements (rows)
	    const size_t num_rows_c = col_map_c->getGlobalNumElements();

		if(rank == 0) {
			cout << "Sampling matrix has " << num_rows_c << " rows and " << C->getGlobalNumCols() << " columns" << endl;
		}

		//remove temporary mtx file
		if(rank == 0) {
			if(std::remove("temp_sampling_mat.mtx") != 0) {
				cout << "Error deleting file" << endl;
			} else {
				cout << "Successfully deleted temp mtx file" << endl;
			}
		}

		//get the max local indices for this process
		//iterate through all local rows and calculate statistics
		const size_t max_local_idx = C->getNodeNumRows();
		//cout << "Rank " << rank << ": has " << max_local_idx << " rows to access!" << endl;
		//every process will open a file to write to, combine it later
		std::ofstream ofs;
		std::string fname;
		int trunc_len = mtx_filename.length() - 18;
		fname = mtx_filename.substr(14, trunc_len) + "_encoding_"  + std::to_string(rank) + "_" + std::to_string(seed)  + ".csv";
		cout << fname << endl;
        ofs = std::ofstream("data/encodings/" + fname, std::ios::out | std::ios::trunc);
		ofs << "mean, max, 1norm, 2norm, maxSD, 1normSD, 2normSD\n";

		for(LO i = 0; i < max_local_idx; i++) {
			size_t rowEntries = C->getNumEntriesInLocalRow(i);
			Teuchos::Array<ST> rowvals (rowEntries);
			Teuchos::Array<GO> rowinds (rowEntries);
			//cout << "Rank " << rank << ": "  << rowEntries << " total entries in local row " << i  << endl;

			//store the appropriate views
	        C->getLocalRowCopy(i, rowinds(), rowvals(), rowEntries);

	        //try storing the row as a vector
	        const LO indexBase = 0;
	        Teuchos::RCP<const row_map_t> vec_map = Teuchos::rcp(new row_map_t(rowEntries, indexBase, t_comm));
	        Tpetra::Vector<ST, GO, LO, NT> temp_row(vec_map, rowvals());

			//get the follow statistics:
			//max, min, mean, mode, sd, skewness, 2norm, 1 norm
	        //get the mean
	        const ST meanValue = temp_row.meanValue();
			const ST maxValue = temp_row.normInf();
			auto norm1 = temp_row.norm1();
			auto norm2 = temp_row.norm2();

			//do this all again with the sd vec
			vec_t x_mean (vec_map);
			x_mean.putScalar(meanValue);
			x_mean.update(1, temp_row, -1);
			x_mean.dot(x_mean);
			vec_t div_v (vec_map);
			div_v.putScalar(1 / rowEntries);
			x_mean.dot(div_v);

			const ST maxValue_sd = x_mean.normInf();
            auto norm1_sd = x_mean.norm1();
            auto norm2_sd = x_mean.norm2();

			//print all the shit
			ofs << meanValue << "," << norm1 << "," << norm2 << "," << maxValue << ",";
			ofs << maxValue_sd << "," << norm1_sd << "," << norm2_sd << "\n";

		}

		ofs.close();
	}

	MPI_Finalize();

    return EXIT_SUCCESS;

}

