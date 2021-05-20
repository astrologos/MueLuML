for matrix in /scratch/ajack/backupmatrices/*
do
    mpirun -np 50 ./optimize $matrix
done
