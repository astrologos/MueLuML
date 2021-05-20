#!/bin/bash

#PBS -A CMDA_Cap_18
#PBS -W group_list=newriver
#PBS -q largemem_q
#PBS -l nodes=1
#PBS -l walltime=08:00:00

source ./loadmods.sh
module load cmake
mpirun -np $PBS_NP ./single_optimal_parallel
