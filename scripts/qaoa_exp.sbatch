#!/bin/bash
# a job submission script to submit a VQE job to the broadwl partition on Midway2

# set the job name to qaoa-3reg-qoc-mpi-4-16
#SBATCH --job-name=qaoa-3reg-qoc-mpi-4-16

# send output to qaoa-3reg-qoc-mpi-4-16.out
#SBATCH --output=qaoa-3reg-qoc-mpi-4-16.out

# receive an email when job starts, ends, and fails
#SBATCH --mail-type=BEGIN,END,FAIL

# this job requests 1 core. Cores can be selected from various nodes.
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=16

# there are a few partitions on Midway2 and it is important to specify which
# partition you want to run your job on. Not having the following line, the 
# sandby partition on Midway1 will be selected as the default partition
#SBATCH --partition=broadwl

# this job needs 32GB of memory per CPU
#SBATCH --mem-per-cpu=2000

# load the required modules
module load Anaconda3/5.0.1
module unload gcc
module load cuda/9.0
module load openmpi

# load virtual environment with quantum optimal control and qiskit toolkit
source activate QOC_qiskit

# Run the process with Python 3. 
pwd
mpirun python experiments/qaoa_exp.py
