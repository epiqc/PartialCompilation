#!/bin/bash
#SBATCH --job-name=lhr7
#SBATCH --time=36:00:00
#SBATCH --partition=broadwl
#SBATCH --exclusive
#SBATCH --ntasks=14
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4500

# Run random search on the slice 0 of the UCCSD LiH circuit.

module load mpi4py/3.0.1a0_py3
. /software/Anaconda3-5.0.1-el7-x86_64/etc/profile.d/conda.sh
module unload gcc
module load cuda/9.0

conda activate fqc
mpirun -n 1 -ppn 14 python ~/repos/fqc/scripts/uccsd/uccsd_lih_slice_hpo_rand.py --slice-index 7 --core-count 14
