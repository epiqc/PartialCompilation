#!/bin/bash

#SBATCH --job-name=strn8ErdosRenyi
#SBATCH --output=ErdosRenyi/arrayJob_%A_%a.out
#SBATCH --error=ErdosRenyi/arrayJob_%A_%a.err
#SBATCH --array=0-223
#SBATCH --time=18:00:00
#SBATCH --partition=broadwl
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=16000

# load the required modules
module unload Anaconda3
module load Anaconda3/5.0.1
module unload gcc
module load cuda/9.0

# load virtual environment with quantum optimal control and qiskit toolkit
source activate QOC_qiskit

echo "Running Array Job: " $SLURM_ARRAY_TASK_ID
python strict_ErdosRenyi_n8.py $SLURM_ARRAY_TASK_ID
