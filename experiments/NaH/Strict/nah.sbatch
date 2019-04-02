#!/bin/bash

#SBATCH --job-name=NaHStrict
#SBATCH --output=arrayJob_%A_%a.out
#SBATCH --error=arrayJob_%A_%a.err
#SBATCH --array=0-320
#SBATCH --time=9:00:00
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

echo "Running Block: " $SLURM_ARRAY_TASK_ID
python nah.py $SLURM_ARRAY_TASK_ID
