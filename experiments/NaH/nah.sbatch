#!/bin/bash

#SBATCH --job-name=NaH
#SBATCH --output=arrayJob_%A_%a.out
#SBATCH --error=arrayJob_%A_%a.err
#SBATCH --array=310,7,260,172,95,48,86,318,306,104,54,182,292,158,82,84,21,31,197,170,208,96,174,89,210,122,240,102,176,308,264,248,156,88,183,207,109,118,154,282,119,280,242,302,290,40,314,312,76,74,72,34,80,64,152,294,256,136,1,66,128,224,194,2,212,262,304,146,230,16
#SBATCH --time=09:00:00
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
