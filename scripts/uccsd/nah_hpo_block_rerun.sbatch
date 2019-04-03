#!/bin/bash
#SBATCH --job-name=nah_b_h
#SBATCH --array=16,27,49,50,52,56,58,59,60,62,63,64,65,66,67,69,70,71,73,74,78,79,80,81,82,83,84,86,87,88,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,110,111,112,113,114,115,116,117,118,119,120,121
#SBATCH --time=1:00:00
#SBATCH --partition=broadwl
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10000

# Run hyperparameter optimization specific blocked UCCSD NaH slices.

module load Anaconda3/5.1.0
. /software/Anaconda3-5.0.1-el7-x86_64/etc/profile.d/conda.sh
module unload gcc
module load cuda/9.0

conda activate fqc
python ~/repos/fqc/scripts/uccsd/uccsd_hpo_blocking.py --molecule NaH --index $SLURM_ARRAY_TASK_ID --core-count 5
