#!/bin/bash
#SBATCH --job-name=nah_b_h
#SBATCH --array=0-123
#SBATCH --time=1:00:00
#SBATCH --partition=broadwl
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1000
#SBATCH --mem=20000

# Run hyperparameter optimization on all of the blocked UCCSD NaH slices.

module load Anaconda3/5.1.0
. /software/Anaconda3-5.0.1-el7-x86_64/etc/profile.d/conda.sh
module unload gcc
module load cuda/9.0

conda activate fqc
python ~/repos/fqc/scripts/uccsd/uccsd_hpo_blocking.py --molecule NaH --index $SLURM_ARRAY_TASK_ID --core-count 5
