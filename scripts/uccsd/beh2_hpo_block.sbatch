#!/bin/bash
#SBATCH --job-name=beh2_b_h
#SBATCH --array=0-129
#SBATCH --time=1:00:00
#SBATCH --partition=broadwl
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3000

# Run hyperparameter optimization on all of the blocked UCCSD BeH2 slices.

module load Anaconda3/5.1.0
. /software/Anaconda3-5.0.1-el7-x86_64/etc/profile.d/conda.sh
module unload gcc
module load cuda/9.0

conda activate fqc
python ~/repos/fqc/scripts/uccsd/uccsd_hpo_blocking.py --molecule BeH2 --index $SLURM_ARRAY_TASK_ID --core-count 5
