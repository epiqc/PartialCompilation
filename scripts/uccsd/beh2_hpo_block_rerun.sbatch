#!/bin/bash
#SBATCH --job-name=beh2_b_h
#SBATCH --array=96,98,100,101,102,104,105,106,107,108,109,110,111,112,118,123
#SBATCH --time=1:00:00
#SBATCH --partition=broadwl
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8000

# Run hyperparameter optimization on specific, blocked UCCSD BeH2 slices.

module load Anaconda3/5.1.0
. /software/Anaconda3-5.0.1-el7-x86_64/etc/profile.d/conda.sh
module unload gcc
module load cuda/9.0

conda activate fqc
python ~/repos/fqc/scripts/uccsd/uccsd_hpo_blocking.py --molecule BeH2 --index $SLURM_ARRAY_TASK_ID --core-count 5
