#!/bin/bash
#SBATCH --job-name=n6r3_p1_hpo
#SBATCH --array=0-7
#SBATCH --time=1:00:00
#SBATCH --partition=broadwl
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1000
#SBATCH --mem=20000

# Run hyperparameter optimization on all of the blocked n6r3 slices.

module load Anaconda3/5.1.0
. /software/Anaconda3-5.0.1-el7-x86_64/etc/profile.d/conda.sh
module unload gcc
module load cuda/9.0

conda activate fqc
python ~/repos/fqc/scripts/qaoa/qaoa_hpo.py --core-count 5 --index $SLURM_ARRAY_TASK_ID --id n6r3 -p 1
