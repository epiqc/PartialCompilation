#!/bin/bash
#SBATCH --job-name=beh2_slice_hpo
#SBATCH --array=0-25
#SBATCH --time=24:00:00
#SBATCH --partition=broadwl
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=9000
#SBATCH --exclusive

# Run hyperparameter optimization on all of the UCCSD LiH slices.

module load Anaconda3/5.1.0
. /software/Anaconda3-5.0.1-el7-x86_64/etc/profile.d/conda.sh
module unload gcc
module load cuda/9.0

conda activate fqc
python ~/repos/fqc/scripts/uccsd/uccsd_hpo.py --molecule BeH2 --slice-index $SLURM_ARRAY_TASK_ID --core-count 5
