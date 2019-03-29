#!/bin/bash
#SBATCH --job-name=lih_hpo_all
#SBATCH --array=0-7
#SBATCH --time=08:00:00
#SBATCH --partition=broadwl
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=5000

# Run hyperparameter optimization on all of the UCCSD LiH slices.

module load Anaconda3/5.1.0
. /software/Anaconda3-5.0.1-el7-x86_64/etc/profile.d/conda.sh
module unload gcc
module load cuda/9.0
module load redis

conda activate fqc
python ~/repos/fqc/scripts/uccsd_slice/uccsd_lih_slice_hpo.py --slice-index $SLURM_ARRAY_TASK_ID --core-count 5
