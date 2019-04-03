#!/bin/bash
#SBATCH --job-name=h2_slice_hpo
#SBATCH --array=0-2
#SBATCH --time=01:00:00
#SBATCH --partition=broadwl
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=6000

# Run hyperparameter optimization on all of the UCCSD H2 slices.

module load Anaconda3/5.1.0
. /software/Anaconda3-5.0.1-el7-x86_64/etc/profile.d/conda.sh
module unload gcc
module load cuda/9.0

conda activate fqc
python ~/repos/fqc/scripts/uccsd/uccsd_hpo.py --molecule H2 --slice-index $SLURM_ARRAY_TASK_ID --core-count 5
