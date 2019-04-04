#!/bin/bash
#SBATCH --job-name=nah_t_b
#SBATCH --array=57,58,59,60,98,102,105,110
#SBATCH --time=1:00:00
#SBATCH --partition=broadwl
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1000
#SBATCH --mem=20000

# Run time binary search on UCCSD NaH slices.

module load Anaconda3/5.1.0
. /software/Anaconda3-5.0.1-el7-x86_64/etc/profile.d/conda.sh
module unload gcc
module load cuda/9.0

conda activate fqc
python ~/repos/fqc/scripts/uccsd/uccsd_time_blocking.py --molecule NaH --rz-index $SLURM_ARRAY_TASK_ID
