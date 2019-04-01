#!/bin/bash
#SBATCH --job-name=h2_full_hpo
#SBATCH --time=01:30:00
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
python ~/repos/fqc/scripts/uccsd/uccsd_hpo.py --molecule H2 --core-count 5
