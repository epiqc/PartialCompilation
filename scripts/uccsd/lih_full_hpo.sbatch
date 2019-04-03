#!/bin/bash
#SBATCH --job-name=lih_full_hpo
#SBATCH --time=08:00:00
#SBATCH --partition=broadwl
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8000

# Run hyperparameter optimization the full UCCSD LiH circuit.

module load Anaconda3/5.1.0
. /software/Anaconda3-5.0.1-el7-x86_64/etc/profile.d/conda.sh
module unload gcc
module load cuda/9.0

conda activate fqc
python ~/repos/fqc/scripts/uccsd/uccsd_hpo.py --molecule LiH --core-count 5
