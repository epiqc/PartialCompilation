#!/bin/bash
#SBATCH --job-name=lih_full_hpo
#SBATCH --time=12:00:00
#SBATCH --partition=broadwl
#SBATCH --exclusive

# Run hyperparameter optimization the full UCCSD LiH circuit.

module load Anaconda3/5.1.0
. /software/Anaconda3-5.0.1-el7-x86_64/etc/profile.d/conda.sh
module unload gcc
module load cuda/9.0

conda activate fqc
python ~/repos/fqc/scripts/uccsd/uccsd_time.py --molecule LiH
