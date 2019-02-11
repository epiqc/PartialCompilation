#!/bin/bash
# a job submission script to submit a VQE job to the broadwl partition on Midway2

# set the job name to hello-uccsdi-qoc
#SBATCH --job-name=hello-uccsd-qoc

# send output to hello-uccsd-qoc.out
#SBATCH --output=hello-uccsd-qoc.out

# receive an email when job starts, ends, and fails
#SBATCH --mail-type=BEGIN,END,FAIL

# this job requests 1 core. Cores can be selected from various nodes.
#SBATCH --ntasks=1

# there are a few partitions on Midway2 and it is important to specify which
# partition you want to run your job on. Not having the following line, the 
# sandby partition on Midway1 will be selected as the default partition
#SBATCH --partition=broadwl

# this job needs 32GB of memory per CPU
#SBATCH --mem-per-cpu=32000

# load the required modules
module load Anaconda3/5.0.1
module unload gcc
module load cuda/9.0

# load virtual environment with quantum optimal control and qiskit toolkit
source activate QOC_qiskit

# Check if python 3 is configured
python -V

# Run the process with Python 3. 
pwd
python demo_uccsd_qoc.py
