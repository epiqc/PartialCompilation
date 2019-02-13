#!/bin/bash

# Get cli args.
THETA=$1

# Get a unique identifier for this run.
NAME="uccsd_slice"
TIME=$(date +'%s')
DATE=$(date)
ID="$NAME_$TIME"

# Create a directory in our shared folder to write data to.
PDIR="/project/ftchong/pc"
DIR="$PDIR/$ID"
mkdir -p $DIR

# Tag the directory with information.
printf "THETA: $THETA\nDATE: $DATE" > "$DIR/info.log"

# Activate python3 on cpu.
conda activate

# Write unitaries to file.

# Switch to python2 on cpu.
conda deactivate
module load 

# Run qoc.



