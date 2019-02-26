#!/bin/bash
PATH_TO_SCRIPT=/home/tcpropson/repos/fqc/scripts/uccsd_slice_qoc.py
PREFIX="cpu_conda && python $PATH_TO_SCRIPT"
ASTART="--angle-start"
ASTOP="--angle-stop"
ASTEP="--angle-step"
SSTART="--slice-start"
SSTOP="--slice-stop"

ANGLE_ARGS="$ASTART 0 $ASTOP 181 $ASTEP 5"
ANGLE_ARGS1="$ASTART 0 $ASTOP 91 $ASTEP 5"
ANGLE_ARGS2="$ASTART 95 $ASTOP 95 $ASTEP 5"

screen -AmdS 0 bash
screen -S 0 -X stuff $"$PREFIX $SSTART 0 $SSTOP 0 $ANGLE_ARGS\r"
