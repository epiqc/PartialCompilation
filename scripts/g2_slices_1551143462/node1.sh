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

screen -AmdS 1 bash
screen -S 1 -X stuff $"$PREFIX $SSTART 1 $SSTOP 1 $ANGLE_ARGS\r"

screen -AmdS 2 bash
screen -S 2 -X stuff $"$PREFIX $SSTART 2 $SSTOP 2 $ANGLE_ARGS\r"

screen -AmdS 3 bash
screen -S 3 -X stuff $"$PREFIX $SSTART 3 $SSTOP 3 $ANGLE_ARGS\r"

screen -AmdS 4 bash
screen -S 4 -X stuff $"$PREFIX $SSTART 4 $SSTOP 4 $ANGLE_ARGS\r"

screen -AmdS 5 bash
screen -S 5 -X stuff $"$PREFIX $SSTART 5 $SSTOP 5 $ANGLE_ARGS1\r"

screen -AmdS 6 bash
screen -S 6 -X stuff $"$PREFIX $SSTART 5 $SSTOP 5 $ANGLE_ARGS2\r"

screen -AmdS 7 bash
screen -S 7 -X stuff $"$PREFIX $SSTART 6 $SSTOP 6 $ANGLE_ARGS1\r"

screen -AmdS 8 bash
screen -S 8 -X stuff $"$PREFIX $SSTART 6 $SSTOP 6 $ANGLE_ARGS2\r"

screen -AmdS 9 bash
screen -S 9 -X stuff $"$PREFIX $SSTART 7 $SSTOP 7 $ANGLE_ARGS\r"

screen -AmdS 10 bash
screen -S 10 -X stuff $"$PREFIX $SSTART 8 $SSTOP 8 $ANGLE_ARGS\r"

screen -AmdS 11 bash
screen -S 11 -X stuff $"$PREFIX $SSTART 9 $SSTOP 9 $ANGLE_ARGS\r"

screen -AmdS 12 bash
screen -S 12 -X stuff $"$PREFIX $SSTART 10 $SSTOP 10 $ANGLE_ARGS\r"

screen -AmdS 13 bash
screen -S 13 -X stuff $"$PREFIX $SSTART 11 $SSTOP 11 $ANGLE_ARGS\r"
