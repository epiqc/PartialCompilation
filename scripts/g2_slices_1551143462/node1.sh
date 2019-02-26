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
ANGLE_ARGS2="$ASTART 95 $ASTOP 181 $ASTEP 5"

screen -AmdS q0 bash
screen -S q0 -X stuff $"$PREFIX $SSTART 0 $SSTOP 0 $ANGLE_ARGS\r"

screen -AmdS q1 bash
screen -S q1 -X stuff $"$PREFIX $SSTART 1 $SSTOP 1 $ANGLE_ARGS\r"

screen -AmdS q2 bash
screen -S q2 -X stuff $"$PREFIX $SSTART 2 $SSTOP 2 $ANGLE_ARGS\r"

screen -AmdS q3 bash
screen -S q3 -X stuff $"$PREFIX $SSTART 3 $SSTOP 3 $ANGLE_ARGS\r"

screen -AmdS q4 bash
screen -S q4 -X stuff $"$PREFIX $SSTART 4 $SSTOP 4 $ANGLE_ARGS\r"

screen -AmdS q5 bash
screen -S q5 -X stuff $"$PREFIX $SSTART 5 $SSTOP 5 $ANGLE_ARGS1\r"

screen -AmdS q6 bash
screen -S q6 -X stuff $"$PREFIX $SSTART 5 $SSTOP 5 $ANGLE_ARGS2\r"

screen -AmdS q7 bash
screen -S q7 -X stuff $"$PREFIX $SSTART 6 $SSTOP 6 $ANGLE_ARGS1\r"

screen -AmdS q8 bash
screen -S q8 -X stuff $"$PREFIX $SSTART 6 $SSTOP 6 $ANGLE_ARGS2\r"

screen -AmdS q9 bash
screen -S q9 -X stuff $"$PREFIX $SSTART 7 $SSTOP 7 $ANGLE_ARGS\r"

screen -AmdS q10 bash
screen -S q10 -X stuff $"$PREFIX $SSTART 8 $SSTOP 8 $ANGLE_ARGS\r"

screen -AmdS q11 bash
screen -S q11 -X stuff $"$PREFIX $SSTART 9 $SSTOP 9 $ANGLE_ARGS\r"

screen -AmdS q12 bash
screen -S q12 -X stuff $"$PREFIX $SSTART 10 $SSTOP 10 $ANGLE_ARGS\r"

screen -AmdS q13 bash
screen -S q13 -X stuff $"$PREFIX $SSTART 11 $SSTOP 11 $ANGLE_ARGS\r"
