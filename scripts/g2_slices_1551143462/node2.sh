#!/bin/bash
PATH_TO_SCRIPT="/home/tcpropson/repos/fqc/scripts/uccsd_slice_qoc.py"
PREFIX="cpu_conda && python $PATH_TO_SCRIPT"
ASTART="--angle-start"
ASTOP="--angle-stop"
ASTEP="--angle-step"
SSTART="--slice-start"
SSTOP="--slice-stop"

ANGLE_ARGS="$ASTART 0 $ASTOP 181 $ASTEP 5"
ANGLE_ARGS1="$ASTART 0 $ASTOP 91 $ASTEP 5"
ANGLE_ARGS2="$ASTART 95 $ASTOP 95 $ASTEP 5"

screen -AmdS 14 bash
screen -S 14 -X stuff $"$PREFIX $SSTART 12 $SSTOP 12 $ANGLE_ARGS1\r"

screen -AmdS 15 bash
screen -S 15 -X stuff $"$PREFIX $SSTART 12 $SSTOP 12 $ANGLE_ARGS2\r"

screen -AmdS 16 bash
screen -S 16 -X stuff $"$PREFIX $SSTART 13 $SSTOP 13 $ANGLE_ARGS1\r"

screen -AmdS 17 bash
screen -S 17 -X stuff $"$PREFIX $SSTART 13 $SSTOP 13 $ANGLE_ARGS2\r"

screen -AmdS 18 bash
screen -S 18 -X stuff $"$PREFIX $SSTART 14 $SSTOP 14 $ANGLE_ARGS\r"

screen -AmdS 19 bash
screen -S 19 -X stuff $"$PREFIX $SSTART 15 $SSTOP 15 $ANGLE_ARGS\r"

screen -AmdS 20 bash
screen -S 20 -X stuff $"$PREFIX $SSTART 16 $SSTOP 16 $ANGLE_ARGS\r"

screen -AmdS 21 bash
screen -S 21 -X stuff $"$PREFIX $SSTART 17 $SSTOP 17 $ANGLE_ARGS\r"

screen -AmdS 22 bash
screen -S 22 -X stuff $"$PREFIX $SSTART 18 $SSTOP 18 $ANGLE_ARGS\r"

screen -AmdS 23 bash
screen -S 23 -X stuff $"$PREFIX $SSTART 19 $SSTOP 19 $ANGLE_ARGS\r"

screen -AmdS 24 bash
screen -S 24 -X stuff $"$PREFIX $SSTART 20 $SSTOP 20 $ANGLE_ARGS1\r"

screen -AmdS 25 bash
screen -S 25 -X stuff $"$PREFIX $SSTART 20 $SSTOP 20 $ANGLE_ARGS2\r"

screen -AmdS 26 bash
screen -S 26 -X stuff $"$PREFIX $SSTART 21 $SSTOP 21 $ANGLE_ARGS\r"

screen -AmdS 27 bash
screen -S 27 -X stuff $"$PREFIX $SSTART 22 $SSTOP 22 $ANGLE_ARGS\r"







