#!/bin/bash
PATH_TO_SCRIPT="/home/tcpropson/repos/fqc/scripts/uccsd_slice_qoc.py"
PREFIX="cpu_conda && python $PATH_TO_SCRIPT"
ASTART="--angle-start"
ASTOP="--angle-stop"
ASTEP="--angle-step"
SSTART="--slice-start"
SSTOP="--slice-stop"

ANGLE_ARGS="$ASTART 185 $ASTOP 356 $ASTEP 5"
ANGLE_ARGS1="$ASTART 185 $ASTOP 271 $ASTEP 5"
ANGLE_ARGS2="$ASTART 275 $ASTOP 356 $ASTEP 5"

screen -AmdS q14 bash
screen -S q14 -X stuff $"$PREFIX $SSTART 12 $SSTOP 12 $ANGLE_ARGS1\r"

screen -AmdS q15 bash
screen -S q15 -X stuff $"$PREFIX $SSTART 12 $SSTOP 12 $ANGLE_ARGS2\r"

screen -AmdS q16 bash
screen -S q16 -X stuff $"$PREFIX $SSTART 13 $SSTOP 13 $ANGLE_ARGS1\r"

screen -AmdS q17 bash
screen -S q17 -X stuff $"$PREFIX $SSTART 13 $SSTOP 13 $ANGLE_ARGS2\r"

screen -AmdS q18 bash
screen -S q18 -X stuff $"$PREFIX $SSTART 14 $SSTOP 14 $ANGLE_ARGS\r"

screen -AmdS q19 bash
screen -S q19 -X stuff $"$PREFIX $SSTART 15 $SSTOP 15 $ANGLE_ARGS\r"

screen -AmdS q20 bash
screen -S q20 -X stuff $"$PREFIX $SSTART 16 $SSTOP 16 $ANGLE_ARGS\r"

screen -AmdS q21 bash
screen -S q21 -X stuff $"$PREFIX $SSTART 17 $SSTOP 17 $ANGLE_ARGS\r"

screen -AmdS q22 bash
screen -S q22 -X stuff $"$PREFIX $SSTART 18 $SSTOP 18 $ANGLE_ARGS\r"

screen -AmdS q23 bash
screen -S q23 -X stuff $"$PREFIX $SSTART 19 $SSTOP 19 $ANGLE_ARGS\r"

screen -AmdS q24 bash
screen -S q24 -X stuff $"$PREFIX $SSTART 20 $SSTOP 20 $ANGLE_ARGS1\r"

screen -AmdS q25 bash
screen -S q25 -X stuff $"$PREFIX $SSTART 20 $SSTOP 20 $ANGLE_ARGS2\r"

screen -AmdS q26 bash
screen -S q26 -X stuff $"$PREFIX $SSTART 21 $SSTOP 21 $ANGLE_ARGS\r"

screen -AmdS q27 bash
screen -S q27 -X stuff $"$PREFIX $SSTART 22 $SSTOP 22 $ANGLE_ARGS\r"







