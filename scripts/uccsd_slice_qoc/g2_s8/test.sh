#!/bin/bash
conda activate fqc
mpirun -n 1 -ppn 14 python /home/tcpropson/repos/fqc/scripts/uccsd_slice_qoc/uccsd_slice_qoc.py --angle-start 0 --angle-stop 2 --angle-step 1 --slice-start 0 --slice-stop 0
