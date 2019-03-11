"""
data.py - A module for storing experimental results and constants.

Notes:
- All times are in nanoseconds.

"""

### GENERAL ###

# See Gate_Times.ipynb for determination of these pulse times.
GATE_TIMES = {'h': 1.4, 'cx': 3.8, 'rz': 0.4, 'rx': 2.5, 'x': 2.5, 'swap': 7.4}

### UCCSD - LiH ###

# A randomly generated theta vector to use for experimental consistency.
UCCSD_LIH_THETA = [0.86203, 3.8037, 3.3223, 1.766, 1.0846, 1.4558, 1.0592,
                   0.091974]

# The pulse times for each of the uccsd lih slices. These times correspond
# to those found by binary search at
# /project/ftchong/qoc/thomas/uccsd_slice_time/lih/
# plus the maximum pulse time for an RZ gate times the number of RZ gates
# in the slice. These slices are the granularity 2, depedence grouping = True
# slices.
_rz = GATE_TIMES['rz']
UCCSD_LIH_SLICE_TIMES = [7.6 + _rz * 2, 2.75 + _rz * 2, 3.55 + _rz * 2,
                         3.1 + _rz * 2, 29.1 + _rz * 8, 39.25 + _rz * 8,
                         56.4 + _rz * 8, 25.55 + _rz * 8]

# This pulse time for the full uccsd lih circuit. This time corresponds
# to that found by binary search at
# /project/ftchong/qoc/thomas/uccsd_full_time/lih/
# plus the maximum pulse time for an RZ gate times the number of RZ gates
# in the circuit.
UCCSD_LIH_FULL_TIME = 72.3 + _rz * 40
