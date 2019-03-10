"""
data.py - A module for storing experimental results and constants.
"""

### UCCSD - LiH ###

# A randomly generated theta vector to use for experiments.
UCCSD_LIH_THETA = [0.86203, 3.8037, 3.3223, 1.766, 1.0846, 1.4558, 1.0592,
                   0.091974]

# The pulse times for each of the uccsd lih slices. These times correspond
# to those found by binary search at
# /project/ftchong/qoc/thomas/uccsd_slice_time/lih/
# plus the maximum pulse time for an RZ gate times the number of RZ gates
# in the slice.
UCCSD_LIH_SLICE_TIMES = []

# This pulse time for the full uccsd lih circuit. This time corresponds
# to that found by binary search at
# /project/ftchong/qoc/thomas/uccsd_full_time/lih/
# plus the maximum pulse time for an RZ gate times the number of RZ gates
# in the circuit.
UCCSD_LIH_FULL_TIME = 0
