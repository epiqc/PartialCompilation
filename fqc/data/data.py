"""
data.py - A module for storing experimental results and constants.

Notes:
- If you want to use one of the configuration dictionaries defined
  herein but want to change a parameter, use the dict.update() method
  in your own code.
- GRAPE_CONFIG does not include all necessary arguments to run grape.
- All times are in nanoseconds.
- All uccsd slices are assumed to be those such that slice_granularity = 2,
  dependence_grouping = True.
- All lists of data in UCCSD_DATA are indexed by slice index.
- THETA is a randomly generated set of gate parameters to use
  for experimental consistency.
- HP means hyperparameters for grape on adam
- TIME means pulse times found by binary search
- CQP means connected qubit pairs
- SCL means states concerned list
- MPA means max pulse amplitude
- If a constant must be referenced inside the data structure
  in which it is defined, an external constant is created.
"""

from copy import deepcopy

import numpy as np
from fqc.uccsd import (get_uccsd_circuit, get_uccsd_slices)
from fqc.util import (get_nearest_neighbor_coupling_list, optimize_circuit)
from quantum_optimal_control.core.hamiltonian import (get_Hops_and_Hnames,
    get_full_states_concerned_list, get_maxA)


### GENERAL CONSTANTS ###

SLICE_GRANULARITY = 2
SLICE_DEPENDENCE_GROUPING = True
NUM_STATES = 2
# pulse steps per nanosecond
SPN = 20.
# nanoseconds per pulse step
NPS = 1 / SPN


### GATE DATA ###

# fqc/experiments/Gate_Times.ipynb
GATE_TIMES = {'h': 1.4, 'cx': 3.8, 'rz': 0.4, 'rx': 2.5, 'x': 2.5, 'swap': 7.4}
_RZ = GATE_TIMES['rz']


### GRAPE CONSTANTS ###

REG_COEFFS = {}

QUBIT2_CQP = get_nearest_neighbor_coupling_list(1, 2, directed=False)
QUBIT2_H0 = np.zeros((NUM_STATES ** 2, NUM_STATES ** 2))
QUBIT2_HOPS, QUBIT2_HNAMES = get_Hops_and_Hnames(2, NUM_STATES, QUBIT2_CQP)
QUBIT2_SCL = get_full_states_concerned_list(2, NUM_STATES)
QUBIT2_MPA = get_maxA(2, NUM_STATES, QUBIT2_CQP)
GRAPE_QUBIT2_CONFIG = {
    "H0": QUBIT2_H0,
    "Hops": QUBIT2_HOPS,
    "Hnames": QUBIT2_HNAMES,
    "states_concerned_list": QUBIT2_SCL,
    "reg_coeffs": REG_COEFFS,
    "maxA": QUBIT2_MPA,
}

QUBIT4_CQP = get_nearest_neighbor_coupling_list(2, 2, directed=False)
QUBIT4_H0 = np.zeros((NUM_STATES ** 4, NUM_STATES ** 4))
QUBIT4_HOPS, QUBIT4_HNAMES = get_Hops_and_Hnames(4, NUM_STATES, QUBIT4_CQP)
QUBIT4_SCL = get_full_states_concerned_list(4, NUM_STATES)
QUBIT4_MPA = get_maxA(4, NUM_STATES, QUBIT4_CQP)
GRAPE_QUBIT4_CONFIG = {
    "H0": QUBIT4_H0,
    "Hops": QUBIT4_HOPS,
    "Hnames": QUBIT4_HNAMES,
    "states_concerned_list": QUBIT4_SCL,
    "reg_coeffs": REG_COEFFS,
    "maxA": QUBIT4_MPA,
}

QUBIT6_CQP = get_nearest_neighbor_coupling_list(2, 3, directed=False)
QUBIT6_H0 = np.zeros((NUM_STATES ** 6, NUM_STATES ** 6))
QUBIT6_HOPS, QUBIT6_HNAMES = get_Hops_and_Hnames(6, NUM_STATES, QUBIT6_CQP)
QUBIT6_SCL = get_full_states_concerned_list(6, NUM_STATES)
QUBIT6_MPA = get_maxA(6, NUM_STATES, QUBIT6_CQP)
GRAPE_QUBIT6_CONFIG = {
    "H0": QUBIT6_H0,
    "Hops": QUBIT6_HOPS,
    "Hnames": QUBIT6_HNAMES,
    "states_concerned_list": QUBIT6_SCL,
    "reg_coeffs": REG_COEFFS,
    "maxA": QUBIT6_MPA,
}


### UCCSD MOLECULE CONSTANTS ###

# H2
UCCSD_H2_THETA = [5.239368082827368, 1.5290813407594008, 4.701843728963671]
UCCSD_H2_FULL_CIRCUIT = optimize_circuit(get_uccsd_circuit("H2", UCCSD_H2_THETA),
                                         QUBIT2_CQP)
UCCSD_H2_SLICES = get_uccsd_slices(UCCSD_H2_FULL_CIRCUIT,
                                   granularity=SLICE_GRANULARITY,
                                   dependence_grouping=SLICE_DEPENDENCE_GROUPING)

# LiH
UCCSD_LIH_THETA = [0.86203, 3.8037, 3.3223, 1.766, 1.0846, 1.4558, 1.0592,
                   0.091974
]
UCCSD_LIH_FULL_CIRCUIT = optimize_circuit(get_uccsd_circuit("LiH", UCCSD_LIH_THETA),
                                          QUBIT4_CQP)
UCCSD_LIH_SLICES = get_uccsd_slices(UCCSD_LIH_FULL_CIRCUIT,
                                    granularity=SLICE_GRANULARITY,
                                    dependence_grouping=SLICE_DEPENDENCE_GROUPING)

# BeH2
UCCSD_BEH2_THETA = [1.910655366933038, 3.0380262019523134, 1.767835033803264,
                    2.351565914908821, 0.2967722640174227, 1.9341032952378827,
                    1.9388072691864795, 0.3910158328333188, 4.73732224179686,
                    2.1482159539540997, 3.946982477487254, 3.894928536026355,
                    3.635723708626727, 0.5989045533715128, 0.8964618741785685,
                    3.242904311964212, 0.4035935811891575, 5.66483862691292,
                    4.656150662864869, 3.680744351841191, 3.2050517907826577,
                    2.2968607361280307, 1.5200151301060538, 5.534557818577588,
                    1.4588597139977681, 1.3356770159523395
]
UCCSD_BEH2_FULL_CIRCUIT = optimize_circuit(get_uccsd_circuit("BeH2", UCCSD_BEH2_THETA),
                                           QUBIT6_CQP)
# TODO: See github issue #6.
# tmp_beh2_circuit = deepcopy(UCCSD_BEH2_FULL_CIRCUIT)
# tmp_beh2_circuit.data = (tmp_beh2_circuit.data[:1965] + tmp_beh2_circuit.data[1999:]
#                          + tmp_beh2_circuit.data[1965:1999])
UCCSD_BEH2_SLICES = get_uccsd_slices(UCCSD_BEH2_FULL_CIRCUIT,
                                     granularity=SLICE_GRANULARITY,
                                     dependence_grouping=SLICE_DEPENDENCE_GROUPING)


### UCCSD DATA ###

UCCSD_DATA = {
    "H2": {
        "INFO": {
            "NUM_QUBITS": 2,
            "NUM_SLICES": 3,
        },
        "THETA": [],
        "GRAPE_CONFIG": GRAPE_QUBIT2_CONFIG,
        "CIRCUIT": UCCSD_H2_FULL_CIRCUIT,
        "SLICES": UCCSD_H2_SLICES,
        # TODO: full time, qoc
        "FULL_DATA": {
            "HP": {
                "lr": 0.2675,
                "decay": 233,
            },
        },
        # TODO: slice hpo, time, qoc
        "SLICE_DATA": {
            "HP": [
                # {"lr":, "decay":},
                # {"lr":, "decay":},
                {"lr": 0.147, "decay": 103},
            ]
        },
    },
    "LiH": {
        "INFO": {
            "NUM_QUBITS": 4,
            "NUM_SLICES": 8,
        },
        "THETA": UCCSD_LIH_THETA,
        "GRAPE_CONFIG": GRAPE_QUBIT4_CONFIG,
        "CIRCUIT": UCCSD_LIH_FULL_CIRCUIT,
        "SLICES": UCCSD_LIH_SLICES,
        # TODO: full qoc
        "FULL_DATA": {
            # /project/ftchong/qoc/thomas/hpo/uccsd_lih/full
            "HP": {
                "lr": 0.264,
                "decay": 400
            },
            # /project/ftchong/qoc/thomas/hpo/uccsd_lih/full
            "TIME": 93.15 * _RZ * 40
        },
        # TODO: hpo, time, qoc s4-s7
        "SLICE_DATA": {
            # /project/ftchong/qoc/thomas/hpo/uccsd_lih/s*
            "HP": [
                {'lr': 2e-2, 'decay': 1e3},
                {'lr': 2e-2, 'decay': 1e3},
                {'lr': 2e-2, 'decay': 1e3},
                {'lr': 2e-2, 'decay': 1e3},
                {'lr': 3.2e-3, 'decay': 2e4},
                {'lr': 6.2e-3, 'decay': 5e3},
                {'lr': 1.2e-2, 'decay': 3e3},
                {'lr': 1.1e-3, 'decay': 3e3},
            ],
            # /project/ftchong/qoc/thomas/time/uccsd_lih/s*
            "TIME": [5.1 + _RZ * 2, 2.85 + _RZ * 2, 3.6 + _RZ * 2,
                     3.4 + _RZ * 2, 29.1 + _RZ * 8, 39.25 + _RZ * 8,
                     56.4 + _RZ * 8, 25.55 + _RZ * 8
                 ],
        },
    },
    "BeH2": {
        "INFO": {
            "NUM_QUBITS": 6,
        },
        "THETA": UCCSD_BEH2_THETA,
        "GRAPE_CONFIG": GRAPE_QUBIT6_CONFIG,
        "CIRCUIT": UCCSD_BEH2_FULL_CIRCUIT,
        "SLICES": UCCSD_BEH2_SLICES,
        # TODO: full hpo, time, qoc
        "FULL_DATA": {},
        # TODO: slice hpo, time, qoc
        "SLICE_DATA": {},
    },
}
