"""Utility Methods"""

from .circuitutil import (get_unitary, optimize_circuit, get_max_pulse_time,
                          squash_circuit, append_gate, merge_rotation_gates,
                          h_cancellation, impose_swap_coupling,
                          get_nearest_neighbor_coupling_list)
from .gateutil import krons, matprods

__all__ = ['get_unitary', 'optimize_circuit', 'get_max_pulse_time',
           'squash_circuit', 'append_gate', 'merge_rotation_gates',
           'h_cancellation', 'impose_swap_coupling',
           'get_nearest_neighbor_coupling_list','krons', 'matprods']
