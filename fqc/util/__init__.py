"""Utility Methods"""

from .circuitutil import (get_unitary, get_max_pulse_time, squash_circuit,
                          redundant, append_gate)
from .gateutil import krons, matprods

__all__ = ['get_unitary', 'get_max_pulse_time', 'squash_circuit', 'append_gate',
           'redundant', 'krons', 'matprods']
