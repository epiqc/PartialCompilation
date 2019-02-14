"""Utility Methods"""

from .circuitutil import (get_unitary, optimize_circuit, get_max_pulse_time,
                          squash_circuit, append_gate, redundant)
from .gateutil import krons, matprods

__all__ = ['get_unitary', 'optimize_circuit', 'get_max_pulse_time',
           'squash_circuit', 'append_gate', 'redundant', 'krons', 'matprods']
