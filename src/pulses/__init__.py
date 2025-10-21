"""
Pulse shape generators for quantum control.

This module provides various pulse shapes commonly used in quantum control:
- Gaussian pulses
- Square (rectangular) pulses
- DRAG (Derivative Removal by Adiabatic Gate) pulses
- Blackman pulses
- Custom pulse envelopes

All pulse functions return time-dependent amplitude arrays that can be
used with control Hamiltonians for qubit gate operations.
"""

from .shapes import (
    gaussian_pulse,
    square_pulse,
    drag_pulse,
    blackman_pulse,
    cosine_pulse,
)

__all__ = [
    "gaussian_pulse",
    "square_pulse",
    "drag_pulse",
    "blackman_pulse",
    "cosine_pulse",
]
