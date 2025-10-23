"""
I/O utilities for quantum control pulses and results.

This module provides export and import functionality for:
- Pulse sequences (JSON, NPZ, CSV formats)
- Optimization results
- Experimental configurations
- Hardware-compatible formats (Qiskit Pulse basic support)

Classes:
    PulseExporter: Export pulses and results to various formats
    PulseLoader: Load pulses and results from files

Functions:
    save_pulse: Convenience function to save pulse data
    load_pulse: Convenience function to load pulse data
    save_optimization_result: Save optimization results
    load_optimization_result: Load optimization results

Example:
    >>> from src.io import save_pulse, load_pulse
    >>> import numpy as np
    >>>
    >>> # Create a simple pulse
    >>> times = np.linspace(0, 100, 1000)
    >>> amplitudes = np.exp(-(times - 50)**2 / 200)
    >>>
    >>> # Save to JSON
    >>> save_pulse("my_pulse.json", times, amplitudes, format="json")
    >>>
    >>> # Load it back
    >>> data = load_pulse("my_pulse.json")
    >>> recovered_times = data["pulse_data"]["times"]
    >>> recovered_amps = data["pulse_data"]["amplitudes"]

Author: QubitPulseOpt Team
Date: 2025-01-28
"""

from .export import (
    PulseExporter,
    PulseLoader,
    save_pulse,
    load_pulse,
    save_optimization_result,
    load_optimization_result,
)

__all__ = [
    "PulseExporter",
    "PulseLoader",
    "save_pulse",
    "load_pulse",
    "save_optimization_result",
    "load_optimization_result",
]

__version__ = "1.0.0"
