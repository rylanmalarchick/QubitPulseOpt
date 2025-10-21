"""
QubitPulseOpt Source Package
=============================

This package contains the core simulation modules for optimal pulse engineering
of single-qubit quantum gates.

Modules
-------
hamiltonian : Physics modules for drift and control Hamiltonians
pulses : Pulse waveform generators
optimization : GRAPE/CRAB optimization algorithms
noise : Decoherence and noise models

Author: Orchestrator Agent
Project: QubitPulseOpt - Quantum Controls Simulation
SOW Reference: Week 1-4 Implementation
"""

__version__ = "0.1.0"
__author__ = "Rylan Malarchick"

# Expose key components at package level
from . import hamiltonian

__all__ = ["hamiltonian", "__version__", "__author__"]
