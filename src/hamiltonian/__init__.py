"""
Hamiltonian Package
===================

This package contains modules for defining and working with quantum Hamiltonians
for single-qubit systems.

Modules
-------
drift
    Drift (free precession) Hamiltonian H₀ = (ω₀/2)σ_z
evolution
    Time evolution under Hamiltonians (analytical and numerical)

Classes
-------
DriftHamiltonian
    Drift Hamiltonian with configurable frequency
TimeEvolution
    Time evolution engine with multiple solver methods

Functions
---------
create_drift_hamiltonian
    Factory function for drift Hamiltonian creation
bloch_coordinates
    Compute Bloch sphere coordinates for a state
bloch_trajectory
    Compute Bloch sphere trajectory for state sequence

Examples
--------
>>> from src.hamiltonian import DriftHamiltonian, TimeEvolution
>>> drift = DriftHamiltonian(omega_0=5.0)
>>> print(drift)
Drift Hamiltonian: H₀ = (ω₀/2)σ_z
  Frequency: ω₀ = 5.0 MHz
  Energy levels: E₀ = -2.500 MHz, E₁ = 2.500 MHz
  Splitting: ΔE = 5.000 MHz
  Period: T = 1.2566 μs

>>> evolver = TimeEvolution(drift.to_qobj())
>>> import numpy as np
>>> import qutip
>>> psi0 = qutip.basis(2, 0)
>>> times = np.linspace(0, drift.precession_period(), 100)
>>> result = evolver.evolve(psi0, times)

Author: Orchestrator Agent
Date: 2025-01-27
SOW Reference: Lines 150-161 (Week 1.2: Drift Dynamics)
"""

from .drift import DriftHamiltonian, create_drift_hamiltonian
from .evolution import TimeEvolution, bloch_coordinates, bloch_trajectory

__all__ = [
    "DriftHamiltonian",
    "create_drift_hamiltonian",
    "TimeEvolution",
    "bloch_coordinates",
    "bloch_trajectory",
]

__version__ = "0.1.0"
