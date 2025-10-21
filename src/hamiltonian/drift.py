"""
Drift Hamiltonian Module
========================

This module implements the drift (free precession) Hamiltonian for a single qubit.
The drift Hamiltonian represents the uncontrolled evolution of the qubit due to
its natural energy splitting.

Physics Background
------------------
The drift Hamiltonian for a two-level system (qubit) is:

    H₀ = (ω₀/2) σ_z

where:
    - ω₀: Qubit transition frequency (angular frequency, rad/s or MHz)
    - σ_z: Pauli Z operator

This Hamiltonian causes the qubit to precess around the z-axis of the Bloch sphere
with angular frequency ω₀. The energy eigenstates are |0⟩ (ground) and |1⟩ (excited)
with eigenvalues -ω₀/2 and +ω₀/2 respectively.

Connection to AirHound Project
------------------------------
Just as the AirHound drone has drift dynamics (yaw rotation without control input
due to IMU bias), the qubit has free precession H₀. This represents the "baseline"
motion before we apply control pulses.

Author: Orchestrator Agent
Date: 2025-01-27
SOW Reference: Lines 150-161 (Week 1.2: Drift Dynamics)
"""

import numpy as np
import qutip


class DriftHamiltonian:
    """
    Drift Hamiltonian for a single qubit.

    This class encapsulates the drift (free precession) Hamiltonian H₀ = (ω₀/2)σ_z,
    providing methods to access its properties and use it in time evolution.

    Attributes
    ----------
    omega_0 : float
        Qubit transition frequency in MHz (angular frequency).
    H : qutip.Qobj
        QuTiP quantum object representing the Hamiltonian operator.

    Examples
    --------
    >>> drift = DriftHamiltonian(omega_0=5.0)
    >>> print(drift.energy_splitting())
    5.0
    >>> E0, E1 = drift.energy_levels()
    >>> print(f"Ground state: {E0:.2f} MHz")
    Ground state: -2.50 MHz
    """

    def __init__(self, omega_0: float = 5.0):
        """
        Initialize drift Hamiltonian.

        Parameters
        ----------
        omega_0 : float, optional
            Qubit transition frequency in MHz (default: 5.0 MHz).
            Typical values for superconducting qubits: 3-8 GHz → 3000-8000 MHz.
            For simplicity, we work in MHz units.

        Raises
        ------
        ValueError
            If omega_0 is not positive.

        Notes
        -----
        The frequency omega_0 represents the energy splitting between |0⟩ and |1⟩.
        In experimental systems, this corresponds to the qubit transition frequency
        measured via spectroscopy.
        """
        if omega_0 <= 0:
            raise ValueError(f"Frequency must be positive, got omega_0={omega_0}")

        self.omega_0 = omega_0

        # Construct Hamiltonian: H₀ = (ω₀/2) σ_z
        sz = qutip.sigmaz()
        self.H = 0.5 * omega_0 * sz

    def to_qobj(self) -> qutip.Qobj:
        """
        Return the Hamiltonian as a QuTiP quantum object.

        Returns
        -------
        qutip.Qobj
            The Hamiltonian operator H₀.

        Examples
        --------
        >>> drift = DriftHamiltonian(omega_0=5.0)
        >>> H = drift.to_qobj()
        >>> print(H.dims)
        [[2], [2]]
        """
        return self.H

    def energy_levels(self) -> tuple[float, float]:
        """
        Compute energy eigenvalues of the Hamiltonian.

        Returns
        -------
        E_ground : float
            Ground state energy (E₀ = -ω₀/2).
        E_excited : float
            Excited state energy (E₁ = +ω₀/2).

        Examples
        --------
        >>> drift = DriftHamiltonian(omega_0=5.0)
        >>> E0, E1 = drift.energy_levels()
        >>> print(f"E₀ = {E0:.2f} MHz, E₁ = {E1:.2f} MHz")
        E₀ = -2.50 MHz, E₁ = 2.50 MHz
        """
        evals = self.H.eigenenergies()
        # Sort to ensure E_ground < E_excited
        evals_sorted = np.sort(evals)
        return evals_sorted[0], evals_sorted[1]

    def energy_splitting(self) -> float:
        """
        Compute energy splitting between ground and excited states.

        Returns
        -------
        float
            Energy difference ΔE = E₁ - E₀ = ω₀.

        Examples
        --------
        >>> drift = DriftHamiltonian(omega_0=5.0)
        >>> print(drift.energy_splitting())
        5.0
        """
        E0, E1 = self.energy_levels()
        return E1 - E0

    def eigenstates(self) -> tuple[qutip.Qobj, qutip.Qobj]:
        """
        Compute energy eigenstates.

        Returns
        -------
        ket_ground : qutip.Qobj
            Ground state |g⟩ (should be |0⟩ for drift Hamiltonian).
        ket_excited : qutip.Qobj
            Excited state |e⟩ (should be |1⟩ for drift Hamiltonian).

        Notes
        -----
        For H₀ = (ω₀/2)σ_z, the eigenstates are the computational basis states:
            |g⟩ = |0⟩ with eigenvalue -ω₀/2
            |e⟩ = |1⟩ with eigenvalue +ω₀/2

        Examples
        --------
        >>> drift = DriftHamiltonian(omega_0=5.0)
        >>> ket_g, ket_e = drift.eigenstates()
        >>> ket0 = qutip.basis(2, 0)
        >>> print(f"Fidelity: {qutip.fidelity(ket_g, ket0):.6f}")
        Fidelity: 1.000000
        """
        evals, evecs = self.H.eigenstates()

        # Sort by energy
        idx_sorted = np.argsort(evals)
        ket_ground = evecs[idx_sorted[0]]
        ket_excited = evecs[idx_sorted[1]]

        return ket_ground, ket_excited

    def precession_period(self) -> float:
        """
        Compute the period of free precession.

        Returns
        -------
        float
            Period T = 2π/ω₀ in microseconds (assuming ω₀ in MHz).

        Examples
        --------
        >>> drift = DriftHamiltonian(omega_0=5.0)
        >>> print(f"Period: {drift.precession_period():.4f} μs")
        Period: 1.2566 μs

        Notes
        -----
        A state starting in |0⟩ will return to |0⟩ (up to global phase) after
        one full period T.
        """
        return 2 * np.pi / self.omega_0

    def commutator_with_sigmaz(self) -> qutip.Qobj:
        """
        Compute commutator [H₀, σ_z].

        Returns
        -------
        qutip.Qobj
            Commutator (should be zero operator).

        Notes
        -----
        Since H₀ ∝ σ_z, we have [H₀, σ_z] = 0. This means σ_z (and thus |0⟩, |1⟩)
        are conserved under drift evolution—states don't rotate in x or y.

        Examples
        --------
        >>> drift = DriftHamiltonian(omega_0=5.0)
        >>> comm = drift.commutator_with_sigmaz()
        >>> print(f"Commutator norm: {comm.norm():.2e}")
        Commutator norm: 0.00e+00
        """
        sz = qutip.sigmaz()
        return qutip.commutator(self.H, sz)

    def evolve_state(self, psi0: qutip.Qobj, times: np.ndarray) -> qutip.solver.Result:
        """
        Evolve an initial state under the drift Hamiltonian.

        Parameters
        ----------
        psi0 : qutip.Qobj
            Initial quantum state (ket).
        times : np.ndarray
            Array of time points for evolution (in μs, assuming ω₀ in MHz).

        Returns
        -------
        qutip.solver.Result
            QuTiP result object containing evolved states at each time point.

        Examples
        --------
        >>> drift = DriftHamiltonian(omega_0=5.0)
        >>> psi0 = qutip.basis(2, 0)  # Start in |0⟩
        >>> times = np.linspace(0, drift.precession_period(), 100)
        >>> result = drift.evolve_state(psi0, times)
        >>> fidelity = qutip.fidelity(result.states[0], result.states[-1])
        >>> print(f"Fidelity after one period: {fidelity:.6f}")
        Fidelity after one period: 1.000000

        Notes
        -----
        Uses QuTiP's Schrödinger equation solver (sesolve) to compute
        unitary evolution: |ψ(t)⟩ = exp(-iH₀t)|ψ(0)⟩.
        """
        result = qutip.sesolve(self.H, psi0, times)
        return result

    def __repr__(self) -> str:
        """String representation of the drift Hamiltonian."""
        return f"DriftHamiltonian(omega_0={self.omega_0} MHz)"

    def __str__(self) -> str:
        """Human-readable string representation."""
        E0, E1 = self.energy_levels()
        return (
            f"Drift Hamiltonian: H₀ = (ω₀/2)σ_z\n"
            f"  Frequency: ω₀ = {self.omega_0} MHz\n"
            f"  Energy levels: E₀ = {E0:.3f} MHz, E₁ = {E1:.3f} MHz\n"
            f"  Splitting: ΔE = {self.energy_splitting():.3f} MHz\n"
            f"  Period: T = {self.precession_period():.4f} μs"
        )


# Convenience function for quick Hamiltonian creation
def create_drift_hamiltonian(omega_0: float = 5.0) -> DriftHamiltonian:
    """
    Factory function to create a drift Hamiltonian.

    Parameters
    ----------
    omega_0 : float, optional
        Qubit transition frequency in MHz (default: 5.0).

    Returns
    -------
    DriftHamiltonian
        Initialized drift Hamiltonian object.

    Examples
    --------
    >>> H_drift = create_drift_hamiltonian(omega_0=6.0)
    >>> print(H_drift)
    Drift Hamiltonian: H₀ = (ω₀/2)σ_z
      Frequency: ω₀ = 6.0 MHz
      Energy levels: E₀ = -3.000 MHz, E₁ = 3.000 MHz
      Splitting: ΔE = 6.000 MHz
      Period: T = 1.0472 μs
    """
    return DriftHamiltonian(omega_0=omega_0)
