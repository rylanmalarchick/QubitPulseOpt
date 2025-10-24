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

# Power of 10 compliance: Import assertion helpers
from ..constants import (
    MIN_ENERGY,
    MAX_ENERGY,
    assert_fidelity_valid,
)


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
        # Rule 5: Parameter validation - use ValueError for user-facing validation
        if omega_0 is None:
            raise ValueError("omega_0 cannot be None")
        if not isinstance(omega_0, (int, float)):
            raise ValueError(f"omega_0 must be numeric, got {type(omega_0)}")
        if omega_0 <= 0:
            raise ValueError(f"Frequency must be positive, got omega_0={omega_0}")
        if not np.isfinite(omega_0):
            raise ValueError(f"omega_0 must be finite, got {omega_0}")
        if not (MIN_ENERGY <= omega_0 <= MAX_ENERGY):
            raise ValueError(
                f"omega_0 {omega_0} outside reasonable bounds [{MIN_ENERGY}, {MAX_ENERGY}] MHz"
            )

        self.omega_0 = omega_0

        # Construct Hamiltonian: H₀ = (ω₀/2) σ_z
        sz = qutip.sigmaz()
        self.H = 0.5 * omega_0 * sz

        # Rule 5: Post-construction invariant checks
        assert self.H is not None, "Hamiltonian construction failed"
        assert isinstance(self.H, qutip.Qobj), "H must be Qobj"
        assert self.H.isherm, "Drift Hamiltonian must be Hermitian"
        assert self.H.shape == (2, 2), f"Expected 2x2 matrix, got {self.H.shape}"

    def to_qobj(self) -> qutip.Qobj:
        """
        Return the Hamiltonian as a QuTiP quantum object.

        Returns
        -------
        qutip.Qobj
            The Hamiltonian operator H₀.
        """
        # Rule 5: Postcondition assertion
        assert self.H is not None, "Hamiltonian is None"
        assert isinstance(self.H, qutip.Qobj), "Hamiltonian must be Qobj"
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

        E0 = evals_sorted[0]
        E1 = evals_sorted[1]

        # Rule 5: Energy level validation
        assert np.isfinite(E0) and np.isfinite(E1), "Energy levels not finite"
        assert E0 < E1, f"Ground state {E0} must be lower than excited {E1}"

        return E0, E1

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
        splitting = E1 - E0

        # Rule 5: Return value validation
        assert np.isfinite(splitting), f"Energy splitting not finite: {splitting}"
        assert splitting > 0, f"Energy splitting must be positive: {splitting}"

        return splitting

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

        # Rule 5: Validate eigenstates
        assert ket_ground is not None, "Ground state is None"
        assert ket_excited is not None, "Excited state is None"
        assert isinstance(ket_ground, qutip.Qobj), "Ground state must be Qobj"
        assert isinstance(ket_excited, qutip.Qobj), "Excited state must be Qobj"

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
        period = 2 * np.pi / self.omega_0

        # Rule 5: Validate period
        assert np.isfinite(period), f"Period not finite: {period}"
        assert period > 0, f"Period must be positive: {period}"

        return period

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
        comm = qutip.commutator(self.H, sz)

        # Rule 5: Validate commutator
        assert comm is not None, "Commutator is None"
        assert isinstance(comm, qutip.Qobj), "Commutator must be Qobj"

        return comm

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
        # Rule 5: Input validation
        if psi0 is None:
            raise ValueError("Initial state psi0 cannot be None")
        if not isinstance(psi0, qutip.Qobj):
            raise ValueError(f"psi0 must be Qobj, got {type(psi0)}")
        if not psi0.isket:
            raise ValueError("psi0 must be a ket state")
        if times is None:
            raise ValueError("times array cannot be None")
        if not isinstance(times, np.ndarray):
            raise ValueError(f"times must be ndarray, got {type(times)}")
        if len(times) == 0:
            raise ValueError("times array must not be empty")
        if not np.all(times >= 0):
            raise ValueError("All times must be non-negative")

        result = qutip.sesolve(self.H, psi0, times)

        # Rule 5: Output validation
        assert result is not None, "Evolution result is None"
        assert hasattr(result, "states"), "Result must have states attribute"
        assert len(result.states) == len(times), "Result states length mismatch"

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
    # Rule 5: Input validation
    if omega_0 is None:
        raise ValueError("omega_0 cannot be None")
    if omega_0 <= 0:
        raise ValueError(f"omega_0 must be positive, got {omega_0}")

    drift = DriftHamiltonian(omega_0)

    # Rule 5: Output validation
    assert drift is not None, "Created DriftHamiltonian is None"
    assert drift.H.isherm, "Created Hamiltonian must be Hermitian"

    return drift


def _validate_drift_parameters(omega_0: float, times: np.ndarray = None) -> None:
    """
    Helper to validate drift Hamiltonian parameters.

    Parameters
    ----------
    omega_0 : float
        Qubit transition frequency.
    times : np.ndarray, optional
        Time array for evolution.
    """
    if omega_0 <= 0:
        raise ValueError(f"omega_0 must be positive, got {omega_0}")
    assert np.isfinite(omega_0), f"omega_0 must be finite"
    if times is not None:
        assert len(times) > 0, "times array must not be empty"
        assert np.all(times >= 0), "All times must be non-negative"
