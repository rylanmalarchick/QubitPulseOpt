"""
Lindblad Master Equation for Open Quantum Systems
==================================================

This module implements the Lindblad master equation for simulating open quantum
systems with decoherence. It includes T1 (energy relaxation) and T2 (dephasing)
decoherence channels that model realistic quantum hardware.

Physical Context:
-----------------
Real quantum systems are never perfectly isolated—they interact with their
environment, leading to decoherence and energy dissipation. The Lindblad
master equation describes this open-system dynamics:

    dρ/dt = -i[H, ρ] + Σ_k L_k ρ L_k† - (1/2){L_k† L_k, ρ}

where:
- ρ is the density matrix (mixed state representation)
- H is the system Hamiltonian
- L_k are Lindblad (jump) operators describing environment coupling
- [·,·] is the commutator, {·,·} is the anticommutator

Decoherence Channels:
---------------------
1. **T1 (Energy Relaxation / Amplitude Damping)**:
   - Describes spontaneous emission: |1⟩ → |0⟩
   - Lindblad operator: L₁ = √(γ₁) σ₋, where γ₁ = 1/T1
   - Physical origin: coupling to thermal bath, photon emission

2. **T2 (Dephasing / Pure Dephasing)**:
   - Describes phase randomization without energy loss
   - Total dephasing rate: γ₂ = 1/T2
   - Pure dephasing: γ_φ = γ₂ - γ₁/2
   - Lindblad operator: L_φ = √(γ_φ) σ_z
   - Physical origin: low-frequency noise, charge fluctuations

3. **Tφ (Pure Dephasing Time)**:
   - Alternative parameterization: 1/T2 = 1/(2T1) + 1/Tφ
   - Tφ captures dephasing not caused by T1

Typical Values (Superconducting Qubits):
----------------------------------------
- T1: 10-100 μs (energy relaxation)
- T2: 1-100 μs (coherence time)
- T2 ≤ 2*T1 (fundamental limit from T1 processes)
- T2* (inhomogeneous dephasing): 1-10 μs

Connection to Control:
----------------------
Decoherence limits gate fidelity and favors faster gates. Optimal control
must balance gate speed (minimize decoherence) vs. control error (minimize
pulse bandwidth, avoid leakage).

References:
-----------
- Breuer & Petruccione, "The Theory of Open Quantum Systems" (2002)
- Nielsen & Chuang, "Quantum Computation and Quantum Information" (2010)
- Krantz et al., Appl. Phys. Rev. 6, 021318 (2019) - Superconducting qubit review
- Magesan & Gambetta, Phys. Rev. A 101, 052308 (2020) - Noise models

Author: Orchestrator Agent
Date: 2025-01-27
SOW Reference: Phase 2.3 - Open System Dynamics
"""

import numpy as np
import qutip as qt
from typing import Union, Callable, Optional, List, Tuple
from dataclasses import dataclass
import warnings


@dataclass
class DecoherenceParams:
    """
    Container for decoherence parameters.

    Attributes
    ----------
    T1 : float
        Energy relaxation time (amplitude damping). Units: same as time evolution.
        Typical: 10-100 μs for superconducting qubits.
    T2 : float
        Total dephasing time (coherence time). Units: same as time evolution.
        Constraint: T2 ≤ 2*T1 (if T2 > 2*T1, pure dephasing is negative).
    Tphi : float, optional
        Pure dephasing time. If provided, T2 is computed as:
            1/T2 = 1/(2*T1) + 1/Tphi
        If None, uses provided T2 value.
    temperature : float, optional
        Environment temperature in energy units (ℏω). Default: 0 (zero temperature).
        For T > 0, thermal excitation |0⟩ → |1⟩ becomes possible.
    """

    T1: float
    T2: Optional[float] = None
    Tphi: Optional[float] = None
    temperature: float = 0.0

    def __post_init__(self):
        """Validate parameters and compute derived quantities."""
        if self.T1 <= 0:
            raise ValueError("T1 must be positive")

        # Compute T2 from Tphi if provided
        if self.Tphi is not None:
            if self.Tphi <= 0:
                raise ValueError("Tphi must be positive")
            self.T2 = 1.0 / (1.0 / (2.0 * self.T1) + 1.0 / self.Tphi)

        if self.T2 is None:
            raise ValueError("Must provide either T2 or Tphi")

        if self.T2 <= 0:
            raise ValueError("T2 must be positive")

        # Check physical constraint: T2 ≤ 2*T1
        if self.T2 > 2.0 * self.T1 + 1e-10:  # Small tolerance for numerical error
            raise ValueError(
                f"T2 ({self.T2}) cannot exceed 2*T1 ({2 * self.T1}). "
                "This violates fundamental quantum mechanics constraints."
            )

        if self.temperature < 0:
            raise ValueError("Temperature must be non-negative")


class LindbladEvolution:
    """
    Lindblad master equation solver for open quantum systems.

    This class simulates density matrix evolution under Hamiltonian dynamics
    plus Lindblad dissipation and dephasing operators.

    Parameters
    ----------
    H : qutip.Qobj or list
        System Hamiltonian. Can be:
        - Time-independent: qutip.Qobj
        - Time-dependent: [H0, [H1, f1(t)], [H2, f2(t)], ...]
    decoherence : DecoherenceParams
        Decoherence parameters (T1, T2, etc.).
    collapse_operators : list of qutip.Qobj, optional
        Custom collapse operators. If None, constructs standard T1/T2 operators.

    Examples
    --------
    >>> # Create Hamiltonian with control
    >>> H0 = 0.5 * 5.0 * qt.sigmaz()  # 5 MHz drift
    >>> H1 = qt.sigmax()
    >>> pulse = lambda t: 0.1 * np.sin(2*np.pi*t/10)
    >>> H = [H0, [H1, pulse]]
    >>>
    >>> # Setup decoherence
    >>> decoherence = DecoherenceParams(T1=50, T2=30)  # μs
    >>>
    >>> # Create solver
    >>> lindblad = LindbladEvolution(H, decoherence)
    >>>
    >>> # Evolve initial state
    >>> rho0 = qt.basis(2, 0) * qt.basis(2, 0).dag()  # |0⟩⟨0|
    >>> times = np.linspace(0, 100, 1000)
    >>> result = lindblad.evolve(rho0, times)
    """

    def __init__(
        self,
        H: Union[qt.Qobj, list],
        decoherence: DecoherenceParams,
        collapse_operators: Optional[List[qt.Qobj]] = None,
    ):
        """Initialize Lindblad evolution solver."""
        self.H = H
        self.decoherence = decoherence

        # Get dimension from Hamiltonian
        if isinstance(H, qt.Qobj):
            self.dim = H.shape[0]
        else:
            # Time-dependent Hamiltonian: first element is H0
            self.dim = H[0].shape[0]

        # Construct collapse operators
        if collapse_operators is not None:
            self.c_ops = collapse_operators
        else:
            self.c_ops = self._construct_collapse_operators()

    def _construct_collapse_operators(self) -> List[qt.Qobj]:
        """
        Construct standard T1/T2 collapse operators.

        Returns
        -------
        list of qutip.Qobj
            Lindblad operators for T1 and T2 processes.
        """
        c_ops = []

        # T1 (amplitude damping): γ₁ = 1/T1
        gamma1 = 1.0 / self.decoherence.T1

        # Lowering operator: σ₋ = |0⟩⟨1|
        sigma_minus = qt.destroy(self.dim)

        # T1 collapse operator (spontaneous emission)
        if self.decoherence.temperature == 0:
            # Zero temperature: only |1⟩ → |0⟩
            c_ops.append(np.sqrt(gamma1) * sigma_minus)
        else:
            # Finite temperature: add thermal excitation |0⟩ → |1⟩
            # Assuming qubit frequency ω₀ is encoded in drift Hamiltonian
            # For simplicity, use Boltzmann factor from temperature parameter
            n_thermal = self.decoherence.temperature

            # Decay: |1⟩ → |0⟩
            c_ops.append(np.sqrt(gamma1 * (1 + n_thermal)) * sigma_minus)

            # Excitation: |0⟩ → |1⟩
            sigma_plus = sigma_minus.dag()
            c_ops.append(np.sqrt(gamma1 * n_thermal) * sigma_plus)

        # Pure dephasing: γ_φ = 1/T2 - 1/(2*T1)
        gamma2 = 1.0 / self.decoherence.T2
        gamma_phi = gamma2 - gamma1 / 2.0

        if gamma_phi > 1e-12:  # Only add if significant
            # Dephasing operator: σ_z
            sigma_z = qt.sigmaz()
            c_ops.append(np.sqrt(gamma_phi) * sigma_z)

        return c_ops

    def evolve(
        self,
        rho0: qt.Qobj,
        times: np.ndarray,
        e_ops: Optional[List[qt.Qobj]] = None,
    ) -> qt.solver.Result:
        """
        Evolve density matrix under Lindblad master equation.

        Solves: dρ/dt = -i[H, ρ] + Σ_k (L_k ρ L_k† - (1/2){L_k† L_k, ρ})

        Parameters
        ----------
        rho0 : qutip.Qobj
            Initial density matrix.
        times : np.ndarray
            Array of time points for evolution.
        e_ops : list of qutip.Qobj, optional
            Operators for expectation value computation.

        Returns
        -------
        qutip.solver.Result
            Result object containing:
            - states: Density matrices at each time point
            - expect: Expectation values (if e_ops provided)
            - times: Time points

        Examples
        --------
        >>> lindblad = LindbladEvolution(H, decoherence)
        >>> rho0 = qt.basis(2, 0) * qt.basis(2, 0).dag()
        >>> times = np.linspace(0, 100, 500)
        >>> result = lindblad.evolve(rho0, times, e_ops=[qt.sigmaz()])
        >>> print(f"Final excited state population: {result.expect[0][-1]:.4f}")
        """
        # Use QuTiP's master equation solver
        result = qt.mesolve(self.H, rho0, times, self.c_ops, e_ops=e_ops)
        return result

    def _get_unitary_hamiltonian(self, H_unitary: Optional[qt.Qobj]) -> qt.Qobj:
        """
        Get Hamiltonian for unitary comparison.

        Parameters
        ----------
        H_unitary : qt.Qobj, optional
            Provided Hamiltonian

        Returns
        -------
        qt.Qobj
            Hamiltonian to use
        """
        if H_unitary is None:
            if isinstance(self.H, qt.Qobj):
                return self.H
            else:
                return self.H[0]  # Use H0 from time-dependent list
        return H_unitary

    def _compute_fidelity_and_purity(
        self,
        lindblad_states: list,
        unitary_states: list,
    ) -> tuple:
        """
        Compute fidelities and purities.

        Parameters
        ----------
        lindblad_states : list
            States from Lindblad evolution
        unitary_states : list
            States from unitary evolution

        Returns
        -------
        fidelities : np.ndarray
            Fidelity at each time
        purities : np.ndarray
            Purity at each time
        """
        fidelities = []
        purities = []

        for rho_lindblad, rho_unitary in zip(lindblad_states, unitary_states):
            # Fidelity between density matrices
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=Warning)
                fid = qt.fidelity(rho_lindblad, rho_unitary)
            fidelities.append(fid)

            # Purity: Tr(ρ²)
            purity = (rho_lindblad * rho_lindblad).tr()
            purities.append(np.real(purity))

        return np.array(fidelities), np.array(purities)

    def compare_with_unitary(
        self,
        rho0: qt.Qobj,
        times: np.ndarray,
        H_unitary: Optional[qt.Qobj] = None,
    ) -> dict:
        """
        Compare open-system evolution with closed-system (unitary) evolution.

        This demonstrates the effect of decoherence by comparing ideal
        (unitary) evolution vs. realistic (Lindblad) evolution.

        Parameters
        ----------
        rho0 : qutip.Qobj
            Initial density matrix.
        times : np.ndarray
            Time points.
        H_unitary : qutip.Qobj, optional
            Hamiltonian for unitary comparison. If None, uses self.H
            (assumes time-independent or extracts H0).

        Returns
        -------
        dict
            Dictionary with keys:
            - 'lindblad_states': States from Lindblad evolution
            - 'unitary_states': States from unitary evolution
            - 'fidelities': Fidelity between evolutions at each time
            - 'purity': Purity Tr(ρ²) for Lindblad evolution (1 = pure)

        Examples
        --------
        >>> comparison = lindblad.compare_with_unitary(rho0, times)
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(times, comparison['fidelities'], label='Fidelity')
        >>> plt.plot(times, comparison['purity'], label='Purity')
        >>> plt.legend()
        >>> plt.show()
        """
        # Lindblad evolution
        result_lindblad = self.evolve(rho0, times)

        # Unitary evolution (no decoherence)
        H_eff = self._get_unitary_hamiltonian(H_unitary)
        result_unitary = qt.sesolve(H_eff, rho0, times)

        # Compute fidelities and purity
        fidelities, purities = self._compute_fidelity_and_purity(
            result_lindblad.states, result_unitary.states
        )

        return {
            "lindblad_states": result_lindblad.states,
            "unitary_states": result_unitary.states,
            "fidelities": fidelities,
            "purity": purities,
        }

    def gate_fidelity_with_decoherence(
        self,
        U_ideal: qt.Qobj,
        rho0: qt.Qobj,
        gate_time: float,
        n_steps: int = 1000,
    ) -> float:
        """
        Compute gate fidelity including decoherence effects.

        Compares the effect of U_ideal on rho0 (ideal gate) vs. the actual
        Lindblad evolution (realistic gate with decoherence).

        Parameters
        ----------
        U_ideal : qutip.Qobj
            Ideal target unitary gate.
        rho0 : qutip.Qobj
            Initial density matrix.
        gate_time : float
            Total gate duration.
        n_steps : int, optional
            Number of time steps for evolution. Default: 1000.

        Returns
        -------
        float
            Gate fidelity F ∈ [0, 1].

        Examples
        --------
        >>> # X-gate with decoherence
        >>> U_target = qt.sigmax()
        >>> rho0 = qt.basis(2, 0) * qt.basis(2, 0).dag()
        >>> fidelity = lindblad.gate_fidelity_with_decoherence(
        ...     U_target, rho0, gate_time=50
        ... )
        >>> print(f"Gate fidelity with decoherence: {fidelity:.6f}")
        """
        times = np.linspace(0, gate_time, n_steps)

        # Lindblad evolution
        result = self.evolve(rho0, times)
        rho_final = result.states[-1]

        # Ideal evolution
        rho_ideal = U_ideal * rho0 * U_ideal.dag()

        # Fidelity
        fidelity = qt.fidelity(rho_final, rho_ideal)
        return fidelity

    def relaxation_curve(
        self,
        initial_state: str = "excited",
        times: Optional[np.ndarray] = None,
        max_time: float = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute T1 relaxation curve (exponential decay).

        For excited state |1⟩, population decays as:
            P₁(t) = exp(-t/T1)

        Parameters
        ----------
        initial_state : str, optional
            Initial state: 'excited' (|1⟩) or 'ground' (|0⟩). Default: 'excited'.
        times : np.ndarray, optional
            Time points for evaluation. If None, auto-generates.
        max_time : float, optional
            Maximum time (in units of T1). Default: 5*T1.

        Returns
        -------
        times : np.ndarray
            Time points.
        populations : np.ndarray
            Excited state population P₁(t).

        Examples
        --------
        >>> times, pops = lindblad.relaxation_curve()
        >>> plt.plot(times, pops, label='Simulated')
        >>> plt.plot(times, np.exp(-times/lindblad.decoherence.T1),
        ...          label='Ideal T1', linestyle='--')
        >>> plt.legend()
        """
        if times is None:
            if max_time is None:
                max_time = 5.0 * self.decoherence.T1
            times = np.linspace(0, max_time, 500)

        # Initial state
        if initial_state == "excited":
            rho0 = qt.basis(2, 1) * qt.basis(2, 1).dag()  # |1⟩⟨1|
        elif initial_state == "ground":
            rho0 = qt.basis(2, 0) * qt.basis(2, 0).dag()  # |0⟩⟨0|
        else:
            raise ValueError("initial_state must be 'excited' or 'ground'")

        # Evolve with only T1 (no Hamiltonian, no dephasing for pure T1 measurement)
        H_zero = 0 * qt.qeye(2)
        gamma1 = 1.0 / self.decoherence.T1
        c_ops_t1 = [np.sqrt(gamma1) * qt.destroy(2)]

        result = qt.mesolve(
            H_zero, rho0, times, c_ops_t1, e_ops=[qt.basis(2, 1) * qt.basis(2, 1).dag()]
        )
        populations = result.expect[0]

        return times, np.real(populations)

    def ramsey_experiment(
        self,
        detuning: float,
        max_time: float = None,
        times: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate Ramsey experiment to measure T2*.

        A Ramsey experiment consists of two π/2 pulses separated by free evolution.
        The oscillations decay with time constant T2* (inhomogeneous dephasing).

        For simplicity, we simulate free evolution in superposition state and
        measure oscillations in σ_x.

        Parameters
        ----------
        detuning : float
            Detuning from qubit frequency (creates oscillations).
        max_time : float, optional
            Maximum free evolution time. Default: 3*T2.
        times : np.ndarray, optional
            Time points for evaluation.

        Returns
        -------
        times : np.ndarray
            Time points.
        signal : np.ndarray
            Ramsey signal (oscillating, decaying).

        Examples
        --------
        >>> times, signal = lindblad.ramsey_experiment(detuning=0.5)
        >>> plt.plot(times, signal)
        >>> plt.xlabel('Time')
        >>> plt.ylabel('⟨σ_x⟩')
        >>> plt.title(f'Ramsey Decay (T2 = {lindblad.decoherence.T2})')
        """
        if times is None:
            if max_time is None:
                max_time = 3.0 * self.decoherence.T2
            times = np.linspace(0, max_time, 500)

        # Initial state: |+⟩ = (|0⟩ + |1⟩)/√2
        psi_plus = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
        rho0 = psi_plus * psi_plus.dag()

        # Hamiltonian with detuning
        H = 0.5 * detuning * qt.sigmaz()

        # Evolve with T2 decoherence
        lindblad_ramsey = LindbladEvolution(H, self.decoherence)
        result = lindblad_ramsey.evolve(rho0, times, e_ops=[qt.sigmax()])

        signal = result.expect[0]
        return times, np.real(signal)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"LindbladEvolution(T1={self.decoherence.T1}, "
            f"T2={self.decoherence.T2}, "
            f"n_collapse_ops={len(self.c_ops)})"
        )


def thermal_state(dim: int, temperature: float, omega: float) -> qt.Qobj:
    """
    Construct thermal (Gibbs) state for a qubit.

    ρ_thermal = exp(-H/kT) / Z

    where Z = Tr[exp(-H/kT)] is the partition function.

    Parameters
    ----------
    dim : int
        Hilbert space dimension (2 for qubit).
    temperature : float
        Temperature in units of ℏω/k_B.
    omega : float
        Qubit frequency.

    Returns
    -------
    qutip.Qobj
        Thermal density matrix.

    Examples
    --------
    >>> rho_thermal = thermal_state(2, temperature=0.1, omega=5.0)
    >>> print(f"Ground state population: {rho_thermal[0,0]:.4f}")
    """
    if temperature == 0:
        # Zero temperature: ground state
        return qt.basis(dim, 0) * qt.basis(dim, 0).dag()

    # Boltzmann factor
    beta = 1.0 / temperature

    # Energy levels (assuming H = ω σ_z / 2)
    energies = np.array([omega / 2, -omega / 2])  # |0⟩, |1⟩

    # Populations
    populations = np.exp(-beta * energies)
    Z = np.sum(populations)  # Partition function
    populations /= Z

    # Construct density matrix (diagonal in computational basis)
    rho = populations[0] * qt.basis(dim, 0) * qt.basis(dim, 0).dag()
    rho += populations[1] * qt.basis(dim, 1) * qt.basis(dim, 1).dag()

    return rho
