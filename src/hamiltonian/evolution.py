"""
Time Evolution Module
=====================

This module implements time evolution for quantum systems under Hamiltonians.
It provides both analytical solutions (where available) and numerical solvers,
enabling validation and performance comparisons.

Physics Background
------------------
The time evolution of a quantum state under a time-independent Hamiltonian H
is given by the Schrödinger equation:

    i ∂/∂t |ψ(t)⟩ = H |ψ(t)⟩

The formal solution is:

    |ψ(t)⟩ = U(t) |ψ(0)⟩

where U(t) = exp(-iHt) is the unitary time evolution operator.

For the drift Hamiltonian H₀ = (ω₀/2)σ_z, the analytical solution is:

    U(t) = exp(-i(ω₀/2)σ_z t) = cos(ω₀t/2)I - i sin(ω₀t/2)σ_z

This causes rotation around the z-axis of the Bloch sphere with angular
frequency ω₀.

Connection to AirHound Project
------------------------------
Time evolution under drift H₀ is analogous to the drone's motion without
control input. Just as we could analytically predict yaw drift from IMU bias,
we can analytically solve for qubit precession. Later, adding control pulses
is like applying corrective torques.

Author: Orchestrator Agent
Date: 2025-01-27
SOW Reference: Lines 150-161 (Week 1.2: Drift Dynamics)
"""

import numpy as np
import qutip
from typing import Optional, Union
import warnings

# Optional agentbible physics validation
try:
    from agentbible import validate_unitary
    HAS_AGENTBIBLE = True
except ImportError:
    HAS_AGENTBIBLE = False
    # Create a no-op decorator if agentbible is not installed
    def validate_unitary(func=None, **kwargs):
        """No-op decorator when agentbible is not installed."""
        if func is not None:
            return func
        return lambda f: f


class TimeEvolution:
    """
    Time evolution engine for quantum systems.

    This class provides methods to evolve quantum states under time-independent
    Hamiltonians, with support for both analytical (exact) and numerical solutions.

    Attributes
    ----------
    hamiltonian : qutip.Qobj
        The Hamiltonian operator governing the evolution.
    method : str
        Evolution method: 'numerical' or 'analytical' (if available).

    Examples
    --------
    >>> H = 0.5 * 5.0 * qutip.sigmaz()  # Drift Hamiltonian
    >>> evolver = TimeEvolution(H)
    >>> psi0 = qutip.basis(2, 0)
    >>> times = np.linspace(0, 1.0, 100)
    >>> result = evolver.evolve(psi0, times)
    """

    def __init__(self, hamiltonian: qutip.Qobj, method: str = "numerical"):
        """
        Initialize time evolution engine.

        Parameters
        ----------
        hamiltonian : qutip.Qobj
            Time-independent Hamiltonian operator.
        method : str, optional
            Evolution method: 'numerical' (default) or 'analytical'.
            Analytical method only works for specific Hamiltonians.

        Raises
        ------
        ValueError
            If method is not recognized or analytical not available for given H.
        """
        if not isinstance(hamiltonian, qutip.Qobj):
            raise TypeError("Hamiltonian must be a qutip.Qobj")

        if hamiltonian.dims != [[2], [2]]:
            raise ValueError("Only single-qubit (2-level) systems currently supported")

        self.hamiltonian = hamiltonian
        self.method = method

        if method == "analytical":
            if not self._is_analytical_available():
                warnings.warn(
                    "Analytical solution not available for this Hamiltonian. "
                    "Falling back to numerical.",
                    UserWarning,
                )
                self.method = "numerical"

    def _is_analytical_available(self) -> bool:
        """
        Check if analytical solution is available for the Hamiltonian.

        Currently supports:
        - Drift Hamiltonian: H ∝ σ_z

        Returns
        -------
        bool
            True if analytical solution is implemented.
        """
        # Check if Hamiltonian is proportional to σ_z
        sz = qutip.sigmaz()

        # If [H, σ_z] = 0 and H is diagonal in computational basis, it's drift-like
        comm = qutip.commutator(self.hamiltonian, sz)

        return comm.norm() < 1e-10

    def evolve(
        self, psi0: qutip.Qobj, times: np.ndarray, e_ops: Optional[list] = None
    ) -> qutip.solver.Result:
        """
        Evolve initial state under the Hamiltonian.

        Parameters
        ----------
        psi0 : qutip.Qobj
            Initial quantum state (ket).
        times : np.ndarray
            Array of time points for evolution.
        e_ops : list of qutip.Qobj, optional
            List of operators for expectation value computation.

        Returns
        -------
        qutip.solver.Result
            Result object containing:
            - states: List of states at each time point
            - expect: Expectation values (if e_ops provided)
            - times: Time points

        Examples
        --------
        >>> H = qutip.sigmaz()
        >>> evolver = TimeEvolution(H)
        >>> psi0 = qutip.basis(2, 0)
        >>> times = np.linspace(0, 1.0, 50)
        >>> result = evolver.evolve(psi0, times)
        >>> print(len(result.states))
        50
        """
        if self.method == "analytical":
            return self._evolve_analytical(psi0, times, e_ops)
        else:
            return self._evolve_numerical(psi0, times, e_ops)

    def _evolve_numerical(
        self, psi0: qutip.Qobj, times: np.ndarray, e_ops: Optional[list] = None
    ) -> qutip.solver.Result:
        """
        Numerical evolution using QuTiP's Schrödinger equation solver.

        Uses adaptive time-stepping ODE solver for accuracy.
        """
        result = qutip.sesolve(self.hamiltonian, psi0, times, e_ops=e_ops)
        return result

    def _evolve_analytical(
        self, psi0: qutip.Qobj, times: np.ndarray, e_ops: Optional[list] = None
    ) -> qutip.solver.Result:
        """
        Analytical evolution for drift Hamiltonian H = (ω/2)σ_z.

        For H ∝ σ_z, the propagator is:
            U(t) = exp(-iHt) = cos(ωt/2)I - i sin(ωt/2)σ_z
        """
        # Extract frequency: H = (ω/2)σ_z
        sz = qutip.sigmaz()
        # H = coeff * sz, solve for coeff
        # Use tr(H * sz) / tr(sz * sz) = coeff
        coeff = (self.hamiltonian * sz).tr() / (sz * sz).tr()
        omega = 2 * coeff  # Since H = (ω/2)σ_z, ω = 2*coeff

        # Evolve state at each time point
        states = []
        for t in times:
            # U(t) = cos(ωt/2)I - i sin(ωt/2)σ_z
            U_t = (
                np.cos(omega * t / 2) * qutip.qeye(2) - 1j * np.sin(omega * t / 2) * sz
            )
            psi_t = U_t * psi0
            states.append(psi_t)

        # Compute expectation values if requested
        expect_vals = []
        if e_ops is not None:
            for op in e_ops:
                expect_vals.append([qutip.expect(op, state) for state in states])

        # Create a simple object to mimic qutip.solver.Result structure
        # QuTiP 5.x Result object requires specific initialization
        class SimpleResult:
            def __init__(self):
                self.states = []
                self.times = None
                self.expect = []

        result = SimpleResult()
        result.states = states
        result.times = times
        if e_ops is not None:
            result.expect = expect_vals
        else:
            result.expect = []

        return result

    def propagator(self, t: float) -> qutip.Qobj:
        """
        Compute the time evolution operator U(t) = exp(-iHt).

        Parameters
        ----------
        t : float
            Time point.

        Returns
        -------
        qutip.Qobj
            Unitary propagator U(t).

        Examples
        --------
        >>> H = qutip.sigmaz()
        >>> evolver = TimeEvolution(H)
        >>> U = evolver.propagator(np.pi)
        >>> print(f"Unitarity check: {(U.dag() * U - qutip.qeye(2)).norm():.2e}")
        Unitarity check: 0.00e+00
        """
        # Use QuTiP's propagator function
        U_t = (-1j * self.hamiltonian * t).expm()
        
        # Validate unitarity using agentbible (if available)
        self._validate_unitary(U_t)
        
        return U_t

    def _validate_unitary(self, U: qutip.Qobj) -> None:
        """Validate that a Qobj is unitary using agentbible validators.
        
        This is a helper method that validates the numpy array representation
        of a qutip.Qobj without changing the return type of the public API.
        """
        if not HAS_AGENTBIBLE:
            return
        
        # Use agentbible's validate_unitary as a function, not decorator
        @validate_unitary(rtol=1e-10, atol=1e-12)
        def _check():
            return U.full()
        
        _check()

    def fidelity_over_time(
        self, psi0: qutip.Qobj, psi_target: qutip.Qobj, times: np.ndarray
    ) -> np.ndarray:
        """
        Compute fidelity with a target state over time.

        Parameters
        ----------
        psi0 : qutip.Qobj
            Initial state.
        psi_target : qutip.Qobj
            Target state to compare against.
        times : np.ndarray
            Time points.

        Returns
        -------
        np.ndarray
            Fidelity values F(|ψ(t)⟩, |ψ_target⟩) at each time point.

        Examples
        --------
        >>> H = qutip.sigmaz()
        >>> evolver = TimeEvolution(H)
        >>> psi0 = qutip.basis(2, 0)
        >>> times = np.linspace(0, 2*np.pi, 100)
        >>> fidelities = evolver.fidelity_over_time(psi0, psi0, times)
        >>> print(f"Fidelity at end: {fidelities[-1]:.6f}")
        Fidelity at end: 1.000000
        """
        result = self.evolve(psi0, times)
        fidelities = np.array(
            [qutip.fidelity(state, psi_target) for state in result.states]
        )
        return fidelities

    def compare_methods(self, psi0: qutip.Qobj, times: np.ndarray) -> dict:
        """
        Compare analytical and numerical evolution methods.

        Parameters
        ----------
        psi0 : qutip.Qobj
            Initial state.
        times : np.ndarray
            Time points.

        Returns
        -------
        dict
            Dictionary with keys:
            - 'numerical_states': States from numerical solver
            - 'analytical_states': States from analytical solver (if available)
            - 'fidelities': Fidelity between methods at each time
            - 'max_error': Maximum fidelity error

        Examples
        --------
        >>> H = 0.5 * 5.0 * qutip.sigmaz()
        >>> evolver = TimeEvolution(H)
        >>> psi0 = qutip.basis(2, 0)
        >>> times = np.linspace(0, 1.0, 50)
        >>> comparison = evolver.compare_methods(psi0, times)
        >>> print(f"Max error: {comparison['max_error']:.2e}")
        Max error: 1.23e-14
        """
        if not self._is_analytical_available():
            raise ValueError("Analytical solution not available for comparison")

        # Numerical evolution
        result_num = self._evolve_numerical(psi0, times)

        # Analytical evolution
        result_ana = self._evolve_analytical(psi0, times)

        # Compute fidelities between methods
        fidelities = np.array(
            [
                qutip.fidelity(state_num, state_ana)
                for state_num, state_ana in zip(result_num.states, result_ana.states)
            ]
        )

        max_error = 1.0 - fidelities.min()

        return {
            "numerical_states": result_num.states,
            "analytical_states": result_ana.states,
            "fidelities": fidelities,
            "max_error": max_error,
        }

    def __repr__(self) -> str:
        """String representation."""
        return f"TimeEvolution(method='{self.method}')"


def bloch_coordinates(psi: qutip.Qobj) -> tuple[float, float, float]:
    """
    Compute Bloch sphere coordinates (x, y, z) for a qubit state.

    Parameters
    ----------
    psi : qutip.Qobj
        Qubit state (ket).

    Returns
    -------
    x, y, z : float
        Bloch sphere coordinates where:
        - x = ⟨σ_x⟩
        - y = ⟨σ_y⟩
        - z = ⟨σ_z⟩

    Examples
    --------
    >>> psi = qutip.basis(2, 0)  # |0⟩
    >>> x, y, z = bloch_coordinates(psi)
    >>> print(f"Bloch vector: ({x:.2f}, {y:.2f}, {z:.2f})")
    Bloch vector: (0.00, 0.00, 1.00)
    """
    sx = qutip.sigmax()
    sy = qutip.sigmay()
    sz = qutip.sigmaz()

    x = qutip.expect(sx, psi)
    y = qutip.expect(sy, psi)
    z = qutip.expect(sz, psi)

    return x, y, z


def bloch_trajectory(states: list[qutip.Qobj]) -> np.ndarray:
    """
    Compute Bloch sphere trajectory for a sequence of states.

    Parameters
    ----------
    states : list of qutip.Qobj
        List of quantum states.

    Returns
    -------
    np.ndarray
        Array of shape (len(states), 3) containing (x, y, z) coordinates.

    Examples
    --------
    >>> H = qutip.sigmaz()
    >>> evolver = TimeEvolution(H)
    >>> psi0 = qutip.basis(2, 0)
    >>> times = np.linspace(0, 1.0, 50)
    >>> result = evolver.evolve(psi0, times)
    >>> trajectory = bloch_trajectory(result.states)
    >>> print(trajectory.shape)
    (50, 3)
    """
    trajectory = np.array([bloch_coordinates(state) for state in states])
    return trajectory
