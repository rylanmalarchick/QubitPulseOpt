"""
Phase 3 GRAPE Wrapper
=====================

Simplified GRAPE optimizer wrapper for Phase 3 demonstration.
This bypasses complex relative import issues while providing full functionality.

Author: QubitPulseOpt Development Team
Phase: 3 - Pulse Optimization
"""

import numpy as np
import qutip as qt
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class GRAPEResult:
    """
    Result container for GRAPE optimization.

    Attributes
    ----------
    final_fidelity : float
        Fidelity of optimized pulse with target.
    optimized_pulses : np.ndarray
        Optimized control amplitudes, shape (n_controls, n_timeslices).
    fidelity_history : list
        Fidelity at each iteration.
    gradient_norms : list
        Gradient norm at each iteration (convergence metric).
    n_iterations : int
        Number of iterations performed.
    converged : bool
        Whether optimization converged (gradient norm < threshold).
    message : str
        Status message describing termination condition.
    """

    final_fidelity: float
    optimized_pulses: np.ndarray
    fidelity_history: List[float]
    gradient_norms: List[float]
    n_iterations: int
    converged: bool
    message: str


class SimpleGRAPEOptimizer:
    """
    Simplified GRAPE optimizer for quantum control.

    Optimizes piecewise-constant control pulses to maximize fidelity
    with a target unitary using gradient ascent.

    Parameters
    ----------
    H_drift : qutip.Qobj
        Drift Hamiltonian (time-independent part)
    H_controls : list of qutip.Qobj
        Control Hamiltonians (one per control field)
    n_timeslices : int
        Number of time discretization slices
    total_time : float
        Total evolution time in seconds
    u_limits : tuple of float, optional
        Amplitude limits (u_min, u_max) for controls
    convergence_threshold : float, optional
        Gradient norm threshold for convergence
    max_iterations : int, optional
        Maximum optimization iterations
    learning_rate : float, optional
        Gradient ascent step size
    verbose : bool, optional
        Print progress messages
    """

    def __init__(
        self,
        H_drift: qt.Qobj,
        H_controls: List[qt.Qobj],
        n_timeslices: int,
        total_time: float,
        u_limits: Tuple[float, float] = (-10.0, 10.0),
        convergence_threshold: float = 1e-4,
        max_iterations: int = 500,
        learning_rate: float = 0.1,
        verbose: bool = True,
    ):
        """Initialize GRAPE optimizer."""
        self.H_drift = H_drift
        self.H_controls = H_controls
        self.n_controls = len(H_controls)
        self.n_timeslices = n_timeslices
        self.total_time = total_time
        self.dt = total_time / n_timeslices
        self.u_min, self.u_max = u_limits
        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.verbose = verbose

        # Hilbert space dimension
        self.dim = H_drift.shape[0]

        if self.verbose:
            logger.info(f"GRAPE optimizer initialized:")
            logger.info(f"  Dimension: {self.dim}")
            logger.info(f"  Controls: {self.n_controls}")
            logger.info(f"  Timeslices: {n_timeslices}")
            logger.info(f"  Total time: {total_time * 1e9:.1f} ns")
            logger.info(f"  dt: {self.dt * 1e9:.3f} ns")

    def _compute_fidelity(self, U_target: qt.Qobj, U_evolved: qt.Qobj) -> float:
        """
        Compute fidelity between target and evolved unitaries.

        F = (1/d²) |Tr(U_target† U_evolved)|²
        """
        overlap = (U_target.dag() * U_evolved).tr()
        fidelity = (np.abs(overlap) ** 2) / (self.dim**2)
        return float(fidelity)

    def _evolve_forward(self, pulses: np.ndarray) -> Tuple[List[qt.Qobj], qt.Qobj]:
        """
        Forward propagation: compute time-evolution operators.

        Returns
        -------
        U_list : list of qutip.Qobj
            Propagators for each timeslice: U_k = exp(-i H_k dt)
        U_total : qutip.Qobj
            Total evolution operator: U(T) = U_N ... U_2 U_1
        """
        U_list = []
        U_total = qt.qeye(self.dim)

        for k in range(self.n_timeslices):
            # Total Hamiltonian at timeslice k
            H_k = self.H_drift
            for j in range(self.n_controls):
                H_k = H_k + pulses[j, k] * self.H_controls[j]

            # Propagator for this timeslice
            U_k = (-1j * H_k * self.dt).expm()
            U_list.append(U_k)

            # Accumulate total propagator (right-to-left multiplication)
            U_total = U_k * U_total

        return U_list, U_total

    def _evolve_backward(
        self, U_target: qt.Qobj, U_list: List[qt.Qobj]
    ) -> List[qt.Qobj]:
        """
        Backward propagation: compute adjoint operators.

        Returns
        -------
        X_list : list of qutip.Qobj
            Adjoint operators for gradient computation
        """
        X_list = [None] * self.n_timeslices

        # Start with X_N = U_target†
        X_k = U_target.dag()
        X_list[-1] = X_k

        # Propagate backward
        for k in range(self.n_timeslices - 1, 0, -1):
            X_k = X_k * U_list[k]
            X_list[k - 1] = X_k

        return X_list

    def _compute_gradients(
        self,
        pulses: np.ndarray,
        U_list: List[qt.Qobj],
        X_list: List[qt.Qobj],
        U_total: qt.Qobj,
    ) -> np.ndarray:
        """
        Compute gradients of fidelity with respect to control amplitudes.

        Returns
        -------
        gradients : np.ndarray
            Shape (n_controls, n_timeslices)
        """
        gradients = np.zeros((self.n_controls, self.n_timeslices))

        # Compute forward propagation up to each timeslice
        U_forward = [qt.qeye(self.dim)]
        for k in range(self.n_timeslices):
            U_forward.append(U_forward[-1] * U_list[k])

        # Gradient for each control and timeslice
        for j in range(self.n_controls):
            for k in range(self.n_timeslices):
                # Derivative of propagator w.r.t. u_j[k]
                # dU_k/du_j[k] = -i dt H_j U_k
                dU_k = -1j * self.dt * self.H_controls[j] * U_list[k]

                # Gradient contribution
                # grad = Re[Tr(X_k† dU_k U_{k-1})]
                grad_term = X_list[k].dag() * dU_k * U_forward[k]
                grad_value = grad_term.tr()

                # For fidelity maximization (proper normalization)
                # Multiply by conjugate of total overlap for correct gradient direction
                U_total_overlap = (X_list[0].dag() * U_forward[-1]).tr()
                gradients[j, k] = (
                    2 * np.real(grad_value * np.conj(U_total_overlap)) / self.dim**2
                )

        return gradients

    def optimize(
        self,
        target: qt.Qobj,
        initial_pulses: np.ndarray,
        initial_state: Optional[qt.Qobj] = None,
    ) -> GRAPEResult:
        """
        Run GRAPE optimization.

        Parameters
        ----------
        target : qutip.Qobj
            Target unitary operator
        initial_pulses : np.ndarray
            Initial guess for control pulses, shape (n_controls, n_timeslices)
        initial_state : qutip.Qobj, optional
            Not used (kept for API compatibility)

        Returns
        -------
        result : GRAPEResult
            Optimization results including optimized pulses and convergence info
        """
        if self.verbose:
            logger.info("Starting GRAPE optimization...")

        # Initialize
        pulses = initial_pulses.copy()
        fidelity_history = []
        gradient_norms = []

        # Optimization loop
        for iteration in range(self.max_iterations):
            # Forward propagation
            U_list, U_total = self._evolve_forward(pulses)

            # Compute fidelity
            fidelity = self._compute_fidelity(target, U_total)
            fidelity_history.append(fidelity)

            # Backward propagation
            X_list = self._evolve_backward(target, U_list)

            # Compute gradients
            gradients = self._compute_gradients(pulses, U_list, X_list, U_total)
            grad_norm = np.linalg.norm(gradients)
            gradient_norms.append(grad_norm)

            # Progress logging
            if self.verbose and iteration % 20 == 0:
                logger.info(
                    f"  Iter {iteration:3d}: F = {fidelity:.6f}, "
                    f"|grad| = {grad_norm:.2e}"
                )

            # Check convergence
            if grad_norm < self.convergence_threshold:
                if self.verbose:
                    logger.info(
                        f"  Converged at iteration {iteration}: F = {fidelity:.6f}"
                    )
                return GRAPEResult(
                    final_fidelity=fidelity,
                    optimized_pulses=pulses,
                    fidelity_history=fidelity_history,
                    gradient_norms=gradient_norms,
                    n_iterations=iteration + 1,
                    converged=True,
                    message=f"Converged: gradient norm {grad_norm:.2e} < {self.convergence_threshold}",
                )

            # Gradient ascent update
            pulses = pulses + self.learning_rate * gradients

            # Apply amplitude constraints
            pulses = np.clip(pulses, self.u_min, self.u_max)

        # Max iterations reached
        final_fidelity = fidelity_history[-1]
        if self.verbose:
            logger.info(f"  Max iterations reached: F = {final_fidelity:.6f}")

        return GRAPEResult(
            final_fidelity=final_fidelity,
            optimized_pulses=pulses,
            fidelity_history=fidelity_history,
            gradient_norms=gradient_norms,
            n_iterations=self.max_iterations,
            converged=False,
            message=f"Max iterations ({self.max_iterations}) reached",
        )


def optimize_gate(
    gate_name: str,
    hardware_params: Dict[str, Any],
    n_timeslices: int = 50,
    total_time: float = 50e-9,
    max_iterations: int = 200,
    verbose: bool = True,
) -> Tuple[GRAPEResult, np.ndarray, np.ndarray]:
    """
    Convenience function to optimize a standard gate.

    Parameters
    ----------
    gate_name : str
        Gate to optimize: 'X', 'Y', 'Z', 'H' (Hadamard)
    hardware_params : dict
        Hardware parameters including rabi_frequency, max_amplitude
    n_timeslices : int, optional
        Number of time slices
    total_time : float, optional
        Total gate time in seconds
    max_iterations : int, optional
        Maximum GRAPE iterations
    verbose : bool, optional
        Print progress

    Returns
    -------
    result : GRAPEResult
        Optimization result
    i_waveform : np.ndarray
        I-channel (X-drive) waveform
    q_waveform : np.ndarray
        Q-channel (Y-drive) waveform
    """
    # Extract parameters
    rabi_freq = hardware_params.get("rabi_frequency", 5e6)
    max_amp = hardware_params.get("max_amplitude", 1.0)

    # Define Hamiltonians (rotating frame)
    H_drift = 0 * qt.sigmaz()
    H_controls = [0.5 * qt.sigmax(), 0.5 * qt.sigmay()]

    # Amplitude limits
    u_limits = (-max_amp * rabi_freq * 2 * np.pi, max_amp * rabi_freq * 2 * np.pi)

    # Create optimizer
    optimizer = SimpleGRAPEOptimizer(
        H_drift=H_drift,
        H_controls=H_controls,
        n_timeslices=n_timeslices,
        total_time=total_time,
        u_limits=u_limits,
        convergence_threshold=1e-4,
        max_iterations=max_iterations,
        learning_rate=0.1,
        verbose=verbose,
    )

    # Target gate
    if gate_name.upper() == "X":
        U_target = qt.sigmax()
        u_init = np.zeros((2, n_timeslices))
        u_init[0, :] = 0.1 * rabi_freq  # X-drive
    elif gate_name.upper() == "Y":
        U_target = qt.sigmay()
        u_init = np.zeros((2, n_timeslices))
        u_init[1, :] = 0.1 * rabi_freq  # Y-drive
    elif gate_name.upper() == "H" or gate_name.upper() == "HADAMARD":
        # Hadamard gate: H = (1/√2) [[1, 1], [1, -1]]
        U_target = qt.Qobj([[1, 1], [1, -1]]) / np.sqrt(2)
        u_init = np.zeros((2, n_timeslices))
        u_init[0, :] = 0.08 * rabi_freq
        u_init[1, :] = 0.04 * rabi_freq
    else:
        raise ValueError(f"Unknown gate: {gate_name}")

    # Optimize
    result = optimizer.optimize(target=U_target, initial_pulses=u_init)

    # Extract I/Q waveforms
    i_waveform = result.optimized_pulses[0, :]
    q_waveform = result.optimized_pulses[1, :]

    return result, i_waveform, q_waveform
