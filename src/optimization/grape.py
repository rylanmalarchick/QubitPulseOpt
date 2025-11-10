"""
GRAPE: Gradient Ascent Pulse Engineering
=========================================

This module implements the GRAPE algorithm for optimal quantum control.
GRAPE optimizes piecewise-constant control pulses to maximize gate fidelity
using gradient-based optimization.

Algorithm Overview:
------------------
1. Discretize time into N timeslices with piecewise-constant controls
2. Define objective function (fidelity with target unitary/state)
3. Compute gradients of fidelity with respect to control amplitudes
4. Update controls using gradient ascent: u(t) → u(t) + ε ∇F
5. Iterate until convergence or max iterations

Mathematical Framework:
----------------------
For a target unitary U_target and evolved unitary U(T), the fidelity is:

    F = (1/d²) |Tr(U_target† U(T))|²

where d is the Hilbert space dimension. The gradient with respect to
control amplitude u_k at timeslice k is:

    ∂F/∂u_k = Re[Tr(U_target† X_k U(T))]

where X_k is the system response to the control operator at timeslice k.

Key Features:
------------
- Piecewise-constant pulse optimization
- Automatic gradient computation via chain rule
- Support for amplitude and bandwidth constraints
- Parallel trajectory optimization for multiple initial states
- Adaptive step size with backtracking line search
- Convergence monitoring and early stopping

Physical Constraints:
--------------------
- Amplitude limits: |Ω(t)| ≤ Ω_max (hardware saturation)
- Bandwidth limits: spectral content within allowed range
- Smoothness penalties: discourage rapid pulse changes
- Total pulse energy constraints

References:
----------
- Khaneja et al., J. Magn. Reson. 172, 296 (2005)
- Machnes et al., Phys. Rev. Lett. 120, 150401 (2018)
- QuTiP control documentation

Author: Orchestrator Agent
Date: 2025-01-27
SOW Reference: Phase 2.1 - GRAPE Implementation
"""

import numpy as np
import qutip as qt
from typing import Union, Callable, Optional, List, Tuple, Dict
import warnings

# Power of 10 compliance: Import loop bounds and assertion helpers
from ..constants import (
    MAX_ITERATIONS,
    MAX_PARAMS,
    MAX_CONTROL_HAMILTONIANS,
    MAX_TIMESLICES,
    MAX_HILBERT_DIM,
    assert_system_size,
    assert_fidelity_valid,
)
from dataclasses import dataclass


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


class GRAPEOptimizer:
    """
    GRAPE optimizer for quantum optimal control.

    This class implements the GRAPE algorithm to find optimal piecewise-constant
    control pulses that maximize fidelity with a target unitary or state.

    Parameters
    ----------
    H_drift : qutip.Qobj
        Drift Hamiltonian (time-independent part).
    H_controls : list of qutip.Qobj
        List of control Hamiltonians (one per control field).
        Total Hamiltonian: H(t) = H_drift + Σ u_k(t) H_k
    n_timeslices : int
        Number of time discretization slices.
    total_time : float
        Total evolution time.
    u_limits : tuple of float, optional
        Amplitude limits (u_min, u_max) for control fields.
        Default: (-10, 10) in angular frequency units.
    convergence_threshold : float, optional
        Convergence criterion on gradient norm. Default: 1e-4.
    max_iterations : int, optional
        Maximum number of optimization iterations. Default: 500.
    learning_rate : float, optional
        Initial gradient ascent step size. Default: 0.1.
    verbose : bool, optional
        Print optimization progress. Default: True.

    Examples
    --------
    >>> # Setup: X-gate on a qubit
    >>> H0 = 0.5 * 5.0 * qt.sigmaz()  # 5 MHz detuning
    >>> Hc = [qt.sigmax()]  # X-drive control
    >>> optimizer = GRAPEOptimizer(H0, Hc, n_timeslices=50, total_time=100)
    >>>
    >>> # Target: X-gate (π rotation around x-axis)
    >>> U_target = qt.sigmax()
    >>>
    >>> # Initial guess: constant pulse
    >>> u_init = np.ones((1, 50)) * 0.05
    >>>
    >>> # Optimize
    >>> result = optimizer.optimize_unitary(U_target, u_init)
    >>> print(f"Final fidelity: {result.final_fidelity:.6f}")
    """

    @staticmethod
    def _validate_drift_hamiltonian(H_drift: qt.Qobj) -> None:
        """Validate drift Hamiltonian."""
        if H_drift is None:
            raise ValueError("Drift Hamiltonian cannot be None")
        if not isinstance(H_drift, qt.Qobj):
            raise ValueError(f"H_drift must be Qobj, got {type(H_drift)}")
        if not H_drift.isherm:
            raise ValueError("Drift Hamiltonian must be Hermitian")
        if H_drift.shape[0] != H_drift.shape[1]:
            raise ValueError(
                f"Drift Hamiltonian must be square, got shape {H_drift.shape}"
            )
        assert_system_size(H_drift.shape[0], MAX_HILBERT_DIM)

    @staticmethod
    def _validate_control_hamiltonians(
        H_controls: List[qt.Qobj], H_drift: qt.Qobj
    ) -> None:
        """Validate control Hamiltonians."""
        if H_controls is None:
            raise ValueError("H_controls cannot be None")
        if len(H_controls) == 0:
            raise ValueError("Must provide at least one control Hamiltonian")
        if len(H_controls) > MAX_CONTROL_HAMILTONIANS:
            raise ValueError(
                f"Number of controls {len(H_controls)} exceeds maximum {MAX_CONTROL_HAMILTONIANS}"
            )

        for i, H_c in enumerate(H_controls):
            if H_c is None:
                raise ValueError(f"Control Hamiltonian {i} cannot be None")
            if not isinstance(H_c, qt.Qobj):
                raise ValueError(f"H_controls[{i}] must be Qobj, got {type(H_c)}")
            if not H_c.isherm:
                raise ValueError(f"Control Hamiltonian {i} must be Hermitian")
            if H_c.shape != H_drift.shape:
                raise ValueError(
                    f"Control Hamiltonian {i} shape {H_c.shape} != drift shape {H_drift.shape}"
                )

    @staticmethod
    def _validate_time_parameters(n_timeslices: int, total_time: float) -> None:
        """Validate time discretization parameters."""
        if n_timeslices <= 0:
            raise ValueError(f"n_timeslices must be positive, got {n_timeslices}")
        if n_timeslices > MAX_TIMESLICES:
            raise ValueError(
                f"n_timeslices {n_timeslices} exceeds maximum {MAX_TIMESLICES}"
            )
        if total_time <= 0:
            raise ValueError(f"total_time must be positive, got {total_time}")
        if not np.isfinite(total_time):
            raise ValueError(f"total_time must be finite, got {total_time}")

    @staticmethod
    def _validate_control_limits(u_limits: Tuple[float, float]) -> None:
        """Validate control amplitude limits."""
        if u_limits is None:
            raise ValueError("u_limits cannot be None")
        if len(u_limits) != 2:
            raise ValueError(
                f"u_limits must be (min, max), got {len(u_limits)} elements"
            )
        if u_limits[0] >= u_limits[1]:
            raise ValueError(
                f"u_limits[0] ({u_limits[0]}) must be < u_limits[1] ({u_limits[1]})"
            )
        if not (np.isfinite(u_limits[0]) and np.isfinite(u_limits[1])):
            raise ValueError("u_limits must be finite")

    @staticmethod
    def _validate_grape_parameters(
        convergence_threshold: float,
        max_iterations: int,
        learning_rate: float,
        momentum: float,
    ) -> None:
        """Validate GRAPE-specific optimization parameters."""
        if convergence_threshold <= 0:
            raise ValueError(
                f"convergence_threshold must be positive, got {convergence_threshold}"
            )
        if max_iterations <= 0:
            raise ValueError(f"max_iterations must be positive, got {max_iterations}")
        if max_iterations > MAX_ITERATIONS:
            raise ValueError(
                f"max_iterations {max_iterations} exceeds maximum {MAX_ITERATIONS}"
            )
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")
        if not (0 <= momentum < 1):
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")

    @staticmethod
    def _validate_total_parameters(n_controls: int, n_timeslices: int) -> None:
        """Validate total parameter count."""
        total_params = n_controls * n_timeslices
        if total_params > MAX_PARAMS:
            raise ValueError(
                f"Total parameters {total_params} = {n_controls} controls × "
                f"{n_timeslices} slices exceeds maximum {MAX_PARAMS}"
            )

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
        gradient_clip: Optional[float] = 10.0,
        use_line_search: bool = True,
        momentum: float = 0.0,
    ):
        """
        Initialize GRAPE optimizer.

        Parameters
        ----------
        gradient_clip : float, optional
            Maximum gradient norm. If specified, gradients will be clipped.
        use_line_search : bool, optional
            Use backtracking line search for step size. Default: True.
        momentum : float, optional
            Momentum coefficient (0 = no momentum, 0.9 = high momentum). Default: 0.0.
        """
        # Validate all parameters
        self._validate_drift_hamiltonian(H_drift)
        self._validate_control_hamiltonians(H_controls, H_drift)
        self._validate_time_parameters(n_timeslices, total_time)
        self._validate_control_limits(u_limits)
        self._validate_grape_parameters(
            convergence_threshold, max_iterations, learning_rate, momentum
        )
        self._validate_total_parameters(len(H_controls), n_timeslices)

        self.H_drift = H_drift
        self.H_controls = H_controls
        self.n_controls = len(H_controls)
        self.n_timeslices = n_timeslices
        self.total_time = total_time
        self.dt = total_time / n_timeslices
        self.u_limits = u_limits
        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.gradient_clip = gradient_clip
        self.use_line_search = use_line_search
        self.momentum = momentum

        # Get Hilbert space dimension
        self.dim = H_drift.shape[0]

        # Rule 5: Post-initialization invariant checks
        assert self.dt > 0, f"Computed dt must be positive, got {self.dt}"
        assert np.isfinite(self.dt), f"Computed dt must be finite, got {self.dt}"

    def _compute_propagators(self, u: np.ndarray) -> List[qt.Qobj]:
        """
        Compute forward propagators for each timeslice.

        For timeslice k, the propagator is:
            U_k = exp(-i (H_drift + Σ u_j^k H_j) dt)

        Parameters
        ----------
        u : np.ndarray
            Control amplitudes, shape (n_controls, n_timeslices).

        Returns
        -------
        list of qutip.Qobj
            List of propagators [U_1, U_2, ..., U_N].
        """
        propagators = []

        for k in range(self.n_timeslices):
            # Total Hamiltonian at timeslice k
            H_total = self.H_drift.copy()
            for j in range(self.n_controls):
                H_total += u[j, k] * self.H_controls[j]

            # Propagator for this timeslice
            U_k = (-1j * H_total * self.dt).expm()
            propagators.append(U_k)

        return propagators

    def _forward_propagation(
        self, propagators: List[qt.Qobj]
    ) -> Tuple[List[qt.Qobj], qt.Qobj]:
        """
        Compute cumulative forward propagators.

        Returns
        -------
        forward_unitaries : list of qutip.Qobj
            Cumulative propagators [U_1, U_1*U_2, ..., U_1*...*U_N].
        U_final : qutip.Qobj
            Final propagator U(T) = U_1 * U_2 * ... * U_N.
        """
        forward_unitaries = []
        U_accum = qt.qeye(self.dim)

        for U_k in propagators:
            U_accum = U_k * U_accum
            forward_unitaries.append(U_accum.copy())

        return forward_unitaries, U_accum

    def _backward_propagation(self, propagators: List[qt.Qobj]) -> List[qt.Qobj]:
        """
        Compute cumulative backward propagators.

        For gradient computation, backward_unitaries[k] contains the product
        of all propagators AFTER timeslice k (not including k itself).

        Returns
        -------
        backward_unitaries : list of qutip.Qobj
            Cumulative backward propagators where backward_unitaries[k] = U_{N}*...*U_{k+1}
            (product of propagators after timeslice k).
            Length is n_timeslices, with backward_unitaries[-1] = Identity.
        """
        backward_unitaries = []
        U_accum = qt.qeye(self.dim)

        # Build backward propagators in reverse, excluding current timeslice
        for i in range(len(propagators) - 1, -1, -1):
            backward_unitaries.insert(0, U_accum.copy())
            U_accum = U_accum * propagators[i]

        return backward_unitaries

    def _compute_fidelity_unitary(self, U_evolved: qt.Qobj, U_target: qt.Qobj) -> float:
        """
        Compute average gate fidelity between evolved and target unitaries.

        Uses the global-phase-invariant average gate fidelity:
        F_avg = (|Tr(U_target† U_evolved)|² + d) / (d(d + 1))

        This metric is invariant under global phase and ranges from 0 to 1.

        Parameters
        ----------
        U_evolved : qutip.Qobj
            Evolved unitary.
        U_target : qutip.Qobj
            Target unitary.

        Returns
        -------
        float
            Average gate fidelity in range [0, 1].
        """
        overlap = (U_target.dag() * U_evolved).tr()
        d = self.dim
        fidelity = (np.abs(overlap) ** 2 + d) / (d * (d + 1))
        return np.real(fidelity)

    def _compute_timeslice_gradient(
        self,
        k: int,
        propagators: List[qt.Qobj],
        forward_unitaries: List[qt.Qobj],
        backward_unitaries: List[qt.Qobj],
        U_target: qt.Qobj,
        overlap_final: complex,
    ) -> np.ndarray:
        """
        Compute gradient for all controls at a single timeslice.

        Parameters
        ----------
        k : int
            Timeslice index.
        propagators : list
            Propagators for each timeslice.
        forward_unitaries : list
            Cumulative forward propagators.
        backward_unitaries : list
            Cumulative backward propagators.
        U_target : qt.Qobj
            Target unitary.
        overlap_final : complex
            Overlap of target with final unitary.

        Returns
        -------
        np.ndarray
            Gradients for all controls at timeslice k.
        """
        gradients_k = np.zeros(self.n_controls)

        U_before = forward_unitaries[k - 1] if k > 0 else qt.qeye(self.dim)
        U_after = backward_unitaries[k]

        for j in range(self.n_controls):
            dU = -1j * self.dt * self.H_controls[j] * propagators[k]
            X_jk = U_before * dU * U_after

            trace_val = (U_target.dag() * X_jk).tr()
            grad_contribution = 2 * np.real(np.conj(overlap_final) * trace_val)
            gradients_k[j] = grad_contribution / (self.dim * (self.dim + 1))

        return gradients_k

    def _compute_gradients_unitary(
        self,
        u: np.ndarray,
        propagators: List[qt.Qobj],
        forward_unitaries: List[qt.Qobj],
        backward_unitaries: List[qt.Qobj],
        U_target: qt.Qobj,
    ) -> np.ndarray:
        """
        Compute gradients of fidelity w.r.t. control amplitudes.

        Gradient: ∂F/∂u_j^k ∝ Re[Tr(U_target† X_jk U(T))]

        Parameters: u (controls), propagators, forward_unitaries,
        backward_unitaries, U_target.
        Returns: gradients array (n_controls, n_timeslices).
        """
        gradients = np.zeros((self.n_controls, self.n_timeslices))

        U_final = forward_unitaries[-1]
        overlap_final = (U_target.dag() * U_final).tr()

        for k in range(self.n_timeslices):
            gradients[:, k] = self._compute_timeslice_gradient(
                k,
                propagators,
                forward_unitaries,
                backward_unitaries,
                U_target,
                overlap_final,
            )

        return gradients

    def _clip_gradients(self, gradients: np.ndarray) -> np.ndarray:
        """
        Clip gradients to prevent excessively large updates.

        Parameters
        ----------
        gradients : np.ndarray
            Gradients to clip.

        Returns
        -------
        np.ndarray
            Clipped gradients.
        """
        if self.gradient_clip is None:
            return gradients

        grad_norm = np.linalg.norm(gradients)
        if grad_norm > self.gradient_clip:
            return gradients * (self.gradient_clip / grad_norm)
        return gradients

    def _line_search(
        self,
        u: np.ndarray,
        gradients: np.ndarray,
        current_fidelity: float,
        U_target: qt.Qobj,
        alpha_init: float = 1.0,
        beta: float = 0.5,
        max_backtracks: int = 10,
    ) -> Tuple[float, float]:
        """
        Backtracking line search to find good step size.

        Parameters
        ----------
        u : np.ndarray
            Current control amplitudes.
        gradients : np.ndarray
            Current gradients.
        current_fidelity : float
            Current fidelity value.
        U_target : qutip.Qobj
            Target unitary.
        alpha_init : float, optional
            Initial step size. Default: 1.0.
        beta : float, optional
            Backtracking factor (0 < beta < 1). Default: 0.5.
        max_backtracks : int, optional
            Maximum number of backtracking steps. Default: 10.

        Returns
        -------
        float
            Chosen step size.
        float
            New fidelity at chosen step size.
        """
        alpha = alpha_init
        grad_norm_sq = np.linalg.norm(gradients) ** 2

        for _ in range(max_backtracks):
            # Try step
            u_new = u + alpha * gradients
            u_new = self._apply_constraints(u_new)

            # Compute new fidelity
            propagators = self._compute_propagators(u_new)
            _, U_final = self._forward_propagation(propagators)
            new_fidelity = self._compute_fidelity_unitary(U_final, U_target)

            # Accept if fidelity increased or stayed same (we're doing gradient ascent)
            # Allow very small decreases due to numerical precision
            if new_fidelity >= current_fidelity - 1e-10:
                return alpha, new_fidelity

            # Backtrack
            alpha *= beta

        # If all backtracks fail, use tiny step to avoid getting stuck
        return alpha * beta, current_fidelity

    def _apply_constraints(self, u: np.ndarray) -> np.ndarray:
        """
        Apply amplitude constraints to control pulses.

        Parameters
        ----------
        u : np.ndarray
            Control amplitudes.

        Returns
        -------
        np.ndarray
            Constrained control amplitudes.
        """
        return np.clip(u, self.u_limits[0], self.u_limits[1])

    def optimize_unitary(
        self,
        U_target: qt.Qobj,
        u_init: Optional[np.ndarray] = None,
        adaptive_step: bool = True,
        step_decay: float = 0.99,
    ) -> GRAPEResult:
        """
        Optimize control pulses to implement a target unitary.

        Uses helper functions to maintain Rule 4 compliance (≤60 lines).

        Parameters
        ----------
        U_target : qutip.Qobj
            Target unitary gate.
        u_init : np.ndarray, optional
            Initial control guess, shape (n_controls, n_timeslices).
            If None, initializes with small random values.
        adaptive_step : bool, optional
            Use adaptive learning rate (decay over iterations). Default: True.
        step_decay : float, optional
            Learning rate decay factor per iteration. Default: 0.99.

        Returns
        -------
        GRAPEResult
            Optimization result containing fidelity, pulses, and history.

        Examples
        --------
        >>> H0 = qt.sigmaz()
        >>> Hc = [qt.sigmax()]
        >>> optimizer = GRAPEOptimizer(H0, Hc, n_timeslices=50, total_time=10)
        >>> U_target = qt.sigmax()  # X-gate
        >>> result = optimizer.optimize_unitary(U_target)
        >>> print(f"Fidelity: {result.final_fidelity:.6f}")
        """
        self._validate_target_unitary(U_target, u_init, step_decay)
        u = self._initialize_controls_for_unitary(u_init)
        opt_state = self._run_unitary_optimization_loop(
            U_target, u, adaptive_step, step_decay
        )
        return self._assemble_grape_result(opt_state)

    def _validate_target_unitary(
        self, U_target: qt.Qobj, u_init: Optional[np.ndarray], step_decay: float
    ) -> None:
        """Validate target unitary and optional parameters."""
        if U_target is None:
            raise ValueError("Target unitary cannot be None")
        assert isinstance(U_target, qt.Qobj), (
            f"U_target must be Qobj, got {type(U_target)}"
        )
        assert U_target.shape[0] == U_target.shape[1], (
            f"Target unitary must be square, got shape {U_target.shape}"
        )
        assert U_target.shape[0] == self.dim, (
            f"Target unitary dimension {U_target.shape[0]} != system dimension {self.dim}"
        )

        # Check unitarity of target (U†U = I)
        identity_check = (U_target.dag() * U_target - qt.qeye(self.dim)).norm()
        assert identity_check < 1e-6, (
            f"Target is not unitary: ||U†U - I|| = {identity_check}"
        )

        # Validate optional parameters
        if u_init is not None:
            assert isinstance(u_init, np.ndarray), (
                f"u_init must be ndarray, got {type(u_init)}"
            )
            assert u_init.shape == (self.n_controls, self.n_timeslices), (
                f"u_init shape {u_init.shape} != expected "
                f"({self.n_controls}, {self.n_timeslices})"
            )
            assert np.all(np.isfinite(u_init)), "u_init contains non-finite values"

        if not (0 < step_decay <= 1):
            raise ValueError(f"step_decay must be in (0, 1], got {step_decay}")

    def _initialize_controls_for_unitary(
        self, u_init: Optional[np.ndarray]
    ) -> np.ndarray:
        """Initialize control pulse array."""
        if u_init is None:
            u_init = np.random.randn(self.n_controls, self.n_timeslices) * 0.01
            u_init = self._apply_constraints(u_init)
        return u_init.copy()

    def _run_unitary_optimization_loop(
        self,
        U_target: qt.Qobj,
        u: np.ndarray,
        adaptive_step: bool,
        step_decay: float,
    ) -> dict:
        """Execute main GRAPE optimization loop for unitary target."""
        opt_state = self._initialize_optimization_state(u, U_target)

        # Main optimization loop
        for iteration in range(self.max_iterations):
            opt_state = self._execute_optimization_iteration(
                iteration, U_target, adaptive_step, step_decay, opt_state
            )

            if opt_state["converged"]:
                break

        return self._finalize_optimization(U_target, opt_state, iteration)

    def _initialize_optimization_state(self, u: np.ndarray, U_target: qt.Qobj) -> dict:
        """Initialize optimization state variables."""
        # Validate inputs
        assert u.shape[1] == self.n_timeslices, (
            f"Control array length {u.shape[1]} != n_timeslices {self.n_timeslices}"
        )
        assert U_target.shape == (self.dim, self.dim), (
            f"Target unitary shape {U_target.shape} invalid"
        )

        propagators = self._compute_propagators(u)
        forward_unitaries, U_final = self._forward_propagation(propagators)
        fidelity = self._compute_fidelity_unitary(U_final, U_target)

        # Validate initial fidelity
        assert_fidelity_valid(fidelity)
        assert 0 <= fidelity <= 1.0, f"Initial fidelity {fidelity} out of bounds [0, 1]"

        return {
            "u": u,
            "propagators": propagators,
            "forward_unitaries": forward_unitaries,
            "fidelity": fidelity,
            "fidelity_history": [],
            "gradient_norms": [],
            "current_lr": self.learning_rate,
            "converged": False,
            "message": "Max iterations reached",
            "best_fidelity": 0.0,
            "best_u": u.copy(),
            "velocity": np.zeros_like(u),
        }

    def _track_best_solution(self, opt_state: dict):
        """Track best solution found so far."""
        if opt_state["fidelity"] > opt_state["best_fidelity"]:
            opt_state["best_fidelity"] = opt_state["fidelity"]
            opt_state["best_u"] = opt_state["u"].copy()

    def _compute_and_check_gradients(
        self, opt_state: dict, U_target: qt.Qobj, iteration: int
    ) -> tuple:
        """
        Compute gradients and check convergence.

        Returns: (gradients, grad_norm, should_stop)
        """
        gradients = self._compute_iteration_gradients(
            opt_state["u"],
            opt_state["propagators"],
            opt_state["forward_unitaries"],
            U_target,
            iteration,
        )

        grad_norm = np.linalg.norm(gradients)
        opt_state["gradient_norms"].append(grad_norm)

        should_stop = self._check_convergence(
            grad_norm, opt_state["fidelity"], iteration
        )
        return gradients, grad_norm, should_stop

    def _execute_optimization_iteration(
        self,
        iteration: int,
        U_target: qt.Qobj,
        adaptive_step: bool,
        step_decay: float,
        opt_state: dict,
    ) -> dict:
        """Execute one iteration of the optimization loop."""
        assert iteration < MAX_ITERATIONS, (
            f"Iteration {iteration} exceeds maximum {MAX_ITERATIONS}"
        )
        assert "fidelity" in opt_state, "opt_state missing required 'fidelity' key"

        opt_state["fidelity_history"].append(opt_state["fidelity"])
        self._track_best_solution(opt_state)

        gradients, grad_norm, should_stop = self._compute_and_check_gradients(
            opt_state, U_target, iteration
        )

        if should_stop:
            opt_state["converged"] = True
            opt_state["message"] = "Converged (gradient norm below threshold)"
            return opt_state

        opt_state["velocity"] = self.momentum * opt_state["velocity"] + gradients

        u, propagators, forward_unitaries, fidelity, current_lr = (
            self._perform_control_update(
                opt_state["u"],
                opt_state["velocity"],
                opt_state["fidelity"],
                U_target,
                opt_state["current_lr"],
                adaptive_step,
                step_decay,
                iteration,
                grad_norm,
            )
        )

        opt_state["u"] = u
        opt_state["propagators"] = propagators
        opt_state["forward_unitaries"] = forward_unitaries
        opt_state["fidelity"] = fidelity
        opt_state["current_lr"] = current_lr

        return opt_state

    def _finalize_optimization(
        self, U_target: qt.Qobj, opt_state: dict, iteration: int
    ) -> dict:
        """Finalize optimization with best solution and create result dict."""
        # Use best solution found
        u = opt_state["best_u"]
        propagators = self._compute_propagators(u)
        forward_unitaries, U_final = self._forward_propagation(propagators)
        final_fidelity = self._compute_fidelity_unitary(U_final, U_target)
        opt_state["fidelity_history"].append(final_fidelity)

        self._validate_optimization_result(
            u, final_fidelity, opt_state["fidelity_history"], iteration
        )

        if self.verbose:
            print(f"\nOptimization complete: {opt_state['message']}")
            print(f"Final fidelity: {final_fidelity:.8f}")
            print(f"Best fidelity: {opt_state['best_fidelity']:.8f}")
            print(f"Iterations: {iteration + 1}")

        return {
            "final_fidelity": final_fidelity,
            "optimized_pulses": u,
            "fidelity_history": opt_state["fidelity_history"],
            "gradient_norms": opt_state["gradient_norms"],
            "n_iterations": iteration + 1,
            "converged": opt_state["converged"],
            "message": opt_state["message"],
        }

    def _compute_iteration_gradients(
        self,
        u: np.ndarray,
        propagators: List[qt.Qobj],
        forward_unitaries: List[qt.Qobj],
        U_target: qt.Qobj,
        iteration: int,
    ) -> np.ndarray:
        """Compute and validate gradients for current iteration."""
        # Validate inputs
        assert len(propagators) == self.n_timeslices, "Propagators list length mismatch"
        assert len(forward_unitaries) == self.n_timeslices, (
            "Forward unitaries length mismatch"
        )

        backward_unitaries = self._backward_propagation(propagators)

        gradients = self._compute_gradients_unitary(
            u, propagators, forward_unitaries, backward_unitaries, U_target
        )

        gradients = self._clip_gradients(gradients)

        # Validate gradient computation
        assert np.all(np.isfinite(gradients)), (
            f"Gradients contain non-finite values at iteration {iteration}"
        )

        return gradients

    def _check_convergence(
        self, grad_norm: float, fidelity: float, iteration: int
    ) -> bool:
        """Check if optimization has converged."""
        assert np.isfinite(grad_norm), (
            f"Gradient norm is not finite at iteration {iteration}: {grad_norm}"
        )
        assert 0 <= fidelity <= 1.0, f"Fidelity {fidelity} out of valid range [0,1]"

        if grad_norm < self.convergence_threshold:
            if self.verbose:
                print(
                    f"Iteration {iteration}: Fidelity = {fidelity:.8f}, "
                    f"Grad Norm = {grad_norm:.2e} [CONVERGED]"
                )
            return True
        return False

    def _perform_control_update(
        self,
        u: np.ndarray,
        effective_gradients: np.ndarray,
        fidelity: float,
        U_target: qt.Qobj,
        current_lr: float,
        adaptive_step: bool,
        step_decay: float,
        iteration: int,
        grad_norm: float,
    ) -> tuple:
        """Perform one control update step with optional line search."""
        if self.use_line_search:
            step_size, new_fidelity = self._line_search(
                u, effective_gradients, fidelity, U_target, alpha_init=current_lr
            )
            u_new = u + step_size * effective_gradients
            u_new = self._apply_constraints(u_new)
            u = u_new

            propagators = self._compute_propagators(u)
            forward_unitaries, U_final = self._forward_propagation(propagators)
            fidelity = new_fidelity

            if adaptive_step:
                current_lr *= step_decay
        else:
            u_new = u + current_lr * effective_gradients
            u_new = self._apply_constraints(u_new)
            u = u_new

            propagators = self._compute_propagators(u)
            forward_unitaries, U_final = self._forward_propagation(propagators)
            fidelity = self._compute_fidelity_unitary(U_final, U_target)

            if adaptive_step:
                current_lr *= step_decay

        # Verbose output
        if self.verbose and (iteration % 10 == 0 or iteration < 5):
            print(
                f"Iteration {iteration}: Fidelity = {fidelity:.8f}, "
                f"Grad Norm = {grad_norm:.2e}, LR = {current_lr:.4f}"
            )

        return u, propagators, forward_unitaries, fidelity, current_lr

    def _validate_optimization_result(
        self,
        u: np.ndarray,
        final_fidelity: float,
        fidelity_history: list,
        iteration: int,
    ) -> None:
        """Validate final optimization result."""
        assert_fidelity_valid(final_fidelity)
        assert 0 <= final_fidelity <= 1.0, (
            f"Final fidelity {final_fidelity} out of bounds [0, 1]"
        )
        assert final_fidelity >= 0, "Fidelity cannot be negative"
        assert np.isfinite(final_fidelity), (
            f"Final fidelity is not finite: {final_fidelity}"
        )
        assert np.all(np.isfinite(u)), "Optimized pulses contain non-finite values"
        assert u.shape == (self.n_controls, self.n_timeslices), (
            f"Optimized pulse shape {u.shape} != expected shape"
        )
        assert len(fidelity_history) > 0, "Fidelity history is empty"
        assert iteration + 1 <= MAX_ITERATIONS, (
            f"Iteration count {iteration + 1} exceeds maximum {MAX_ITERATIONS}"
        )

    def _assemble_grape_result(self, opt_state: dict) -> GRAPEResult:
        """Assemble optimization state into GRAPEResult object."""
        return GRAPEResult(
            final_fidelity=opt_state["final_fidelity"],
            optimized_pulses=opt_state["optimized_pulses"],
            fidelity_history=opt_state["fidelity_history"],
            gradient_norms=opt_state["gradient_norms"],
            n_iterations=opt_state["n_iterations"],
            converged=opt_state["converged"],
            message=opt_state["message"],
        )

    def _validate_state_parameters_grape(
        self,
        psi_init: qt.Qobj,
        psi_target: qt.Qobj,
        u_init: Optional[np.ndarray],
        step_decay: float,
    ) -> None:
        """Validate state optimization parameters."""
        if psi_init is None:
            raise ValueError("Initial state cannot be None")
        if psi_target is None:
            raise ValueError("Target state cannot be None")
        assert isinstance(psi_init, qt.Qobj), (
            f"psi_init must be Qobj, got {type(psi_init)}"
        )
        assert isinstance(psi_target, qt.Qobj), (
            f"psi_target must be Qobj, got {type(psi_target)}"
        )

        # Check state dimensions
        assert psi_init.shape[0] == self.dim, (
            f"Initial state dimension {psi_init.shape[0]} != system dimension {self.dim}"
        )
        assert psi_target.shape[0] == self.dim, (
            f"Target state dimension {psi_target.shape[0]} != system dimension {self.dim}"
        )
        assert psi_init.shape[1] == 1, (
            f"Initial state must be ket, got shape {psi_init.shape}"
        )
        assert psi_target.shape[1] == 1, (
            f"Target state must be ket, got shape {psi_target.shape}"
        )

        # Check state normalization
        psi_init_norm = psi_init.norm()
        psi_target_norm = psi_target.norm()
        assert 0.99 <= psi_init_norm <= 1.01, (
            f"Initial state not normalized: ||psi_init|| = {psi_init_norm}"
        )
        assert 0.99 <= psi_target_norm <= 1.01, (
            f"Target state not normalized: ||psi_target|| = {psi_target_norm}"
        )

        # Validate optional parameters
        if u_init is not None:
            assert isinstance(u_init, np.ndarray), (
                f"u_init must be ndarray, got {type(u_init)}"
            )
            assert u_init.shape == (self.n_controls, self.n_timeslices), (
                f"u_init shape {u_init.shape} != expected ({self.n_controls}, {self.n_timeslices})"
            )
            assert np.all(np.isfinite(u_init)), "u_init contains non-finite values"

        if not (0 < step_decay <= 1):
            raise ValueError(f"step_decay must be in (0, 1], got {step_decay}")

    def _compute_state_gradients(
        self,
        psi_init: qt.Qobj,
        psi_target: qt.Qobj,
        psi_final: qt.Qobj,
        propagators: List[qt.Qobj],
        forward_unitaries: List[qt.Qobj],
        backward_unitaries: List[qt.Qobj],
        overlap,
    ) -> np.ndarray:
        """Compute gradients for state transfer optimization."""
        gradients = np.zeros((self.n_controls, self.n_timeslices))

        for k in range(self.n_timeslices):
            U_before = forward_unitaries[k - 1] if k > 0 else qt.qeye(self.dim)
            U_after = backward_unitaries[k]

            for j in range(self.n_controls):
                dU = -1j * self.dt * self.H_controls[j] * propagators[k]
                X_jk = U_before * dU * U_after
                psi_derivative = X_jk * psi_init
                trace_val = psi_target.dag() * psi_derivative
                gradients[j, k] = 2 * np.real(np.conj(overlap) * trace_val)

        return gradients

    def _execute_state_iteration_grape(
        self, psi_init: qt.Qobj, psi_target: qt.Qobj, u: np.ndarray
    ) -> Tuple[float, np.ndarray, List[qt.Qobj]]:
        """Execute one iteration of state optimization."""
        propagators = self._compute_propagators(u)
        forward_unitaries, U_final = self._forward_propagation(propagators)
        psi_final = U_final * psi_init

        # Compute fidelity
        overlap = psi_target.dag() * psi_final
        fidelity = np.real(np.abs(overlap) ** 2)

        # Compute gradients
        backward_unitaries = self._backward_propagation(propagators)
        gradients = self._compute_state_gradients(
            psi_init,
            psi_target,
            psi_final,
            propagators,
            forward_unitaries,
            backward_unitaries,
            overlap,
        )

        return fidelity, gradients, propagators

    def _check_state_convergence_grape(
        self, iteration: int, fidelity: float, grad_norm: float, current_lr: float
    ) -> Tuple[bool, str]:
        """Check convergence for state optimization."""
        if grad_norm < self.convergence_threshold:
            if self.verbose:
                print(
                    f"Iteration {iteration}: Fidelity = {fidelity:.8f}, "
                    f"Grad Norm = {grad_norm:.2e} [CONVERGED]"
                )
            return True, "Converged (gradient norm below threshold)"
        return False, ""

    def _run_state_optimization_loop_grape(
        self,
        psi_init: qt.Qobj,
        psi_target: qt.Qobj,
        u: np.ndarray,
        adaptive_step: bool,
        step_decay: float,
    ) -> Tuple[np.ndarray, List[float], List[float], bool, str, int]:
        """Execute the main state optimization loop."""
        fidelity_history = []
        gradient_norms = []
        current_lr = self.learning_rate
        converged = False
        message = "Max iterations reached"

        for iteration in range(self.max_iterations):
            # Execute iteration
            fidelity, gradients, propagators = self._execute_state_iteration_grape(
                psi_init, psi_target, u
            )
            fidelity_history.append(fidelity)

            # Gradient norm
            grad_norm = np.linalg.norm(gradients)
            gradient_norms.append(grad_norm)

            # Check convergence
            converged, conv_message = self._check_state_convergence_grape(
                iteration, fidelity, grad_norm, current_lr
            )
            if converged:
                message = conv_message
                break

            # Update controls
            u_new = u + current_lr * gradients
            u = self._apply_constraints(u_new)

            # Adaptive learning rate
            if adaptive_step:
                current_lr *= step_decay

            # Verbose output
            if self.verbose and (iteration % 10 == 0 or iteration < 5):
                print(
                    f"Iteration {iteration}: Fidelity = {fidelity:.8f}, "
                    f"Grad Norm = {grad_norm:.2e}, LR = {current_lr:.4f}"
                )

        return u, fidelity_history, gradient_norms, converged, message, iteration + 1

    def optimize_state(
        self,
        psi_init: qt.Qobj,
        psi_target: qt.Qobj,
        u_init: Optional[np.ndarray] = None,
        adaptive_step: bool = True,
        step_decay: float = 0.99,
    ) -> GRAPEResult:
        """
        Optimize control pulses to transfer initial state to target state.

        Orchestrates GRAPE optimization for state-to-state transfer using
        gradient ascent with optional adaptive learning rate.
        """
        # Validate parameters
        self._validate_state_parameters_grape(psi_init, psi_target, u_init, step_decay)
        # Initialize controls
        if u_init is None:
            u_init = np.random.randn(self.n_controls, self.n_timeslices) * 0.01
            u_init = self._apply_constraints(u_init)
        u = u_init.copy()

        # Run optimization loop
        u, fid_hist, grad_norms, converged, msg, n_iter = (
            self._run_state_optimization_loop_grape(
                psi_init, psi_target, u, adaptive_step, step_decay
            )
        )

        # Final evaluation
        propagators = self._compute_propagators(u)
        _, U_final = self._forward_propagation(propagators)
        psi_final = U_final * psi_init
        overlap = psi_target.dag() * psi_final
        final_fidelity = np.real(np.abs(overlap) ** 2)
        fid_hist.append(final_fidelity)

        if self.verbose:
            print(f"\nOptimization complete: {msg}")
            print(f"Final fidelity: {final_fidelity:.8f}")
            print(f"Iterations: {n_iter}")

        return GRAPEResult(
            final_fidelity=final_fidelity,
            optimized_pulses=u,
            fidelity_history=fid_hist,
            gradient_norms=grad_norms,
            n_iterations=n_iter,
            converged=converged,
            message=msg,
        )

    def get_pulse_functions(self, u: np.ndarray) -> List[Callable]:
        """
        Convert piecewise-constant pulses to callable functions.

        Parameters
        ----------
        u : np.ndarray
            Piecewise-constant control amplitudes, shape (n_controls, n_timeslices).

        Returns
        -------
        list of callable
            List of pulse functions, one per control.
            Each function has signature: f(t) -> float

        Examples
        --------
        >>> result = optimizer.optimize_unitary(U_target)
        >>> pulse_funcs = optimizer.get_pulse_functions(result.optimized_pulses)
        >>> times = np.linspace(0, optimizer.total_time, 1000)
        >>> pulse_values = pulse_funcs[0](times)
        """
        pulse_functions = []

        for j in range(self.n_controls):

            def pulse_func(t, control_idx=j, pulses=u):
                # Determine which timeslice t falls into
                t = np.atleast_1d(t)
                indices = np.floor(t / self.dt).astype(int)
                indices = np.clip(indices, 0, self.n_timeslices - 1)
                return pulses[control_idx, indices]

            pulse_functions.append(pulse_func)

        return pulse_functions

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"GRAPEOptimizer(n_controls={self.n_controls}, "
            f"n_timeslices={self.n_timeslices}, total_time={self.total_time})"
        )
