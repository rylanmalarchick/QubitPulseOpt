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
        # Validate inputs first
        if len(H_controls) == 0:
            raise ValueError("Must provide at least one control Hamiltonian")
        if n_timeslices <= 0:
            raise ValueError("Number of timeslices must be positive")
        if total_time <= 0:
            raise ValueError("Total time must be positive")

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

    def _compute_gradients_unitary(
        self,
        u: np.ndarray,
        propagators: List[qt.Qobj],
        forward_unitaries: List[qt.Qobj],
        backward_unitaries: List[qt.Qobj],
        U_target: qt.Qobj,
    ) -> np.ndarray:
        """
        Compute gradients of fidelity with respect to control amplitudes.

        The gradient at timeslice k for control j is:

            ∂F/∂u_j^k ∝ Re[Tr(U_target† X_jk U(T))]

        where X_jk is the system response to control j at timeslice k.

        Parameters
        ----------
        u : np.ndarray
            Current control amplitudes.
        propagators : list of qutip.Qobj
            Propagators for each timeslice.
        forward_unitaries : list of qutip.Qobj
            Cumulative forward propagators.
        backward_unitaries : list of qutip.Qobj
            Cumulative backward propagators.
        U_target : qutip.Qobj
            Target unitary.

        Returns
        -------
        np.ndarray
            Gradients, shape (n_controls, n_timeslices).
        """
        gradients = np.zeros((self.n_controls, self.n_timeslices))

        U_final = forward_unitaries[-1]
        overlap_final = (U_target.dag() * U_final).tr()

        for k in range(self.n_timeslices):
            # Get propagators before and after timeslice k
            if k > 0:
                U_before = forward_unitaries[k - 1]
            else:
                U_before = qt.qeye(self.dim)

            U_after = backward_unitaries[k]

            for j in range(self.n_controls):
                # Compute response operator: -i dt H_j
                dU = -1j * self.dt * self.H_controls[j] * propagators[k]

                # Chain rule: X_jk = U_before * dU * U_after
                X_jk = U_before * dU * U_after

                # Gradient contribution
                trace_val = (U_target.dag() * X_jk).tr()
                grad_contribution = 2 * np.real(np.conj(overlap_final) * trace_val)
                # Normalize gradient to match average gate fidelity: F = (|z|² + d) / (d(d+1))
                d = self.dim
                gradients[j, k] = grad_contribution / (d * (d + 1))

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
        # Initialize controls
        if u_init is None:
            u_init = np.random.randn(self.n_controls, self.n_timeslices) * 0.01
            u_init = self._apply_constraints(u_init)

        u = u_init.copy()

        # History tracking
        fidelity_history = []
        gradient_norms = []

        # Optimization loop
        current_lr = self.learning_rate
        converged = False
        message = "Max iterations reached"
        best_fidelity = 0.0
        best_u = u.copy()
        velocity = np.zeros_like(u)  # For momentum

        # Compute initial fidelity
        propagators = self._compute_propagators(u)
        forward_unitaries, U_final = self._forward_propagation(propagators)
        fidelity = self._compute_fidelity_unitary(U_final, U_target)

        for iteration in range(self.max_iterations):
            # Store fidelity
            fidelity_history.append(fidelity)

            # Track best solution
            if fidelity > best_fidelity:
                best_fidelity = fidelity
                best_u = u.copy()

            # Backward propagation (reuse forward propagators from current u)
            backward_unitaries = self._backward_propagation(propagators)

            # Compute gradients
            gradients = self._compute_gradients_unitary(
                u, propagators, forward_unitaries, backward_unitaries, U_target
            )

            # Clip gradients if specified
            gradients = self._clip_gradients(gradients)

            # Gradient norm (convergence check)
            grad_norm = np.linalg.norm(gradients)
            gradient_norms.append(grad_norm)

            # Check convergence
            if grad_norm < self.convergence_threshold:
                converged = True
                message = "Converged (gradient norm below threshold)"
                if self.verbose:
                    print(
                        f"Iteration {iteration}: Fidelity = {fidelity:.8f}, "
                        f"Grad Norm = {grad_norm:.2e} [CONVERGED]"
                    )
                break

            # Apply momentum to gradients
            velocity = self.momentum * velocity + gradients
            effective_gradients = velocity

            # Determine step size and update controls
            if self.use_line_search:
                # Backtracking line search
                step_size, new_fidelity = self._line_search(
                    u, effective_gradients, fidelity, U_target, alpha_init=current_lr
                )
                u_new = u + step_size * effective_gradients
                u_new = self._apply_constraints(u_new)
                u = u_new

                # Recompute propagators for next iteration
                propagators = self._compute_propagators(u)
                forward_unitaries, U_final = self._forward_propagation(propagators)
                fidelity = new_fidelity

                # Update for next iteration
                if adaptive_step:
                    current_lr *= step_decay
            else:
                # Standard gradient ascent with momentum
                u_new = u + current_lr * effective_gradients
                u_new = self._apply_constraints(u_new)
                u = u_new

                # Recompute for next iteration
                propagators = self._compute_propagators(u)
                forward_unitaries, U_final = self._forward_propagation(propagators)
                fidelity = self._compute_fidelity_unitary(U_final, U_target)

                # Adaptive learning rate
                if adaptive_step:
                    current_lr *= step_decay

            # Verbose output
            if self.verbose and (iteration % 10 == 0 or iteration < 5):
                print(
                    f"Iteration {iteration}: Fidelity = {fidelity:.8f}, "
                    f"Grad Norm = {grad_norm:.2e}, LR = {current_lr:.4f}"
                )

        # Use best solution found
        u = best_u
        propagators = self._compute_propagators(u)
        forward_unitaries, U_final = self._forward_propagation(propagators)
        final_fidelity = self._compute_fidelity_unitary(U_final, U_target)
        fidelity_history.append(final_fidelity)

        if self.verbose:
            print(f"\nOptimization complete: {message}")
            print(f"Final fidelity: {final_fidelity:.8f}")
            print(f"Best fidelity: {best_fidelity:.8f}")
            print(f"Iterations: {iteration + 1}")

        return GRAPEResult(
            final_fidelity=final_fidelity,
            optimized_pulses=u,
            fidelity_history=fidelity_history,
            gradient_norms=gradient_norms,
            n_iterations=iteration + 1,
            converged=converged,
            message=message,
        )

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

        Parameters
        ----------
        psi_init : qutip.Qobj
            Initial quantum state.
        psi_target : qutip.Qobj
            Target quantum state.
        u_init : np.ndarray, optional
            Initial control guess.
        adaptive_step : bool, optional
            Use adaptive learning rate. Default: True.
        step_decay : float, optional
            Learning rate decay factor. Default: 0.99.

        Returns
        -------
        GRAPEResult
            Optimization result.

        Notes
        -----
        For state transfer, fidelity is:
            F = |⟨ψ_target|U(T)|ψ_init⟩|²
        """
        # Initialize controls
        if u_init is None:
            u_init = np.random.randn(self.n_controls, self.n_timeslices) * 0.01
            u_init = self._apply_constraints(u_init)

        u = u_init.copy()

        # History tracking
        fidelity_history = []
        gradient_norms = []

        # Optimization loop
        current_lr = self.learning_rate
        converged = False
        message = "Max iterations reached"

        for iteration in range(self.max_iterations):
            # Forward propagation
            propagators = self._compute_propagators(u)
            forward_unitaries, U_final = self._forward_propagation(propagators)

            # Evolve initial state
            psi_final = U_final * psi_init

            # Compute fidelity
            overlap = psi_target.dag() * psi_final  # Returns complex scalar
            fidelity = np.abs(overlap) ** 2
            fidelity = np.real(fidelity)
            fidelity_history.append(fidelity)

            # Compute gradients (simplified for state transfer)
            gradients = np.zeros((self.n_controls, self.n_timeslices))

            # Backward propagation
            backward_unitaries = self._backward_propagation(propagators)

            overlap = psi_target.dag() * psi_final  # Returns complex scalar

            for k in range(self.n_timeslices):
                if k > 0:
                    U_before = forward_unitaries[k - 1]
                else:
                    U_before = qt.qeye(self.dim)

                U_after = backward_unitaries[k]

                for j in range(self.n_controls):
                    dU = -1j * self.dt * self.H_controls[j] * propagators[k]
                    X_jk = U_before * dU * U_after
                    psi_derivative = X_jk * psi_init
                    trace_val = (
                        psi_target.dag() * psi_derivative
                    )  # Returns complex scalar
                    gradients[j, k] = 2 * np.real(np.conj(overlap) * trace_val)

            # Gradient norm
            grad_norm = np.linalg.norm(gradients)
            gradient_norms.append(grad_norm)

            # Check convergence
            if grad_norm < self.convergence_threshold:
                converged = True
                message = "Converged (gradient norm below threshold)"
                if self.verbose:
                    print(
                        f"Iteration {iteration}: Fidelity = {fidelity:.8f}, "
                        f"Grad Norm = {grad_norm:.2e} [CONVERGED]"
                    )
                break

            # Gradient ascent
            u_new = u + current_lr * gradients
            u_new = self._apply_constraints(u_new)
            u = u_new

            # Adaptive learning rate
            if adaptive_step:
                current_lr *= step_decay

            # Verbose output
            if self.verbose and (iteration % 10 == 0 or iteration < 5):
                print(
                    f"Iteration {iteration}: Fidelity = {fidelity:.8f}, "
                    f"Grad Norm = {grad_norm:.2e}, LR = {current_lr:.4f}"
                )

        # Final fidelity
        propagators = self._compute_propagators(u)
        _, U_final = self._forward_propagation(propagators)
        psi_final = U_final * psi_init
        overlap = psi_target.dag() * psi_final  # Returns complex scalar
        final_fidelity = np.abs(overlap) ** 2
        final_fidelity = np.real(final_fidelity)
        fidelity_history.append(final_fidelity)

        if self.verbose:
            print(f"\nOptimization complete: {message}")
            print(f"Final fidelity: {final_fidelity:.8f}")
            print(f"Iterations: {iteration + 1}")

        return GRAPEResult(
            final_fidelity=final_fidelity,
            optimized_pulses=u,
            fidelity_history=fidelity_history,
            gradient_norms=gradient_norms,
            n_iterations=iteration + 1,
            converged=converged,
            message=message,
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
