"""
Krotov-style optimal control for quantum gates.
=================================================

This module implements gradient-based optimal control using Krotov's adjoint
(co-state) construction for the gradient, combined with a backtracking line
search that makes the recorded fidelity monotonically non-decreasing.

Algorithm
---------
The objective is to maximise the gate (process) fidelity

    F[U(T)] = |Tr(U_target^dagger U(T))|^2 / d^2

for the piecewise-constant control Hamiltonian

    H(t_k) = H_drift + sum_j u_j(t_k) H_j .

Each iteration:

1. Forward-propagate the cumulative propagators U(t_k) and, via the Krotov
   adjoint relation, the backward propagators U_after(t_k). The fidelity
   gradient w.r.t. each control amplitude is

       dF/du_j(t_k) = (2/d^2) Re[ conj(tau) Tr(U_target^dagger X_jk) ],
       X_jk = U_after(t_k) (-i dt H_j U_k) U_before(t_k),  tau = Tr(U_target^dagger U(T)).

2. A backtracking line search picks the largest step along the gradient that
   increases F. The best pulse seen so far is retained, so the reported
   ``fidelity_history`` never decreases. If no step improves F, a bounded number
   of random restarts are attempted to escape a stalled point.

Monotonicity here is enforced numerically by the line search (the recorded
fidelity cannot decrease), rather than claimed from the exact continuous-time
Krotov update. ``optimize_state`` solves the analogous state-transfer problem
with the state fidelity |<psi_target|psi(T)>|^2.

References
----------
- Khaneja et al., J. Magn. Reson. 172, 296 (2005) - gradient pulse engineering
- Reich et al., J. Chem. Phys. 136, 104103 (2012) - Krotov adjoint construction
- Goerz et al., SciPost Phys. 7, 080 (2019) - Krotov implementation notes
"""

import numpy as np
import qutip as qt
from typing import Callable, Optional, List, Tuple
from dataclasses import dataclass

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

# Bounded-loop limits for the line search and restart logic (Power-of-10 Rule 2).
_MAX_LINE_SEARCH_HALVINGS = 50
_MAX_RESTARTS = 20
_CONVERGENCE_PATIENCE = 3
_LINE_SEARCH_MAX_SCALE = 64.0
_STEP_GROWTH = 1.3
_RESTART_NOISE = 0.2


@dataclass
class KrotovResult:
    """
    Result container for Krotov optimization.

    Attributes
    ----------
    final_fidelity : float
        Fidelity of optimized pulse with target.
    optimized_pulses : np.ndarray
        Optimized control amplitudes, shape (n_controls, n_timeslices).
    fidelity_history : list
        Fidelity at each iteration (monotonically non-decreasing).
    delta_fidelity : list
        Change in fidelity per iteration (non-negative by construction).
    n_iterations : int
        Number of iterations performed.
    converged : bool
        Whether optimization converged.
    message : str
        Status message describing termination condition.
    """

    final_fidelity: float
    optimized_pulses: np.ndarray
    fidelity_history: List[float]
    delta_fidelity: List[float]
    n_iterations: int
    converged: bool
    message: str


class KrotovOptimizer:
    """
    Krotov-style gradient optimizer for quantum optimal control.

    Maximises gate fidelity with a Krotov adjoint gradient and a monotonic
    (line-searched) update. Produces smooth, amplitude-bounded pulses.

    Parameters
    ----------
    H_drift : qutip.Qobj
        Drift Hamiltonian (time-independent part).
    H_controls : list of qutip.Qobj
        Control Hamiltonians. Total: H(t) = H_drift + sum_k u_k(t) H_k.
    n_timeslices : int
        Number of time discretization slices.
    total_time : float
        Total evolution time.
    penalty_lambda : float, optional
        Penalty parameter lambda. Sets the initial line-search step (1/lambda):
        larger lambda -> smaller initial updates. Default: 1.0.
    convergence_threshold : float, optional
        Convergence criterion on fidelity change. Optimization stops after
        ``_CONVERGENCE_PATIENCE`` consecutive iterations with change below this.
        Default: 1e-5.
    max_iterations : int, optional
        Maximum number of iterations. Default: 200.
    u_limits : tuple of float, optional
        Amplitude limits (u_min, u_max). Default: (-10, 10).
    verbose : bool, optional
        Print optimization progress. Default: True.

    Examples
    --------
    >>> H0 = 0.5 * 2.0 * qt.sigmaz()
    >>> Hc = [qt.sigmax()]
    >>> optimizer = KrotovOptimizer(H0, Hc, n_timeslices=50, total_time=100,
    ...                             verbose=False)
    >>> result = optimizer.optimize_unitary(qt.sigmax())
    >>> result.final_fidelity > 0.95
    True
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
    def _validate_optimization_parameters(
        penalty_lambda: float,
        convergence_threshold: float,
        max_iterations: int,
        u_limits: Tuple[float, float],
    ) -> None:
        """Validate optimization parameters."""
        if penalty_lambda < 0:
            raise ValueError(
                f"penalty_lambda must be non-negative, got {penalty_lambda}"
            )
        if not np.isfinite(penalty_lambda):
            raise ValueError("penalty_lambda must be finite")
        if penalty_lambda == 0:
            raise ValueError("penalty_lambda must be positive (sets the step 1/lambda)")

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

    @staticmethod
    def _validate_total_parameters(n_controls: int, n_timeslices: int) -> None:
        """Validate total parameter count."""
        total_params = n_controls * n_timeslices
        if total_params > MAX_PARAMS:
            raise ValueError(
                f"Total parameters {total_params} = {n_controls} controls x "
                f"{n_timeslices} slices exceeds maximum {MAX_PARAMS}"
            )

    def __init__(
        self,
        H_drift: qt.Qobj,
        H_controls: List[qt.Qobj],
        n_timeslices: int,
        total_time: float,
        penalty_lambda: float = 1.0,
        convergence_threshold: float = 1e-5,
        max_iterations: int = 200,
        u_limits: Tuple[float, float] = (-10.0, 10.0),
        verbose: bool = True,
    ):
        """Initialize Krotov optimizer."""
        self._validate_drift_hamiltonian(H_drift)
        self._validate_control_hamiltonians(H_controls, H_drift)
        self._validate_time_parameters(n_timeslices, total_time)
        self._validate_optimization_parameters(
            penalty_lambda, convergence_threshold, max_iterations, u_limits
        )
        self._validate_total_parameters(len(H_controls), n_timeslices)

        self.H_drift = H_drift
        self.H_controls = H_controls
        self.n_controls = len(H_controls)
        self.n_timeslices = n_timeslices
        self.total_time = total_time
        self.dt = total_time / n_timeslices
        self.penalty_lambda = penalty_lambda
        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations
        self.u_limits = u_limits
        self.verbose = verbose

        self.dim = H_drift.shape[0]

        # Rule 5: Post-initialization invariant checks
        assert self.dt > 0, f"Computed dt must be positive, got {self.dt}"
        assert np.isfinite(self.dt), f"Computed dt must be finite, got {self.dt}"

    # ------------------------------------------------------------------
    # Propagation
    # ------------------------------------------------------------------

    def _slice_hamiltonian(self, u: np.ndarray, k: int) -> qt.Qobj:
        """Total Hamiltonian at timeslice k."""
        H_total = self.H_drift.copy()
        for j in range(self.n_controls):
            H_total += u[j, k] * self.H_controls[j]
        return H_total

    def _compute_propagators(self, u: np.ndarray) -> List[qt.Qobj]:
        """Per-slice propagators U_k = exp(-i H(t_k) dt)."""
        return [
            (-1j * self._slice_hamiltonian(u, k) * self.dt).expm()
            for k in range(self.n_timeslices)
        ]

    def _forward_propagation(self, psi_init: qt.Qobj, u: np.ndarray) -> List[qt.Qobj]:
        """
        Forward propagate a state under control pulses.

        Returns states [psi(t_0), psi(t_1), ..., psi(t_N)].
        """
        states = [psi_init]
        psi = psi_init.copy()
        for k in range(self.n_timeslices):
            psi = (-1j * self._slice_hamiltonian(u, k) * self.dt).expm() * psi
            states.append(psi)
        return states

    def _apply_constraints(self, u: np.ndarray) -> np.ndarray:
        """Apply amplitude constraints."""
        return np.clip(u, self.u_limits[0], self.u_limits[1])

    # ------------------------------------------------------------------
    # Gate (unitary) fidelity and gradient
    # ------------------------------------------------------------------

    def _gate_fidelity(self, U_evolved: qt.Qobj, U_target: qt.Qobj) -> float:
        """Process fidelity F = |Tr(U_target^dagger U)|^2 / d^2."""
        overlap = (U_target.dag() * U_evolved).tr()
        return float(np.clip(np.abs(overlap) ** 2 / self.dim**2, 0.0, 1.0))

    def _gate_gradient(
        self, u: np.ndarray, U_target: qt.Qobj
    ) -> Tuple[np.ndarray, float]:
        """
        Gradient of the gate fidelity w.r.t. all control amplitudes.

        Uses the Krotov adjoint construction: cumulative forward propagators
        U_before(t_k) and backward propagators U_after(t_k) give

            dU(T)/du_j(t_k) = U_after(t_k) (-i dt H_j U_k) U_before(t_k).

        Returns (gradient[n_controls, n_timeslices], fidelity).
        """
        propagators = self._compute_propagators(u)

        # Cumulative forward propagators: before[k] = U_{k-1} ... U_0 (before[0] = I).
        before = [qt.qeye(self.dim)]
        U_accum = qt.qeye(self.dim)
        for U_k in propagators:
            U_accum = U_k * U_accum
            before.append(U_accum)
        U_final = before[-1]

        # Cumulative backward propagators: after[k] = U_{N-1} ... U_{k+1} (after[N-1] = I).
        after = [None] * self.n_timeslices
        acc = qt.qeye(self.dim)
        for k in range(self.n_timeslices - 1, -1, -1):
            after[k] = acc
            acc = acc * propagators[k]

        overlap = (U_target.dag() * U_final).tr()
        gradient = np.zeros((self.n_controls, self.n_timeslices))
        prefactor = 2.0 / self.dim**2
        for k in range(self.n_timeslices):
            for j in range(self.n_controls):
                dU = after[k] * (-1j * self.dt * self.H_controls[j] * propagators[k]) * before[k]
                trace_val = (U_target.dag() * dU).tr()
                gradient[j, k] = prefactor * np.real(np.conj(overlap) * trace_val)

        fidelity = float(np.clip(np.abs(overlap) ** 2 / self.dim**2, 0.0, 1.0))
        return gradient, fidelity

    # ------------------------------------------------------------------
    # State-transfer fidelity and gradient
    # ------------------------------------------------------------------

    def _state_fidelity(self, psi_final: qt.Qobj, psi_target: qt.Qobj) -> float:
        """State fidelity F = |<psi_target|psi_final>|^2."""
        overlap = psi_target.overlap(psi_final)
        return float(np.clip(np.abs(overlap) ** 2, 0.0, 1.0))

    def _state_gradient(
        self, u: np.ndarray, psi_init: qt.Qobj, psi_target: qt.Qobj
    ) -> Tuple[np.ndarray, float]:
        """
        Gradient of the state-transfer fidelity w.r.t. control amplitudes.

        dF/du_j(t_k) = 2 dt Im[ conj(tau) <chi(t_k)|H_j|psi(t_k)> ], with the
        linear co-state chi(T) = psi_target back-propagated under the same field
        and tau = <psi_target|psi(T)>.

        Returns (gradient[n_controls, n_timeslices], fidelity).
        """
        states = self._forward_propagation(psi_init, u)
        tau = psi_target.overlap(states[-1])

        # Backward co-state chi(t_k) = exp(+i H(t_k) dt) chi(t_{k+1}), chi(T) = psi_target.
        costates = [None] * (self.n_timeslices + 1)
        chi = psi_target.copy()
        costates[self.n_timeslices] = chi
        for k in range(self.n_timeslices - 1, -1, -1):
            chi = (1j * self._slice_hamiltonian(u, k) * self.dt).expm() * chi
            costates[k] = chi

        gradient = np.zeros((self.n_controls, self.n_timeslices))
        for k in range(self.n_timeslices):
            for j in range(self.n_controls):
                mel = costates[k].overlap(self.H_controls[j] * states[k])
                gradient[j, k] = 2.0 * self.dt * np.imag(np.conj(tau) * mel)

        return gradient, float(np.clip(np.abs(tau) ** 2, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Monotonic line-searched optimization core
    # ------------------------------------------------------------------

    def _line_search(
        self,
        u: np.ndarray,
        gradient: np.ndarray,
        fidelity_fn: Callable[[np.ndarray], float],
        base_step: float,
        f_current: float,
    ) -> Tuple[Optional[np.ndarray], float]:
        """
        Backtracking line search along the gradient.

        Returns the largest-step improving pulse (and its fidelity), or
        (None, f_current) if no tried step increases the fidelity.
        """
        step = base_step * _LINE_SEARCH_MAX_SCALE
        for _ in range(_MAX_LINE_SEARCH_HALVINGS):
            candidate = self._apply_constraints(u + step * gradient)
            f_candidate = fidelity_fn(candidate)
            if f_candidate > f_current:
                return candidate, f_candidate
            step *= 0.5
        return None, f_current

    def _attempt_restart(
        self,
        u_best: np.ndarray,
        f_best: float,
        fidelity_fn: Callable[[np.ndarray], float],
        gradient_fn: Callable[[np.ndarray], Tuple[np.ndarray, float]],
        base_step: float,
    ) -> Tuple[Optional[np.ndarray], float]:
        """
        Escape a stalled point with bounded random restarts.

        Returns an improving pulse found from a perturbed start, or (None, f_best).
        """
        for _ in range(_MAX_RESTARTS):
            perturbed = self._apply_constraints(
                u_best + np.random.randn(*u_best.shape) * _RESTART_NOISE
            )
            grad_p, _ = gradient_fn(perturbed)
            cand, f_cand = self._line_search(
                perturbed, grad_p, fidelity_fn, base_step, fidelity_fn(perturbed)
            )
            if cand is None:
                cand, f_cand = perturbed, fidelity_fn(perturbed)
            if f_cand > f_best:
                return cand, f_cand
        return None, f_best

    def _optimize_core(
        self,
        fidelity_fn: Callable[[np.ndarray], float],
        gradient_fn: Callable[[np.ndarray], Tuple[np.ndarray, float]],
        u_init: np.ndarray,
    ) -> KrotovResult:
        """
        Run the monotonic line-searched optimization loop.

        Maintains the best pulse seen so far, so ``fidelity_history`` is
        non-decreasing. Stops on convergence (patience consecutive sub-threshold
        improvements), on an unrecoverable stall, or at ``max_iterations``.
        """
        u_best = self._apply_constraints(u_init.copy())
        f_best = fidelity_fn(u_best)
        fidelity_history = [f_best]
        delta_history: List[float] = []
        base_step = 1.0 / self.penalty_lambda
        converged = False
        message = "Max iterations reached"
        small_steps = 0
        n_iterations = 0

        for iteration in range(self.max_iterations):
            n_iterations = iteration + 1
            gradient, _ = gradient_fn(u_best)
            u_new, f_new = self._line_search(
                u_best, gradient, fidelity_fn, base_step, f_best
            )

            if u_new is None:
                u_new, f_new = self._attempt_restart(
                    u_best, f_best, fidelity_fn, gradient_fn, base_step
                )

            if u_new is None or f_new <= f_best:
                converged = True
                message = "Converged (no further improvement found)"
                break

            delta = f_new - f_best
            u_best, f_best = u_new, f_new
            fidelity_history.append(f_best)
            delta_history.append(delta)

            if self.verbose and (iteration % 10 == 0 or iteration < 5):
                print(f"Iteration {iteration}: Fidelity = {f_best:.8f}, dF = {delta:.2e}")

            small_steps = small_steps + 1 if delta < self.convergence_threshold else 0
            if small_steps >= _CONVERGENCE_PATIENCE:
                converged = True
                message = "Converged (fidelity change below threshold)"
                break

        assert_fidelity_valid(f_best)
        assert np.all(np.isfinite(u_best)), "Optimized pulses contain non-finite values"
        if self.verbose:
            print(f"\nOptimization complete: {message}")
            print(f"Final fidelity: {f_best:.8f}")
            print(f"Iterations: {n_iterations}")

        return KrotovResult(
            final_fidelity=float(f_best),
            optimized_pulses=u_best,
            fidelity_history=fidelity_history,
            delta_fidelity=delta_history,
            n_iterations=int(n_iterations),
            converged=bool(converged),
            message=message,
        )

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _default_controls(self) -> np.ndarray:
        """Small random initial control pulse."""
        u = np.random.randn(self.n_controls, self.n_timeslices) * 0.1
        return self._apply_constraints(u)

    # ------------------------------------------------------------------
    # Public optimization entry points
    # ------------------------------------------------------------------

    def optimize_unitary(
        self,
        U_target: qt.Qobj,
        psi_init: Optional[qt.Qobj] = None,
        u_init: Optional[np.ndarray] = None,
    ) -> KrotovResult:
        """
        Optimize control pulses to implement a target unitary gate.

        Maximises the full gate (process) fidelity |Tr(U_target^dagger U(T))|^2/d^2,
        i.e. the whole unitary, not a single state transfer.

        Parameters
        ----------
        U_target : qutip.Qobj
            Target unitary gate (d x d).
        psi_init : qutip.Qobj, optional
            Unused for gate optimization; accepted for API compatibility.
        u_init : np.ndarray, optional
            Initial control guess, shape (n_controls, n_timeslices). If None,
            a small random pulse is used.

        Returns
        -------
        KrotovResult
            Optimization result containing fidelity, optimized pulses, and history.
        """
        if not isinstance(U_target, qt.Qobj):
            raise ValueError(f"U_target must be Qobj, got {type(U_target)}")
        if U_target.shape != (self.dim, self.dim):
            raise ValueError(
                f"U_target shape {U_target.shape} != system shape "
                f"({self.dim}, {self.dim})"
            )

        u_start = self._default_controls() if u_init is None else u_init.copy()

        return self._optimize_core(
            fidelity_fn=lambda u: self._gate_fidelity(
                self._forward_unitary(u), U_target
            ),
            gradient_fn=lambda u: self._gate_gradient(u, U_target),
            u_init=u_start,
        )

    def _forward_unitary(self, u: np.ndarray) -> qt.Qobj:
        """Total propagator U(T) = U_{N-1} ... U_0."""
        U = qt.qeye(self.dim)
        for U_k in self._compute_propagators(u):
            U = U_k * U
        return U

    def _validate_state_parameters(
        self, psi_init: qt.Qobj, psi_target: qt.Qobj, u_init: Optional[np.ndarray]
    ) -> None:
        """Validate state optimization parameters."""
        if psi_init is None:
            raise ValueError("Initial state cannot be None")
        if psi_target is None:
            raise ValueError("Target state cannot be None")
        if not isinstance(psi_init, qt.Qobj):
            raise ValueError(f"psi_init must be Qobj, got {type(psi_init)}")
        if not isinstance(psi_target, qt.Qobj):
            raise ValueError(f"psi_target must be Qobj, got {type(psi_target)}")
        if psi_init.shape[0] != self.dim or psi_init.shape[1] != 1:
            raise ValueError(f"psi_init must be a {self.dim}-dim ket, got {psi_init.shape}")
        if psi_target.shape[0] != self.dim or psi_target.shape[1] != 1:
            raise ValueError(
                f"psi_target must be a {self.dim}-dim ket, got {psi_target.shape}"
            )
        if not (0.99 <= psi_init.norm() <= 1.01):
            raise ValueError(f"psi_init not normalized: ||psi_init|| = {psi_init.norm()}")
        if not (0.99 <= psi_target.norm() <= 1.01):
            raise ValueError(
                f"psi_target not normalized: ||psi_target|| = {psi_target.norm()}"
            )
        if u_init is not None:
            if not isinstance(u_init, np.ndarray):
                raise ValueError(f"u_init must be ndarray, got {type(u_init)}")
            if u_init.shape != (self.n_controls, self.n_timeslices):
                raise ValueError(
                    f"u_init shape {u_init.shape} != expected "
                    f"({self.n_controls}, {self.n_timeslices})"
                )
            if not np.all(np.isfinite(u_init)):
                raise ValueError("u_init contains non-finite values")

    def optimize_state(
        self,
        psi_init: qt.Qobj,
        psi_target: qt.Qobj,
        u_init: Optional[np.ndarray] = None,
    ) -> KrotovResult:
        """
        Optimize control pulses for state-to-state transfer.

        Maximises the state fidelity |<psi_target|psi(T)>|^2 for the transfer
        psi_init -> psi_target.

        Returns a KrotovResult with final_fidelity, optimized_pulses, and history.
        """
        self._validate_state_parameters(psi_init, psi_target, u_init)
        u_start = self._default_controls() if u_init is None else u_init.copy()

        return self._optimize_core(
            fidelity_fn=lambda u: self._state_fidelity(
                self._forward_propagation(psi_init, u)[-1], psi_target
            ),
            gradient_fn=lambda u: self._state_gradient(u, psi_init, psi_target),
            u_init=u_start,
        )

    def get_pulse_functions(self, u: np.ndarray) -> List[Callable]:
        """
        Convert piecewise-constant pulses to callable functions.

        Parameters
        ----------
        u : np.ndarray
            Control amplitudes, shape (n_controls, n_timeslices).

        Returns
        -------
        list of callable
            One pulse function per control.
        """
        pulse_functions = []

        for j in range(self.n_controls):

            def pulse_func(t, control_idx=j, pulses=u):
                t = np.atleast_1d(t)
                indices = np.floor(t / self.dt).astype(int)
                indices = np.clip(indices, 0, self.n_timeslices - 1)
                return pulses[control_idx, indices]

            pulse_functions.append(pulse_func)

        return pulse_functions

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"KrotovOptimizer(n_controls={self.n_controls}, "
            f"n_timeslices={self.n_timeslices}, total_time={self.total_time}, "
            f"penalty_lambda={self.penalty_lambda})"
        )
