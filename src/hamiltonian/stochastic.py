"""
Stochastic Master Equation (SME) for Quantum Feedback Control
=============================================================

This module implements the Stochastic Master Equation (SME) for simulating
individual quantum trajectories under continuous weak measurement. This is the
foundation for simulating real-time quantum feedback control.

Physical Context:
-----------------
While the Lindblad master equation describes the *average* evolution of an
ensemble, the SME describes the evolution of a *single* system conditioned on
a measurement record I(t).

    dρ = -i[H, ρ]dt + L[ρ]dt + √η H[ρ]dW

where:
- dW is the Wiener process (Gaussian white noise, dW ~ √dt)
- H[ρ] is the measurement superoperator (backaction)
- η is the measurement efficiency

This requires Itô calculus to simulate correctly.

Classes:
--------
- StochasticEvolution: Simulates single trajectories using homodyne/heterodyne detection.

Dependencies:
-------------
- QuTiP (stochastic solvers)
- NumPy
- SciPy

Author: Orchestrator Agent (Ghost of Itô)
Date: 2026-01-20
"""

import numpy as np
import qutip as qt
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass
from .lindblad import DecoherenceParams

@dataclass
class MeasurementParams:
    """
    Parameters for continuous weak measurement.
    
    Attributes
    ----------
    efficiency : float
        Measurement efficiency η ∈ [0, 1].
        η=1: Perfect measurement (pure state remains pure).
        η=0: No information (Lindblad evolution).
    strength : float
        Measurement strength κ (rate of information gain).
    axis : str
        Measurement axis ('x', 'y', 'z').
    method : str
        'homodyne' (measures one quadrature) or 'heterodyne' (measures both).
    """
    efficiency: float = 1.0
    strength: float = 1.0
    axis: str = 'z'
    method: str = 'homodyne'


@dataclass
class FeedbackResult:
    """
    Result of a measurement-conditioned feedback trajectory.

    Attributes
    ----------
    times : np.ndarray
        Time grid (length N).
    states : list of qutip.Qobj
        Conditioned density matrix at each time (length N).
    measurement_record : np.ndarray
        Homodyne photocurrent J_k fed to the controller at each step (length N-1).
    controls : np.ndarray
        Feedback control amplitude applied at each step (length N-1).
    expect : dict
        Pauli expectation trajectories with keys 'x', 'y', 'z' (each length N).
    """

    times: np.ndarray
    states: List[qt.Qobj]
    measurement_record: np.ndarray
    controls: np.ndarray
    expect: dict


class StochasticEvolution:
    """
    Stochastic Master Equation solver for quantum trajectories.
    
    Simulates:
        dρ_c = -i[H, ρ_c]dt + D[L]ρ_c dt + √η H[L]ρ_c dW
        
    where ρ_c is the conditioned density matrix.
    """
    
    def __init__(
        self,
        H: Union[qt.Qobj, list],
        decoherence: DecoherenceParams,
        measurement: MeasurementParams
    ):
        self.H = H
        self.decoherence = decoherence
        self.measurement = measurement
        
        # System dimension
        self.dim = H[0].shape[0] if isinstance(H, list) else H.shape[0]
        
        # Build operators
        self.c_ops = self._build_collapse_ops()
        self.sc_ops = self._build_stochastic_ops()
        
    def _build_collapse_ops(self) -> List[qt.Qobj]:
        """Standard environmental noise (unmonitored)."""
        # Re-use logic from Lindblad, but exclude the monitored channel
        # For simplicity, we assume T1/T2 are separate from the measurement channel
        gamma1 = 1.0 / self.decoherence.T1
        c_ops = [np.sqrt(gamma1) * qt.destroy(self.dim)]
        
        if self.decoherence.T2:
            gamma2 = 1.0 / self.decoherence.T2
            gamma_phi = gamma2 - gamma1/2
            if gamma_phi > 0:
                c_ops.append(np.sqrt(gamma_phi) * qt.sigmaz())
                
        return c_ops

    def _build_stochastic_ops(self) -> List[qt.Qobj]:
        """Measurement operators (monitored channels)."""
        # Measurement operator M associated with strength κ
        # L_meas = √κ * σ_axis
        kappa = self.measurement.strength
        
        if self.measurement.axis == 'z':
            op = qt.sigmaz()
        elif self.measurement.axis == 'x':
            op = qt.sigmax()
        elif self.measurement.axis == 'y':
            op = qt.sigmay()
        else:
            raise ValueError(f"Invalid axis: {self.measurement.axis}")
            
        return [np.sqrt(kappa) * op]

    def evolve(
        self,
        rho0: qt.Qobj,
        times: np.ndarray,
        n_trajectories: int = 1,
        seed: Optional[int] = None
    ) -> qt.solver.Result:
        """
        Simulate stochastic trajectories.
        
        Parameters
        ----------
        rho0 : Initial state
        times : Time array
        n_trajectories : Number of trajectories to average/store
        seed : Random seed
        
        Returns
        -------
        result : QuTiP Result object
        """
        if seed is not None:
            np.random.seed(seed)

        if len(times) < 2:
            raise ValueError("times must contain at least two points")

        # QuTiP 5 moved nsubsteps/store_measurement into options and replaced the
        # 'method' string with the heterodyne flag.
        dt_sub = (times[1] - times[0]) / 10.0  # substepping for Ito convergence
        options = {"dt": dt_sub, "store_measurement": True}

        if self.measurement.method in ("homodyne", "heterodyne"):
            return qt.smesolve(
                self.H, rho0, times,
                c_ops=self.c_ops,
                sc_ops=self.sc_ops,
                heterodyne=(self.measurement.method == "heterodyne"),
                e_ops=[qt.sigmax(), qt.sigmay(), qt.sigmaz()],
                ntraj=n_trajectories,
                options=options,
            )

        raise NotImplementedError(
            f"Measurement method '{self.measurement.method}' is not supported; "
            "use 'homodyne' or 'heterodyne'."
        )

    @staticmethod
    def _dissipator(c: qt.Qobj, rho: qt.Qobj) -> qt.Qobj:
        """Lindblad dissipator D[c]rho = c rho c^dag - 1/2 {c^dag c, rho}."""
        cdc = c.dag() * c
        return c * rho * c.dag() - 0.5 * (cdc * rho + rho * cdc)

    def simulate_feedback_loop(
        self,
        rho0: qt.Qobj,
        times: np.ndarray,
        feedback_func: callable,
        feedback_operator: Optional[qt.Qobj] = None,
        seed: Optional[int] = None,
        n_substeps: int = 10,
    ) -> FeedbackResult:
        """
        Simulate measurement-conditioned feedback by explicit stochastic stepping.

        Integrates the homodyne stochastic master equation in positivity-preserving
        measurement-operator (Kraus) form. Over each substep dt_s, with the
        monitored channel ``c_meas = sqrt(kappa) sigma_axis`` and unmonitored
        environmental channels ``c_k`` (T1/dephasing),

            M   = I + (-i H(t) - 1/2 sum_all c^dag c) dt_s + sqrt(eta) c_meas dW
            rho'= M rho M^dag + sum_k (c_k rho c_k^dag) dt_s
                                + (1-eta) (c_meas rho c_meas^dag) dt_s
            rho = rho' / Tr(rho').

        ``M rho M^dag`` is positive semidefinite by construction, so the
        conditioned state stays physical; the nonlinear measurement term arises
        from the normalization. The control is held constant over each ``times``
        interval and is chosen from the homodyne photocurrent measured over the
        *previous* interval (a one-interval feedback latency),

            J = 2 sqrt(eta) <c_meas> + (sum dW)/dt,

        driving the feedback Hamiltonian ``H_fb = u * feedback_operator`` added to H.

        Parameters
        ----------
        rho0 : qutip.Qobj
            Initial density matrix.
        times : np.ndarray
            Feedback-update grid (length >= 2); the SME is substepped within each
            interval. The step size may be non-uniform.
        feedback_func : callable
            Maps the measured photocurrent J (float) to a control amplitude (float).
        feedback_operator : qutip.Qobj, optional
            Hermitian operator driven by the feedback. Defaults to sigma_x.
        seed : int, optional
            Seed for the Wiener increments (reproducible trajectories).
        n_substeps : int, optional
            SME substeps per feedback interval (Ito stability). Default 10.

        Returns
        -------
        FeedbackResult
            Conditioned states, photocurrent record, controls, and Pauli expectations.

        Raises
        ------
        RuntimeError
            If the integration diverges (state trace -> 0), e.g. from too coarse a
            step or too large a feedback gain.
        """
        if seed is not None:
            np.random.seed(seed)
        if len(times) < 2:
            raise ValueError("times must contain at least two points")
        if n_substeps < 1:
            raise ValueError(f"n_substeps must be >= 1, got {n_substeps}")

        H_base = self.H[0] if isinstance(self.H, list) else self.H
        if feedback_operator is None:
            feedback_operator = qt.sigmax()
        eta = self.measurement.efficiency
        dims = rho0.dims
        n = len(times)

        # numpy arrays for the substep hot loop
        H_arr = H_base.full()
        F_arr = feedback_operator.full()
        eye = np.eye(self.dim, dtype=complex)
        cm = self.sc_ops[0].full()
        c_envs = [c.full() for c in self.c_ops]
        cm_dag = cm.conj().T
        # no-jump decay term for all dissipative channels (monitored + unmonitored)
        decay = 0.5 * (cm_dag @ cm + sum(c.conj().T @ c for c in c_envs))

        # Pauli expectations are defined for the qubit case this module targets.
        pauli_arrays = (
            {"x": qt.sigmax().full(), "y": qt.sigmay().full(), "z": qt.sigmaz().full()}
            if self.dim == 2
            else {}
        )
        expect = {key: np.zeros(n) for key in pauli_arrays}

        def _record_expect(step: int, state: np.ndarray) -> None:
            for key, op in pauli_arrays.items():
                expect[key][step] = np.real(np.trace(op @ state))

        rho = rho0.full().astype(complex)
        states = [qt.Qobj(rho.copy(), dims=dims)]
        record = np.zeros(n - 1)
        controls = np.zeros(n - 1)
        _record_expect(0, rho)

        # Initial photocurrent (no noise yet) seeds the first control decision.
        current = 2.0 * np.sqrt(eta) * np.real(np.trace(cm @ rho))

        for k in range(n - 1):
            dt = times[k + 1] - times[k]
            dt_s = dt / n_substeps
            u = float(feedback_func(current))
            if not np.isfinite(u):
                raise RuntimeError(
                    f"feedback_func returned a non-finite control ({u}) at step {k}."
                )
            controls[k] = u

            generator = -1j * (H_arr + u * F_arr) - decay
            dW_interval = 0.0
            meas_accum = 0.0
            for _ in range(n_substeps):
                dW = np.random.normal(0.0, np.sqrt(dt_s))
                dW_interval += dW
                meas_accum += np.real(np.trace(cm @ rho))

                M = eye + generator * dt_s + np.sqrt(eta) * cm * dW
                rho_un = M @ rho @ M.conj().T
                for c in c_envs:
                    rho_un += (c @ rho @ c.conj().T) * dt_s
                if eta < 1.0:
                    rho_un += (1.0 - eta) * (cm @ rho @ cm_dag) * dt_s

                # The Kraus update keeps Tr(rho') > 0 by construction; a
                # non-finite trace means the control itself diverged (e.g.
                # feedback_func returned inf/nan or overflowed).
                trace_val = np.trace(rho_un)
                if not np.isfinite(trace_val):
                    raise RuntimeError(
                        f"Feedback SME integration produced a non-finite state at "
                        f"step {k}; check that feedback_func returns finite values."
                    )
                rho = rho_un / trace_val
                rho = 0.5 * (rho + rho.conj().T)  # enforce Hermiticity

            # Photocurrent measured over this interval drives the next control.
            current = 2.0 * np.sqrt(eta) * (meas_accum / n_substeps) + dW_interval / dt
            record[k] = current

            states.append(qt.Qobj(rho.copy(), dims=dims))
            _record_expect(k + 1, rho)

        return FeedbackResult(
            times=np.asarray(times),
            states=states,
            measurement_record=record,
            controls=controls,
            expect=expect,
        )
