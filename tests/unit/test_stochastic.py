"""
Unit tests for the stochastic master equation / feedback module
(src/hamiltonian/stochastic.py).

Covers:
- Construction of collapse and measurement (stochastic) operators
- The QuTiP-5 homodyne/heterodyne ``evolve`` wrapper
- The explicit measurement-conditioned feedback integrator, including the
  physical invariants it must satisfy (trace, Hermiticity, positivity) and a
  closed-loop stabilization demonstration.
"""

import numpy as np
import pytest
import qutip as qt

from src.hamiltonian.stochastic import (
    StochasticEvolution,
    MeasurementParams,
    FeedbackResult,
)
from src.hamiltonian.lindblad import DecoherenceParams


def make_evolution(axis="z", strength=3.0, efficiency=1.0, drive=0.3, T1=30.0, T2=20.0):
    """A qubit with a coherent drift (drives off |0>) under monitored sigma_z."""
    H = 2 * np.pi * drive * qt.sigmax()
    dec = DecoherenceParams(T1=T1, T2=T2)
    meas = MeasurementParams(efficiency=efficiency, strength=strength, axis=axis)
    return StochasticEvolution(H, dec, meas)


def min_eigenvalue(state: qt.Qobj) -> float:
    """Smallest eigenvalue of a density matrix (for positivity checks)."""
    herm = 0.5 * (state.full() + state.full().conj().T)
    return float(np.min(np.linalg.eigvalsh(herm)).real)


class TestMeasurementParams:
    def test_defaults(self):
        params = MeasurementParams()
        assert params.efficiency == 1.0
        assert params.strength == 1.0
        assert params.axis == "z"
        assert params.method == "homodyne"


class TestOperatorConstruction:
    def test_collapse_ops_include_relaxation_and_dephasing(self):
        ev = make_evolution(T1=20.0, T2=10.0)
        # gamma_phi = 1/T2 - 1/(2 T1) = 0.1 - 0.025 > 0, so two channels.
        assert len(ev.c_ops) == 2

    def test_collapse_ops_relaxation_rate(self):
        ev = make_evolution(T1=25.0, T2=40.0)
        # First collapse op is sqrt(1/T1) * destroy; check its rate.
        c0 = ev.c_ops[0]
        rate = (c0.dag() * c0).full()[1, 1].real  # <1|c^dag c|1> = gamma1
        assert rate == pytest.approx(1.0 / 25.0, rel=1e-6)

    @pytest.mark.parametrize(
        "axis,op", [("x", qt.sigmax()), ("y", qt.sigmay()), ("z", qt.sigmaz())]
    )
    def test_stochastic_op_axis(self, axis, op):
        ev = make_evolution(axis=axis, strength=4.0)
        expected = np.sqrt(4.0) * op
        assert (ev.sc_ops[0] - expected).norm() < 1e-12

    def test_invalid_axis_raises(self):
        with pytest.raises(ValueError, match="Invalid axis"):
            make_evolution(axis="w")

    def test_dissipator_is_traceless(self):
        # D[c]rho must be traceless (trace-preserving generator).
        c = qt.destroy(2)
        rho = qt.Qobj([[0.7, 0.2 + 0.1j], [0.2 - 0.1j, 0.3]])
        d = StochasticEvolution._dissipator(c, rho)
        assert abs(d.tr()) < 1e-12


class TestEvolve:
    def test_homodyne_runs_and_returns_expectations(self):
        ev = make_evolution(drive=0.0)
        res = ev.evolve(qt.basis(2, 0).proj(), np.linspace(0, 1, 40), n_trajectories=2, seed=1)
        expect = np.array(res.expect)
        assert expect.shape == (3, 40)  # <sx>, <sy>, <sz>
        assert np.all(np.isfinite(expect))

    def test_heterodyne_runs(self):
        ev = make_evolution(drive=0.0)
        ev.measurement.method = "heterodyne"
        res = ev.evolve(qt.basis(2, 0).proj(), np.linspace(0, 1, 30), n_trajectories=1, seed=2)
        assert np.all(np.isfinite(np.array(res.expect)))

    def test_unsupported_method_raises(self):
        ev = make_evolution()
        ev.measurement.method = "photodetection"
        with pytest.raises(NotImplementedError, match="not supported"):
            ev.evolve(qt.basis(2, 0).proj(), np.linspace(0, 1, 10))

    def test_too_few_times_raises(self):
        ev = make_evolution()
        with pytest.raises(ValueError, match="at least two points"):
            ev.evolve(qt.basis(2, 0).proj(), np.array([0.0]))


class TestFeedbackResultStructure:
    def test_result_fields_and_shapes(self):
        ev = make_evolution()
        times = np.linspace(0, 1, 25)
        res = ev.simulate_feedback_loop(
            qt.basis(2, 0).proj(), times, lambda J: 0.0, seed=0, n_substeps=5
        )
        assert isinstance(res, FeedbackResult)
        assert len(res.states) == len(times)
        assert res.measurement_record.shape == (len(times) - 1,)
        assert res.controls.shape == (len(times) - 1,)
        for key in ("x", "y", "z"):
            assert res.expect[key].shape == (len(times),)

    def test_controls_record_feedback_outputs(self):
        ev = make_evolution()
        times = np.linspace(0, 1, 15)
        res = ev.simulate_feedback_loop(
            qt.basis(2, 0).proj(), times, lambda J: 1.5, seed=0, n_substeps=5
        )
        # Constant feedback law -> all recorded controls equal 1.5.
        assert np.allclose(res.controls, 1.5)


class TestFeedbackInvariants:
    def test_trace_preserved(self):
        ev = make_evolution()
        res = ev.simulate_feedback_loop(
            qt.basis(2, 0).proj(), np.linspace(0, 2, 80), lambda J: -0.3 * J, seed=3
        )
        assert max(abs(s.tr() - 1.0) for s in res.states) < 1e-9

    def test_states_hermitian(self):
        ev = make_evolution()
        res = ev.simulate_feedback_loop(
            qt.basis(2, 0).proj(), np.linspace(0, 2, 80), lambda J: -0.3 * J, seed=4
        )
        assert max((s - s.dag()).norm() for s in res.states) < 1e-9

    def test_states_positive_semidefinite(self):
        # The Kraus-form update must keep the conditioned state physical.
        ev = make_evolution(strength=4.0)
        res = ev.simulate_feedback_loop(
            qt.basis(2, 0).proj(), np.linspace(0, 2, 100), lambda J: -0.4 * J, seed=5
        )
        assert min(min_eigenvalue(s) for s in res.states) > -1e-9

    def test_inefficient_measurement_stays_physical(self):
        # eta < 1 adds the unmonitored refill term; the state must remain a
        # valid, positive density matrix.
        ev = make_evolution(strength=3.0, efficiency=0.5)
        res = ev.simulate_feedback_loop(
            qt.basis(2, 0).proj(), np.linspace(0, 2, 80), lambda J: -0.3 * J, seed=6
        )
        assert max(abs(s.tr() - 1.0) for s in res.states) < 1e-9
        assert min(min_eigenvalue(s) for s in res.states) > -1e-9

    def test_deterministic_with_seed(self):
        ev = make_evolution()
        times = np.linspace(0, 2, 60)
        r1 = ev.simulate_feedback_loop(qt.basis(2, 0).proj(), times, lambda J: -0.3 * J, seed=7)
        r2 = ev.simulate_feedback_loop(qt.basis(2, 0).proj(), times, lambda J: -0.3 * J, seed=7)
        assert np.allclose(r1.expect["z"], r2.expect["z"])

    def test_feedback_changes_trajectory(self):
        # The control must actually couple into the dynamics.
        ev = make_evolution()
        times = np.linspace(0, 2, 60)
        no_fb = ev.simulate_feedback_loop(qt.basis(2, 0).proj(), times, lambda J: 0.0, seed=8)
        with_fb = ev.simulate_feedback_loop(qt.basis(2, 0).proj(), times, lambda J: 3.0, seed=8)
        assert not np.allclose(no_fb.expect["z"], with_fb.expect["z"])

    def test_too_few_times_raises(self):
        ev = make_evolution()
        with pytest.raises(ValueError, match="at least two points"):
            ev.simulate_feedback_loop(qt.basis(2, 0).proj(), np.array([0.0]), lambda J: 0.0)

    def test_invalid_substeps_raises(self):
        ev = make_evolution()
        with pytest.raises(ValueError, match="n_substeps"):
            ev.simulate_feedback_loop(
                qt.basis(2, 0).proj(), np.linspace(0, 1, 10), lambda J: 0.0, n_substeps=0
            )

    def test_nonfinite_feedback_raises(self):
        # A controller returning a non-finite value must be detected, not
        # silently propagated as a NaN state.
        ev = make_evolution()
        with pytest.raises(RuntimeError, match="non-finite"):
            ev.simulate_feedback_loop(
                qt.basis(2, 0).proj(), np.linspace(0, 1, 10), lambda J: np.inf, seed=0
            )


class TestFeedbackStabilization:
    @pytest.mark.slow
    def test_feedback_improves_state_retention(self):
        """
        Closed-loop demonstration: a coherent drift rotates the qubit off |0>;
        proportional feedback on the homodyne current counteracts it.

        Averaged over fixed seeds, the mean fidelity to |0> with feedback must
        exceed the no-feedback baseline by a clear margin. (Empirically the
        margin is ~0.2; we require >0.1 to stay well clear of trajectory noise.)
        """
        ev = make_evolution(drive=0.3, strength=3.0, T1=30.0, T2=20.0)
        times = np.linspace(0, 3, 120)
        rho0 = qt.basis(2, 0).proj()
        seeds = range(10)

        def fidelity_to_ground(result):
            return (1.0 + result.expect["z"][-1]) / 2.0

        with_fb = [
            fidelity_to_ground(
                ev.simulate_feedback_loop(rho0, times, lambda J: -0.3 * J, seed=s)
            )
            for s in seeds
        ]
        no_fb = [
            fidelity_to_ground(
                ev.simulate_feedback_loop(rho0, times, lambda J: 0.0, seed=s)
            )
            for s in seeds
        ]

        assert np.mean(with_fb) > np.mean(no_fb) + 0.1
