"""
Unit tests for adiabatic pulse techniques.

Tests STIRAP, Landau-Zener sweeps, and adiabaticity checking.
"""

import pytest
import numpy as np
import qutip as qt
from src.pulses.adiabatic import (
    LandauZenerParameters,
    STIRAPParameters,
    AdiabaticityMetrics,
    LandauZenerSweep,
    STIRAPulse,
    AdiabaticChecker,
    create_landau_zener_sweep,
    create_stirap_pulse,
)


class TestLandauZenerParameters:
    """Test Landau-Zener parameter validation."""

    def test_valid_parameters(self):
        """Test valid parameter initialization."""
        params = LandauZenerParameters(
            delta_initial=-10.0,
            delta_final=10.0,
            sweep_time=5.0,
            coupling=2.0,
            sweep_function="linear",
        )
        assert params.delta_initial == -10.0
        assert params.delta_final == 10.0
        assert params.sweep_time == 5.0
        assert params.coupling == 2.0
        assert params.sweep_function == "linear"

    def test_invalid_sweep_time(self):
        """Test that negative sweep time raises error."""
        with pytest.raises(ValueError, match="sweep_time must be positive"):
            LandauZenerParameters(
                delta_initial=-10.0,
                delta_final=10.0,
                sweep_time=-1.0,
                coupling=2.0,
            )

    def test_invalid_coupling(self):
        """Test that negative coupling raises error."""
        with pytest.raises(ValueError, match="coupling must be positive"):
            LandauZenerParameters(
                delta_initial=-10.0,
                delta_final=10.0,
                sweep_time=5.0,
                coupling=-1.0,
            )

    def test_invalid_sweep_function(self):
        """Test that invalid sweep function raises error."""
        with pytest.raises(ValueError, match="Unknown sweep_function"):
            LandauZenerParameters(
                delta_initial=-10.0,
                delta_final=10.0,
                sweep_time=5.0,
                coupling=2.0,
                sweep_function="invalid",
            )


class TestSTIRAPParameters:
    """Test STIRAP parameter validation."""

    def test_valid_parameters(self):
        """Test valid parameter initialization."""
        params = STIRAPParameters(
            omega_pump=5.0,
            omega_stokes=5.0,
            pulse_duration=10.0,
            delay=-1.0,
            detuning=0.0,
            pulse_shape="gaussian",
        )
        assert params.omega_pump == 5.0
        assert params.omega_stokes == 5.0
        assert params.pulse_duration == 10.0
        assert params.delay == -1.0
        assert params.pulse_shape == "gaussian"

    def test_invalid_omega_pump(self):
        """Test that negative omega_pump raises error."""
        with pytest.raises(ValueError, match="omega_pump must be positive"):
            STIRAPParameters(
                omega_pump=-1.0,
                omega_stokes=5.0,
                pulse_duration=10.0,
            )

    def test_invalid_omega_stokes(self):
        """Test that negative omega_stokes raises error."""
        with pytest.raises(ValueError, match="omega_stokes must be positive"):
            STIRAPParameters(
                omega_pump=5.0,
                omega_stokes=-1.0,
                pulse_duration=10.0,
            )

    def test_invalid_pulse_shape(self):
        """Test that invalid pulse shape raises error."""
        with pytest.raises(ValueError, match="Unknown pulse_shape"):
            STIRAPParameters(
                omega_pump=5.0,
                omega_stokes=5.0,
                pulse_duration=10.0,
                pulse_shape="invalid",
            )


class TestLandauZenerSweep:
    """Test Landau-Zener sweep functionality."""

    def test_linear_detuning(self):
        """Test linear detuning profile."""
        params = LandauZenerParameters(
            delta_initial=-10.0,
            delta_final=10.0,
            sweep_time=5.0,
            coupling=2.0,
            sweep_function="linear",
        )
        sweep = LandauZenerSweep(params)

        # Check endpoints
        assert sweep.detuning(0.0) == pytest.approx(-10.0)
        assert sweep.detuning(5.0) == pytest.approx(10.0)

        # Check midpoint
        assert sweep.detuning(2.5) == pytest.approx(0.0)

    def test_tanh_detuning(self):
        """Test tanh detuning profile."""
        params = LandauZenerParameters(
            delta_initial=-10.0,
            delta_final=10.0,
            sweep_time=5.0,
            coupling=2.0,
            sweep_function="tanh",
        )
        sweep = LandauZenerSweep(params)

        # Check endpoints (approximately)
        assert sweep.detuning(0.0) < -9.0
        assert sweep.detuning(5.0) > 9.0

        # Check midpoint is near zero
        assert abs(sweep.detuning(2.5)) < 0.1

    def test_gaussian_detuning(self):
        """Test Gaussian (erf) detuning profile."""
        params = LandauZenerParameters(
            delta_initial=-10.0,
            delta_final=10.0,
            sweep_time=5.0,
            coupling=2.0,
            sweep_function="gaussian",
        )
        sweep = LandauZenerSweep(params)

        # Check smooth sweep
        assert sweep.detuning(0.0) < -9.0
        assert sweep.detuning(5.0) > 9.0

    def test_linear_detuning_rate(self):
        """Test linear sweep has constant rate."""
        params = LandauZenerParameters(
            delta_initial=-10.0,
            delta_final=10.0,
            sweep_time=5.0,
            coupling=2.0,
            sweep_function="linear",
        )
        sweep = LandauZenerSweep(params)

        expected_rate = 4.0  # (10 - (-10)) / 5
        assert sweep.detuning_rate(0.0) == pytest.approx(expected_rate)
        assert sweep.detuning_rate(2.5) == pytest.approx(expected_rate)
        assert sweep.detuning_rate(5.0) == pytest.approx(expected_rate)

    def test_energy_gap(self):
        """Test energy gap calculation."""
        params = LandauZenerParameters(
            delta_initial=-10.0,
            delta_final=10.0,
            sweep_time=5.0,
            coupling=2.0,
            sweep_function="linear",
        )
        sweep = LandauZenerSweep(params)

        # At resonance (delta=0), gap = coupling
        gap_resonance = sweep.energy_gap(2.5)
        assert gap_resonance == pytest.approx(2.0)

        # Far from resonance, gap ≈ |delta|
        gap_start = sweep.energy_gap(0.0)
        assert gap_start == pytest.approx(np.sqrt(10**2 + 2**2))

    def test_landau_zener_probability(self):
        """Test Landau-Zener transition probability."""
        # Slow sweep (adiabatic)
        params_slow = LandauZenerParameters(
            delta_initial=-10.0,
            delta_final=10.0,
            sweep_time=100.0,  # Very slow
            coupling=2.0,
            sweep_function="linear",
        )
        sweep_slow = LandauZenerSweep(params_slow)
        p_slow = sweep_slow.landau_zener_probability()
        assert p_slow < 0.1  # Should be mostly adiabatic

        # Fast sweep (diabatic)
        params_fast = LandauZenerParameters(
            delta_initial=-10.0,
            delta_final=10.0,
            sweep_time=0.1,  # Very fast
            coupling=2.0,
            sweep_function="linear",
        )
        sweep_fast = LandauZenerSweep(params_fast)
        p_fast = sweep_fast.landau_zener_probability()
        assert p_fast > 0.5  # Should be mostly diabatic

    def test_adiabaticity_parameter(self):
        """Test adiabaticity parameter calculation."""
        params = LandauZenerParameters(
            delta_initial=-10.0,
            delta_final=10.0,
            sweep_time=5.0,
            coupling=2.0,
            sweep_function="linear",
        )
        sweep = LandauZenerSweep(params)

        # At resonance, gap is smallest, so gamma is smallest
        gamma_resonance = sweep.adiabaticity_parameter(2.5)
        gamma_start = sweep.adiabaticity_parameter(0.0)

        assert gamma_resonance < gamma_start

    def test_check_adiabaticity(self):
        """Test adiabaticity checking."""
        params = LandauZenerParameters(
            delta_initial=-10.0,
            delta_final=10.0,
            sweep_time=10.0,
            coupling=2.0,
            sweep_function="linear",
        )
        sweep = LandauZenerSweep(params)

        metrics = sweep.check_adiabaticity(threshold=1.0)

        assert isinstance(metrics, AdiabaticityMetrics)
        assert metrics.min_adiabaticity >= 0
        assert metrics.max_diabatic_rate >= 0
        assert metrics.transition_probability >= 0
        assert metrics.transition_probability <= 1
        assert len(metrics.adiabatic_times) > 0

    def test_simulate(self):
        """Test Landau-Zener sweep simulation."""
        params = LandauZenerParameters(
            delta_initial=-10.0,
            delta_final=10.0,
            sweep_time=5.0,
            coupling=2.0,
            sweep_function="linear",
        )
        sweep = LandauZenerSweep(params)

        times, states = sweep.simulate(n_points=50)

        assert len(times) == 50
        assert len(states) == 50
        assert times[0] == 0.0
        assert times[-1] == pytest.approx(5.0)

        # Check states are normalized
        for state in states:
            assert state.norm() == pytest.approx(1.0)


class TestSTIRAPulse:
    """Test STIRAP pulse functionality."""

    def test_pulse_envelope_gaussian(self):
        """Test Gaussian pulse envelope."""
        params = STIRAPParameters(
            omega_pump=5.0,
            omega_stokes=5.0,
            pulse_duration=10.0,
            pulse_shape="gaussian",
        )
        pulse = STIRAPulse(params)

        # Peak should be at center
        peak_time = 5.0
        peak_value = pulse.pulse_envelope(peak_time, peak_time)
        assert peak_value == pytest.approx(1.0)

        # Should decay away from peak
        off_peak = pulse.pulse_envelope(peak_time + 2.0, peak_time)
        assert off_peak < peak_value

    def test_pulse_envelope_sech(self):
        """Test sech pulse envelope."""
        params = STIRAPParameters(
            omega_pump=5.0,
            omega_stokes=5.0,
            pulse_duration=10.0,
            pulse_shape="sech",
        )
        pulse = STIRAPulse(params)

        peak_time = 5.0
        peak_value = pulse.pulse_envelope(peak_time, peak_time)
        assert peak_value == pytest.approx(1.0)

    def test_counter_intuitive_ordering(self):
        """Test counter-intuitive pulse ordering (Stokes before pump)."""
        params = STIRAPParameters(
            omega_pump=5.0,
            omega_stokes=5.0,
            pulse_duration=10.0,
            delay=-1.0,  # Counter-intuitive
            pulse_shape="gaussian",
        )
        pulse = STIRAPulse(params)

        # At early time, Stokes should be stronger
        t_early = 3.0
        assert pulse.stokes_amplitude(t_early) > pulse.pump_amplitude(t_early)

        # At late time, pump should be stronger
        t_late = 7.0
        assert pulse.pump_amplitude(t_late) > pulse.stokes_amplitude(t_late)

    def test_mixing_angle(self):
        """Test mixing angle calculation."""
        params = STIRAPParameters(
            omega_pump=5.0,
            omega_stokes=5.0,
            pulse_duration=10.0,
            delay=-1.0,
            pulse_shape="gaussian",
        )
        pulse = STIRAPulse(params)

        # At early time, theta should be small (dominated by Stokes)
        theta_early = pulse.mixing_angle(2.0)
        assert 0 <= theta_early < np.pi / 4

        # At late time, theta should be large (dominated by pump)
        theta_late = pulse.mixing_angle(8.0)
        assert np.pi / 4 < theta_late <= np.pi / 2

    def test_dark_state(self):
        """Test dark state calculation."""
        params = STIRAPParameters(
            omega_pump=5.0,
            omega_stokes=5.0,
            pulse_duration=10.0,
            delay=-1.0,
            pulse_shape="gaussian",
        )
        pulse = STIRAPulse(params)

        # Dark state should be normalized
        dark = pulse.dark_state(5.0)
        assert dark.norm() == pytest.approx(1.0)

        # Dark state at start should be mostly |1⟩
        dark_start = pulse.dark_state(0.0)
        overlap_1_start = qt.basis(3, 0).dag() * dark_start
        # Inner product returns Qobj for ket result
        if isinstance(overlap_1_start, qt.Qobj):
            pop_1_start = abs(overlap_1_start.full()[0, 0]) ** 2
        else:
            pop_1_start = abs(overlap_1_start) ** 2
        assert pop_1_start > 0.9

        # Dark state at end should be mostly |3⟩
        dark_end = pulse.dark_state(params.pulse_duration)
        overlap_3_end = qt.basis(3, 2).dag() * dark_end
        if isinstance(overlap_3_end, qt.Qobj):
            pop_3_end = abs(overlap_3_end.full()[0, 0]) ** 2
        else:
            pop_3_end = abs(overlap_3_end) ** 2
        assert pop_3_end > 0.9

    def test_adiabaticity_parameter(self):
        """Test STIRAP adiabaticity parameter."""
        params = STIRAPParameters(
            omega_pump=5.0,
            omega_stokes=5.0,
            pulse_duration=10.0,
            delay=-1.0,
            pulse_shape="gaussian",
        )
        pulse = STIRAPulse(params)

        gamma = pulse.adiabaticity_parameter(5.0)
        assert gamma > 0

    def test_check_adiabaticity(self):
        """Test STIRAP adiabaticity checking."""
        params = STIRAPParameters(
            omega_pump=5.0,
            omega_stokes=5.0,
            pulse_duration=10.0,
            delay=-1.0,
            pulse_shape="gaussian",
        )
        pulse = STIRAPulse(params)

        metrics = pulse.check_adiabaticity(threshold=1.0, n_points=100)

        assert isinstance(metrics, AdiabaticityMetrics)
        assert metrics.min_adiabaticity >= 0
        assert 0 <= metrics.transition_probability <= 1

    def test_simulate(self):
        """Test STIRAP simulation."""
        params = STIRAPParameters(
            omega_pump=5.0,
            omega_stokes=5.0,
            pulse_duration=10.0,
            delay=-1.0,
            pulse_shape="gaussian",
        )
        pulse = STIRAPulse(params)

        times, states = pulse.simulate(n_points=50)

        assert len(times) == 50
        assert len(states) == 50

        # Check normalization
        for state in states:
            assert state.norm() == pytest.approx(1.0, abs=1e-6)

    def test_simulate_with_loss(self):
        """Test STIRAP simulation with spontaneous emission."""
        params = STIRAPParameters(
            omega_pump=5.0,
            omega_stokes=5.0,
            pulse_duration=10.0,
            delay=-1.0,
            pulse_shape="gaussian",
        )
        pulse = STIRAPulse(params)

        times, states = pulse.simulate(n_points=50, include_loss=True, loss_rate=0.1)

        assert len(times) == 50
        assert len(states) == 50

        # With loss/collapse operators, mesolve returns density matrices
        # Density matrices have trace = 1, but norm can be > 1
        for state in states:
            if state.isket:
                assert 0 < state.norm() <= 1.0
            else:
                # Density matrix: check trace instead
                assert 0 < abs(state.tr()) <= 1.1  # Allow small numerical error

    def test_transfer_efficiency(self):
        """Test population transfer efficiency calculation."""
        # Good STIRAP parameters should give high efficiency
        params_good = STIRAPParameters(
            omega_pump=10.0,  # Strong coupling
            omega_stokes=10.0,
            pulse_duration=20.0,  # Slow enough to be adiabatic
            delay=-2.0,  # Counter-intuitive
            pulse_shape="gaussian",
        )
        pulse_good = STIRAPulse(params_good)
        efficiency_good = pulse_good.transfer_efficiency(n_points=100)

        # Should transfer most population
        assert efficiency_good > 0.7

        # Bad STIRAP (intuitive ordering) should give lower efficiency
        params_bad = STIRAPParameters(
            omega_pump=10.0,
            omega_stokes=10.0,
            pulse_duration=20.0,
            delay=2.0,  # Intuitive ordering (wrong!)
            pulse_shape="gaussian",
        )
        pulse_bad = STIRAPulse(params_bad)
        efficiency_bad = pulse_bad.transfer_efficiency(n_points=100)

        # Counter-intuitive should be better than intuitive
        assert efficiency_good > efficiency_bad


class TestAdiabaticChecker:
    """Test general adiabaticity checker."""

    def test_instantaneous_eigensystem(self):
        """Test eigenvalue/eigenvector computation."""
        H = qt.sigmaz()
        eigvals, eigvecs = AdiabaticChecker.instantaneous_eigensystem(H)

        assert len(eigvals) == 2
        assert len(eigvecs) == 2

        # Eigenvalues should be ±1
        assert set(np.round(eigvals, 5)) == {-1.0, 1.0}

    def test_adiabatic_condition_static(self):
        """Test adiabatic condition for static Hamiltonian."""
        # Static H should be perfectly adiabatic (no transitions)
        H = qt.sigmaz()
        H_list = [H] * 10
        times = np.linspace(0, 1, 10)

        result = AdiabaticChecker.adiabatic_condition(H_list, times, state_index=0)

        # For static H, dH/dt = 0, so transition rate should be tiny
        assert result["max_transition_rate"] < 1e-10

    def test_adiabatic_condition_time_varying(self):
        """Test adiabatic condition for time-varying Hamiltonian."""
        # Create slowly varying Hamiltonian
        times = np.linspace(0, 10, 50)
        H_list = []
        for t in times:
            # H(t) = σ_z + ε(t) σ_x, where ε varies slowly
            epsilon = 0.1 * np.sin(t)
            H = qt.sigmaz() + epsilon * qt.sigmax()
            H_list.append(H)

        result = AdiabaticChecker.adiabatic_condition(H_list, times, state_index=0)

        assert "min_gap" in result
        assert "max_transition_rate" in result
        assert "min_adiabaticity" in result
        assert result["min_gap"] > 0

    def test_optimize_sweep_time(self):
        """Test sweep time optimization."""

        def sweep_builder(T):
            """Build sweep with given time."""
            params = LandauZenerParameters(
                delta_initial=-10.0,
                delta_final=10.0,
                sweep_time=T,
                coupling=2.0,
                sweep_function="linear",
            )
            return LandauZenerSweep(params)

        result = AdiabaticChecker.optimize_sweep_time(
            sweep_builder, min_time=1.0, max_time=50.0, target_adiabaticity=5.0
        )

        assert "optimal_time" in result
        assert "min_adiabaticity" in result
        assert result["optimal_time"] >= 1.0
        assert result["optimal_time"] <= 50.0
        assert result["success"]


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_create_landau_zener_sweep(self):
        """Test Landau-Zener sweep creation."""
        sweep = create_landau_zener_sweep(
            delta_range=(-10.0, 10.0),
            coupling=2.0,
            sweep_time=5.0,
            sweep_type="linear",
        )

        assert isinstance(sweep, LandauZenerSweep)
        assert sweep.params.delta_initial == -10.0
        assert sweep.params.delta_final == 10.0
        assert sweep.params.sweep_time == 5.0
        assert sweep.params.coupling == 2.0

    def test_create_stirap_pulse(self):
        """Test STIRAP pulse creation."""
        pulse = create_stirap_pulse(
            omega_pump=5.0,
            omega_stokes=5.0,
            pulse_duration=10.0,
            delay=-1.0,
            pulse_shape="gaussian",
        )

        assert isinstance(pulse, STIRAPulse)
        assert pulse.params.omega_pump == 5.0
        assert pulse.params.omega_stokes == 5.0
        assert pulse.params.pulse_duration == 10.0
        assert pulse.params.delay == -1.0


class TestEdgeCases:
    """Test edge cases and robustness."""

    def test_zero_coupling_landau_zener(self):
        """Test Landau-Zener with zero coupling."""
        params = LandauZenerParameters(
            delta_initial=-10.0,
            delta_final=10.0,
            sweep_time=5.0,
            coupling=1e-10,  # Nearly zero
            sweep_function="linear",
        )
        sweep = LandauZenerSweep(params)

        # Should still compute (though unphysical)
        gap = sweep.energy_gap(2.5)
        assert gap >= 0

    def test_very_slow_stirap(self):
        """Test STIRAP with very slow pulses (highly adiabatic)."""
        params = STIRAPParameters(
            omega_pump=1.0,
            omega_stokes=1.0,
            pulse_duration=100.0,  # Very slow
            delay=-10.0,
            pulse_shape="gaussian",
        )
        pulse = STIRAPulse(params)

        metrics = pulse.check_adiabaticity(threshold=10.0)
        # Should be very adiabatic
        assert metrics.min_adiabaticity > 1.0

    def test_very_fast_stirap(self):
        """Test STIRAP with very fast pulses (non-adiabatic)."""
        params = STIRAPParameters(
            omega_pump=1.0,
            omega_stokes=1.0,
            pulse_duration=0.5,  # Very fast
            delay=-0.05,
            pulse_shape="gaussian",
        )
        pulse = STIRAPulse(params)

        # Should still run without error
        metrics = pulse.check_adiabaticity(threshold=10.0)
        assert metrics.min_adiabaticity >= 0


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_stirap_vs_landau_zener_concepts(self):
        """
        Conceptual test: both STIRAP and LZ are adiabatic methods.
        Verify both can achieve high-fidelity transfer when parameters are good.
        """
        # Good STIRAP
        stirap = create_stirap_pulse(
            omega_pump=10.0,
            omega_stokes=10.0,
            pulse_duration=20.0,
            delay=-2.0,
        )
        stirap_efficiency = stirap.transfer_efficiency(n_points=100)

        # Good Landau-Zener (slow sweep)
        lz = create_landau_zener_sweep(
            delta_range=(-20.0, 20.0),
            coupling=5.0,
            sweep_time=20.0,
            sweep_type="linear",
        )
        lz_prob = lz.landau_zener_probability()

        # Both should be in adiabatic regime
        assert stirap_efficiency > 0.7  # High transfer
        assert lz_prob < 0.1  # Low diabatic probability

    def test_adiabaticity_scaling(self):
        """
        Test that adiabaticity improves with sweep time.
        """
        times = [1.0, 5.0, 10.0, 20.0]
        probabilities = []

        for T in times:
            sweep = create_landau_zener_sweep(
                delta_range=(-10.0, 10.0),
                coupling=2.0,
                sweep_time=T,
                sweep_type="linear",
            )
            prob = sweep.landau_zener_probability()
            probabilities.append(prob)

        # Longer sweep times should give lower diabatic probability
        assert probabilities[-1] < probabilities[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
