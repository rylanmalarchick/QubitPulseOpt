"""
Unit Tests for Lindblad Master Equation Solver
===============================================

Comprehensive test suite for open quantum system dynamics including
T1 (energy relaxation) and T2 (dephasing) decoherence.

Test Coverage:
-------------
1. DecoherenceParams validation
2. Collapse operator construction
3. T1 relaxation dynamics
4. T2 dephasing dynamics
5. Lindblad evolution
6. Comparison with unitary evolution
7. Gate fidelity with decoherence
8. Relaxation curve (T1 measurement)
9. Ramsey experiment (T2 measurement)
10. Thermal states
11. Purity tracking
12. Edge cases and physical constraints

Author: Orchestrator Agent
Date: 2025-01-27
SOW Reference: Phase 2.3 - Open System Dynamics Tests
"""

import pytest
import numpy as np
import qutip as qt
from src.hamiltonian.lindblad import (
    LindbladEvolution,
    DecoherenceParams,
    thermal_state,
)


class TestDecoherenceParams:
    """Test decoherence parameter validation."""

    def test_valid_t1_t2(self):
        """Test valid T1 and T2 parameters."""
        params = DecoherenceParams(T1=50.0, T2=30.0)
        assert params.T1 == 50.0
        assert params.T2 == 30.0
        assert params.temperature == 0.0

    def test_t2_from_tphi(self):
        """Test T2 computation from Tphi."""
        params = DecoherenceParams(T1=100.0, Tphi=150.0)

        # 1/T2 = 1/(2*T1) + 1/Tphi
        expected_T2 = 1.0 / (1.0 / (2.0 * 100.0) + 1.0 / 150.0)
        assert np.isclose(params.T2, expected_T2)

    def test_t2_limit(self):
        """Test T2 ≤ 2*T1 constraint."""
        # Valid: T2 = 2*T1 (limit case)
        params = DecoherenceParams(T1=50.0, T2=100.0)
        assert params.T2 == 100.0

        # Invalid: T2 > 2*T1
        with pytest.raises(ValueError, match="cannot exceed 2\\*T1"):
            DecoherenceParams(T1=50.0, T2=101.0)

    def test_negative_t1(self):
        """Test error for negative T1."""
        with pytest.raises(ValueError, match="T1 must be positive"):
            DecoherenceParams(T1=-10.0, T2=5.0)

    def test_negative_t2(self):
        """Test error for negative T2."""
        with pytest.raises(ValueError, match="T2 must be positive"):
            DecoherenceParams(T1=50.0, T2=-10.0)

    def test_missing_t2_and_tphi(self):
        """Test error when neither T2 nor Tphi provided."""
        with pytest.raises(ValueError, match="Must provide either T2 or Tphi"):
            DecoherenceParams(T1=50.0)

    def test_temperature(self):
        """Test temperature parameter."""
        params = DecoherenceParams(T1=50.0, T2=30.0, temperature=0.1)
        assert params.temperature == 0.1

    def test_negative_temperature(self):
        """Test error for negative temperature."""
        with pytest.raises(ValueError, match="Temperature must be non-negative"):
            DecoherenceParams(T1=50.0, T2=30.0, temperature=-0.1)


class TestCollapseOperators:
    """Test collapse operator construction."""

    def test_t1_collapse_operator(self):
        """Test T1 (amplitude damping) operator construction."""
        H = qt.sigmaz()
        decoherence = DecoherenceParams(T1=50.0, T2=30.0)
        lindblad = LindbladEvolution(H, decoherence)

        # Should have T1 and dephasing operators
        assert len(lindblad.c_ops) >= 1

        # Check T1 operator is proportional to sigma_minus
        gamma1 = 1.0 / 50.0
        expected_op = np.sqrt(gamma1) * qt.destroy(2)

        # First collapse operator should be T1
        assert np.allclose(lindblad.c_ops[0].full(), expected_op.full(), atol=1e-10)

    def test_pure_dephasing_operator(self):
        """Test pure dephasing operator construction."""
        H = qt.sigmaz()
        # T2 < 2*T1 means pure dephasing is present
        decoherence = DecoherenceParams(T1=50.0, T2=30.0)
        lindblad = LindbladEvolution(H, decoherence)

        # Should have 2 operators: T1 and dephasing
        assert len(lindblad.c_ops) == 2

        # Dephasing rate
        gamma1 = 1.0 / 50.0
        gamma2 = 1.0 / 30.0
        gamma_phi = gamma2 - gamma1 / 2.0

        expected_dephasing = np.sqrt(gamma_phi) * qt.sigmaz()
        assert np.allclose(
            lindblad.c_ops[1].full(), expected_dephasing.full(), atol=1e-10
        )

    def test_no_pure_dephasing(self):
        """Test case where T2 = 2*T1 (no pure dephasing)."""
        H = qt.sigmaz()
        decoherence = DecoherenceParams(T1=50.0, T2=100.0)
        lindblad = LindbladEvolution(H, decoherence)

        # Only T1 operator (pure dephasing is zero)
        assert len(lindblad.c_ops) == 1

    def test_custom_collapse_operators(self):
        """Test with custom collapse operators."""
        H = qt.sigmaz()
        decoherence = DecoherenceParams(T1=50.0, T2=30.0)

        custom_cops = [0.1 * qt.sigmax(), 0.1 * qt.sigmay()]
        lindblad = LindbladEvolution(H, decoherence, collapse_operators=custom_cops)

        assert len(lindblad.c_ops) == 2
        assert lindblad.c_ops == custom_cops


class TestLindbladEvolution:
    """Test Lindblad master equation evolution."""

    def test_basic_evolution(self):
        """Test basic density matrix evolution."""
        H = 0.5 * 5.0 * qt.sigmaz()
        decoherence = DecoherenceParams(T1=100.0, T2=50.0)
        lindblad = LindbladEvolution(H, decoherence)

        rho0 = qt.basis(2, 0) * qt.basis(2, 0).dag()
        times = np.linspace(0, 10, 50)

        result = lindblad.evolve(rho0, times)

        assert len(result.states) == 50
        for rho in result.states:
            assert isinstance(rho, qt.Qobj)
            assert rho.shape == (2, 2)
            # Check trace preservation
            assert np.isclose(rho.tr(), 1.0, atol=1e-10)

    def test_expectation_values(self):
        """Test expectation value computation during evolution."""
        H = qt.sigmaz()
        decoherence = DecoherenceParams(T1=50.0, T2=30.0)
        lindblad = LindbladEvolution(H, decoherence)

        rho0 = qt.basis(2, 1) * qt.basis(2, 1).dag()  # Excited state
        times = np.linspace(0, 100, 200)

        # Track excited state population
        e_ops = [qt.basis(2, 1) * qt.basis(2, 1).dag()]
        result = lindblad.evolve(rho0, times, e_ops=e_ops)

        assert len(result.expect) == 1
        assert len(result.expect[0]) == 200
        # Population should decay
        assert result.expect[0][-1] < result.expect[0][0]

    def test_time_dependent_hamiltonian(self):
        """Test evolution with time-dependent Hamiltonian."""
        H0 = qt.sigmaz()
        H1 = qt.sigmax()
        pulse = lambda t, args: 0.1 * np.sin(2 * np.pi * t / 10)
        H = [H0, [H1, pulse]]

        decoherence = DecoherenceParams(T1=50.0, T2=30.0)
        lindblad = LindbladEvolution(H, decoherence)

        rho0 = qt.basis(2, 0) * qt.basis(2, 0).dag()
        times = np.linspace(0, 50, 100)

        result = lindblad.evolve(rho0, times)

        assert len(result.states) == 100
        # Should show Rabi-like oscillations with decay
        pops = [
            qt.expect(qt.basis(2, 1) * qt.basis(2, 1).dag(), rho)
            for rho in result.states
        ]
        assert max(pops) > 0.007  # Some population transfer


class TestT1Relaxation:
    """Test T1 (energy relaxation) dynamics."""

    def test_t1_decay(self):
        """Test exponential T1 decay."""
        H = qt.sigmaz()
        T1 = 50.0
        decoherence = DecoherenceParams(T1=T1, T2=T1)
        lindblad = LindbladEvolution(H, decoherence)

        times, populations = lindblad.relaxation_curve(
            initial_state="excited", max_time=5 * T1
        )

        # Check exponential decay: P(t) = exp(-t/T1)
        expected = np.exp(-times / T1)

        # Should match within a few percent
        assert np.allclose(populations, expected, rtol=0.1)

    def test_t1_from_ground_state(self):
        """Test T1 starting from ground state (should stay)."""
        H = qt.sigmaz()
        T1 = 50.0
        decoherence = DecoherenceParams(T1=T1, T2=T1)
        lindblad = LindbladEvolution(H, decoherence)

        times, populations = lindblad.relaxation_curve(
            initial_state="ground", max_time=3 * T1
        )

        # Population should remain near zero
        assert np.all(populations < 0.01)

    def test_t1_timescale(self):
        """Test that decay timescale matches T1."""
        H = 0 * qt.sigmaz()  # No Hamiltonian
        T1 = 100.0
        decoherence = DecoherenceParams(T1=T1, T2=T1)
        lindblad = LindbladEvolution(H, decoherence)

        times, populations = lindblad.relaxation_curve(
            initial_state="excited", max_time=T1
        )

        # At t=T1, population should be ~1/e ≈ 0.368
        idx_t1 = np.argmin(np.abs(times - T1))
        assert np.isclose(populations[idx_t1], 1.0 / np.e, rtol=0.1)


class TestT2Dephasing:
    """Test T2 (dephasing) dynamics."""

    def test_ramsey_oscillations(self):
        """Test Ramsey experiment with T2 decay."""
        H = qt.sigmaz()
        T2 = 30.0
        decoherence = DecoherenceParams(T1=100.0, T2=T2)
        lindblad = LindbladEvolution(H, decoherence)

        detuning = 0.5
        times, signal = lindblad.ramsey_experiment(detuning=detuning, max_time=3 * T2)

        # Signal should oscillate and decay
        assert len(signal) > 0
        assert np.max(signal) > 0.5  # Some oscillation amplitude

        # Later times should have smaller amplitude
        early_amplitude = np.std(signal[: len(signal) // 4])
        late_amplitude = np.std(signal[3 * len(signal) // 4 :])
        assert late_amplitude < early_amplitude

    def test_ramsey_decay_envelope(self):
        """Test Ramsey decay envelope matches T2."""
        H = qt.sigmaz()
        T2 = 50.0
        decoherence = DecoherenceParams(T1=200.0, T2=T2)
        lindblad = LindbladEvolution(H, decoherence)

        detuning = 0.3
        times, signal = lindblad.ramsey_experiment(detuning=detuning)

        # Envelope should decay roughly as exp(-t/T2)
        # Check that signal decays significantly
        assert np.abs(signal[-1]) < np.abs(signal[0]) * 0.5


class TestComparisonWithUnitary:
    """Test comparison between open and closed system evolution."""

    def test_compare_with_unitary(self):
        """Test comparison utility."""
        H = 0.5 * 5.0 * qt.sigmaz()
        decoherence = DecoherenceParams(T1=100.0, T2=50.0)
        lindblad = LindbladEvolution(H, decoherence)

        rho0 = qt.basis(2, 0) * qt.basis(2, 0).dag()
        times = np.linspace(0, 50, 100)

        comparison = lindblad.compare_with_unitary(rho0, times)

        assert "lindblad_states" in comparison
        assert "unitary_states" in comparison
        assert "fidelities" in comparison
        assert "purity" in comparison

        assert len(comparison["fidelities"]) == 100
        assert len(comparison["purity"]) == 100

    def test_fidelity_decreases(self):
        """Test that fidelity decreases due to decoherence."""
        H = qt.sigmaz()
        decoherence = DecoherenceParams(T1=50.0, T2=30.0)
        lindblad = LindbladEvolution(H, decoherence)

        rho0 = qt.basis(2, 0) * qt.basis(2, 0).dag()
        times = np.linspace(0, 100, 100)

        comparison = lindblad.compare_with_unitary(rho0, times)
        fidelities = comparison["fidelities"]

        # Fidelity should start at 1 and decrease
        assert fidelities[0] > 0.99
        assert fidelities[-1] < fidelities[0]

    def test_purity_decreases(self):
        """Test that purity decreases (mixed state formation)."""
        H = qt.sigmaz()
        decoherence = DecoherenceParams(T1=50.0, T2=30.0)
        lindblad = LindbladEvolution(H, decoherence)

        # Start with excited state
        rho0 = qt.basis(2, 1) * qt.basis(2, 1).dag()
        times = np.linspace(0, 100, 100)

        comparison = lindblad.compare_with_unitary(rho0, times)
        purity = comparison["purity"]

        # Purity should start at 1 (pure state) and decrease
        assert purity[0] > 0.99
        assert purity[-1] < purity[0]
        assert purity[-1] < 1.0  # Becomes mixed


class TestGateFidelityWithDecoherence:
    """Test gate fidelity including decoherence."""

    def test_x_gate_fidelity(self):
        """Test X-gate fidelity with decoherence."""
        # Create Hamiltonian for X-gate
        H1 = qt.sigmax()
        pulse = lambda t, args: 0.1  # Constant amplitude
        H = [0 * qt.qeye(2), [H1, pulse]]

        decoherence = DecoherenceParams(T1=100.0, T2=50.0)
        lindblad = LindbladEvolution(H, decoherence)

        U_target = qt.sigmax()
        rho0 = qt.basis(2, 0) * qt.basis(2, 0).dag()

        gate_time = np.pi / 0.1  # π-pulse duration
        fidelity = lindblad.gate_fidelity_with_decoherence(
            U_target, rho0, gate_time, n_steps=500
        )

        # Should have reasonable fidelity (less than ideal due to decoherence)
        assert 0.5 < fidelity < 1.0

    def test_shorter_gate_better_fidelity(self):
        """Test that shorter gates have higher fidelity."""
        H1 = qt.sigmax()
        decoherence = DecoherenceParams(T1=50.0, T2=30.0)

        U_target = qt.sigmax()
        rho0 = qt.basis(2, 0) * qt.basis(2, 0).dag()

        # Fast gate
        pulse_fast = lambda t, args: 0.2
        H_fast = [0 * qt.qeye(2), [H1, pulse_fast]]
        lindblad_fast = LindbladEvolution(H_fast, decoherence)
        gate_time_fast = np.pi / (2 * 0.2)  # Correct time for X-gate: t = π/(2Ω)
        fid_fast = lindblad_fast.gate_fidelity_with_decoherence(
            U_target, rho0, gate_time_fast
        )

        # Slow gate
        pulse_slow = lambda t, args: 0.05
        H_slow = [0 * qt.qeye(2), [H1, pulse_slow]]
        lindblad_slow = LindbladEvolution(H_slow, decoherence)
        gate_time_slow = np.pi / (2 * 0.05)  # Correct time for X-gate: t = π/(2Ω)
        fid_slow = lindblad_slow.gate_fidelity_with_decoherence(
            U_target, rho0, gate_time_slow
        )

        # Faster gate should have higher fidelity
        assert fid_fast > fid_slow


class TestThermalState:
    """Test thermal state construction."""

    def test_zero_temperature(self):
        """Test thermal state at T=0 (ground state)."""
        rho_thermal = thermal_state(2, temperature=0.0, omega=5.0)

        # Should be ground state |0⟩⟨0|
        expected = qt.basis(2, 0) * qt.basis(2, 0).dag()
        assert np.allclose(rho_thermal.full(), expected.full(), atol=1e-10)

    def test_high_temperature(self):
        """Test thermal state at high temperature (maximally mixed)."""
        rho_thermal = thermal_state(2, temperature=100.0, omega=5.0)

        # At high T, should be close to maximally mixed
        ground_pop = rho_thermal[0, 0].real
        excited_pop = rho_thermal[1, 1].real

        # Populations should be similar
        assert np.abs(ground_pop - excited_pop) < 0.2
        # Both should be close to 0.5
        assert 0.3 < ground_pop < 0.7
        assert 0.3 < excited_pop < 0.7

    def test_thermal_state_trace(self):
        """Test that thermal state is properly normalized."""
        rho_thermal = thermal_state(2, temperature=1.0, omega=5.0)
        assert np.isclose(rho_thermal.tr(), 1.0, atol=1e-10)


class TestLindbladRepr:
    """Test string representation."""

    def test_repr(self):
        """Test LindbladEvolution string representation."""
        H = qt.sigmaz()
        decoherence = DecoherenceParams(T1=50.0, T2=30.0)
        lindblad = LindbladEvolution(H, decoherence)

        repr_str = repr(lindblad)
        assert "LindbladEvolution" in repr_str
        assert "T1=50.0" in repr_str
        assert "T2=30.0" in repr_str
        assert "n_collapse_ops" in repr_str


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_long_t1(self):
        """Test with very long T1 (nearly unitary)."""
        H = qt.sigmaz()
        decoherence = DecoherenceParams(T1=1e6, T2=1e6)
        lindblad = LindbladEvolution(H, decoherence)

        rho0 = qt.basis(2, 0) * qt.basis(2, 0).dag()
        times = np.linspace(0, 10, 50)

        result = lindblad.evolve(rho0, times)

        # Should remain nearly pure
        final_purity = (result.states[-1] * result.states[-1]).tr()
        assert np.abs(final_purity - 1.0) < 0.01

    def test_short_t1_fast_decay(self):
        """Test with very short T1 (fast decay)."""
        H = 0 * qt.sigmaz()
        decoherence = DecoherenceParams(T1=1.0, T2=1.0)
        lindblad = LindbladEvolution(H, decoherence)

        rho0 = qt.basis(2, 1) * qt.basis(2, 1).dag()
        times = np.linspace(0, 10, 100)

        result = lindblad.evolve(rho0, times)

        # Should decay to ground state quickly
        final_excited_pop = qt.expect(
            qt.basis(2, 1) * qt.basis(2, 1).dag(), result.states[-1]
        )
        assert final_excited_pop < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
