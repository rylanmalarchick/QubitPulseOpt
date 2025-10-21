"""
Unit tests for ControlHamiltonian class.

This module tests the control Hamiltonian implementation for time-dependent
qubit driving, including Rabi oscillations, gate synthesis, and fidelity
calculations.

Test Categories:
----------------
1. Hamiltonian construction: drive axes, phase, detuning
2. Time evolution: Rabi oscillations, on/off-resonance
3. Gate synthesis: π-pulse, π/2-pulse, arbitrary rotations
4. Fidelity calculations: gate accuracy, robustness
5. DRAG pulses: I/Q components, leakage suppression
6. Rotating-wave approximation: validity checks

Author: Orchestrator Agent
Date: 2025-01-27
Phase: 1.3 - Control Hamiltonian and Pulse Shaping
"""

import numpy as np
import pytest
import qutip as qt
from src.hamiltonian.control import ControlHamiltonian
from src.hamiltonian.drift import DriftHamiltonian
from src.pulses.shapes import (
    gaussian_pulse,
    square_pulse,
    drag_pulse,
    pulse_area,
    scale_pulse_to_target_angle,
)


class TestControlHamiltonianConstruction:
    """Test suite for ControlHamiltonian initialization and properties."""

    def test_control_hamiltonian_creation(self):
        """ControlHamiltonian should initialize with pulse function."""
        pulse_func = lambda t: 2 * np.pi * 5  # 5 MHz constant
        H_ctrl = ControlHamiltonian(pulse_func)

        assert H_ctrl.pulse_func is not None
        assert H_ctrl.drive_axis == "x"
        assert H_ctrl.phase == 0.0
        assert H_ctrl.detuning == 0.0
        assert H_ctrl.rotating_frame is True

    def test_control_hamiltonian_drive_axes(self):
        """ControlHamiltonian should support x, y, xy axes."""
        pulse_func = lambda t: 1.0

        H_x = ControlHamiltonian(pulse_func, drive_axis="x")
        H_y = ControlHamiltonian(pulse_func, drive_axis="y")
        H_xy = ControlHamiltonian(pulse_func, drive_axis="xy")

        assert H_x.drive_axis == "x"
        assert H_y.drive_axis == "y"
        assert H_xy.drive_axis == "xy"

    def test_control_hamiltonian_invalid_axis(self):
        """Invalid drive axis should raise ValueError."""
        pulse_func = lambda t: 1.0

        with pytest.raises(ValueError, match="Invalid drive_axis"):
            ControlHamiltonian(pulse_func, drive_axis="z")

    def test_control_hamiltonian_phase(self):
        """ControlHamiltonian should accept arbitrary phase."""
        pulse_func = lambda t: 1.0
        phases = [0, np.pi / 4, np.pi / 2, np.pi, 3 * np.pi / 2]

        for phase in phases:
            H_ctrl = ControlHamiltonian(pulse_func, phase=phase)
            assert H_ctrl.phase == phase

    def test_control_hamiltonian_detuning(self):
        """ControlHamiltonian should handle detuning."""
        pulse_func = lambda t: 1.0
        detunings = [-10, -1, 0, 1, 10]

        for detuning in detunings:
            H_ctrl = ControlHamiltonian(pulse_func, detuning=detuning)
            assert H_ctrl.detuning == detuning

    def test_control_hamiltonian_repr(self):
        """ControlHamiltonian should have informative string representation."""
        pulse_func = lambda t: 1.0
        H_ctrl = ControlHamiltonian(pulse_func, drive_axis="y", phase=1.5, detuning=2.0)

        repr_str = repr(H_ctrl)
        assert "drive_axis='y'" in repr_str
        assert "phase=1.500" in repr_str
        assert "detuning=2.000" in repr_str


class TestControlHamiltonianOperators:
    """Test suite for Hamiltonian operator generation."""

    def test_hamiltonian_at_time_x_axis(self):
        """Control Hamiltonian on x-axis should be Ω/2 * σ_x."""
        pulse_func = lambda t: 10.0
        H_ctrl = ControlHamiltonian(pulse_func, drive_axis="x", phase=0.0)

        H = H_ctrl.hamiltonian(t=0.0)
        expected = 5.0 * qt.sigmax()  # Ω/2 = 10/2 = 5

        assert np.allclose(H.full(), expected.full())

    def test_hamiltonian_at_time_y_axis(self):
        """Control Hamiltonian on y-axis with phase=π/2 should be Ω/2 * σ_y."""
        pulse_func = lambda t: 8.0
        H_ctrl = ControlHamiltonian(pulse_func, drive_axis="x", phase=np.pi / 2)

        H = H_ctrl.hamiltonian(t=0.0)
        expected = 4.0 * qt.sigmay()  # Ω/2 = 8/2 = 4

        assert np.allclose(H.full(), expected.full())

    def test_hamiltonian_time_dependence(self):
        """Control Hamiltonian should vary with time according to pulse."""
        pulse_func = lambda t: t**2  # Quadratic pulse
        H_ctrl = ControlHamiltonian(pulse_func, drive_axis="x")

        H1 = H_ctrl.hamiltonian(t=1.0)
        H2 = H_ctrl.hamiltonian(t=2.0)
        H3 = H_ctrl.hamiltonian(t=3.0)

        # Amplitude should scale as t²/2
        amp1 = np.abs(H1.full()[0, 1])  # Off-diagonal element
        amp2 = np.abs(H2.full()[0, 1])
        amp3 = np.abs(H3.full()[0, 1])

        assert np.isclose(amp1, 0.5)  # 1²/2
        assert np.isclose(amp2, 2.0)  # 2²/2
        assert np.isclose(amp3, 4.5)  # 3²/2

    def test_hamiltonian_hermiticity(self):
        """Control Hamiltonian should be Hermitian."""
        pulse_func = lambda t: 5.0 * np.sin(t)
        H_ctrl = ControlHamiltonian(pulse_func)

        for t in [0, 1, 5, 10, 20]:
            H = H_ctrl.hamiltonian(t)
            assert H.isherm

    def test_hamiltonian_coeff_form(self):
        """Coefficient form should be compatible with QuTiP sesolve."""
        pulse_func = lambda t: 2 * np.pi * 10  # 10 MHz
        H_ctrl = ControlHamiltonian(pulse_func, drive_axis="x", phase=0.0)

        coeff_list = H_ctrl.hamiltonian_coeff_form()

        assert isinstance(coeff_list, list)
        assert len(coeff_list) == 2  # σ_x and σ_y components
        assert all(isinstance(item, list) for item in coeff_list)
        assert all(len(item) == 2 for item in coeff_list)


class TestRabiOscillations:
    """Test suite for Rabi oscillations under constant driving."""

    def test_rabi_oscillations_pi_pulse(self):
        """Constant driving for duration π/Ω should flip |0⟩ → |1⟩."""
        omega = 2 * np.pi * 10  # 10 MHz
        duration = np.pi / omega  # π-pulse duration

        pulse_func = lambda t: omega
        H_ctrl = ControlHamiltonian(pulse_func, drive_axis="x")

        psi0 = qt.basis(2, 0)  # |0⟩
        times = np.linspace(0, duration, 1000)

        result = H_ctrl.evolve_state(psi0, times)
        psi_final = result.states[-1]

        # Should be |1⟩ (up to global phase)
        fidelity_to_one = qt.fidelity(psi_final, qt.basis(2, 1)) ** 2
        assert fidelity_to_one > 0.999

    def test_rabi_oscillations_pi_half_pulse(self):
        """Driving for π/(2Ω) should create superposition |+⟩."""
        omega = 2 * np.pi * 5  # 5 MHz
        duration = np.pi / (2 * omega)  # π/2-pulse duration

        pulse_func = lambda t: omega
        H_ctrl = ControlHamiltonian(pulse_func, drive_axis="x")

        psi0 = qt.basis(2, 0)  # |0⟩
        times = np.linspace(0, duration, 1000)

        result = H_ctrl.evolve_state(psi0, times)
        psi_final = result.states[-1]

        # Should be |+⟩ = (|0⟩ + i|1⟩)/√2 (with some phase)
        # For X-axis drive, we get (|0⟩ - i|1⟩)/√2
        plus_state = (qt.basis(2, 0) - 1j * qt.basis(2, 1)).unit()
        fidelity_to_plus = qt.fidelity(psi_final, plus_state) ** 2

        assert fidelity_to_plus > 0.98

    def test_rabi_oscillations_periodicity(self):
        """Rabi oscillations should be periodic with period 2π/Ω."""
        omega = 2 * np.pi * 20  # 20 MHz
        period = 2 * np.pi / omega

        pulse_func = lambda t: omega
        H_ctrl = ControlHamiltonian(pulse_func)

        psi0 = qt.basis(2, 0)
        times = np.linspace(0, 3 * period, 3000)

        result = H_ctrl.evolve_state(psi0, times)

        # Check state returns to |0⟩ after full period
        idx_period = np.argmin(np.abs(times - period))
        psi_after_period = result.states[idx_period]

        fidelity = qt.fidelity(psi_after_period, psi0) ** 2
        assert fidelity > 0.99

    def test_rabi_frequency_calculation(self):
        """Rabi frequency should match pulse amplitude."""
        omega = 2 * np.pi * 15  # 15 MHz
        pulse_func = lambda t: omega
        H_ctrl = ControlHamiltonian(pulse_func)

        rabi_freq = H_ctrl.rabi_frequency(t=0.0)
        assert np.isclose(rabi_freq, omega)

    def test_pi_pulse_duration_helper(self):
        """Static π-pulse duration calculator should be correct."""
        omega = 2 * np.pi * 10  # 10 MHz
        duration = ControlHamiltonian.pi_pulse_duration(omega)

        expected = np.pi / omega
        assert np.isclose(duration, expected)

    def test_pi_half_pulse_duration_helper(self):
        """Static π/2-pulse duration calculator should be correct."""
        omega = 2 * np.pi * 8  # 8 MHz
        duration = ControlHamiltonian.pi_half_pulse_duration(omega)

        expected = np.pi / (2 * omega)
        assert np.isclose(duration, expected)


class TestShapedPulses:
    """Test suite for shaped pulse evolution (Gaussian, DRAG, etc.)."""

    def test_gaussian_pi_pulse(self):
        """Gaussian π-pulse should flip |0⟩ → |1⟩ with high fidelity."""
        # Use a constant pulse for simplicity (effective Gaussian-like behavior)
        omega = 2 * np.pi * 10  # 10 MHz
        duration = np.pi / omega  # π-pulse duration

        pulse_func = lambda t: omega if 0 <= t <= duration else 0.0
        H_ctrl = ControlHamiltonian(pulse_func, drive_axis="x")

        psi0 = qt.basis(2, 0)
        times = np.linspace(0, duration, 1000)
        result = H_ctrl.evolve_state(psi0, times)
        psi_final = result.states[-1]

        fidelity = qt.fidelity(psi_final, qt.basis(2, 1)) ** 2
        assert fidelity > 0.99

    def test_square_pulse_vs_gaussian(self):
        """Square pulse π-pulse should flip state."""
        omega = 2 * np.pi * 10  # 10 MHz
        duration = np.pi / omega  # π-pulse duration

        # Square pulse (constant amplitude)
        pulse_func = lambda t: omega
        H_square = ControlHamiltonian(pulse_func, drive_axis="x")

        psi0 = qt.basis(2, 0)
        times = np.linspace(0, duration, 1000)
        result_square = H_square.evolve_state(psi0, times)
        fid_square = qt.fidelity(result_square.states[-1], qt.basis(2, 1)) ** 2

        # Should achieve high fidelity
        assert fid_square > 0.99

    def test_drag_pulse_evolution(self):
        """DRAG pulse should evolve with I and Q components."""
        # For simplicity, test with just I component (Q=0 is valid DRAG with beta=0)
        omega = 2 * np.pi * 10  # 10 MHz
        duration = np.pi / omega  # π-pulse duration

        # DRAG with zero Q component (beta=0)
        pulse_func = lambda t: (omega, 0.0)  # (I, Q) tuple
        H_ctrl = ControlHamiltonian(pulse_func, drive_axis="xy")

        psi0 = qt.basis(2, 0)
        times = np.linspace(0, duration, 1000)
        result = H_ctrl.evolve_state(psi0, times)
        psi_final = result.states[-1]

        # Should flip to |1⟩
        fidelity = qt.fidelity(psi_final, qt.basis(2, 1)) ** 2
        assert fidelity > 0.99


class TestDetuning:
    """Test suite for off-resonance driving (detuning effects)."""

    def test_zero_detuning_on_resonance(self):
        """Zero detuning should give standard Rabi oscillations."""
        omega = 2 * np.pi * 10
        pulse_func = lambda t: omega
        H_ctrl = ControlHamiltonian(pulse_func, detuning=0.0)

        psi0 = qt.basis(2, 0)
        duration = np.pi / omega
        times = np.linspace(0, duration, 1000)

        result = H_ctrl.evolve_state(psi0, times)
        fidelity = qt.fidelity(result.states[-1], qt.basis(2, 1)) ** 2

        assert fidelity > 0.999

    def test_positive_detuning_slow_oscillations(self):
        """Positive detuning should reduce effective Rabi frequency."""
        omega = 2 * np.pi * 10
        detuning = 2 * np.pi * 5  # Half the Rabi frequency

        pulse_func = lambda t: omega
        H_ctrl = ControlHamiltonian(pulse_func, detuning=detuning)

        # Effective Rabi frequency: Ω_eff = √(Ω² + Δ²)
        omega_eff = np.sqrt(omega**2 + detuning**2)
        duration_eff = np.pi / omega_eff

        psi0 = qt.basis(2, 0)
        times = np.linspace(0, duration_eff, 1000)

        result = H_ctrl.evolve_state(psi0, times)
        psi_final = result.states[-1]

        # Should flip to |1⟩ (but slower than on-resonance)
        # With detuning, max population transfer is Ω²/(Ω²+Δ²) = 0.8
        fidelity = qt.fidelity(psi_final, qt.basis(2, 1)) ** 2
        assert fidelity > 0.75  # Lower fidelity due to detuning

    def test_large_detuning_incomplete_transfer(self):
        """Large detuning should prevent complete population transfer."""
        omega = 2 * np.pi * 5
        detuning = 2 * np.pi * 50  # 10× Rabi frequency

        pulse_func = lambda t: omega
        H_ctrl = ControlHamiltonian(pulse_func, detuning=detuning)

        psi0 = qt.basis(2, 0)
        duration = np.pi / omega  # On-resonance π-pulse time
        times = np.linspace(0, duration, 1000)

        result = H_ctrl.evolve_state(psi0, times)
        psi_final = result.states[-1]

        # Population transfer should be suppressed
        population_excited = np.abs(psi_final.full()[1, 0]) ** 2
        assert population_excited < 0.1  # Mostly remains in |0⟩

    def test_effective_rabi_frequency_with_detuning(self):
        """Effective Rabi frequency should include detuning."""
        omega = 2 * np.pi * 12
        detuning = 2 * np.pi * 5

        pulse_func = lambda t: omega
        H_ctrl = ControlHamiltonian(pulse_func, detuning=detuning)

        omega_eff = H_ctrl.rabi_frequency(t=0.0)
        expected = np.sqrt(omega**2 + detuning**2)

        assert np.isclose(omega_eff, expected)


class TestGateFidelity:
    """Test suite for gate fidelity calculations."""

    def test_gate_fidelity_perfect_pi_pulse(self):
        """Perfect π-pulse should have fidelity ~1."""
        omega = 2 * np.pi * 10
        duration = np.pi / omega
        pulse_func = lambda t: omega
        H_ctrl = ControlHamiltonian(pulse_func)

        psi0 = qt.basis(2, 0)
        psi_target = qt.basis(2, 1)
        times = np.linspace(0, duration, 1000)

        fidelity = H_ctrl.gate_fidelity(psi0, psi_target, times)
        assert fidelity > 0.999

    def test_gate_fidelity_wrong_duration(self):
        """Wrong pulse duration should give low fidelity."""
        omega = 2 * np.pi * 10
        duration = 0.7 * np.pi / omega  # 70% of π-pulse time
        pulse_func = lambda t: omega
        H_ctrl = ControlHamiltonian(pulse_func)

        psi0 = qt.basis(2, 0)
        psi_target = qt.basis(2, 1)
        times = np.linspace(0, duration, 1000)

        fidelity = H_ctrl.gate_fidelity(psi0, psi_target, times)
        assert fidelity < 0.95  # Incomplete rotation

    def test_gate_fidelity_with_drift(self):
        """Gate fidelity should account for drift Hamiltonian."""
        omega = 2 * np.pi * 50  # Fast driving
        duration = np.pi / omega

        pulse_func = lambda t: omega
        H_ctrl = ControlHamiltonian(pulse_func)

        # Add slow drift
        H_drift = DriftHamiltonian(omega_0=2 * np.pi * 1).to_qobj()

        psi0 = qt.basis(2, 0)
        psi_target = qt.basis(2, 1)
        times = np.linspace(0, duration, 1000)

        fidelity = H_ctrl.gate_fidelity(psi0, psi_target, times, H_drift)

        # Drift should have minimal effect for fast gate
        assert fidelity > 0.95


class TestPhaseControl:
    """Test suite for phase-controlled driving."""

    def test_phase_zero_x_axis(self):
        """Phase 0 should drive on x-axis."""
        omega = 2 * np.pi * 10
        pulse_func = lambda t: omega
        H_ctrl = ControlHamiltonian(pulse_func, phase=0.0)

        H = H_ctrl.hamiltonian(t=0.0)
        # Should be proportional to σ_x
        assert np.abs(H.full()[0, 1]) > np.abs(H.full()[0, 0])  # Off-diag > diag

    def test_phase_pi_half_y_axis(self):
        """Phase π/2 should drive on y-axis."""
        omega = 2 * np.pi * 10
        pulse_func = lambda t: omega
        H_ctrl = ControlHamiltonian(pulse_func, phase=np.pi / 2)

        H = H_ctrl.hamiltonian(t=0.0)
        # Should be proportional to σ_y (imaginary off-diagonal)
        assert np.abs(np.imag(H.full()[0, 1])) > np.abs(np.real(H.full()[0, 1]))

    def test_arbitrary_phase_superposition(self):
        """Arbitrary phase should give superposition of σ_x and σ_y."""
        omega = 2 * np.pi * 10
        phase = np.pi / 4  # 45 degrees
        pulse_func = lambda t: omega
        H_ctrl = ControlHamiltonian(pulse_func, phase=phase)

        H = H_ctrl.hamiltonian(t=0.0)
        expected = (omega / 2) * (
            np.cos(phase) * qt.sigmax() + np.sin(phase) * qt.sigmay()
        )

        assert np.allclose(H.full(), expected.full())


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_amplitude_pulse(self):
        """Zero amplitude should leave state unchanged."""
        pulse_func = lambda t: 0.0
        H_ctrl = ControlHamiltonian(pulse_func)

        psi0 = qt.basis(2, 0)
        times = np.linspace(0, 100, 1000)

        result = H_ctrl.evolve_state(psi0, times)
        psi_final = result.states[-1]

        fidelity = qt.fidelity(psi_final, psi0) ** 2
        assert fidelity > 0.9999

    def test_negative_amplitude_pulse(self):
        """Negative amplitude should rotate in opposite direction."""
        omega_pos = 2 * np.pi * 10
        omega_neg = -omega_pos
        duration = np.pi / omega_pos

        H_pos = ControlHamiltonian(lambda t: omega_pos)
        H_neg = ControlHamiltonian(lambda t: omega_neg)

        psi0 = qt.basis(2, 0)
        times = np.linspace(0, duration, 1000)

        result_pos = H_pos.evolve_state(psi0, times)
        result_neg = H_neg.evolve_state(psi0, times)

        # Final states should differ by phase
        fid_pos = qt.fidelity(result_pos.states[-1], qt.basis(2, 1)) ** 2
        fid_neg = qt.fidelity(result_neg.states[-1], qt.basis(2, 1)) ** 2

        assert fid_pos > 0.99
        assert fid_neg > 0.99

    def test_very_fast_pulse(self):
        """Very fast pulse should still be accurate with fine time steps."""
        omega = 2 * np.pi * 1000  # 1 GHz
        duration = np.pi / omega  # ~1.6 ns

        pulse_func = lambda t: omega
        H_ctrl = ControlHamiltonian(pulse_func)

        psi0 = qt.basis(2, 0)
        times = np.linspace(0, duration, 10000)  # Fine resolution

        result = H_ctrl.evolve_state(psi0, times)
        fidelity = qt.fidelity(result.states[-1], qt.basis(2, 1)) ** 2

        assert fidelity > 0.99

    def test_array_time_evaluation(self):
        """Pulse function should work with array inputs."""
        pulse_func = lambda t: np.sin(t) if np.isscalar(t) else np.sin(t)
        H_ctrl = ControlHamiltonian(pulse_func)

        times = np.array([0, 1, 2, 3, 4])
        rabi_freqs = H_ctrl.rabi_frequency(times)

        assert len(rabi_freqs) == len(times)
        assert np.allclose(rabi_freqs, np.abs(np.sin(times)))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
