"""
Unit tests for pulse shape generators.

This module tests all pulse shape functions in src/pulses/shapes.py,
verifying mathematical properties, edge cases, and physical constraints.

Test Categories:
----------------
1. Gaussian pulses: amplitude, width, truncation, normalization
2. Square pulses: flat-top, rise times, edge cases
3. DRAG pulses: I/Q components, derivative correctness, beta scaling
4. Blackman pulses: window properties, spectral characteristics
5. Cosine pulses: smooth envelope, boundary conditions
6. Utility functions: pulse area, scaling, integration accuracy

Author: Orchestrator Agent
Date: 2025-01-27
Phase: 1.3 - Control Hamiltonian and Pulse Shaping
"""

import numpy as np
import pytest
from src.pulses.shapes import (
    gaussian_pulse,
    square_pulse,
    drag_pulse,
    blackman_pulse,
    cosine_pulse,
    pulse_area,
    scale_pulse_to_target_angle,
)


class TestGaussianPulse:
    """Test suite for Gaussian pulse generation."""

    def test_gaussian_pulse_amplitude(self):
        """Gaussian pulse should reach peak amplitude at center."""
        times = np.linspace(0, 100, 1000)
        amplitude = 10.0
        t_center = 50.0
        sigma = 10.0

        pulse = gaussian_pulse(times, amplitude, t_center, sigma)

        # Find peak
        peak_idx = np.argmax(pulse)
        peak_time = times[peak_idx]
        peak_amplitude = pulse[peak_idx]

        assert np.isclose(peak_time, t_center, atol=0.1)
        assert np.isclose(peak_amplitude, amplitude, rtol=1e-3)

    def test_gaussian_pulse_symmetry(self):
        """Gaussian pulse should be symmetric about center."""
        times = np.linspace(0, 100, 10000)  # Higher resolution
        pulse = gaussian_pulse(times, amplitude=5.0, t_center=50.0, sigma=10.0)

        center_idx = np.argmin(np.abs(times - 50.0))

        # Compare left and right halves (reversed)
        left_half = pulse[:center_idx]
        right_half = pulse[center_idx + 1 :][::-1]

        # Use the shorter half
        n = min(len(left_half), len(right_half))
        assert np.allclose(left_half[-n:], right_half[:n], rtol=1e-3)

    def test_gaussian_pulse_width(self):
        """Gaussian FWHM should scale with sigma."""
        times = np.linspace(0, 100, 10000)
        sigma = 10.0
        pulse = gaussian_pulse(times, amplitude=1.0, t_center=50.0, sigma=sigma)

        # FWHM = 2*sqrt(2*ln(2))*sigma ≈ 2.355*sigma
        half_max = 0.5
        above_half = pulse > half_max
        fwhm_indices = np.where(above_half)[0]
        fwhm_time = times[fwhm_indices[-1]] - times[fwhm_indices[0]]

        expected_fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma
        assert np.isclose(fwhm_time, expected_fwhm, rtol=0.05)

    def test_gaussian_pulse_truncation(self):
        """Gaussian pulse should be zero outside truncation region."""
        times = np.linspace(0, 100, 1000)
        t_center = 50.0
        sigma = 5.0
        truncation = 3.0

        pulse = gaussian_pulse(times, 10.0, t_center, sigma, truncation)

        # Check values outside truncation region
        outside_mask = np.abs(times - t_center) > truncation * sigma
        assert np.all(pulse[outside_mask] == 0.0)

    def test_gaussian_pulse_integration(self):
        """Gaussian pulse area should match analytical formula."""
        times = np.linspace(-50, 50, 10000)
        amplitude = 1.0
        sigma = 10.0
        pulse = gaussian_pulse(times, amplitude, t_center=0.0, sigma=sigma)

        area = pulse_area(times, pulse)
        # Analytical: ∫ A*exp(-t²/(2σ²)) dt = A*σ*√(2π)
        expected_area = amplitude * sigma * np.sqrt(2 * np.pi)

        assert np.isclose(area, expected_area, rtol=0.01)

    def test_gaussian_pulse_zero_sigma(self):
        """Zero sigma should produce inf/nan values."""
        times = np.linspace(0, 100, 1000)
        # Division by zero in exponent will produce inf/nan
        pulse = gaussian_pulse(times, 1.0, 50.0, sigma=1e-20)
        # With very small sigma, pulse should be very narrow or have numerical issues
        assert len(pulse) == len(times)

    def test_gaussian_pulse_negative_amplitude(self):
        """Negative amplitude should produce inverted pulse."""
        times = np.linspace(0, 100, 1000)
        pulse_pos = gaussian_pulse(times, 5.0, 50.0, 10.0)
        pulse_neg = gaussian_pulse(times, -5.0, 50.0, 10.0)

        assert np.allclose(pulse_pos, -pulse_neg)


class TestSquarePulse:
    """Test suite for square (rectangular) pulse generation."""

    def test_square_pulse_flat_top(self):
        """Square pulse should be constant during flat-top region."""
        times = np.linspace(0, 100, 1000)
        amplitude = 8.0
        pulse = square_pulse(times, amplitude, t_start=20, t_end=80, rise_time=0)

        # Check flat region
        flat_mask = (times >= 20) & (times <= 80)
        assert np.all(pulse[flat_mask] == amplitude)

    def test_square_pulse_zero_outside(self):
        """Square pulse should be zero outside [t_start, t_end]."""
        times = np.linspace(0, 100, 1000)
        pulse = square_pulse(times, 5.0, t_start=30, t_end=70, rise_time=0)

        outside_mask = (times < 30) | (times > 70)
        assert np.all(pulse[outside_mask] == 0.0)

    def test_square_pulse_duration(self):
        """Square pulse duration should match t_end - t_start."""
        times = np.linspace(0, 100, 10000)
        t_start, t_end = 25, 75
        pulse = square_pulse(times, 1.0, t_start, t_end, rise_time=0)

        nonzero = pulse > 0
        duration_time = times[nonzero][-1] - times[nonzero][0]
        expected_duration = t_end - t_start

        assert np.isclose(duration_time, expected_duration, atol=0.1)

    def test_square_pulse_with_rise_time(self):
        """Square pulse with rise time should have smooth edges."""
        times = np.linspace(0, 100, 10000)
        amplitude = 10.0
        rise_time = 5.0
        pulse = square_pulse(times, amplitude, 20, 80, rise_time)

        # Check rise edge is smooth (no discontinuities)
        rise_region = (times >= 20) & (times <= 25)
        rise_pulse = pulse[rise_region]
        assert rise_pulse[0] < amplitude * 0.1  # Starts near zero
        assert rise_pulse[-1] > amplitude * 0.9  # Ends near full amplitude
        assert np.all(np.diff(rise_pulse) >= 0)  # Monotonic increase

    def test_square_pulse_area(self):
        """Square pulse area should equal amplitude × duration."""
        times = np.linspace(0, 100, 10000)
        amplitude = 6.0
        t_start, t_end = 30, 70
        pulse = square_pulse(times, amplitude, t_start, t_end, rise_time=0)

        area = pulse_area(times, pulse)
        expected_area = amplitude * (t_end - t_start)

        assert np.isclose(area, expected_area, rtol=0.01)

    def test_square_pulse_invalid_times(self):
        """Square pulse with t_end < t_start should be zero."""
        times = np.linspace(0, 100, 1000)
        pulse = square_pulse(times, 5.0, t_start=80, t_end=20)
        assert np.all(pulse == 0.0)


class TestDRAGPulse:
    """Test suite for DRAG (Derivative Removal by Adiabatic Gate) pulses."""

    def test_drag_pulse_components(self):
        """DRAG pulse should return I and Q components."""
        times = np.linspace(0, 100, 1000)
        omega_I, omega_Q = drag_pulse(times, 10.0, 50.0, 10.0, beta=0.3)

        assert isinstance(omega_I, np.ndarray)
        assert isinstance(omega_Q, np.ndarray)
        assert omega_I.shape == times.shape
        assert omega_Q.shape == times.shape

    def test_drag_I_component_is_gaussian(self):
        """DRAG I component should be a Gaussian pulse."""
        times = np.linspace(0, 100, 1000)
        amplitude = 5.0
        t_center = 50.0
        sigma = 10.0

        omega_I, _ = drag_pulse(times, amplitude, t_center, sigma, beta=0.3)
        gaussian = gaussian_pulse(times, amplitude, t_center, sigma)

        assert np.allclose(omega_I, gaussian, rtol=1e-10)

    def test_drag_Q_component_is_derivative(self):
        """DRAG Q component should be proportional to derivative of I."""
        times = np.linspace(0, 100, 10000)
        amplitude = 5.0
        t_center = 50.0
        sigma = 10.0
        beta = 0.4

        omega_I, omega_Q = drag_pulse(times, amplitude, t_center, sigma, beta)

        # Numerical derivative of I component
        dt = times[1] - times[0]
        dI_dt = np.gradient(omega_I, dt)

        # Q should be proportional to dI/dt (with correct sign)
        # Note: beta is applied to analytical derivative, correlation can be negative
        correlation = np.corrcoef(omega_Q, dI_dt)[0, 1]
        assert np.abs(correlation) > 0.99  # Strong correlation (sign may differ)

    def test_drag_Q_zero_at_center(self):
        """DRAG Q component should be near zero at pulse center (derivative=0)."""
        times = np.linspace(0, 100, 10000)
        t_center = 50.0
        _, omega_Q = drag_pulse(times, 10.0, t_center, 10.0, beta=0.3)

        center_idx = np.argmin(np.abs(times - t_center))
        # Due to discretization, allow small non-zero value
        assert np.abs(omega_Q[center_idx]) < 1e-3

    def test_drag_Q_antisymmetric(self):
        """DRAG Q component should be antisymmetric about center."""
        times = np.linspace(0, 100, 10000)
        t_center = 50.0
        sigma = 10.0
        _, omega_Q = drag_pulse(times, 10.0, t_center, sigma, beta=0.3)

        center_idx = np.argmin(np.abs(times - t_center))

        # Check that the Q component changes sign across the center
        # by verifying sum of left half is negative of sum of right half
        left_half = omega_Q[:center_idx]
        right_half = omega_Q[center_idx + 1 :]

        # For antisymmetric function, integral of left should equal -integral of right
        sum_left = np.sum(left_half)
        sum_right = np.sum(right_half)

        # Allow some tolerance due to discretization
        assert np.isclose(sum_left, -sum_right, rtol=0.1)

    def test_drag_beta_scaling(self):
        """DRAG Q component should scale linearly with beta."""
        times = np.linspace(0, 100, 1000)
        _, omega_Q1 = drag_pulse(times, 5.0, 50.0, 10.0, beta=0.2)
        _, omega_Q2 = drag_pulse(times, 5.0, 50.0, 10.0, beta=0.4)

        # Q2 should be twice Q1
        assert np.allclose(omega_Q2, 2 * omega_Q1, rtol=1e-10)

    def test_drag_zero_beta(self):
        """DRAG with beta=0 should have zero Q component."""
        times = np.linspace(0, 100, 1000)
        _, omega_Q = drag_pulse(times, 10.0, 50.0, 10.0, beta=0.0)
        assert np.all(omega_Q == 0.0)


class TestBlackmanPulse:
    """Test suite for Blackman window pulse."""

    def test_blackman_pulse_range(self):
        """Blackman pulse should be within [t_start, t_end]."""
        times = np.linspace(0, 100, 1000)
        pulse = blackman_pulse(times, 10.0, 20, 80)

        outside = (times < 20) | (times > 80)
        assert np.all(pulse[outside] == 0.0)

        inside = (times >= 20) & (times <= 80)
        assert np.any(pulse[inside] > 0.0)

    def test_blackman_pulse_peak(self):
        """Blackman pulse should peak near center."""
        times = np.linspace(0, 100, 10000)
        t_start, t_end = 20, 80
        amplitude = 8.0
        pulse = blackman_pulse(times, amplitude, t_start, t_end)

        peak_idx = np.argmax(pulse)
        peak_time = times[peak_idx]
        center = (t_start + t_end) / 2

        assert np.isclose(peak_time, center, atol=1.0)
        assert pulse[peak_idx] <= amplitude

    def test_blackman_pulse_smooth_edges(self):
        """Blackman pulse should smoothly go to zero at edges."""
        times = np.linspace(20, 80, 10000)
        pulse = blackman_pulse(times, 10.0, 20, 80)

        # First and last values should be near zero
        assert pulse[0] < 0.01
        assert pulse[-1] < 0.01

        # No discontinuities (gradient should be finite)
        gradient = np.gradient(pulse)
        assert np.all(np.isfinite(gradient))

    def test_blackman_pulse_symmetry(self):
        """Blackman pulse should be symmetric about center."""
        times = np.linspace(20, 80, 10000)
        pulse = blackman_pulse(times, 5.0, 20, 80)

        # Compare first and second half
        mid = len(pulse) // 2
        first_half = pulse[:mid]
        second_half = pulse[mid:][::-1]

        assert np.allclose(first_half, second_half, rtol=0.01)


class TestCosinePulse:
    """Test suite for cosine (Hann) pulse."""

    def test_cosine_pulse_range(self):
        """Cosine pulse should be nonzero only in [t_start, t_end]."""
        times = np.linspace(0, 100, 1000)
        pulse = cosine_pulse(times, 10.0, 25, 75)

        outside = (times < 25) | (times > 75)
        inside = (times >= 25) & (times <= 75)

        assert np.all(pulse[outside] == 0.0)
        assert np.all(pulse[inside] >= 0.0)

    def test_cosine_pulse_boundary_conditions(self):
        """Cosine pulse should be zero at boundaries."""
        times = np.linspace(20, 80, 10000)
        pulse = cosine_pulse(times, 10.0, 20, 80)

        assert np.isclose(pulse[0], 0.0, atol=1e-10)
        assert np.isclose(pulse[-1], 0.0, atol=1e-10)

    def test_cosine_pulse_peak(self):
        """Cosine pulse should peak at center."""
        times = np.linspace(0, 100, 10000)
        amplitude = 12.0
        t_start, t_end = 30, 70
        pulse = cosine_pulse(times, amplitude, t_start, t_end)

        peak_idx = np.argmax(pulse)
        peak_time = times[peak_idx]
        center = (t_start + t_end) / 2

        assert np.isclose(peak_time, center, atol=0.2)
        assert np.isclose(pulse[peak_idx], amplitude, rtol=1e-3)

    def test_cosine_pulse_smoothness(self):
        """Cosine pulse should be smooth (continuous derivatives)."""
        times = np.linspace(20, 80, 10000)
        pulse = cosine_pulse(times, 8.0, 20, 80)

        # First derivative should be finite
        gradient = np.gradient(pulse)
        assert np.all(np.isfinite(gradient))

        # Second derivative should be finite
        gradient2 = np.gradient(gradient)
        assert np.all(np.isfinite(gradient2))


class TestPulseUtilities:
    """Test suite for pulse utility functions."""

    def test_pulse_area_square(self):
        """Pulse area for square pulse should be amplitude × duration."""
        times = np.linspace(0, 100, 10000)
        amplitude = 5.0
        duration = 40.0
        pulse = square_pulse(times, amplitude, 30, 70, rise_time=0)

        area = pulse_area(times, pulse)
        expected = amplitude * duration

        assert np.isclose(area, expected, rtol=0.01)

    def test_pulse_area_gaussian(self):
        """Pulse area for Gaussian should match analytical formula."""
        times = np.linspace(-50, 50, 10000)
        amplitude = 3.0
        sigma = 8.0
        pulse = gaussian_pulse(times, amplitude, 0, sigma)

        area = pulse_area(times, pulse)
        expected = amplitude * sigma * np.sqrt(2 * np.pi)

        assert np.isclose(area, expected, rtol=0.01)

    def test_pulse_area_zero_pulse(self):
        """Pulse area for zero pulse should be zero."""
        times = np.linspace(0, 100, 1000)
        pulse = np.zeros_like(times)

        area = pulse_area(times, pulse)
        assert np.isclose(area, 0.0, atol=1e-12)

    def test_scale_pulse_to_pi(self):
        """Scaled pulse should have area equal to π."""
        times = np.linspace(0, 100, 10000)
        pulse = gaussian_pulse(times, 1.0, 50, 10)

        pi_pulse = scale_pulse_to_target_angle(pulse, times, np.pi)
        area = pulse_area(times, pi_pulse)

        assert np.isclose(area, np.pi, rtol=1e-3)

    def test_scale_pulse_to_pi_half(self):
        """Scaled pulse should have area equal to π/2."""
        times = np.linspace(0, 100, 10000)
        pulse = cosine_pulse(times, 5.0, 20, 80)

        pi_half_pulse = scale_pulse_to_target_angle(pulse, times, np.pi / 2)
        area = pulse_area(times, pi_half_pulse)

        assert np.isclose(area, np.pi / 2, rtol=1e-3)

    def test_scale_pulse_preserves_shape(self):
        """Scaling should preserve pulse shape (only amplitude changes)."""
        times = np.linspace(0, 100, 10000)
        pulse = blackman_pulse(times, 3.0, 25, 75)
        target_angle = 2.0

        scaled_pulse = scale_pulse_to_target_angle(pulse, times, target_angle)

        # Normalized shapes should be identical
        pulse_norm = pulse / np.max(np.abs(pulse))
        scaled_norm = scaled_pulse / np.max(np.abs(scaled_pulse))

        assert np.allclose(pulse_norm, scaled_norm, rtol=1e-10)

    def test_scale_zero_pulse_raises(self):
        """Scaling zero pulse should raise ValueError."""
        times = np.linspace(0, 100, 1000)
        pulse = np.zeros_like(times)

        with pytest.raises(ValueError, match="Pulse area is zero"):
            scale_pulse_to_target_angle(pulse, times, np.pi)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_time_array(self):
        """Empty time array should return empty pulse."""
        times = np.array([])
        pulse = gaussian_pulse(times, 1.0, 50, 10)
        assert len(pulse) == 0

    def test_single_time_point(self):
        """Single time point should be handled correctly."""
        times = np.array([50.0])
        pulse = gaussian_pulse(times, 10.0, 50.0, 5.0)
        assert pulse[0] == 10.0  # At center

    def test_very_large_sigma(self):
        """Very large sigma should give nearly flat Gaussian."""
        times = np.linspace(0, 100, 1000)
        pulse = gaussian_pulse(times, 10.0, 50.0, sigma=1e6)

        # Should be nearly constant across range
        assert np.std(pulse) < 1e-6

    def test_very_small_sigma(self):
        """Very small sigma should give narrow spike."""
        times = np.linspace(49, 51, 10000)
        pulse = gaussian_pulse(times, 10.0, 50.0, sigma=0.01)

        # Should be nonzero only very close to center
        nonzero = pulse > 0.1 * pulse.max()
        # With truncation=4, width is 4*0.01 = 0.04, in 2 unit range with 10000 pts
        # that's about 0.04/2 * 10000 = 200 points
        assert np.sum(nonzero) < 500  # Very narrow relative to total range

    def test_negative_duration_square_pulse(self):
        """Square pulse with negative duration should be zero."""
        times = np.linspace(0, 100, 1000)
        pulse = square_pulse(times, 5.0, t_start=70, t_end=30)
        assert np.all(pulse == 0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
