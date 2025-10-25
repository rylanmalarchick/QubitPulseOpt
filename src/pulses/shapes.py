"""
Pulse shape generators for quantum control.

This module implements various pulse envelope functions used in quantum control
experiments. Each function generates time-dependent amplitude profiles that
modulate the control Hamiltonian to achieve high-fidelity quantum gates.

Physical Context:
-----------------
In quantum control, we apply time-dependent electromagnetic fields to drive
transitions between qubit states. The control Hamiltonian has the form:
    H_c(t) = Ω(t) σ_x
where Ω(t) is the Rabi frequency (pulse envelope) and σ_x is the Pauli-X matrix.

The pulse shape Ω(t) determines:
- Gate fidelity (how close we get to the target unitary)
- Leakage to non-computational states
- Robustness to noise and parameter variations
- Total gate time (speed vs. accuracy trade-off)

Common Pulse Shapes:
-------------------
1. Gaussian: Smooth rise/fall, minimal spectral leakage
2. Square: Constant amplitude, fastest but harsh transitions
3. DRAG: Gaussian + derivative correction, suppresses leakage
4. Blackman: Excellent spectral properties, gentle envelope
5. Cosine: Simple smooth envelope, good for adiabatic gates

References:
-----------
- Motzoi et al., PRL 103, 110501 (2009) - DRAG pulses
- Nielsen & Chuang, "Quantum Computation and Quantum Information" (2010)
- Gambetta et al., PRA 83, 012308 (2011) - Pulse shaping for superconducting qubits
"""

import numpy as np
from typing import Union, Optional, Tuple

# Power of 10 compliance: Import bounds and assertion helpers
from ..constants import (
    MIN_PULSE_AMPLITUDE,
    MAX_PULSE_AMPLITUDE,
    MIN_TIME,
    MAX_TIME,
)


def _validate_times_array(times: np.ndarray) -> None:
    """Validate times array parameter."""
    if times is None:
        raise ValueError("times array cannot be None")
    if not isinstance(times, np.ndarray):
        raise TypeError(f"times must be ndarray, got {type(times)}")
    if len(times) > 0 and not np.all(np.isfinite(times)):
        raise ValueError("times must contain only finite values")


def _validate_pulse_amplitude(amplitude: float) -> None:
    """Validate pulse amplitude parameter."""
    if not isinstance(amplitude, (int, float)):
        raise TypeError(f"amplitude must be numeric, got {type(amplitude)}")
    if not np.isfinite(amplitude):
        raise ValueError(f"amplitude must be finite, got {amplitude}")
    if not (MIN_PULSE_AMPLITUDE <= amplitude <= MAX_PULSE_AMPLITUDE):
        raise ValueError(
            f"amplitude {amplitude} outside bounds [{MIN_PULSE_AMPLITUDE}, {MAX_PULSE_AMPLITUDE}]"
        )


def _validate_pulse_center(t_center: float) -> None:
    """Validate pulse center time parameter."""
    if not isinstance(t_center, (int, float)):
        raise TypeError(f"t_center must be numeric, got {type(t_center)}")
    if not np.isfinite(t_center):
        raise ValueError(f"t_center must be finite, got {t_center}")
    if t_center < 0:
        raise ValueError(f"t_center must be non-negative, got {t_center}")


def _validate_pulse_sigma(sigma: float) -> None:
    """Validate pulse width parameter."""
    if not isinstance(sigma, (int, float)):
        raise TypeError(f"sigma must be numeric, got {type(sigma)}")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if not np.isfinite(sigma):
        raise ValueError(f"sigma must be finite, got {sigma}")


def _validate_truncation(truncation: float) -> None:
    """Validate truncation parameter."""
    if truncation <= 0:
        raise ValueError(f"truncation must be positive, got {truncation}")


def _compute_gaussian_envelope(
    times: np.ndarray,
    amplitude: float,
    t_center: float,
    sigma: float,
    truncation: float,
) -> np.ndarray:
    """Compute Gaussian pulse envelope."""
    pulse = np.zeros_like(times)
    mask = np.abs(times - t_center) <= truncation * sigma
    pulse[mask] = amplitude * np.exp(-((times[mask] - t_center) ** 2) / (2 * sigma**2))
    return pulse


def gaussian_pulse(
    times: np.ndarray,
    amplitude: float,
    t_center: float,
    sigma: float,
    truncation: float = 4.0,
) -> np.ndarray:
    """
    Generate a Gaussian pulse envelope.

    Gaussian pulses minimize spectral leakage and unwanted transitions.
    Form: Ω(t) = A * exp(-(t-t_c)²/(2σ²)) for |t-t_c| ≤ truncation*σ
    """
    # Validate parameters
    _validate_times_array(times)
    if len(times) == 0:
        return np.array([])

    _validate_pulse_amplitude(amplitude)
    _validate_pulse_center(t_center)
    _validate_pulse_sigma(sigma)
    _validate_truncation(truncation)

    # Compute pulse
    pulse = _compute_gaussian_envelope(times, amplitude, t_center, sigma, truncation)

    # Postcondition checks
    assert np.all(np.isfinite(pulse)), "Pulse contains non-finite values"
    assert pulse.shape == times.shape, (
        f"Output shape {pulse.shape} != input shape {times.shape}"
    )

    return pulse


def _validate_square_pulse_params(
    times: np.ndarray,
    amplitude: float,
    t_start: float,
    t_end: float,
    rise_time: float,
) -> None:
    """
    Validate square pulse parameters.

    Parameters
    ----------
    times : np.ndarray
        Array of time points
    amplitude : float
        Pulse amplitude
    t_start : float
        Start time
    t_end : float
        End time
    rise_time : float
        Rise/fall time

    Raises
    ------
    ValueError, TypeError
        If parameters are invalid
    """
    if times is None:
        raise ValueError("times array cannot be None")
    if not isinstance(times, np.ndarray):
        raise TypeError(f"times must be ndarray, got {type(times)}")
    if len(times) == 0:
        raise ValueError("times array must not be empty")
    if not np.all(np.isfinite(times)):
        raise ValueError("times must contain only finite values")

    if not isinstance(amplitude, (int, float)):
        raise TypeError(f"amplitude must be numeric, got {type(amplitude)}")
    if not np.isfinite(amplitude):
        raise ValueError(f"amplitude must be finite, got {amplitude}")

    if not isinstance(t_start, (int, float)):
        raise TypeError(f"t_start must be numeric, got {type(t_start)}")
    if not isinstance(t_end, (int, float)):
        raise TypeError(f"t_end must be numeric, got {type(t_end)}")
    if not (np.isfinite(t_start) and np.isfinite(t_end)):
        raise ValueError("t_start and t_end must be finite")

    if rise_time < 0:
        raise ValueError(f"rise_time must be non-negative, got {rise_time}")


def _compute_square_envelope_smooth(
    times: np.ndarray,
    amplitude: float,
    t_start: float,
    t_end: float,
    rise_time: float,
) -> np.ndarray:
    """
    Compute square pulse with smooth rise/fall.

    Parameters
    ----------
    times : np.ndarray
        Time points
    amplitude : float
        Pulse amplitude
    t_start : float
        Start time
    t_end : float
        End time
    rise_time : float
        Rise/fall time

    Returns
    -------
    np.ndarray
        Pulse envelope
    """
    pulse = np.zeros_like(times)
    for i, t in enumerate(times):
        if t_start <= t <= t_start + rise_time:
            pulse[i] = amplitude * (t - t_start) / rise_time
        elif t_start + rise_time < t < t_end - rise_time:
            pulse[i] = amplitude
        elif t_end - rise_time <= t <= t_end:
            pulse[i] = amplitude * (t_end - t) / rise_time
    return pulse


def _compute_square_envelope_hard(
    times: np.ndarray,
    amplitude: float,
    t_start: float,
    t_end: float,
) -> np.ndarray:
    """
    Compute square pulse with hard edges.

    Parameters
    ----------
    times : np.ndarray
        Time points
    amplitude : float
        Pulse amplitude
    t_start : float
        Start time
    t_end : float
        End time

    Returns
    -------
    np.ndarray
        Pulse envelope
    """
    pulse = np.zeros_like(times)
    mask = (times >= t_start) & (times <= t_end)
    pulse[mask] = amplitude
    return pulse


def square_pulse(
    times: np.ndarray,
    amplitude: float,
    t_start: float,
    t_end: float,
    rise_time: float = 0.0,
) -> np.ndarray:
    """
    Generate a square (rectangular) pulse envelope.

    Square pulses provide constant driving amplitude for a specified duration.
    They are the fastest way to achieve a target rotation angle but have
    harsh spectral properties (sinc spectrum) that can excite unwanted transitions.
    Optional rise/fall times can smooth the transitions.

    Mathematical Form:
        Ω(t) = A  for t_start ≤ t ≤ t_end
             = 0  otherwise
    (or with smooth rise/fall if rise_time > 0)

    Parameters
    ----------
    times : np.ndarray
        Array of time points.
    amplitude : float
        Pulse amplitude during the 'on' period.
    t_start : float
        Start time of the pulse.
    t_end : float
        End time of the pulse.
    rise_time : float, optional
        Rise/fall time for smooth edges (default: 0.0 for hard edges).

    Returns
    -------
    np.ndarray
        Pulse amplitude array.
    """
    # Validate parameters
    _validate_square_pulse_params(times, amplitude, t_start, t_end, rise_time)

    # Compute envelope
    if rise_time > 0:
        pulse = _compute_square_envelope_smooth(
            times, amplitude, t_start, t_end, rise_time
        )
    else:
        pulse = _compute_square_envelope_hard(times, amplitude, t_start, t_end)

    # Postcondition assertions
    assert np.all(np.isfinite(pulse)), "Pulse contains non-finite values"
    assert pulse.shape == times.shape, f"Output shape mismatch"

    return pulse


def blackman_pulse(
    times: np.ndarray,
    amplitude: float,
    t_center: float,
    duration: float,
) -> np.ndarray:
    """
    Generate a Blackman window pulse envelope.

    Parameters
    ----------
    times : np.ndarray
        Array of time points.
    amplitude : float
        Peak pulse amplitude.
    t_center : float
        Center time of the pulse.
    duration : float
        Total duration of the pulse.

    Returns
    -------
    np.ndarray
        Pulse amplitude array.

    Examples
    --------
    >>> times = np.linspace(0, 100, 1000)
    >>> pulse = blackman_pulse(times, amplitude=2*np.pi*5, t_center=50, duration=60)
    """
    t_start = t_center - duration / 2
    t_end = t_center + duration / 2

    pulse = np.zeros_like(times)
    mask = (times >= t_start) & (times <= t_end)

    if not np.any(mask):
        return pulse

    t_masked = times[mask]
    x = (t_masked - t_start) / duration
    window = _compute_blackman_window(x)
    pulse[mask] = amplitude * window

    return pulse


def _compute_drag_i_component(
    times: np.ndarray,
    amplitude: float,
    t_center: float,
    sigma: float,
    truncation: float,
) -> np.ndarray:
    """
    Compute in-phase (I) component of DRAG pulse.

    Parameters
    ----------
    times : np.ndarray
        Time points
    amplitude : float
        Peak amplitude
    t_center : float
        Center time
    sigma : float
        Gaussian width
    truncation : float
        Truncation parameter

    Returns
    -------
    np.ndarray
        I component (Gaussian envelope)
    """
    omega_I = np.zeros_like(times)
    mask = np.abs(times - t_center) <= truncation * sigma
    t_masked = times[mask]
    gaussian = amplitude * np.exp(-((t_masked - t_center) ** 2) / (2 * sigma**2))
    omega_I[mask] = gaussian
    return omega_I


def _compute_drag_q_component(
    times: np.ndarray,
    amplitude: float,
    t_center: float,
    sigma: float,
    beta: float,
    truncation: float,
) -> np.ndarray:
    """
    Compute quadrature (Q) component of DRAG pulse.

    Parameters
    ----------
    times : np.ndarray
        Time points
    amplitude : float
        Peak amplitude
    t_center : float
        Center time
    sigma : float
        Gaussian width
    beta : float
        DRAG coefficient
    truncation : float
        Truncation parameter

    Returns
    -------
    np.ndarray
        Q component (derivative correction)
    """
    omega_Q = np.zeros_like(times)
    mask = np.abs(times - t_center) <= truncation * sigma
    t_masked = times[mask]
    gaussian = amplitude * np.exp(-((t_masked - t_center) ** 2) / (2 * sigma**2))
    derivative = (t_masked - t_center) / (sigma**2) * gaussian
    omega_Q[mask] = beta * derivative
    return omega_Q


def drag_pulse(
    times: np.ndarray,
    amplitude: float,
    t_center: float,
    sigma: float,
    beta: float,
    truncation: float = 4.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate DRAG pulse for leakage suppression in anharmonic qubits.

    DRAG adds quadrature component (derivative) to cancel |2⟩ leakage.
    Form: Ω_I(t) = A*exp(-(t-t_c)²/(2σ²)), Ω_Q(t) = -β*dΩ_I/dt

    Parameters
    ----------
    times : np.ndarray
        Time points.
    amplitude : float
        Peak I-component amplitude.
    t_center : float
        Pulse center time.
    sigma : float
        Pulse width (std deviation).
    beta : float
        DRAG coefficient (typical: 0.1-0.5).
    truncation : float, optional
        Truncation in units of σ. Default 4.0.

    Returns
    -------
    omega_I : np.ndarray
        In-phase Gaussian component.
    omega_Q : np.ndarray
        Quadrature derivative component.
    """
    # Compute I and Q components
    omega_I = _compute_drag_i_component(times, amplitude, t_center, sigma, truncation)
    omega_Q = _compute_drag_q_component(
        times, amplitude, t_center, sigma, beta, truncation
    )

    return omega_I, omega_Q


def _compute_blackman_window(x: np.ndarray) -> np.ndarray:
    """
    Compute Blackman window coefficients.

    Parameters
    ----------
    x : np.ndarray
        Normalized position in [0, 1]

    Returns
    -------
    np.ndarray
        Window values
    """
    a0 = 0.42
    a1 = 0.5
    a2 = 0.08
    return a0 - a1 * np.cos(2 * np.pi * x) + a2 * np.cos(4 * np.pi * x)


def blackman_pulse(
    times: np.ndarray,
    amplitude: float,
    t_start: float,
    t_end: float,
) -> np.ndarray:
    """
    Generate a Blackman window pulse envelope.

    The Blackman window has excellent spectral properties with minimal
    side-lobes, making it ideal for avoiding unwanted transitions.
    It's smoother than a Gaussian at the edges.

    Mathematical Form:
        Ω(t) = A * [0.42 - 0.5*cos(2π*x) + 0.08*cos(4π*x)]
    where x = (t - t_start) / (t_end - t_start) ∈ [0, 1]

    Parameters
    ----------
    times : np.ndarray
        Array of time points.
    amplitude : float
        Peak amplitude of the pulse.
    t_start : float
        Start time of the pulse.
    t_end : float
        End time of the pulse.

    Returns
    -------
    np.ndarray
        Pulse amplitude as a function of time.

    Examples
    --------
    >>> times = np.linspace(0, 100, 1000)
    >>> pulse = blackman_pulse(times, amplitude=2*np.pi*8, t_start=10, t_end=90)

    Notes
    -----
    The Blackman window has side-lobes <60 dB below the main lobe in the
    frequency domain, providing excellent spectral containment.
    """
    pulse = np.zeros_like(times)
    mask = (times >= t_start) & (times <= t_end)

    if not np.any(mask):
        return pulse

    t_masked = times[mask]
    duration = t_end - t_start
    x = (t_masked - t_start) / duration  # Normalize to [0, 1]

    # Compute Blackman window
    window = _compute_blackman_window(x)
    pulse[mask] = amplitude * window

    return pulse


def cosine_pulse(
    times: np.ndarray,
    amplitude: float,
    t_start: float,
    t_end: float,
) -> np.ndarray:
    """
    Generate a raised cosine (Hann) pulse envelope.

    A simple, smooth pulse shape that's easy to implement experimentally.
    The cosine envelope provides a good balance between speed and spectral
    purity.

    Mathematical Form:
        Ω(t) = A * sin²(π * (t - t_start) / (t_end - t_start))
             = A * 0.5 * (1 - cos(2π * (t - t_start) / (t_end - t_start)))

    Parameters
    ----------
    times : np.ndarray
        Array of time points.
    amplitude : float
        Peak amplitude of the pulse (at center).
    t_start : float
        Start time of the pulse.
    t_end : float
        End time of the pulse.

    Returns
    -------
    np.ndarray
        Pulse amplitude as a function of time.

    Examples
    --------
    >>> times = np.linspace(0, 100, 1000)
    >>> pulse = cosine_pulse(times, amplitude=2*np.pi*6, t_start=20, t_end=80)
    >>> # Smooth cosine envelope from 20 to 80 ns

    Notes
    -----
    Also known as a Hann window or sin² pulse. The envelope naturally goes
    to zero at both endpoints, ensuring no discontinuities.
    """
    pulse = np.zeros_like(times)
    mask = (times >= t_start) & (times <= t_end)

    if not np.any(mask):
        return pulse

    t_masked = times[mask]
    duration = t_end - t_start
    phase = np.pi * (t_masked - t_start) / duration

    pulse[mask] = amplitude * np.sin(phase) ** 2

    return pulse


def custom_pulse(
    times: np.ndarray,
    control_points: np.ndarray,
    control_times: np.ndarray,
    interpolation: str = "cubic",
) -> np.ndarray:
    """
    Generate custom pulse from control points via interpolation.

    Parameters: times (array), control_points (array), control_times (array),
    interpolation ('linear', 'cubic', or 'pchip', default='cubic').
    Returns: interpolated pulse array.
    """
    from scipy.interpolate import interp1d, PchipInterpolator

    if len(control_points) != len(control_times):
        raise ValueError("control_points and control_times must have same length")

    if interpolation == "pchip":
        # PCHIP preserves monotonicity and avoids overshoots
        interpolator = PchipInterpolator(control_times, control_points)
        pulse = interpolator(times)
    else:
        # Linear or cubic spline
        interpolator = interp1d(
            control_times,
            control_points,
            kind=interpolation,
            bounds_error=False,
            fill_value=0.0,
        )
        pulse = interpolator(times)

    return pulse


def pulse_area(times: np.ndarray, pulse: np.ndarray) -> float:
    """
    Calculate the integrated area under a pulse envelope.

    The pulse area determines the total rotation angle:
        θ = ∫ Ω(t) dt

    For a π-pulse: area = π
    For a π/2-pulse: area = π/2

    Parameters
    ----------
    times : np.ndarray
        Time points (must be uniformly spaced).
    pulse : np.ndarray
        Pulse amplitude at each time point.

    Returns
    -------
    float
        Integrated pulse area (units: rad if Ω is in rad/s).

    Examples
    --------
    >>> times = np.linspace(0, 100, 1000)
    >>> pulse = gaussian_pulse(times, amplitude=2*np.pi*5, t_center=50, sigma=10)
    >>> area = pulse_area(times, pulse)
    >>> print(f"Rotation angle: {area/np.pi:.3f}π")
    """
    return np.trapezoid(pulse, times)


def scale_pulse_to_target_angle(
    pulse: np.ndarray,
    times: np.ndarray,
    target_angle: float,
) -> np.ndarray:
    """
    Scale a pulse envelope to achieve a target rotation angle.

    Given an arbitrary pulse shape, compute the scaling factor needed
    to achieve a desired total rotation angle (pulse area).

    Parameters
    ----------
    pulse : np.ndarray
        Original pulse amplitude array.
    times : np.ndarray
        Time points corresponding to pulse.
    target_angle : float
        Desired rotation angle in radians (e.g., π for π-pulse).

    Returns
    -------
    np.ndarray
        Scaled pulse with correct area.

    Examples
    --------
    >>> times = np.linspace(0, 100, 1000)
    >>> pulse = gaussian_pulse(times, amplitude=1.0, t_center=50, sigma=10)
    >>> pi_pulse = scale_pulse_to_target_angle(pulse, times, np.pi)
    >>> # Now pi_pulse will drive a |0⟩ → |1⟩ transition

    Notes
    -----
    This is equivalent to:
        pulse_scaled = pulse * (target_angle / pulse_area(pulse))
    """
    current_area = pulse_area(times, pulse)
    if np.abs(current_area) < 1e-12:
        raise ValueError("Pulse area is zero; cannot scale")

    scaling_factor = target_angle / current_area
    return pulse * scaling_factor
