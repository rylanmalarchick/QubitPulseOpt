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


def gaussian_pulse(
    times: np.ndarray,
    amplitude: float,
    t_center: float,
    sigma: float,
    truncation: float = 4.0,
) -> np.ndarray:
    """
    Generate a Gaussian pulse envelope.

    The Gaussian pulse is one of the most commonly used shapes in quantum control
    due to its smooth rise and fall, which minimizes spectral leakage and
    unwanted transitions. The pulse is typically truncated at ±4σ to ensure
    it decays to near-zero at the boundaries.

    Mathematical Form:
        Ω(t) = A * exp(-(t - t_c)² / (2σ²))  for |t - t_c| ≤ truncation*σ
             = 0                              otherwise

    Parameters
    ----------
    times : np.ndarray
        Array of time points at which to evaluate the pulse.
    amplitude : float
        Peak amplitude of the pulse (maximum Rabi frequency).
        Units: angular frequency (rad/s or 2π × Hz).
    t_center : float
        Center time of the pulse (peak location).
    sigma : float
        Standard deviation of the Gaussian (controls pulse width).
        A larger σ gives a slower, wider pulse.
    truncation : float, optional
        Number of standard deviations at which to truncate the pulse.
        Default is 4.0 (pulse decays to ~0.034% of peak at edges).

    Returns
    -------
    np.ndarray
        Pulse amplitude as a function of time, same shape as `times`.

    Examples
    --------
    >>> import numpy as np
    >>> times = np.linspace(0, 100, 1000)  # 100 ns duration
    >>> pulse = gaussian_pulse(times, amplitude=2*np.pi*10, t_center=50, sigma=10)
    >>> # Peak at t=50 ns, width ~40 ns (4σ), peak Rabi frequency = 10 MHz

    Notes
    -----
    For a π-pulse (|0⟩ → |1⟩ transition), the integrated pulse area must satisfy:
        ∫ Ω(t) dt = π
    For a Gaussian pulse, this gives:
        A * σ * √(2π) ≈ π  →  A ≈ √(π/(2σ²))
    """
    pulse = np.zeros_like(times)
    mask = np.abs(times - t_center) <= truncation * sigma
    pulse[mask] = amplitude * np.exp(-((times[mask] - t_center) ** 2) / (2 * sigma**2))
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
        Array of time points at which to evaluate the pulse.
    amplitude : float
        Constant amplitude during the pulse (Rabi frequency).
    t_start : float
        Start time of the pulse.
    t_end : float
        End time of the pulse.
    rise_time : float, optional
        Duration of smooth rise and fall edges (cosine smoothing).
        If 0 (default), pulse has sharp on/off transitions.
        If > 0, edges are smoothed over this duration.

    Returns
    -------
    np.ndarray
        Pulse amplitude as a function of time.

    Examples
    --------
    >>> times = np.linspace(0, 100, 1000)
    >>> pulse = square_pulse(times, amplitude=2*np.pi*5, t_start=20, t_end=80)
    >>> # 60 ns flat-top pulse at 5 MHz Rabi frequency

    Notes
    -----
    For a π-pulse with square envelope:
        A * (t_end - t_start) = π  →  duration = π / A
    """
    pulse = np.zeros_like(times)

    if rise_time == 0:
        # Sharp edges
        mask = (times >= t_start) & (times <= t_end)
        pulse[mask] = amplitude
    else:
        # Smooth rise and fall with cosine edges
        t_rise_end = t_start + rise_time
        t_fall_start = t_end - rise_time

        # Rise edge
        rise_mask = (times >= t_start) & (times < t_rise_end)
        if np.any(rise_mask):
            phase = (times[rise_mask] - t_start) / rise_time
            pulse[rise_mask] = amplitude * 0.5 * (1 - np.cos(np.pi * phase))

        # Flat top
        flat_mask = (times >= t_rise_end) & (times <= t_fall_start)
        pulse[flat_mask] = amplitude

        # Fall edge
        fall_mask = (times > t_fall_start) & (times <= t_end)
        if np.any(fall_mask):
            phase = (times[fall_mask] - t_fall_start) / rise_time
            pulse[fall_mask] = amplitude * 0.5 * (1 + np.cos(np.pi * phase))

    return pulse


def drag_pulse(
    times: np.ndarray,
    amplitude: float,
    t_center: float,
    sigma: float,
    beta: float,
    truncation: float = 4.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a DRAG (Derivative Removal by Adiabatic Gate) pulse.

    DRAG pulses correct for leakage to non-computational states in weakly
    anharmonic qubits (e.g., transmons). The technique adds a quadrature
    component (derivative of the in-phase pulse) to cancel leakage to the
    |2⟩ state during fast gates.

    Mathematical Form:
        Ω_I(t) = A * exp(-(t - t_c)² / (2σ²))
        Ω_Q(t) = -β * dΩ_I/dt = β * A * (t - t_c)/σ² * exp(-(t - t_c)² / (2σ²))

    The total control Hamiltonian becomes:
        H_c(t) = Ω_I(t) σ_x + Ω_Q(t) σ_y

    Parameters
    ----------
    times : np.ndarray
        Array of time points.
    amplitude : float
        Peak amplitude of the in-phase (I) component.
    t_center : float
        Center time of the pulse.
    sigma : float
        Standard deviation (pulse width).
    beta : float
        DRAG coefficient (dimensionless).
        Optimal value: β ≈ -α / (2Ω_max) where α is the anharmonicity.
        Typical range: 0.1 to 0.5 for superconducting qubits.
    truncation : float, optional
        Truncation parameter (number of σ). Default 4.0.

    Returns
    -------
    omega_I : np.ndarray
        In-phase (I) component (Gaussian envelope).
    omega_Q : np.ndarray
        Quadrature (Q) component (derivative correction).

    Examples
    --------
    >>> times = np.linspace(0, 100, 1000)
    >>> omega_I, omega_Q = drag_pulse(times, amplitude=2*np.pi*10, t_center=50,
    ...                                sigma=10, beta=0.3)
    >>> # Apply as: H_c(t) = omega_I(t)*σ_x + omega_Q(t)*σ_y

    References
    ----------
    Motzoi, F. et al., "Simple pulses for elimination of leakage in weakly
    nonlinear qubits," Physical Review Letters 103, 110501 (2009).

    Notes
    -----
    The DRAG correction is a first-order perturbative solution. For very fast
    gates or strong driving, higher-order corrections may be needed.
    """
    # In-phase component (standard Gaussian)
    omega_I = np.zeros_like(times)
    mask = np.abs(times - t_center) <= truncation * sigma
    t_masked = times[mask]
    gaussian = amplitude * np.exp(-((t_masked - t_center) ** 2) / (2 * sigma**2))
    omega_I[mask] = gaussian

    # Quadrature component (derivative of Gaussian)
    omega_Q = np.zeros_like(times)
    derivative = (t_masked - t_center) / (sigma**2) * gaussian
    omega_Q[mask] = beta * derivative

    return omega_I, omega_Q


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

    # Blackman window coefficients
    a0 = 0.42
    a1 = 0.5
    a2 = 0.08

    window = a0 - a1 * np.cos(2 * np.pi * x) + a2 * np.cos(4 * np.pi * x)
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
    Generate a custom pulse from control points using interpolation.

    This function allows arbitrary pulse shapes defined by a set of
    control points. Useful for implementing optimized pulses from
    GRAPE or other optimal control algorithms.

    Parameters
    ----------
    times : np.ndarray
        Array of time points at which to evaluate the pulse.
    control_points : np.ndarray
        Amplitude values at control times.
    control_times : np.ndarray
        Time points corresponding to control_points.
        Must be same length as control_points.
    interpolation : str, optional
        Interpolation method: 'linear', 'cubic', or 'pchip'.
        Default is 'cubic' (smooth, no oscillations).

    Returns
    -------
    np.ndarray
        Interpolated pulse amplitude at all time points.

    Examples
    --------
    >>> times = np.linspace(0, 100, 1000)
    >>> control_times = np.linspace(0, 100, 20)
    >>> control_points = np.random.randn(20) * 2*np.pi
    >>> pulse = custom_pulse(times, control_points, control_times, 'cubic')

    Notes
    -----
    For optimal control implementations (GRAPE, Krotov), this function can
    reconstruct smooth pulses from piecewise-constant or discrete control
    vectors.
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
