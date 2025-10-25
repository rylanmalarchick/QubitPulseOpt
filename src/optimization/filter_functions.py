"""
Filter Functions for Noise Spectroscopy and Analysis

This module implements filter function formalism for characterizing pulse
sensitivity to noise at different frequencies. Filter functions provide a
powerful framework for:
- Analyzing noise susceptibility of control sequences
- Optimizing pulses for specific noise environments
- Predicting infidelity from noise power spectral densities

Mathematical Framework:
- Filter function: F(ω) = |∫₀ᵀ y(t)e^(iωt) dt|²
- Noise infidelity: χ = (1/2π) ∫ F(ω)S(ω) dω
- Control modulation: y(t) = Ω(t) for amplitude noise, δ(t) for frequency noise

References:
- Green et al., PRL 109, 020501 (2012) - Filter function formalism
- Cywinski et al., PRB 77, 174509 (2008) - Dephasing noise
- Biercuk et al., Nature 458, 996 (2009) - Dynamical decoupling
"""

import numpy as np
import qutip as qt
from dataclasses import dataclass
from typing import Callable, Optional, Union, Dict, List, Tuple
from scipy.integrate import simpson, quad
from scipy.optimize import minimize
import warnings


@dataclass
class FilterFunctionResult:
    """Results from filter function analysis.

    Attributes:
        frequencies: Frequency array (rad/s)
        filter_function: Filter function F(ω) values
        pulse_duration: Total pulse duration
        noise_infidelity: Infidelity from noise (if noise PSD provided)
        noise_type: Type of noise analyzed
        metadata: Additional analysis metadata
    """

    frequencies: np.ndarray
    filter_function: np.ndarray
    pulse_duration: float
    noise_infidelity: Optional[float] = None
    noise_type: Optional[str] = None
    metadata: Optional[Dict] = None

    def __repr__(self):
        base = (
            f"FilterFunctionResult(duration={self.pulse_duration:.3e}s, "
            f"freq_range=[{self.frequencies[0]:.2e}, {self.frequencies[-1]:.2e}] rad/s)"
        )
        if self.noise_infidelity is not None:
            base += f"\n  Noise infidelity χ = {self.noise_infidelity:.4e} ({self.noise_type})"
        return base


class FilterFunctionCalculator:
    """Calculate filter functions for various control sequences.

    The filter function characterizes the frequency response of a quantum
    control sequence to noise. It quantifies how strongly noise at frequency ω
    affects the control fidelity.

    For a control pulse with modulation y(t), the filter function is:
        F(ω) = |∫₀ᵀ y(t) exp(iωt) dt|²

    Different noise types correspond to different modulation functions:
    - Amplitude noise: y(t) = Ω(t) (control amplitude)
    - Detuning noise: y(t) = 1 (constant, for always-on detuning)
    - Phase noise: y(t) = ∫₀ᵗ Ω(s) ds (accumulated phase)
    """

    def __init__(
        self, n_freq: int = 200, freq_range: Optional[Tuple[float, float]] = None
    ):
        """Initialize filter function calculator.

        Args:
            n_freq: Number of frequency points to evaluate
            freq_range: (min_freq, max_freq) in rad/s. If None, auto-determined
        """
        self.n_freq = n_freq
        self.freq_range = freq_range

    def compute_filter_function(
        self,
        times: np.ndarray,
        modulation: np.ndarray,
        noise_type: str = "amplitude",
        frequencies: Optional[np.ndarray] = None,
    ) -> FilterFunctionResult:
        """Compute filter function for a given control modulation.

        Args:
            times: Time array
            modulation: Control modulation y(t) at each time point
            noise_type: Type of noise ('amplitude', 'detuning', 'phase')
            frequencies: Frequency array to evaluate at (if None, auto-generated)

        Returns:
            FilterFunctionResult with F(ω) and metadata
        """
        # Validate inputs
        if len(times) != len(modulation):
            raise ValueError("times and modulation must have same length")
        if len(times) < 2:
            raise ValueError("Need at least 2 time points")

        pulse_duration = times[-1] - times[0]

        # Generate frequency array if not provided
        if frequencies is None:
            frequencies = self._generate_frequencies(times, pulse_duration)

        # Compute filter function via discrete Fourier transform
        filter_func = self._compute_ff_fft(times, modulation, frequencies)

        return FilterFunctionResult(
            frequencies=frequencies,
            filter_function=filter_func,
            pulse_duration=pulse_duration,
            noise_type=noise_type,
            metadata={
                "n_points": len(times),
                "dt": np.mean(np.diff(times)),
                "max_modulation": np.max(np.abs(modulation)),
            },
        )

    def compute_from_pulse(
        self,
        times: np.ndarray,
        amplitudes: np.ndarray,
        noise_type: str = "amplitude",
        frequencies: Optional[np.ndarray] = None,
    ) -> FilterFunctionResult:
        """Compute filter function from pulse amplitudes.

        Convenience method that extracts the appropriate modulation function
        from pulse amplitudes based on noise type.

        Args:
            times: Time array
            amplitudes: Pulse amplitude Ω(t) at each time
            noise_type: 'amplitude', 'detuning', or 'phase'
            frequencies: Frequency array (if None, auto-generated)

        Returns:
            FilterFunctionResult
        """
        # Extract modulation function based on noise type
        if noise_type == "amplitude":
            modulation = amplitudes
        elif noise_type == "detuning":
            # For detuning noise, modulation is constant (always-on coupling)
            modulation = np.ones_like(amplitudes)
        elif noise_type == "phase":
            # For phase noise, modulation is accumulated phase
            dt = np.diff(times)
            # Cumulative integration of Ω(t)
            modulation = np.concatenate([[0], np.cumsum(amplitudes[:-1] * dt)])
        else:
            raise ValueError(f"Unknown noise_type: {noise_type}")

        return self.compute_filter_function(times, modulation, noise_type, frequencies)

    def _generate_frequencies(self, times: np.ndarray, duration: float) -> np.ndarray:
        """Generate frequency array for filter function evaluation."""
        if self.freq_range is not None:
            return np.linspace(self.freq_range[0], self.freq_range[1], self.n_freq)

        # Auto-determine frequency range based on pulse characteristics
        dt = np.mean(np.diff(times))
        f_max = np.pi / dt  # Nyquist frequency
        f_min = 2 * np.pi / duration  # Fundamental frequency

        # Use logarithmic spacing for better coverage
        return np.logspace(np.log10(f_min), np.log10(f_max), self.n_freq)

    def _compute_ff_fft(
        self, times: np.ndarray, modulation: np.ndarray, frequencies: np.ndarray
    ) -> np.ndarray:
        """Compute filter function using discrete Fourier transform."""
        dt = times[1] - times[0]  # Assume uniform sampling
        filter_func = np.zeros(len(frequencies))

        for i, omega in enumerate(frequencies):
            # Compute Fourier transform: ∫ y(t) exp(iωt) dt
            integrand = modulation * np.exp(1j * omega * times)
            ft = simpson(integrand, x=times)
            # Filter function is |FT|²
            filter_func[i] = np.abs(ft) ** 2

        return filter_func


class NoisePSD:
    """Noise power spectral density models.

    Provides common noise PSD models for quantum systems:
    - White noise: S(ω) = S₀
    - 1/f noise: S(ω) = A/|ω|
    - 1/f^α noise: S(ω) = A/|ω|^α
    - Lorentzian: S(ω) = A / (1 + (ω/ω_c)²)
    - Ohmic bath: S(ω) = γ·ω (for ω > 0)
    """

    @staticmethod
    def white_noise(amplitude: float = 1.0) -> Callable[[np.ndarray], np.ndarray]:
        """White noise: S(ω) = S₀."""

        def psd(omega):
            return amplitude * np.ones_like(omega)

        return psd

    @staticmethod
    def one_over_f(
        amplitude: float = 1.0, alpha: float = 1.0
    ) -> Callable[[np.ndarray], np.ndarray]:
        """1/f^α noise: S(ω) = A/|ω|^α."""

        def psd(omega):
            # Avoid division by zero
            omega_safe = np.where(np.abs(omega) > 1e-10, np.abs(omega), 1e-10)
            return amplitude / omega_safe**alpha

        return psd

    @staticmethod
    def lorentzian(
        amplitude: float = 1.0, cutoff: float = 1.0
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Lorentzian noise: S(ω) = A / (1 + (ω/ω_c)²)."""

        def psd(omega):
            return amplitude / (1 + (omega / cutoff) ** 2)

        return psd

    @staticmethod
    def ohmic(gamma: float = 1.0) -> Callable[[np.ndarray], np.ndarray]:
        """Ohmic bath: S(ω) = γ·ω for ω > 0."""

        def psd(omega):
            return gamma * np.maximum(omega, 0)

        return psd

    @staticmethod
    def gaussian(
        amplitude: float = 1.0, center: float = 1.0, width: float = 0.1
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Gaussian noise peak: S(ω) = A·exp(-(ω-ω₀)²/2σ²)."""

        def psd(omega):
            return amplitude * np.exp(-((omega - center) ** 2) / (2 * width**2))

        return psd


class NoiseInfidelityCalculator:
    """Calculate noise-induced infidelity from filter functions and PSDs.

    The noise infidelity is:
        χ = (1/2π) ∫ F(ω) S(ω) dω

    This quantifies the reduction in fidelity due to noise with PSD S(ω)
    when using a control with filter function F(ω).
    """

    def __init__(self, ff_calculator: Optional[FilterFunctionCalculator] = None):
        """Initialize noise infidelity calculator.

        Args:
            ff_calculator: FilterFunctionCalculator instance (creates new if None)
        """
        self.ff_calc = ff_calculator or FilterFunctionCalculator()

    def compute_infidelity(
        self,
        ff_result: FilterFunctionResult,
        noise_psd: Callable[[np.ndarray], np.ndarray],
        integration_method: str = "trapz",
    ) -> float:
        """Compute noise-induced infidelity.

        Args:
            ff_result: FilterFunctionResult with F(ω)
            noise_psd: Function S(ω) returning noise PSD at given frequencies
            integration_method: 'trapz', 'simpson', or 'quad'

        Returns:
            Noise infidelity χ
        """
        frequencies = ff_result.frequencies
        filter_func = ff_result.filter_function

        # Evaluate noise PSD at frequencies
        psd_values = noise_psd(frequencies)

        # Integrand: F(ω) * S(ω)
        integrand = filter_func * psd_values

        # Integrate
        if integration_method == "trapz":
            integral = np.trapz(integrand, frequencies)
        elif integration_method == "simpson":
            integral = simpson(integrand, x=frequencies)
        elif integration_method == "quad":
            # Use scipy.quad for adaptive integration (requires interpolation)
            from scipy.interpolate import interp1d

            integrand_func = interp1d(
                frequencies, integrand, kind="cubic", fill_value="extrapolate"
            )
            integral, _ = quad(integrand_func, frequencies[0], frequencies[-1])
        else:
            raise ValueError(f"Unknown integration method: {integration_method}")

        # Normalize by 2π
        chi = integral / (2 * np.pi)

        return chi

    def compute_from_pulse(
        self,
        times: np.ndarray,
        amplitudes: np.ndarray,
        noise_psd: Callable[[np.ndarray], np.ndarray],
        noise_type: str = "amplitude",
    ) -> FilterFunctionResult:
        """Compute filter function and infidelity from pulse.

        Args:
            times: Time array
            amplitudes: Pulse amplitudes
            noise_psd: Noise PSD function
            noise_type: Type of noise

        Returns:
            FilterFunctionResult with infidelity computed
        """
        # Compute filter function
        ff_result = self.ff_calc.compute_from_pulse(times, amplitudes, noise_type)

        # Compute infidelity
        chi = self.compute_infidelity(ff_result, noise_psd)

        # Update result with infidelity
        ff_result.noise_infidelity = chi

        return ff_result

    def compare_pulses(
        self,
        pulses: Dict[str, Tuple[np.ndarray, np.ndarray]],
        noise_psd: Callable[[np.ndarray], np.ndarray],
        noise_type: str = "amplitude",
    ) -> Dict[str, FilterFunctionResult]:
        """Compare multiple pulses under the same noise.

        Args:
            pulses: Dict mapping pulse names to (times, amplitudes) tuples
            noise_psd: Noise PSD function
            noise_type: Type of noise

        Returns:
            Dict mapping pulse names to FilterFunctionResult
        """
        results = {}
        for name, (times, amplitudes) in pulses.items():
            results[name] = self.compute_from_pulse(
                times, amplitudes, noise_psd, noise_type
            )

        return results


class NoiseTailoredOptimizer:
    """Optimize pulses for specific noise environments.

    This optimizer modifies pulse parameters to minimize noise infidelity
    for a given noise PSD. It can be used to find pulses that are robust
    against specific noise sources.
    """

    def __init__(
        self,
        ff_calculator: Optional[FilterFunctionCalculator] = None,
        infidelity_calc: Optional[NoiseInfidelityCalculator] = None,
    ):
        """Initialize noise-tailored optimizer.

        Args:
            ff_calculator: FilterFunctionCalculator instance
            infidelity_calc: NoiseInfidelityCalculator instance
        """
        self.ff_calc = ff_calculator or FilterFunctionCalculator()
        self.infid_calc = infidelity_calc or NoiseInfidelityCalculator(self.ff_calc)

    def _create_objective_function(
        self, times: np.ndarray, noise_type: str, noise_psd: Callable
    ) -> Callable:
        """
        Create objective function for pulse optimization.

        Args:
            times: Time array
            noise_type: Type of noise
            noise_psd: Noise PSD function

        Returns:
            Objective function
        """

        def objective(amps):
            try:
                ff_result = self.ff_calc.compute_from_pulse(times, amps, noise_type)
                chi = self.infid_calc.compute_infidelity(ff_result, noise_psd)
                return chi
            except Exception as e:
                warnings.warn(f"Error in objective: {e}")
                return 1e10

        return objective

    def _create_scipy_constraints(self, times: np.ndarray, constraints: Dict) -> list:
        """
        Create scipy-compatible constraints.

        Args:
            times: Time array
            constraints: Constraint dictionary

        Returns:
            List of scipy constraint dictionaries
        """
        scipy_constraints = []
        if "area" in constraints:
            target_area = constraints["area"]
            scipy_constraints.append(
                {"type": "eq", "fun": lambda amps: np.trapz(amps, times) - target_area}
            )
        return scipy_constraints

    def optimize_pulse_shape(
        self,
        times: np.ndarray,
        initial_amplitudes: np.ndarray,
        noise_psd: Callable[[np.ndarray], np.ndarray],
        noise_type: str = "amplitude",
        constraints: Optional[Dict] = None,
        method: str = "L-BFGS-B",
        max_iter: int = 100,
    ) -> Dict:
        """Optimize pulse amplitudes to minimize noise infidelity.

        Args:
            times: Time array (fixed)
            initial_amplitudes: Initial pulse shape
            noise_psd: Target noise PSD
            noise_type: Type of noise to optimize against
            constraints: Dict with 'max_amplitude', 'area', etc.
            method: Optimization method for scipy.optimize.minimize
            max_iter: Maximum iterations

        Returns:
            Dict with 'amplitudes', 'infidelity', 'ff_result', 'success'
        """
        constraints = constraints or {}
        max_amp = constraints.get("max_amplitude", 10.0)

        # Create objective and constraints
        objective = self._create_objective_function(times, noise_type, noise_psd)
        bounds = [(-max_amp, max_amp) for _ in initial_amplitudes]
        scipy_constraints = self._create_scipy_constraints(times, constraints)

        # Optimize
        result = minimize(
            objective,
            initial_amplitudes,
            method=method,
            bounds=bounds,
            constraints=scipy_constraints if scipy_constraints else None,
            options={"maxiter": max_iter},
        )

        # Compute final filter function
        optimal_amps = result.x
        ff_result = self.infid_calc.compute_from_pulse(
            times, optimal_amps, noise_psd, noise_type
        )

        return {
            "amplitudes": optimal_amps,
            "infidelity": result.fun,
            "ff_result": ff_result,
            "success": result.success,
            "message": result.message,
            "iterations": result.nit,
        }

    def optimize_pulse_timing(
        self,
        pulse_shape: Callable[[np.ndarray], np.ndarray],
        duration: float,
        n_points: int,
        noise_psd: Callable[[np.ndarray], np.ndarray],
        noise_type: str = "amplitude",
    ) -> Dict:
        """Optimize pulse timing/duration for given shape and noise.

        Args:
            pulse_shape: Function that generates pulse from time array
            duration: Initial pulse duration
            n_points: Number of time points
            noise_psd: Noise PSD
            noise_type: Type of noise

        Returns:
            Dict with optimal duration and infidelity
        """

        def objective(duration_param):
            dur = duration_param[0]
            if dur <= 0:
                return 1e10

            times = np.linspace(0, dur, n_points)
            amps = pulse_shape(times)

            try:
                ff_result = self.ff_calc.compute_from_pulse(times, amps, noise_type)
                chi = self.infid_calc.compute_infidelity(ff_result, noise_psd)
                return chi
            except Exception:
                return 1e10

        result = minimize(
            objective,
            [duration],
            method="Nelder-Mead",
            bounds=[(duration * 0.1, duration * 10)],
        )

        optimal_duration = result.x[0]
        times = np.linspace(0, optimal_duration, n_points)
        amps = pulse_shape(times)
        ff_result = self.infid_calc.compute_from_pulse(
            times, amps, noise_psd, noise_type
        )

        return {
            "duration": optimal_duration,
            "times": times,
            "amplitudes": amps,
            "infidelity": result.fun,
            "ff_result": ff_result,
        }


def _plot_filter_function_data(
    ax: object,
    frequencies: np.ndarray,
    ff: np.ndarray,
    log_scale: bool,
) -> None:
    """
    Plot filter function data on axes.

    Args:
        ax: Matplotlib axes
        frequencies: Frequency array
        ff: Filter function values
        log_scale: Use log-log scale
    """
    freq_hz = frequencies / (2 * np.pi)
    if log_scale:
        ax.loglog(freq_hz, ff, "b-", linewidth=2, label="Filter Function F(ω)")
    else:
        ax.plot(freq_hz, ff, "b-", linewidth=2, label="Filter Function F(ω)")


def _plot_noise_psd_overlay(
    ax: object,
    frequencies: np.ndarray,
    noise_psd: Callable[[np.ndarray], np.ndarray],
    log_scale: bool,
) -> object:
    """
    Plot noise PSD on secondary y-axis.

    Args:
        ax: Primary matplotlib axes
        frequencies: Frequency array
        noise_psd: Noise PSD function
        log_scale: Use log-log scale

    Returns:
        Secondary axes object
    """
    psd_values = noise_psd(frequencies)
    ax2 = ax.twinx()
    freq_hz = frequencies / (2 * np.pi)

    if log_scale:
        ax2.loglog(
            freq_hz, psd_values, "r--", linewidth=2, alpha=0.7, label="Noise PSD S(ω)"
        )
    else:
        ax2.plot(
            freq_hz, psd_values, "r--", linewidth=2, alpha=0.7, label="Noise PSD S(ω)"
        )

    ax2.set_ylabel("Noise PSD S(ω)", color="r")
    ax2.tick_params(axis="y", labelcolor="r")
    ax2.legend(loc="upper right")
    return ax2


def _configure_filter_function_plot(
    ax: object,
    ff_result: FilterFunctionResult,
) -> None:
    """
    Configure axes labels and formatting.

    Args:
        ax: Matplotlib axes
        ff_result: Filter function result
    """
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Filter Function F(ω)")
    ax.set_title(
        f"Filter Function Analysis\n({ff_result.noise_type} noise, T={ff_result.pulse_duration:.2e}s)"
    )
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)


def visualize_filter_function(
    ff_result: FilterFunctionResult,
    noise_psd: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    log_scale: bool = True,
    ax: Optional[object] = None,
) -> object:
    """Visualize filter function and optionally noise PSD.

    Args:
        ff_result: FilterFunctionResult to plot
        noise_psd: Optional noise PSD to overlay
        log_scale: Use log-log scale
        ax: Matplotlib axes (creates new if None)

    Returns:
        Matplotlib axes object
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for visualization")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    frequencies = ff_result.frequencies
    ff = ff_result.filter_function

    # Plot filter function
    _plot_filter_function_data(ax, frequencies, ff, log_scale)

    # Optionally overlay noise PSD
    if noise_psd is not None:
        _plot_noise_psd_overlay(ax, frequencies, noise_psd, log_scale)

    # Configure plot
    _configure_filter_function_plot(ax, ff_result)

    return ax


def compute_filter_function_sum_rule(ff_result: FilterFunctionResult) -> float:
    """Compute filter function sum rule integral.

    The sum rule states:
        ∫₀^∞ F(ω) dω = T·∫₀ᵀ |y(t)|² dt

    This is useful for validating filter function calculations.

    Args:
        ff_result: FilterFunctionResult

    Returns:
        Integral value ∫ F(ω) dω
    """
    return simpson(ff_result.filter_function, x=ff_result.frequencies)


# Convenience functions
def analyze_pulse_noise_sensitivity(
    times: np.ndarray,
    amplitudes: np.ndarray,
    noise_models: Optional[Dict[str, Callable]] = None,
    noise_type: str = "amplitude",
) -> Dict[str, FilterFunctionResult]:
    """Analyze pulse sensitivity to multiple noise models.

    Convenience function for quick analysis of a pulse under various noise.

    Args:
        times: Time array
        amplitudes: Pulse amplitudes
        noise_models: Dict of noise name -> PSD function (uses defaults if None)
        noise_type: Type of noise

    Returns:
        Dict of noise name -> FilterFunctionResult with infidelity
    """
    if noise_models is None:
        noise_models = {
            "white": NoisePSD.white_noise(1.0),
            "1/f": NoisePSD.one_over_f(1.0, 1.0),
            "lorentzian": NoisePSD.lorentzian(1.0, 1.0),
        }

    infid_calc = NoiseInfidelityCalculator()
    results = {}

    for name, psd in noise_models.items():
        results[name] = infid_calc.compute_from_pulse(
            times, amplitudes, psd, noise_type
        )

    return results
