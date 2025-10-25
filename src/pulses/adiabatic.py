"""
Adiabatic pulse techniques for quantum control.

This module implements adiabatic passage methods including:
- STIRAP (Stimulated Raman Adiabatic Passage)
- Landau-Zener sweeps
- Adiabaticity criteria and validation
- Sweep optimization and robustness analysis

Author: Quantum Controls Team
License: MIT
"""

from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, List, Dict, Any
import numpy as np
from numpy.typing import NDArray
import qutip as qt
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar


@dataclass
class LandauZenerParameters:
    """
    Parameters for Landau-Zener sweep.

    Attributes:
        delta_initial: Initial detuning (rad/s)
        delta_final: Final detuning (rad/s)
        sweep_time: Total sweep duration (s)
        coupling: Coupling strength Ω (rad/s)
        sweep_function: Optional custom sweep profile ('linear', 'tanh', 'gaussian')
    """

    delta_initial: float
    delta_final: float
    sweep_time: float
    coupling: float
    sweep_function: str = "linear"

    def __post_init__(self):
        """Validate parameters."""
        if self.sweep_time <= 0:
            raise ValueError("sweep_time must be positive")
        if self.coupling <= 0:
            raise ValueError("coupling must be positive")
        if self.sweep_function not in ["linear", "tanh", "gaussian"]:
            raise ValueError(f"Unknown sweep_function: {self.sweep_function}")


@dataclass
class STIRAPParameters:
    """
    Parameters for STIRAP (Stimulated Raman Adiabatic Passage).

    Three-level system: |1⟩ → |3⟩ via intermediate |2⟩
    Uses counter-intuitive pulse ordering: Stokes before pump.

    Attributes:
        omega_pump: Peak Rabi frequency for pump pulse (|1⟩↔|2⟩, rad/s)
        omega_stokes: Peak Rabi frequency for Stokes pulse (|2⟩↔|3⟩, rad/s)
        pulse_duration: Total duration of each pulse (s)
        delay: Time delay between Stokes and pump peaks (s, negative = counter-intuitive)
        detuning: Two-photon detuning (rad/s)
        pulse_shape: Shape function ('gaussian', 'sech', 'sin_squared')
    """

    omega_pump: float
    omega_stokes: float
    pulse_duration: float
    delay: float = -1.0  # Counter-intuitive ordering (negative delay)
    detuning: float = 0.0
    pulse_shape: str = "gaussian"

    def __post_init__(self):
        """Validate parameters."""
        if self.omega_pump <= 0:
            raise ValueError("omega_pump must be positive")
        if self.omega_stokes <= 0:
            raise ValueError("omega_stokes must be positive")
        if self.pulse_duration <= 0:
            raise ValueError("pulse_duration must be positive")
        if self.pulse_shape not in ["gaussian", "sech", "sin_squared"]:
            raise ValueError(f"Unknown pulse_shape: {self.pulse_shape}")


@dataclass
class AdiabaticityMetrics:
    """
    Metrics characterizing adiabaticity of a pulse sequence.

    Attributes:
        min_adiabaticity: Minimum adiabaticity parameter γ(t) = |⟨n|dH/dt|m⟩| / (E_m - E_n)²
        max_diabatic_rate: Maximum diabatic transition rate
        adiabatic_times: Times where adiabaticity condition is satisfied
        violations: Times where adiabaticity is violated
        transition_probability: Estimated diabatic transition probability
        robustness_factor: Ratio of min energy gap to max transition rate
    """

    min_adiabaticity: float
    max_diabatic_rate: float
    adiabatic_times: NDArray[np.float64]
    violations: NDArray[np.float64]
    transition_probability: float
    robustness_factor: float


class LandauZenerSweep:
    """
    Landau-Zener sweep for two-level system.

    Models avoided crossing sweep with configurable sweep profiles.
    Computes transition probabilities and adiabaticity metrics.
    """

    def __init__(self, params: LandauZenerParameters):
        """
        Initialize Landau-Zener sweep.

        Args:
            params: Landau-Zener parameters
        """
        self.params = params
        self._sweep_rate = (
            params.delta_final - params.delta_initial
        ) / params.sweep_time

    def detuning(self, t: float) -> float:
        """
        Compute detuning Δ(t) at time t.

        Args:
            t: Time (s)

        Returns:
            Detuning (rad/s)
        """
        p = self.params

        if p.sweep_function == "linear":
            return p.delta_initial + self._sweep_rate * t

        elif p.sweep_function == "tanh":
            # Smooth S-curve
            x = 4 * (2 * t / p.sweep_time - 1)  # Map to [-4, 4]
            progress = (np.tanh(x) + 1) / 2  # Map to [0, 1]
            return p.delta_initial + (p.delta_final - p.delta_initial) * progress

        elif p.sweep_function == "gaussian":
            # Error function sweep (integral of Gaussian)
            from scipy.special import erf

            x = 4 * (2 * t / p.sweep_time - 1)
            progress = (erf(x) + 1) / 2
            return p.delta_initial + (p.delta_final - p.delta_initial) * progress

        return p.delta_initial

    def detuning_rate(self, t: float) -> float:
        """
        Compute dΔ/dt at time t.

        Args:
            t: Time (s)

        Returns:
            Sweep rate (rad/s²)
        """
        p = self.params

        if p.sweep_function == "linear":
            return self._sweep_rate

        elif p.sweep_function == "tanh":
            x = 4 * (2 / p.sweep_time) * (2 * t / p.sweep_time - 1)
            sech_sq = 1 / np.cosh(x) ** 2
            return (p.delta_final - p.delta_initial) * (4 / p.sweep_time) * sech_sq

        elif p.sweep_function == "gaussian":
            x = 4 * (2 * t / p.sweep_time - 1)
            gauss = np.exp(-(x**2)) * (8 / p.sweep_time) / np.sqrt(np.pi)
            return (p.delta_final - p.delta_initial) * gauss

        return 0.0

    def landau_zener_probability(self) -> float:
        """
        Compute diabatic transition probability using Landau-Zener formula.

        For linear sweep: P_LZ = exp(-π Ω² / (2 |dΔ/dt|))

        Returns:
            Transition probability (0 = fully adiabatic, 1 = fully diabatic)
        """
        omega = self.params.coupling

        if self.params.sweep_function == "linear":
            sweep_rate = abs(self._sweep_rate)
        else:
            # Use maximum sweep rate for nonlinear sweeps
            times = np.linspace(0, self.params.sweep_time, 1000)
            rates = np.array([abs(self.detuning_rate(t)) for t in times])
            sweep_rate = np.max(rates)

        if sweep_rate == 0:
            return 1.0

        # Landau-Zener formula
        exponent = -np.pi * omega**2 / (2 * sweep_rate)
        return np.exp(exponent)

    def energy_gap(self, t: float) -> float:
        """
        Compute instantaneous energy gap between dressed states.

        E_gap = √(Δ² + Ω²)

        Args:
            t: Time (s)

        Returns:
            Energy gap (rad/s)
        """
        delta = self.detuning(t)
        omega = self.params.coupling
        return np.sqrt(delta**2 + omega**2)

    def adiabaticity_parameter(self, t: float) -> float:
        """
        Compute adiabaticity parameter γ(t).

        γ = E_gap² / |dΔ/dt|

        Large γ ≫ 1 indicates adiabatic regime.

        Args:
            t: Time (s)

        Returns:
            Adiabaticity parameter (dimensionless)
        """
        gap = self.energy_gap(t)
        rate = abs(self.detuning_rate(t))

        if rate == 0:
            return np.inf

        return gap**2 / rate

    def check_adiabaticity(
        self, threshold: float = 10.0, n_points: int = 1000
    ) -> AdiabaticityMetrics:
        """
        Check adiabaticity condition over full sweep.

        Args:
            threshold: Minimum γ for adiabatic regime
            n_points: Number of time points to sample

        Returns:
            Adiabaticity metrics
        """
        times = np.linspace(0, self.params.sweep_time, n_points)
        gamma_values = np.array([self.adiabaticity_parameter(t) for t in times])

        # Handle infinite values near zero sweep rate
        finite_mask = np.isfinite(gamma_values)
        gamma_finite = gamma_values[finite_mask]

        min_gamma = np.min(gamma_finite) if len(gamma_finite) > 0 else 0.0

        # Find violations
        adiabatic_mask = gamma_values >= threshold
        adiabatic_times = times[adiabatic_mask]
        violation_times = times[~adiabatic_mask & finite_mask]

        # Estimate transition probability
        p_lz = self.landau_zener_probability()

        # Robustness factor: minimum gap / maximum transition rate
        gaps = np.array([self.energy_gap(t) for t in times])
        rates = np.array([abs(self.detuning_rate(t)) for t in times])
        rates_nonzero = rates[rates > 0]

        min_gap = np.min(gaps)
        max_rate = np.max(rates_nonzero) if len(rates_nonzero) > 0 else 1.0
        robustness = min_gap / max_rate if max_rate > 0 else np.inf

        return AdiabaticityMetrics(
            min_adiabaticity=min_gamma,
            max_diabatic_rate=max_rate,
            adiabatic_times=adiabatic_times,
            violations=violation_times,
            transition_probability=p_lz,
            robustness_factor=robustness,
        )

    def simulate(
        self, initial_state: Optional[qt.Qobj] = None, n_points: int = 100
    ) -> Tuple[NDArray[np.float64], List[qt.Qobj]]:
        """
        Simulate sweep dynamics using QuTiP.

        Args:
            initial_state: Initial state (default: ground state)
            n_points: Number of time points

        Returns:
            Tuple of (times, states)
        """
        if initial_state is None:
            initial_state = qt.basis(2, 0)

        times = np.linspace(0, self.params.sweep_time, n_points)

        # Build time-dependent Hamiltonian
        sigma_x = qt.sigmax()
        sigma_z = qt.sigmaz()

        # H(t) = Δ(t)/2 σ_z + Ω/2 σ_x
        H0 = 0.5 * self.params.coupling * sigma_x
        H1 = [0.5 * sigma_z, lambda t, args: self.detuning(t)]

        H = [H0, H1]

        # Solve
        result = qt.mesolve(H, initial_state, times, [], [])

        return times, result.states


class STIRAPulse:
    """
    STIRAP (Stimulated Raman Adiabatic Passage) for three-level systems.

    Implements counter-intuitive pulse sequence for robust population transfer
    from |1⟩ to |3⟩ via intermediate state |2⟩.
    """

    def __init__(self, params: STIRAPParameters):
        """
        Initialize STIRAP pulse.

        Args:
            params: STIRAP parameters
        """
        self.params = params

    def pulse_envelope(self, t: float, peak_time: float) -> float:
        """
        Compute pulse envelope at time t.

        Args:
            t: Time (s)
            peak_time: Time of pulse peak (s)

        Returns:
            Envelope amplitude (0 to 1)
        """
        p = self.params
        tau = p.pulse_duration / 4  # Width parameter

        if p.pulse_shape == "gaussian":
            return np.exp(-(((t - peak_time) / tau) ** 2))

        elif p.pulse_shape == "sech":
            return 1 / np.cosh((t - peak_time) / tau)

        elif p.pulse_shape == "sin_squared":
            # Sin² pulse with smooth turn-on/off
            if abs(t - peak_time) > p.pulse_duration / 2:
                return 0.0
            phase = np.pi * (t - peak_time) / p.pulse_duration + np.pi / 2
            return np.sin(phase) ** 2

        return 0.0

    def pump_amplitude(self, t: float) -> float:
        """
        Compute pump pulse Ω_p(t) (couples |1⟩↔|2⟩).

        Args:
            t: Time (s)

        Returns:
            Pump Rabi frequency (rad/s)
        """
        # Pump peaks at t = T/2 (for counter-intuitive ordering with negative delay)
        peak_time = self.params.pulse_duration / 2
        return self.params.omega_pump * self.pulse_envelope(t, peak_time)

    def stokes_amplitude(self, t: float) -> float:
        """
        Compute Stokes pulse Ω_s(t) (couples |2⟩↔|3⟩).

        Args:
            t: Time (s)

        Returns:
            Stokes Rabi frequency (rad/s)
        """
        # Stokes peaks earlier (counter-intuitive)
        peak_time = self.params.pulse_duration / 2 + self.params.delay
        return self.params.omega_stokes * self.pulse_envelope(t, peak_time)

    def mixing_angle(self, t: float) -> float:
        """
        Compute instantaneous mixing angle θ(t) of dark state.

        tan(θ) = Ω_p / Ω_s

        Dark state: |D(t)⟩ = cos(θ)|1⟩ - sin(θ)|3⟩

        Args:
            t: Time (s)

        Returns:
            Mixing angle (radians)
        """
        omega_p = self.pump_amplitude(t)
        omega_s = self.stokes_amplitude(t)

        if omega_s == 0 and omega_p == 0:
            return 0.0

        return np.arctan2(omega_p, omega_s)

    def dark_state(self, t: float) -> qt.Qobj:
        """
        Compute instantaneous dark state |D(t)⟩.

        |D(t)⟩ = cos(θ(t))|1⟩ - sin(θ(t))|3⟩

        Args:
            t: Time (s)

        Returns:
            Dark state as Qobj
        """
        theta = self.mixing_angle(t)

        # Three-level basis: |1⟩, |2⟩, |3⟩
        psi = np.zeros(3, dtype=complex)
        psi[0] = np.cos(theta)  # |1⟩
        psi[2] = -np.sin(theta)  # |3⟩

        return qt.Qobj(psi)

    def adiabaticity_parameter(self, t: float, epsilon: float = 1e-10) -> float:
        """
        Compute adiabaticity parameter for STIRAP.

        γ(t) = Ω_eff / |dθ/dt|
        where Ω_eff = √(Ω_p² + Ω_s²)

        Args:
            t: Time (s)
            epsilon: Small time step for numerical derivative

        Returns:
            Adiabaticity parameter
        """
        omega_p = self.pump_amplitude(t)
        omega_s = self.stokes_amplitude(t)
        omega_eff = np.sqrt(omega_p**2 + omega_s**2)

        # Numerical derivative of mixing angle
        theta_plus = self.mixing_angle(t + epsilon)
        theta_minus = self.mixing_angle(t - epsilon)
        dtheta_dt = (theta_plus - theta_minus) / (2 * epsilon)

        if abs(dtheta_dt) < 1e-15:
            return np.inf

        return omega_eff / abs(dtheta_dt)

    def check_adiabaticity(
        self, threshold: float = 10.0, n_points: int = 1000
    ) -> AdiabaticityMetrics:
        """
        Check adiabaticity over full STIRAP sequence.

        Args:
            threshold: Minimum γ for adiabatic regime
            n_points: Number of time points

        Returns:
            Adiabaticity metrics
        """
        times = np.linspace(0, self.params.pulse_duration, n_points)
        gamma_values = np.array([self.adiabaticity_parameter(t) for t in times])

        # Handle infinite values
        finite_mask = np.isfinite(gamma_values)
        gamma_finite = gamma_values[finite_mask]

        min_gamma = np.min(gamma_finite) if len(gamma_finite) > 0 else 0.0

        # Find violations
        adiabatic_mask = gamma_values >= threshold
        adiabatic_times = times[adiabatic_mask]
        violation_times = times[~adiabatic_mask & finite_mask]

        # Estimate transition probability from minimum γ
        # Rough estimate: P ∝ exp(-γ_min)
        p_transition = np.exp(-min_gamma) if min_gamma < 10 else 1e-4

        # Robustness: minimum effective Rabi frequency
        omega_eff = np.array(
            [
                np.sqrt(self.pump_amplitude(t) ** 2 + self.stokes_amplitude(t) ** 2)
                for t in times
            ]
        )

        # Max rate of angle change
        dtheta = np.diff([self.mixing_angle(t) for t in times])
        dt = times[1] - times[0]
        max_rate = np.max(np.abs(dtheta / dt)) if len(dtheta) > 0 else 1.0

        min_omega = np.min(omega_eff[omega_eff > 0]) if np.any(omega_eff > 0) else 0.0
        robustness = min_omega / max_rate if max_rate > 0 else 0.0

        return AdiabaticityMetrics(
            min_adiabaticity=min_gamma,
            max_diabatic_rate=max_rate,
            adiabatic_times=adiabatic_times,
            violations=violation_times,
            transition_probability=p_transition,
            robustness_factor=robustness,
        )

    def simulate(
        self,
        initial_state: Optional[qt.Qobj] = None,
        n_points: int = 100,
        include_loss: bool = False,
        loss_rate: float = 0.0,
    ) -> Tuple[NDArray[np.float64], List[qt.Qobj]]:
        """
        Simulate STIRAP dynamics.

        Args:
            initial_state: Initial state (default: |1⟩)
            n_points: Number of time points
            include_loss: Include spontaneous emission from |2⟩
            loss_rate: Decay rate γ from |2⟩ (rad/s)

        Returns:
            Tuple of (times, states)
        """
        if initial_state is None:
            initial_state = qt.basis(3, 0)  # |1⟩

        times = np.linspace(0, self.params.pulse_duration, n_points)

        # Build Hamiltonian for three-level Lambda system
        # States: |1⟩ = ground, |2⟩ = intermediate, |3⟩ = target

        # Coupling operators
        sigma_12 = qt.basis(3, 1) * qt.basis(3, 0).dag()  # |2⟩⟨1|
        sigma_23 = qt.basis(3, 2) * qt.basis(3, 1).dag()  # |3⟩⟨2|

        # Time-dependent coefficients
        def pump_coeff(t, args):
            return self.pump_amplitude(t)

        def stokes_coeff(t, args):
            return self.stokes_amplitude(t)

        # Hamiltonian: H = Ω_p(t)/2 (|2⟩⟨1| + h.c.) + Ω_s(t)/2 (|3⟩⟨2| + h.c.) + Δ|2⟩⟨2|
        H = [
            [0.5 * (sigma_12 + sigma_12.dag()), pump_coeff],
            [0.5 * (sigma_23 + sigma_23.dag()), stokes_coeff],
        ]

        # Add detuning if present
        if self.params.detuning != 0:
            sigma_22 = qt.basis(3, 1) * qt.basis(3, 1).dag()
            H.append(self.params.detuning * sigma_22)

        # Collapse operators for loss
        c_ops = []
        if include_loss and loss_rate > 0:
            # Spontaneous emission from |2⟩
            c_ops.append(np.sqrt(loss_rate) * qt.basis(3, 0) * qt.basis(3, 1).dag())

        # Solve
        result = qt.mesolve(H, initial_state, times, c_ops, [])

        return times, result.states

    def transfer_efficiency(
        self, initial_state: Optional[qt.Qobj] = None, n_points: int = 100
    ) -> float:
        """
        Compute population transfer efficiency from |1⟩ to |3⟩.

        Args:
            initial_state: Initial state (default: |1⟩)
            n_points: Number of time points

        Returns:
            Final population in |3⟩ (0 to 1)
        """
        times, states = self.simulate(initial_state, n_points)
        final_state = states[-1]

        # Project onto |3⟩
        target = qt.basis(3, 2)

        # Handle both ket and density matrix cases
        if final_state.isket:
            overlap = target.dag() * final_state
            # Inner product returns complex scalar, not Qobj
            if isinstance(overlap, qt.Qobj):
                population = abs(overlap.full()[0, 0]) ** 2
            else:
                population = abs(overlap) ** 2
        else:
            # Density matrix case: tr(ρ |3⟩⟨3|)
            proj = target * target.dag()
            population = abs((proj * final_state).tr())

        return population


class AdiabaticChecker:
    """
    General adiabaticity checker for time-dependent Hamiltonians.

    Analyzes instantaneous eigenstates and checks quantum adiabatic theorem criteria.
    """

    @staticmethod
    def instantaneous_eigensystem(
        H: qt.Qobj,
    ) -> Tuple[NDArray[np.float64], List[qt.Qobj]]:
        """
        Compute instantaneous eigenvalues and eigenstates.

        Args:
            H: Hamiltonian (time-independent snapshot)

        Returns:
            Tuple of (eigenvalues, eigenstates)
        """
        eigvals, eigvecs = H.eigenstates()
        return eigvals, eigvecs

    @staticmethod
    def _compute_hamiltonian_derivative(
        H_list: List[qt.Qobj], times: NDArray[np.float64], i: int
    ) -> qt.Qobj:
        """
        Compute numerical time derivative of Hamiltonian.

        Args:
            H_list: List of Hamiltonian snapshots
            times: Time array
            i: Current time index

        Returns:
            dH/dt at time i
        """
        dt_back = times[i] - times[i - 1]
        dt_fwd = times[i + 1] - times[i]
        return (H_list[i + 1] - H_list[i - 1]) / (dt_back + dt_fwd)

    @staticmethod
    def _compute_transition_metrics(
        eigvals: np.ndarray,
        eigvecs: List[qt.Qobj],
        dH_dt: qt.Qobj,
        state_index: int,
    ) -> tuple:
        """
        Compute transition rates and gaps for all states.

        Args:
            eigvals: Eigenvalues
            eigvecs: Eigenvectors
            dH_dt: Hamiltonian time derivative
            state_index: Index of state to track

        Returns:
            gaps, transition_rates, adiabaticity_params
        """
        gaps = []
        transition_rates = []
        adiabaticity_params = []
        n = state_index

        for m in range(len(eigvals)):
            if m == n:
                continue

            gap = abs(eigvals[m] - eigvals[n])
            if gap < 1e-10:
                continue

            # Compute matrix element: ⟨m|dH/dt|n⟩
            result = eigvecs[m].dag() * dH_dt * eigvecs[n]
            matrix_element = abs(result.tr()) if hasattr(result, "tr") else abs(result)

            transition_rate = matrix_element / gap

            if matrix_element > 1e-15:
                gamma = gap**2 / matrix_element
                adiabaticity_params.append(gamma)

            gaps.append(gap)
            transition_rates.append(transition_rate)

        return gaps, transition_rates, adiabaticity_params

    @staticmethod
    def _aggregate_adiabaticity_metrics(
        gaps: list, transition_rates: list, adiabaticity_params: list
    ) -> Dict[str, Any]:
        """
        Aggregate adiabaticity metrics into result dictionary.

        Args:
            gaps: Energy gaps
            transition_rates: Transition rates
            adiabaticity_params: Adiabaticity parameters

        Returns:
            Dictionary with metrics
        """
        return {
            "min_gap": np.min(gaps) if gaps else 0.0,
            "max_transition_rate": np.max(transition_rates)
            if transition_rates
            else 0.0,
            "min_adiabaticity": np.min(adiabaticity_params)
            if adiabaticity_params
            else 0.0,
            "adiabatic": np.min(adiabaticity_params) > 10.0
            if adiabaticity_params
            else False,
            "n_violations": sum(1 for g in adiabaticity_params if g < 10.0),
        }

    @staticmethod
    def adiabatic_condition(
        H_list: List[qt.Qobj], times: NDArray[np.float64], state_index: int = 0
    ) -> Dict[str, Any]:
        """
        Check adiabatic condition for time-dependent Hamiltonian.

        Condition: |⟨m|dH/dt|n⟩| ≪ (E_n - E_m)²

        Args:
            H_list: List of Hamiltonian snapshots at each time
            times: Corresponding times
            state_index: Index of state to track (0 = ground state)

        Returns:
            Dictionary with adiabaticity metrics
        """
        n_times = len(times)
        if n_times < 3:
            raise ValueError("Need at least 3 time points")

        all_gaps = []
        all_transition_rates = []
        all_adiabaticity_params = []

        for i in range(1, n_times - 1):
            # Current eigendecomposition
            eigvals, eigvecs = H_list[i].eigenstates()

            # Compute Hamiltonian derivative
            dH_dt = AdiabaticChecker._compute_hamiltonian_derivative(H_list, times, i)

            # Compute transition metrics
            gaps, rates, params = AdiabaticChecker._compute_transition_metrics(
                eigvals, eigvecs, dH_dt, state_index
            )

            all_gaps.extend(gaps)
            all_transition_rates.extend(rates)
            all_adiabaticity_params.extend(params)

        return AdiabaticChecker._aggregate_adiabaticity_metrics(
            all_gaps, all_transition_rates, all_adiabaticity_params
        )

    @staticmethod
    def optimize_sweep_time(
        sweep_builder: Callable[[float], Any],
        min_time: float = 1.0,
        max_time: float = 100.0,
        target_adiabaticity: float = 10.0,
    ) -> Dict[str, Any]:
        """
        Optimize sweep time to satisfy adiabaticity with minimum duration.

        Args:
            sweep_builder: Function that takes sweep_time and returns sweep object with
                          check_adiabaticity() method
            min_time: Minimum sweep time to consider
            max_time: Maximum sweep time to consider
            target_adiabaticity: Target minimum adiabaticity parameter

        Returns:
            Dictionary with optimized sweep time and metrics
        """

        def cost(T: float) -> float:
            """Cost function: penalize short times with low adiabaticity."""
            sweep = sweep_builder(T)
            metrics = sweep.check_adiabaticity(threshold=target_adiabaticity)

            # Want high adiabaticity and short time
            # Cost = -γ_min + α*T (minimizing gives high γ and low T)
            alpha = 0.1  # Weight for time penalty
            return -metrics.min_adiabaticity + alpha * T

        result = minimize_scalar(cost, bounds=(min_time, max_time), method="bounded")

        optimal_time = result.x
        sweep = sweep_builder(optimal_time)
        metrics = sweep.check_adiabaticity(threshold=target_adiabaticity)

        return {
            "optimal_time": optimal_time,
            "min_adiabaticity": metrics.min_adiabaticity,
            "transition_probability": metrics.transition_probability,
            "robustness_factor": metrics.robustness_factor,
            "success": result.success,
        }


def create_landau_zener_sweep(
    delta_range: Tuple[float, float],
    coupling: float,
    sweep_time: float,
    sweep_type: str = "linear",
) -> LandauZenerSweep:
    """
    Convenience function to create Landau-Zener sweep.

    Args:
        delta_range: (initial_detuning, final_detuning) in rad/s
        coupling: Coupling strength Ω in rad/s
        sweep_time: Sweep duration in seconds
        sweep_type: 'linear', 'tanh', or 'gaussian'

    Returns:
        LandauZenerSweep object
    """
    params = LandauZenerParameters(
        delta_initial=delta_range[0],
        delta_final=delta_range[1],
        sweep_time=sweep_time,
        coupling=coupling,
        sweep_function=sweep_type,
    )
    return LandauZenerSweep(params)


def create_stirap_pulse(
    omega_pump: float,
    omega_stokes: float,
    pulse_duration: float,
    delay: float = -1.0,
    pulse_shape: str = "gaussian",
) -> STIRAPulse:
    """
    Convenience function to create STIRAP pulse.

    Args:
        omega_pump: Peak pump Rabi frequency (rad/s)
        omega_stokes: Peak Stokes Rabi frequency (rad/s)
        pulse_duration: Total pulse duration (s)
        delay: Delay between pulses (negative = counter-intuitive)
        pulse_shape: 'gaussian', 'sech', or 'sin_squared'

    Returns:
        STIRAPulse object
    """
    params = STIRAPParameters(
        omega_pump=omega_pump,
        omega_stokes=omega_stokes,
        pulse_duration=pulse_duration,
        delay=delay,
        pulse_shape=pulse_shape,
    )
    return STIRAPulse(params)
