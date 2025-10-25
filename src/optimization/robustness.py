"""
Robustness Testing for Quantum Control
=======================================

This module provides utilities for testing the robustness of quantum control
pulses against various sources of uncertainty and noise:

1. Parameter detuning (frequency errors, amplitude miscalibration)
2. Amplitude noise (Gaussian, 1/f noise, systematic drifts)
3. Phase noise (dephasing, clock jitter)
4. Control parameter sweeps (optimization validation)

Robustness is critical for real quantum hardware where:
- Qubit frequencies drift due to temperature, flux noise
- Control amplitudes have systematic calibration errors
- Environmental noise couples to control lines
- Hardware imperfections introduce phase errors

Mathematical Framework:
----------------------
For a nominal control u₀(t) achieving fidelity F₀, we test:
    F(u₀ + δu, H₀ + δH) ≈ F₀ - sensitivity × δ

where:
- δu: control amplitude error
- δH: Hamiltonian parameter error
- sensitivity: ∂F/∂parameter (computed numerically)

Metrics:
--------
1. **Average Fidelity**: Mean over parameter distribution
2. **Worst-Case Fidelity**: Minimum over parameter range
3. **Fidelity Variance**: Spread of fidelity distribution
4. **Robustness Radius**: Maximum δ for F > threshold

References:
-----------
- Motzoi et al., Phys. Rev. Lett. 103, 110501 (2009)
- Tripathi et al., Phys. Rev. A 100, 012301 (2019) - Robustness metrics
- Egger & Wilhelm, Phys. Rev. Applied 11, 014017 (2019)
- Green et al., Phys. Rev. Lett. 114, 120502 (2015) - Composite pulses

Author: Orchestrator Agent
Date: 2025-01-27
SOW Reference: Phase 2.4 - Robustness Testing
"""

import numpy as np
import qutip as qt
from typing import Union, Callable, Optional, List, Tuple, Dict
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
import warnings


@dataclass
class RobustnessResult:
    """
    Result container for robustness analysis.

    Attributes
    ----------
    parameter_values : np.ndarray
        Values of the swept parameter.
    fidelities : np.ndarray
        Fidelity at each parameter value.
    mean_fidelity : float
        Average fidelity over parameter range.
    std_fidelity : float
        Standard deviation of fidelity.
    min_fidelity : float
        Worst-case fidelity.
    nominal_fidelity : float
        Fidelity at nominal (zero error) parameter.
    robustness_radius : float
        Maximum parameter deviation for F > threshold (if computed).
    parameter_name : str
        Name of the parameter being swept.
    """

    parameter_values: np.ndarray
    fidelities: np.ndarray
    mean_fidelity: float
    std_fidelity: float
    min_fidelity: float
    nominal_fidelity: float
    robustness_radius: Optional[float]
    parameter_name: str


class RobustnessTester:
    """
    Robustness testing for quantum control pulses.

    This class provides methods to test pulse robustness against various
    error sources: detuning, amplitude errors, noise, etc.

    Parameters
    ----------
    H_drift : qutip.Qobj
        Drift Hamiltonian.
    H_controls : list of qutip.Qobj
        Control Hamiltonians.
    pulse_amplitudes : np.ndarray
        Nominal pulse amplitudes, shape (n_controls, n_timeslices).
    total_time : float
        Total pulse duration.
    U_target : qutip.Qobj, optional
        Target unitary (for unitary gate fidelity).
    psi_init : qutip.Qobj, optional
        Initial state (for state transfer fidelity).
    psi_target : qutip.Qobj, optional
        Target state (for state transfer fidelity).

    Examples
    --------
    >>> # Setup nominal pulse
    >>> H0 = 0.5 * 5.0 * qt.sigmaz()
    >>> Hc = [qt.sigmax()]
    >>> pulse = np.ones((1, 100)) * 0.1
    >>>
    >>> # Create tester
    >>> tester = RobustnessTester(H0, Hc, pulse, total_time=50,
    ...                           U_target=qt.sigmax())
    >>>
    >>> # Test detuning robustness
    >>> result = tester.sweep_detuning(
    ...     detuning_range=np.linspace(-1, 1, 21)
    ... )
    >>> print(f"Mean fidelity: {result.mean_fidelity:.4f}")
    """

    def __init__(
        self,
        H_drift: qt.Qobj,
        H_controls: List[qt.Qobj],
        pulse_amplitudes: np.ndarray,
        total_time: float,
        U_target: Optional[qt.Qobj] = None,
        psi_init: Optional[qt.Qobj] = None,
        psi_target: Optional[qt.Qobj] = None,
    ):
        """Initialize robustness tester."""
        self.H_drift = H_drift
        self.H_controls = H_controls
        self.pulse_amplitudes = pulse_amplitudes
        self.total_time = total_time
        self.U_target = U_target
        self.psi_init = psi_init
        self.psi_target = psi_target

        self.n_controls = len(H_controls)
        self.n_timeslices = pulse_amplitudes.shape[1]
        self.dt = total_time / self.n_timeslices
        self.dim = H_drift.shape[0]

        # Determine fidelity type
        if U_target is not None:
            self.fidelity_type = "unitary"
        elif psi_init is not None and psi_target is not None:
            self.fidelity_type = "state"
        else:
            raise ValueError(
                "Must provide either U_target (for unitary fidelity) "
                "or (psi_init, psi_target) for state fidelity"
            )

    def _build_time_dependent_hamiltonian(
        self, H_drift: qt.Qobj, pulse_amplitudes: np.ndarray
    ) -> list:
        """
        Build time-dependent Hamiltonian list for QuTiP.

        Parameters
        ----------
        H_drift : qt.Qobj
            Drift Hamiltonian.
        pulse_amplitudes : np.ndarray
            Control pulse amplitudes.

        Returns
        -------
        list
            QuTiP Hamiltonian list format.
        """
        H_list = [H_drift]
        for j in range(self.n_controls):

            def coeff_func(t, args=None, ctrl_idx=j, pulses=pulse_amplitudes):
                idx = int(np.clip(t / self.dt, 0, self.n_timeslices - 1))
                return pulses[ctrl_idx, idx]

            H_list.append([self.H_controls[j], coeff_func])
        return H_list

    def _compute_unitary_fidelity(self, H: list, times: np.ndarray) -> float:
        """
        Compute unitary gate fidelity.

        Parameters
        ----------
        H : list
            Time-dependent Hamiltonian.
        times : np.ndarray
            Time points.

        Returns
        -------
        float
            Unitary fidelity.
        """
        psi0 = qt.basis(self.dim, 0)
        result = qt.sesolve(H, psi0, times)
        U_evolved_col0 = result.states[-1]

        if self.dim == 2:
            psi1 = qt.basis(self.dim, 1)
            result = qt.sesolve(H, psi1, times)
            U_evolved_col1 = result.states[-1]

            U_evolved = qt.Qobj(
                np.column_stack([U_evolved_col0.full(), U_evolved_col1.full()])
            )

            overlap = (self.U_target.dag() * U_evolved).tr()
            fidelity = np.abs(overlap) ** 2 / self.dim**2
            return np.real(fidelity)
        else:
            warnings.warn("Unitary fidelity for dim > 2 uses approximation")
            return qt.fidelity(U_evolved_col0, self.U_target * psi0) ** 2

    def _compute_state_fidelity(self, H: list, times: np.ndarray) -> float:
        """
        Compute state transfer fidelity.

        Parameters
        ----------
        H : list
            Time-dependent Hamiltonian.
        times : np.ndarray
            Time points.

        Returns
        -------
        float
            State fidelity.
        """
        result = qt.sesolve(H, self.psi_init, times)
        psi_final = result.states[-1]
        return qt.fidelity(psi_final, self.psi_target) ** 2

    def _compute_fidelity(
        self,
        H_drift: qt.Qobj,
        pulse_amplitudes: np.ndarray,
    ) -> float:
        """
        Compute fidelity for given Hamiltonian and pulse.

        Parameters
        ----------
        H_drift : qutip.Qobj
            Modified drift Hamiltonian (with errors).
        pulse_amplitudes : np.ndarray
            Control pulse amplitudes (possibly with noise).

        Returns
        -------
        float
            Fidelity.
        """
        H = self._build_time_dependent_hamiltonian(H_drift, pulse_amplitudes)
        times = np.linspace(0, self.total_time, self.n_timeslices + 1)

        if self.fidelity_type == "unitary":
            return self._compute_unitary_fidelity(H, times)
        else:
            return self._compute_state_fidelity(H, times)

    def _sweep_detuning_range(self, detuning_range: np.ndarray) -> tuple:
        """
        Sweep over detuning range and compute fidelities.

        Parameters
        ----------
        detuning_range : np.ndarray
            Array of detuning values

        Returns
        -------
        fidelities : np.ndarray
            Fidelities for each detuning
        nominal_fidelity : float or None
            Fidelity at zero detuning
        """
        fidelities = []
        nominal_fidelity = None

        for detuning in detuning_range:
            # Add detuning to drift Hamiltonian: δH = (δω/2) σ_z
            H_drift_detuned = self.H_drift + 0.5 * detuning * qt.sigmaz()

            # Compute fidelity
            fid = self._compute_fidelity(H_drift_detuned, self.pulse_amplitudes)
            fidelities.append(fid)

            # Store nominal fidelity (at zero detuning)
            if np.abs(detuning) < 1e-10:
                nominal_fidelity = fid

        return np.array(fidelities), nominal_fidelity

    def _compute_robustness_radius(
        self,
        parameter_range: np.ndarray,
        fidelities: np.ndarray,
        fidelity_threshold: float,
    ) -> Optional[float]:
        """
        Compute robustness radius from sweep results.

        Parameters
        ----------
        parameter_range : np.ndarray
            Parameter values
        fidelities : np.ndarray
            Fidelities at each parameter value
        fidelity_threshold : float
            Threshold fidelity

        Returns
        -------
        float or None
            Maximum parameter magnitude where F > threshold
        """
        valid_indices = np.where(fidelities >= fidelity_threshold)[0]
        if len(valid_indices) > 0:
            valid_params = np.abs(parameter_range[valid_indices])
            return np.max(valid_params)
        return None

    def sweep_detuning(
        self,
        detuning_range: np.ndarray,
        fidelity_threshold: float = 0.99,
    ) -> RobustnessResult:
        """
        Test robustness against frequency detuning errors.

        Detuning errors arise from:
        - Qubit frequency drift (flux noise, temperature)
        - Drive frequency miscalibration
        - Stark shifts from nearby qubits

        Parameters
        ----------
        detuning_range : np.ndarray
            Array of detuning values to test (relative to nominal).
            Units: angular frequency (rad/s or 2π×Hz).
        fidelity_threshold : float, optional
            Threshold for computing robustness radius. Default: 0.99.

        Returns
        -------
        RobustnessResult
            Robustness analysis results.

        Examples
        --------
        >>> detunings = np.linspace(-0.5, 0.5, 31)  # ±0.5 MHz
        >>> result = tester.sweep_detuning(detunings)
        >>> plt.plot(result.parameter_values, result.fidelities)
        >>> plt.xlabel('Detuning (MHz)')
        >>> plt.ylabel('Fidelity')
        """
        # Sweep detuning range
        fidelities, nominal_fidelity = self._sweep_detuning_range(detuning_range)

        # Compute robustness radius
        robustness_radius = None
        if nominal_fidelity is not None:
            robustness_radius = self._compute_robustness_radius(
                detuning_range, fidelities, fidelity_threshold
            )

        return RobustnessResult(
            parameter_values=detuning_range,
            fidelities=fidelities,
            mean_fidelity=np.mean(fidelities),
            std_fidelity=np.std(fidelities),
            min_fidelity=np.min(fidelities),
            nominal_fidelity=nominal_fidelity or fidelities[len(fidelities) // 2],
            robustness_radius=robustness_radius,
            parameter_name="detuning",
        )

    def _sweep_amplitude_error_range(self, amplitude_error_range: np.ndarray) -> tuple:
        """
        Sweep over amplitude error range and compute fidelities.

        Parameters
        ----------
        amplitude_error_range : np.ndarray
            Array of amplitude errors

        Returns
        -------
        fidelities : np.ndarray
            Fidelities for each error
        nominal_fidelity : float or None
            Fidelity at zero error
        """
        fidelities = []
        nominal_fidelity = None

        for error in amplitude_error_range:
            # Scale pulse amplitudes: u → u * (1 + ε)
            pulse_scaled = self.pulse_amplitudes * (1.0 + error)

            # Compute fidelity
            fid = self._compute_fidelity(self.H_drift, pulse_scaled)
            fidelities.append(fid)

            # Nominal fidelity
            if np.abs(error) < 1e-10:
                nominal_fidelity = fid

        return np.array(fidelities), nominal_fidelity

    def sweep_amplitude_error(
        self,
        amplitude_error_range: np.ndarray,
        fidelity_threshold: float = 0.99,
    ) -> RobustnessResult:
        """
        Test robustness against control amplitude calibration errors.

        Amplitude errors arise from:
        - Systematic DAC calibration errors
        - Attenuator drift
        - Amplifier gain variations
        - Cable losses

        Parameters
        ----------
        amplitude_error_range : np.ndarray
            Array of fractional amplitude errors to test.
            Example: [-0.1, 0, 0.1] tests -10% to +10% error.
        fidelity_threshold : float, optional
            Threshold for robustness radius. Default: 0.99.

        Returns
        -------
        RobustnessResult
            Robustness analysis results.

        Examples
        --------
        >>> errors = np.linspace(-0.2, 0.2, 41)  # ±20%
        >>> result = tester.sweep_amplitude_error(errors)
        >>> print(f"Worst-case fidelity: {result.min_fidelity:.4f}")
        """
        # Sweep amplitude error range
        fidelities, nominal_fidelity = self._sweep_amplitude_error_range(
            amplitude_error_range
        )

        # Compute robustness radius
        robustness_radius = None
        if nominal_fidelity is not None:
            robustness_radius = self._compute_robustness_radius(
                amplitude_error_range, fidelities, fidelity_threshold
            )

        return RobustnessResult(
            parameter_values=amplitude_error_range,
            fidelities=fidelities,
            mean_fidelity=np.mean(fidelities),
            std_fidelity=np.std(fidelities),
            min_fidelity=np.min(fidelities),
            nominal_fidelity=nominal_fidelity or fidelities[len(fidelities) // 2],
            robustness_radius=robustness_radius,
            parameter_name="amplitude_error",
        )

    def _run_noise_realizations(self, noise_level: float, n_realizations: int) -> tuple:
        """
        Run multiple noise realizations.

        Parameters
        ----------
        noise_level : float
            Standard deviation of Gaussian noise
        n_realizations : int
            Number of realizations

        Returns
        -------
        fidelities : np.ndarray
            Fidelities for each realization
        noise_levels_actual : np.ndarray
            Actual noise levels (RMS)
        """
        fidelities = []
        noise_levels_actual = []

        for _ in range(n_realizations):
            # Generate Gaussian noise
            noise = np.random.randn(*self.pulse_amplitudes.shape) * noise_level
            pulse_noisy = self.pulse_amplitudes + noise

            # Compute fidelity
            fid = self._compute_fidelity(self.H_drift, pulse_noisy)
            fidelities.append(fid)

            # Track actual noise level (RMS)
            noise_levels_actual.append(np.std(noise))

        return np.array(fidelities), np.array(noise_levels_actual)

    def add_gaussian_noise(
        self,
        noise_level: float,
        n_realizations: int = 100,
        seed: Optional[int] = None,
    ) -> RobustnessResult:
        """
        Test robustness against Gaussian white noise on control amplitudes.

        White noise represents fast fluctuations from:
        - Thermal noise in electronics
        - Quantization noise in DACs
        - Shot noise in amplifiers

        Parameters
        ----------
        noise_level : float
            Standard deviation of Gaussian noise (relative to mean amplitude).
        n_realizations : int, optional
            Number of noise realizations to average over. Default: 100.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        RobustnessResult
            Robustness analysis with noise realizations.

        Examples
        --------
        >>> result = tester.add_gaussian_noise(noise_level=0.05, n_realizations=50)
        >>> print(f"Mean fidelity with noise: {result.mean_fidelity:.4f}")
        >>> print(f"Fidelity std dev: {result.std_fidelity:.4f}")
        """
        if seed is not None:
            np.random.seed(seed)

        # Nominal fidelity (no noise)
        nominal_fidelity = self._compute_fidelity(self.H_drift, self.pulse_amplitudes)

        # Run noise realizations
        fidelities, noise_levels_actual = self._run_noise_realizations(
            noise_level, n_realizations
        )

        return RobustnessResult(
            parameter_values=noise_levels_actual,
            fidelities=fidelities,
            mean_fidelity=np.mean(fidelities),
            std_fidelity=np.std(fidelities),
            min_fidelity=np.min(fidelities),
            nominal_fidelity=nominal_fidelity,
            robustness_radius=None,  # Not applicable for stochastic noise
            parameter_name="gaussian_noise",
        )

    def _apply_parameter_modification(
        self, param_name: str, param_value: float
    ) -> tuple:
        """
        Apply parameter modification based on name.

        Parameters
        ----------
        param_name : str
            Name of parameter to modify
        param_value : float
            Value of parameter

        Returns
        -------
        H_drift_mod : qt.Qobj
            Modified drift Hamiltonian
        pulse_mod : np.ndarray
            Modified pulse amplitudes
        """
        if param_name == "detuning":
            H_drift_mod = self.H_drift + 0.5 * param_value * qt.sigmaz()
        else:
            H_drift_mod = self.H_drift

        if param_name == "amplitude_error":
            pulse_mod = self.pulse_amplitudes * (1.0 + param_value)
        else:
            pulse_mod = self.pulse_amplitudes

        return H_drift_mod, pulse_mod

    def _sweep_2d_grid(
        self,
        param1_range: np.ndarray,
        param2_range: np.ndarray,
        param1_name: str,
        param2_name: str,
    ) -> np.ndarray:
        """
        Sweep over 2D parameter grid.

        Parameters
        ----------
        param1_range, param2_range : np.ndarray
            Parameter ranges
        param1_name, param2_name : str
            Parameter names

        Returns
        -------
        np.ndarray
            2D array of fidelities
        """
        fidelities = np.zeros((len(param1_range), len(param2_range)))

        for i, param1 in enumerate(param1_range):
            for j, param2 in enumerate(param2_range):
                fidelities[i, j] = self._compute_2d_parameter_fidelity(
                    param1_name, param1, param2_name, param2
                )

        return fidelities

    def _compute_2d_parameter_fidelity(
        self, param1_name: str, param1: float, param2_name: str, param2: float
    ) -> float:
        """
        Compute fidelity for a single point in 2D parameter space.

        Parameters
        ----------
        param1_name, param2_name : str
            Parameter names
        param1, param2 : float
            Parameter values

        Returns
        -------
        float
            Fidelity at this parameter point
        """
        # Apply parameter 1
        H_drift_mod, pulse_mod = self._apply_parameter_modification(param1_name, param1)

        # Apply parameter 2 (may override pulse_mod)
        if param2_name == "amplitude_error":
            pulse_mod = pulse_mod * (1.0 + param2)
        elif param2_name == "detuning" and param1_name != "detuning":
            H_drift_mod = H_drift_mod + 0.5 * param2 * qt.sigmaz()

        return self._compute_fidelity(H_drift_mod, pulse_mod)

    def sweep_2d_parameters(
        self,
        param1_range: np.ndarray,
        param2_range: np.ndarray,
        param1_name: str = "detuning",
        param2_name: str = "amplitude_error",
    ) -> Dict[str, np.ndarray]:
        """
        Test robustness over 2D parameter space.

        Useful for visualizing coupled effects of multiple error sources.

        Parameters
        ----------
        param1_range : np.ndarray
            Range for first parameter (e.g., detuning).
        param2_range : np.ndarray
            Range for second parameter (e.g., amplitude error).
        param1_name : str, optional
            Name of first parameter. Default: 'detuning'.
        param2_name : str, optional
            Name of second parameter. Default: 'amplitude_error'.

        Returns
        -------
        dict
            Dictionary with keys:
            - 'param1': Parameter 1 values
            - 'param2': Parameter 2 values
            - 'fidelities': 2D array of fidelities, shape (len(param1), len(param2))
            - 'param1_name': Name of parameter 1
            - 'param2_name': Name of parameter 2

        Examples
        --------
        >>> detunings = np.linspace(-0.5, 0.5, 21)
        >>> amp_errors = np.linspace(-0.1, 0.1, 21)
        >>> result = tester.sweep_2d_parameters(detunings, amp_errors)
        >>>
        >>> # Plot heatmap
        >>> plt.imshow(result['fidelities'], extent=[
        ...     amp_errors[0], amp_errors[-1],
        ...     detunings[0], detunings[-1]
        ... ], aspect='auto', origin='lower')
        >>> plt.colorbar(label='Fidelity')
        """
        fidelities = self._sweep_2d_grid(
            param1_range, param2_range, param1_name, param2_name
        )

        return {
            "param1": param1_range,
            "param2": param2_range,
            "fidelities": fidelities,
            "param1_name": param1_name,
            "param2_name": param2_name,
        }

    def compute_sensitivity(
        self,
        parameter_name: str,
        delta: float = 1e-4,
    ) -> float:
        """
        Compute numerical sensitivity ∂F/∂parameter.

        Sensitivity quantifies how much fidelity changes for small parameter
        variations. Higher sensitivity → less robust.

        Parameters
        ----------
        parameter_name : str
            Parameter to compute sensitivity for:
            - 'detuning': Frequency error sensitivity
            - 'amplitude': Amplitude error sensitivity
        delta : float, optional
            Finite difference step size. Default: 1e-4.

        Returns
        -------
        float
            Sensitivity |∂F/∂parameter|.

        Examples
        --------
        >>> sens_detuning = tester.compute_sensitivity('detuning')
        >>> sens_amplitude = tester.compute_sensitivity('amplitude')
        >>> print(f"Detuning sensitivity: {sens_detuning:.6f}")
        >>> print(f"Amplitude sensitivity: {sens_amplitude:.6f}")
        """
        # Nominal fidelity
        F0 = self._compute_fidelity(self.H_drift, self.pulse_amplitudes)

        # Perturbed fidelity
        if parameter_name == "detuning":
            H_drift_pert = self.H_drift + 0.5 * delta * qt.sigmaz()
            F_pert = self._compute_fidelity(H_drift_pert, self.pulse_amplitudes)
        elif parameter_name == "amplitude":
            pulse_pert = self.pulse_amplitudes * (1.0 + delta)
            F_pert = self._compute_fidelity(self.H_drift, pulse_pert)
        else:
            raise ValueError(f"Unknown parameter: {parameter_name}")

        # Numerical derivative
        sensitivity = np.abs((F_pert - F0) / delta)
        return sensitivity

    def _compute_perturbed_state(self, parameter_name: str, delta: float) -> qt.Qobj:
        """
        Compute perturbed quantum state.

        Parameters
        ----------
        parameter_name : str
            Parameter to perturb
        delta : float
            Perturbation magnitude

        Returns
        -------
        qt.Qobj
            Perturbed state

        Raises
        ------
        ValueError
            If parameter name is unknown
        """
        if parameter_name == "detuning":
            H_pert = self.H_drift + 0.5 * delta * qt.sigmaz()
            return self._evolve_state(H_pert, self.pulse_amplitudes)
        elif parameter_name == "amplitude":
            pulse_pert = self.pulse_amplitudes * (1.0 + delta)
            return self._evolve_state(self.H_drift, pulse_pert)
        else:
            raise ValueError(f"Unknown parameter: {parameter_name}")

    def _compute_fisher_from_states(
        self, rho_0: qt.Qobj, rho_delta: qt.Qobj, delta: float
    ) -> float:
        """
        Compute Fisher information from states.

        Parameters
        ----------
        rho_0 : qt.Qobj
            Unperturbed state
        rho_delta : qt.Qobj
            Perturbed state
        delta : float
            Perturbation step size

        Returns
        -------
        float
            Fisher information
        """
        # Numerical derivative of state
        drho_dtheta = (rho_delta - rho_0) / delta

        # Classical Fisher information approximation:
        # F ≈ 4 * Tr[(∂ρ/∂θ)²]
        fisher = 4 * np.real((drho_dtheta.dag() * drho_dtheta).tr())
        return fisher

    def compute_fisher_information(
        self,
        parameter_name: str,
        delta: float = 1e-5,
    ) -> float:
        """
        Compute Fisher information for parameter estimation.

        Fisher information quantifies the amount of information about a parameter
        that can be extracted from measurements. Higher Fisher information means
        the parameter can be estimated more precisely.

        For a quantum state ρ(θ), the quantum Fisher information is:
            F(θ) = Tr[ρ(θ) L²] where L is the symmetric logarithmic derivative.

        We approximate this via numerical differentiation of the state.

        Parameters
        ----------
        parameter_name : str
            Parameter to compute Fisher information for:
            - 'detuning': Frequency error
            - 'amplitude': Amplitude error
        delta : float, optional
            Finite difference step size. Default: 1e-5.

        Returns
        -------
        float
            Fisher information F(θ).

        Examples
        --------
        >>> fisher = tester.compute_fisher_information('detuning')
        >>> print(f"Fisher information: {fisher:.6e}")

        References
        ----------
        - Braunstein & Caves, PRL 72, 3439 (1994)
        - Paris, Int. J. Quantum Inf. 7, 125 (2009)
        """
        # Compute unperturbed and perturbed states
        rho_0 = self._evolve_state(self.H_drift, self.pulse_amplitudes)
        rho_delta = self._compute_perturbed_state(parameter_name, delta)

        # Compute Fisher information
        return self._compute_fisher_from_states(rho_0, rho_delta, delta)

    def _evolve_state(
        self,
        H_drift: qt.Qobj,
        pulse_amplitudes: np.ndarray,
        initial_state: Optional[qt.Qobj] = None,
    ) -> qt.Qobj:
        """
        Evolve state under Hamiltonian and return final density matrix.

        Parameters
        ----------
        H_drift : qt.Qobj
            Drift Hamiltonian.
        pulse_amplitudes : np.ndarray
            Control pulse amplitudes.
        initial_state : qt.Qobj, optional
            Initial state (default: ground state).

        Returns
        -------
        qt.Qobj
            Final state (density matrix).
        """
        if initial_state is None:
            initial_state = qt.basis(2, 0)

        # Convert to density matrix if ket
        if initial_state.type == "ket":
            rho_0 = qt.ket2dm(initial_state)
        else:
            rho_0 = initial_state

        # Build Hamiltonian list
        times = np.linspace(0, self.total_time, self.n_timeslices + 1)
        H_list = [H_drift]
        for j in range(self.n_controls):
            # Create time-dependent coefficient function
            def coeff_func(t, args=None, ctrl_idx=j):
                idx = int(t / self.dt)
                idx = min(idx, pulse_amplitudes.shape[1] - 1)
                return pulse_amplitudes[ctrl_idx, idx]

            H_list.append([self.H_controls[j], coeff_func])

        # Evolve
        result = qt.mesolve(H_list, rho_0, times, [], [])

        return result.states[-1]

    def find_worst_case_parameters(
        self,
        param_ranges: Dict[str, Tuple[float, float]],
        n_samples: int = 20,
        method: str = "grid",
    ) -> Dict:
        """
        Find worst-case parameter combination that minimizes fidelity.

        This is useful for robust optimization: design pulses that maximize
        the worst-case fidelity over parameter uncertainties.

        Parameters
        ----------
        param_ranges : dict
            Dict mapping parameter names to (min, max) ranges.
            Supported: 'detuning', 'amplitude_error'.
        n_samples : int, optional
            Number of samples per parameter (for grid search). Default: 20.
        method : str, optional
            Search method: 'grid' or 'random'. Default: 'grid'.

        Returns
        -------
        dict
            Dictionary with:
            - 'worst_case_params': Dict of parameter values at worst case
            - 'worst_case_fidelity': Minimum fidelity found
            - 'all_fidelities': Array of all fidelities tested
            - 'all_params': List of all parameter combinations tested

        Examples
        --------
        >>> param_ranges = {
        ...     'detuning': (-0.1, 0.1),
        ...     'amplitude_error': (-0.05, 0.05)
        ... }
        >>> result = tester.find_worst_case_parameters(param_ranges)
        >>> print(f"Worst-case fidelity: {result['worst_case_fidelity']:.6f}")

        References
        ----------
        - Vandersypen & Chuang, Rev. Mod. Phys. 76, 1037 (2005)
        - Motzoi et al., PRL 103, 110501 (2009)
        """
        if method == "grid":
            return self._worst_case_grid_search(param_ranges, n_samples)
        elif method == "random":
            return self._worst_case_random_search(param_ranges, n_samples)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _worst_case_grid_search(
        self, param_ranges: Dict[str, Tuple[float, float]], n_samples: int
    ) -> Dict:
        """Grid search for worst-case parameters."""
        param_names = list(param_ranges.keys())
        param_grids = [
            np.linspace(param_ranges[name][0], param_ranges[name][1], n_samples)
            for name in param_names
        ]

        # Create meshgrid
        if len(param_names) == 1:
            grids = [param_grids[0]]
            param_combinations = [(p,) for p in grids[0]]
        elif len(param_names) == 2:
            grids = np.meshgrid(*param_grids)
            param_combinations = [
                (grids[0].flat[i], grids[1].flat[i]) for i in range(grids[0].size)
            ]
        else:
            # For >2 parameters, use itertools
            import itertools

            param_combinations = list(itertools.product(*param_grids))

        # Evaluate fidelity at each combination
        fidelities = []
        all_params = []

        for param_vals in param_combinations:
            # Build parameter dict
            params = {name: val for name, val in zip(param_names, param_vals)}
            all_params.append(params)

            # Modify Hamiltonian and pulse
            H_drift_mod = self.H_drift
            pulse_mod = self.pulse_amplitudes.copy()

            if "detuning" in params:
                H_drift_mod = H_drift_mod + 0.5 * params["detuning"] * qt.sigmaz()

            if "amplitude_error" in params:
                pulse_mod = pulse_mod * (1.0 + params["amplitude_error"])

            # Compute fidelity
            fid = self._compute_fidelity(H_drift_mod, pulse_mod)
            fidelities.append(fid)

        fidelities = np.array(fidelities)
        worst_idx = np.argmin(fidelities)

        return {
            "worst_case_params": all_params[worst_idx],
            "worst_case_fidelity": fidelities[worst_idx],
            "all_fidelities": fidelities,
            "all_params": all_params,
        }

    def _worst_case_random_search(
        self, param_ranges: Dict[str, Tuple[float, float]], n_samples: int
    ) -> Dict:
        """Random search for worst-case parameters."""
        param_names = list(param_ranges.keys())
        fidelities = []
        all_params = []

        for _ in range(n_samples):
            # Random sample from ranges
            params = {}
            for name in param_names:
                low, high = param_ranges[name]
                params[name] = np.random.uniform(low, high)

            all_params.append(params)

            # Modify Hamiltonian and pulse
            H_drift_mod = self.H_drift
            pulse_mod = self.pulse_amplitudes.copy()

            if "detuning" in params:
                H_drift_mod = H_drift_mod + 0.5 * params["detuning"] * qt.sigmaz()

            if "amplitude_error" in params:
                pulse_mod = pulse_mod * (1.0 + params["amplitude_error"])

            # Compute fidelity
            fid = self._compute_fidelity(H_drift_mod, pulse_mod)
            fidelities.append(fid)

        fidelities = np.array(fidelities)
        worst_idx = np.argmin(fidelities)

        return {
            "worst_case_params": all_params[worst_idx],
            "worst_case_fidelity": fidelities[worst_idx],
            "all_fidelities": fidelities,
            "all_params": all_params,
        }

    def _apply_parameter_modifications(
        self, param_name: str, param_value: float
    ) -> Tuple:
        """
        Apply parameter modifications to Hamiltonian and pulse.

        Parameters
        ----------
        param_name : str
            Name of parameter to modify.
        param_value : float
            Value of parameter.

        Returns
        -------
        tuple
            Modified (H_drift, pulse_amplitudes).
        """
        H_drift_mod = self.H_drift
        pulse_mod = self.pulse_amplitudes.copy()

        if param_name == "detuning":
            H_drift_mod = H_drift_mod + 0.5 * param_value * qt.sigmaz()
        elif param_name == "amplitude_error":
            pulse_mod = pulse_mod * (1.0 + param_value)

        return H_drift_mod, pulse_mod

    def _compute_landscape_1d(
        self, param_name: str, param_range: Tuple[float, float], n_points: int
    ) -> Dict:
        """
        Compute 1D robustness landscape.

        Parameters
        ----------
        param_name : str
            Parameter name.
        param_range : tuple
            (min, max) range for parameter.
        n_points : int
            Number of points to sample.

        Returns
        -------
        dict
            Landscape data with fidelities and statistics.
        """
        low, high = param_range
        param_values = np.linspace(low, high, n_points)

        fidelities = []
        for val in param_values:
            H_drift_mod, pulse_mod = self._apply_parameter_modifications(
                param_name, val
            )
            fid = self._compute_fidelity(H_drift_mod, pulse_mod)
            fidelities.append(fid)

        fidelities = np.array(fidelities)

        return {
            "param_names": [param_name],
            "param_grids": [param_values],
            "fidelity_grid": fidelities,
            "mean_fidelity": np.mean(fidelities),
            "std_fidelity": np.std(fidelities),
            "min_fidelity": np.min(fidelities),
        }

    def _compute_landscape_2d(
        self,
        param_names: List[str],
        param_ranges: Dict[str, Tuple[float, float]],
        n_points: int,
    ) -> Dict:
        """
        Compute 2D robustness landscape.

        Parameters
        ----------
        param_names : list
            Two parameter names.
        param_ranges : dict
            Parameter ranges.
        n_points : int
            Number of points per axis.

        Returns
        -------
        dict
            Landscape data with 2D fidelity grid and statistics.
        """
        name1, name2 = param_names
        low1, high1 = param_ranges[name1]
        low2, high2 = param_ranges[name2]

        param1_values = np.linspace(low1, high1, n_points)
        param2_values = np.linspace(low2, high2, n_points)

        fidelity_grid = np.zeros((n_points, n_points))

        for i, val1 in enumerate(param1_values):
            for j, val2 in enumerate(param2_values):
                H_drift_mod = self.H_drift
                pulse_mod = self.pulse_amplitudes.copy()

                H_drift_mod, pulse_mod = self._apply_parameter_modifications(
                    name1, val1
                )
                H_drift_tmp, pulse_tmp = self._apply_parameter_modifications(
                    name2, val2
                )

                # Combine modifications
                if name2 == "detuning":
                    H_drift_mod = H_drift_tmp
                elif name2 == "amplitude_error":
                    pulse_mod = pulse_tmp

                fidelity_grid[i, j] = self._compute_fidelity(H_drift_mod, pulse_mod)

        return {
            "param_names": param_names,
            "param_grids": [param1_values, param2_values],
            "fidelity_grid": fidelity_grid,
            "mean_fidelity": np.mean(fidelity_grid),
            "std_fidelity": np.std(fidelity_grid),
            "min_fidelity": np.min(fidelity_grid),
        }

    def compute_robustness_landscape(
        self,
        param_ranges: Dict[str, Tuple[float, float]],
        n_points: int = 30,
    ) -> Dict:
        """
        Compute full fidelity landscape over parameter space.

        This provides a comprehensive view of how fidelity varies across
        the parameter space, useful for understanding robustness regions.

        Parameters
        ----------
        param_ranges : dict
            Dict mapping parameter names to (min, max) ranges.
            Currently supports up to 2 parameters for visualization.
        n_points : int, optional
            Number of points per parameter axis. Default: 30.

        Returns
        -------
        dict
            Dictionary with landscape data suitable for plotting:
            - 'param_names': List of parameter names
            - 'param_grids': List of parameter arrays
            - 'fidelity_grid': Array of fidelity values
            - 'mean_fidelity': Mean over landscape
            - 'std_fidelity': Std deviation over landscape
            - 'min_fidelity': Minimum (worst-case) fidelity

        Examples
        --------
        >>> param_ranges = {'detuning': (-0.2, 0.2), 'amplitude_error': (-0.1, 0.1)}
        >>> landscape = tester.compute_robustness_landscape(param_ranges, n_points=50)
        >>> print(f"Mean fidelity: {landscape['mean_fidelity']:.6f}")
        >>> print(f"Worst-case: {landscape['min_fidelity']:.6f}")
        """
        if len(param_ranges) > 2:
            raise ValueError("Landscape computation supports up to 2 parameters")

        param_names = list(param_ranges.keys())

        if len(param_names) == 1:
            return self._compute_landscape_1d(
                param_names[0], param_ranges[param_names[0]], n_points
            )
        else:
            return self._compute_landscape_2d(param_names, param_ranges, n_points)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"RobustnessTester(fidelity_type='{self.fidelity_type}', "
            f"n_controls={self.n_controls}, n_timeslices={self.n_timeslices})"
        )


def compare_pulse_robustness(
    testers: List[RobustnessTester],
    labels: List[str],
    detuning_range: np.ndarray,
) -> Dict[str, List]:
    """
    Compare robustness of multiple pulses (e.g., GRAPE vs. Krotov).

    Parameters
    ----------
    testers : list of RobustnessTester
        List of robustness testers for different pulses.
    labels : list of str
        Labels for each pulse (for plotting).
    detuning_range : np.ndarray
        Detuning values to sweep.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'labels': Pulse labels
        - 'results': List of RobustnessResult objects
        - 'detuning_range': Detuning values

    Examples
    --------
    >>> tester1 = RobustnessTester(H0, Hc, pulse_grape, T, U_target=U)
    >>> tester2 = RobustnessTester(H0, Hc, pulse_krotov, T, U_target=U)
    >>> comparison = compare_pulse_robustness(
    ...     [tester1, tester2],
    ...     ['GRAPE', 'Krotov'],
    ...     np.linspace(-1, 1, 21)
    ... )
    >>>
    >>> for label, result in zip(comparison['labels'], comparison['results']):
    ...     plt.plot(result.parameter_values, result.fidelities, label=label)
    >>> plt.legend()
    """
    results = []

    for tester in testers:
        result = tester.sweep_detuning(detuning_range)
        results.append(result)

    return {
        "labels": labels,
        "results": results,
        "detuning_range": detuning_range,
    }
