"""
DRAG (Derivative Removal by Adiabatic Gate) Pulse Module.

This module provides advanced DRAG pulse functionality including:
- Optimal β parameter calculation for given anharmonicity
- Leakage error quantification in multi-level systems
- Comparison with standard Gaussian pulses
- Integration with GRAPE/Krotov optimizers

DRAG pulses are essential for high-fidelity gates in weakly anharmonic qubits
(e.g., superconducting transmons) where leakage to |2⟩ can limit gate fidelity.

Physical Background:
-------------------
In a transmon qubit, the energy levels are not perfectly evenly spaced due to
anharmonicity α = (E₂ - E₁) - (E₁ - E₀). During a fast gate, off-resonant
driving can populate the |2⟩ level, causing leakage error.

DRAG corrects this by adding a quadrature component proportional to the
derivative of the in-phase pulse, effectively canceling leakage to first order
in the perturbation theory.

References:
-----------
[1] Motzoi, F. et al., "Simple pulses for elimination of leakage in weakly
    nonlinear qubits," Phys. Rev. Lett. 103, 110501 (2009).
[2] Gambetta, J. M. et al., "Analytic control methods for high-fidelity unitary
    operations in a weakly nonlinear oscillator," Phys. Rev. A 83, 012308 (2011).
[3] Chen, Z. et al., "Measuring and suppressing quantum state leakage in a
    superconducting qubit," Phys. Rev. Lett. 116, 020501 (2016).
"""

import numpy as np
import qutip as qt
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class DRAGParameters:
    """
    Parameters for DRAG pulse generation.

    Attributes
    ----------
    amplitude : float
        Peak Rabi frequency (MHz or rad/s)
    sigma : float
        Gaussian width parameter (ns or dimensionless)
    beta : float
        DRAG coefficient (dimensionless)
        Optimal value: β = -1/(2α) where α is anharmonicity in angular freq (rad/ns)
    detuning : float
        Detuning from qubit resonance (MHz)
    anharmonicity : float, optional
        Qubit anharmonicity α = ω₁₂ - ω₀₁ (MHz)
        If provided, enables automatic β optimization
    truncation : float
        Pulse truncation in units of σ (default 4.0)
    """

    amplitude: float
    sigma: float
    beta: float
    detuning: float = 0.0
    anharmonicity: Optional[float] = None
    truncation: float = 4.0

    def __post_init__(self):
        """Validate parameters."""
        if self.amplitude <= 0:
            raise ValueError("Amplitude must be positive")
        if self.sigma <= 0:
            raise ValueError("Sigma must be positive")
        if self.truncation <= 0:
            raise ValueError("Truncation must be positive")


class DRAGPulse:
    """
    DRAG pulse generator with optimization and analysis tools.

    This class provides:
    - I/Q pulse envelope generation
    - Optimal β parameter calculation
    - Leakage error estimation
    - Comparison with standard Gaussian pulses

    Examples
    --------
    >>> # Create DRAG pulse with manual β
    >>> params = DRAGParameters(amplitude=10.0, sigma=5.0, beta=0.3)
    >>> drag = DRAGPulse(params)
    >>> times = np.linspace(0, 50, 500)
    >>> omega_I, omega_Q = drag.envelope(times, t_center=25.0)

    >>> # Optimize β for given anharmonicity
    >>> params = DRAGParameters(amplitude=10.0, sigma=5.0, beta=0.0,
    ...                         anharmonicity=-200.0)
    >>> drag = DRAGPulse(params)
    >>> optimal_beta = drag.optimize_beta()
    >>> print(f"Optimal β = {optimal_beta:.4f}")
    """

    def __init__(self, parameters: DRAGParameters):
        """
        Initialize DRAG pulse generator.

        Parameters
        ----------
        parameters : DRAGParameters
            DRAG pulse parameters
        """
        self.params = parameters

    def envelope(
        self, times: np.ndarray, t_center: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate DRAG pulse envelopes (I and Q components).

        Mathematical form:
            Ω_I(t) = A * exp(-(t - t_c)² / (2σ²))
            Ω_Q(t) = β * dΩ_I/dt = β * A * (t - t_c)/σ² * exp(-(t - t_c)² / (2σ²))

        Parameters
        ----------
        times : np.ndarray
            Time points for pulse evaluation
        t_center : float
            Center time of the pulse

        Returns
        -------
        omega_I : np.ndarray
            In-phase (I) component
        omega_Q : np.ndarray
            Quadrature (Q) component
        """
        # Initialize arrays
        omega_I = np.zeros_like(times, dtype=float)
        omega_Q = np.zeros_like(times, dtype=float)

        # Apply truncation mask
        mask = np.abs(times - t_center) <= self.params.truncation * self.params.sigma
        t_masked = times[mask]

        # Calculate Gaussian envelope (I component)
        exponent = -((t_masked - t_center) ** 2) / (2 * self.params.sigma**2)
        gaussian = self.params.amplitude * np.exp(exponent)
        omega_I[mask] = gaussian

        # Calculate derivative correction (Q component)
        # dΩ_I/dt = A * (-(t-tc)/σ²) * exp(-(t-tc)²/(2σ²))
        derivative_factor = -(t_masked - t_center) / (self.params.sigma**2)
        omega_Q[mask] = self.params.beta * derivative_factor * gaussian

        return omega_I, omega_Q

    def derivative_check(
        self, times: np.ndarray, t_center: float, epsilon: float = 1e-6
    ) -> float:
        """
        Verify that Q component matches numerical derivative of I component.

        This is a validation check: Ω_Q should equal β * dΩ_I/dt

        Parameters
        ----------
        times : np.ndarray
            Time points for evaluation
        t_center : float
            Center time of the pulse
        epsilon : float
            Finite difference step size

        Returns
        -------
        max_error : float
            Maximum normalized error between analytical and numerical derivative
        """
        omega_I, omega_Q = self.envelope(times, t_center)

        # Compute numerical derivative of I component
        dt = times[1] - times[0]
        dI_dt_numerical = np.gradient(omega_I, dt)

        # Analytical derivative (should match Q/beta)
        dI_dt_analytical = (
            omega_Q / self.params.beta if self.params.beta != 0 else omega_Q
        )

        # Compute normalized error (avoid division by zero)
        mask = np.abs(omega_I) > 1e-10 * self.params.amplitude
        if not np.any(mask):
            return 0.0

        abs_diff = np.abs(dI_dt_numerical[mask] - dI_dt_analytical[mask])
        max_deriv = np.max(np.abs(dI_dt_analytical[mask]))

        normalized_error = np.max(abs_diff) / (max_deriv + 1e-12)

        return normalized_error

    def optimize_beta(self) -> float:
        """
        Calculate optimal β parameter for given anharmonicity.

        The optimal DRAG coefficient for suppressing leakage to first order is:
            β_opt = -α / (2 * Ω_max)

        where α is the anharmonicity and Ω_max is the peak Rabi frequency.

        .. warning::
            Both ``anharmonicity`` and ``amplitude`` must be in the **same
            angular-frequency units** (e.g., both in rad/ns or both in rad/s).
            If anharmonicity is provided in MHz and amplitude in rad/ns, the
            caller must convert first:
            ``anharmonicity_radns = 2π × anharmonicity_MHz × 1e-3``

        Typical DRAG β values are O(1) for transmon qubits. If the returned
        value is >> 10, the units are likely mismatched.

        Returns
        -------
        beta_opt : float
            Optimal β parameter (dimensionless)

        Raises
        ------
        ValueError
            If anharmonicity is not set in parameters

        References
        ----------
        Motzoi et al., PRL 103, 110501 (2009), Eq. 3
        """
        if self.params.anharmonicity is None:
            raise ValueError("Anharmonicity must be set to optimize β")

        # Optimal β formula from Motzoi et al. PRL 103, 110501 (2009)
        # β = -1/(2α) where α is anharmonicity in angular frequency (rad/ns)
        # NOTE: self.params.anharmonicity must be in angular frequency units (rad/ns)
        # If in MHz, convert first: α_rad = 2π × f_MHz × 1e-3
        beta_opt = -1.0 / (2.0 * self.params.anharmonicity)

        if abs(beta_opt) > 5:
            import warnings
            warnings.warn(
                f"Computed β = {beta_opt:.2f} is unusually large (expected ~0.4 for transmon). "
                f"Verify anharmonicity ({self.params.anharmonicity}) is in rad/ns. "
                f"For α/2π = -200 MHz, α = -1.257 rad/ns, β = 0.398.",
                stacklevel=2,
            )

        return beta_opt

    def pulse_area(self, times: np.ndarray, t_center: float) -> Tuple[float, float]:
        """
        Calculate integrated pulse areas for I and Q components.

        For the I component, this determines the rotation angle:
            θ = ∫ Ω_I(t) dt

        Parameters
        ----------
        times : np.ndarray
            Time points
        t_center : float
            Center time of the pulse

        Returns
        -------
        area_I : float
            Integrated area of I component
        area_Q : float
            Integrated area of Q component (should be ~0 for symmetric pulse)
        """
        omega_I, omega_Q = self.envelope(times, t_center)

        area_I = np.trapezoid(omega_I, times)
        area_Q = np.trapezoid(omega_Q, times)

        return area_I, area_Q

    def hamiltonian_coefficients(
        self, times: np.ndarray, t_center: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate time-dependent Hamiltonian coefficients for QuTiP.

        These coefficients can be used with QuTiP's time-dependent Hamiltonian:
            H(t) = H_drift + Ω_I(t) * H_x + Ω_Q(t) * H_y

        Parameters
        ----------
        times : np.ndarray
            Time points
        t_center : float
            Center time of the pulse

        Returns
        -------
        coeff_I : np.ndarray
            Coefficients for σ_x control
        coeff_Q : np.ndarray
            Coefficients for σ_y control
        """
        return self.envelope(times, t_center)

    def _create_control_hamiltonians(self, n_levels: int) -> tuple[qt.Qobj, qt.Qobj]:
        """
        Create control Hamiltonians for given number of levels.

        Parameters
        ----------
        n_levels : int
            Number of qubit levels.

        Returns
        -------
        tuple
            (H_x, H_y) control Hamiltonians.
        """
        from qutip import sigmax, sigmay

        if n_levels == 2:
            H_x = sigmax()
            H_y = sigmay()
        else:
            # Multi-level: use ladder operators in computational subspace
            H_x = qt.create(n_levels) + qt.destroy(n_levels)
            H_y = 1j * (qt.create(n_levels) - qt.destroy(n_levels))

        return H_x, H_y

    def _create_pulse_coefficients(self, times: np.ndarray, t_center: float) -> tuple:
        """
        Create interpolated coefficient functions for QuTiP simulation.

        Parameters
        ----------
        times : np.ndarray
            Time points.
        t_center : float
            Pulse center time.

        Returns
        -------
        tuple
            (coeff_I, coeff_Q) functions that return scalar values.
        """
        from scipy.interpolate import interp1d

        omega_I, omega_Q = self.envelope(times, t_center)

        coeff_I_interp = interp1d(
            times, omega_I, kind="linear", fill_value=0.0, bounds_error=False
        )
        coeff_Q_interp = interp1d(
            times, omega_Q, kind="linear", fill_value=0.0, bounds_error=False
        )

        def coeff_I(t, args=None):
            return float(coeff_I_interp(t))

        def coeff_Q(t, args=None):
            return float(coeff_Q_interp(t))

        return coeff_I, coeff_Q

    def _simulate_drag_and_gaussian(
        self,
        H_drift: qt.Qobj,
        H_x: qt.Qobj,
        H_y: qt.Qobj,
        coeff_I,
        coeff_Q,
        times: np.ndarray,
        n_levels: int,
    ) -> tuple[qt.Qobj, qt.Qobj]:
        """
        Simulate evolution with DRAG and Gaussian pulses.

        Parameters
        ----------
        H_drift : qt.Qobj
            Drift Hamiltonian.
        H_x, H_y : qt.Qobj
            Control Hamiltonians.
        coeff_I, coeff_Q : callable
            Coefficient functions.
        times : np.ndarray
            Time points.
        n_levels : int
            Number of levels.

        Returns
        -------
        tuple
            (U_drag, U_gauss) final states.
        """
        from qutip import mesolve

        psi0 = qt.basis(n_levels, 0)

        # Evolve with DRAG
        H_drag = [H_drift, [H_x, coeff_I], [H_y, coeff_Q]]
        result_drag = mesolve(H_drag, psi0, times, [], [])
        U_drag = result_drag.states[-1]

        # Evolve with Gaussian (β=0)
        H_gauss = [H_drift, [H_x, coeff_I]]
        result_gauss = mesolve(H_gauss, psi0, times, [], [])
        U_gauss = result_gauss.states[-1]

        return U_drag, U_gauss

    def _compute_fidelities(
        self,
        U_drag: qt.Qobj,
        U_gauss: qt.Qobj,
        U_target: qt.Qobj,
        n_levels: int,
    ) -> tuple[float, float]:
        """
        Compute fidelities for DRAG and Gaussian pulses.

        Parameters
        ----------
        U_drag, U_gauss : qt.Qobj
            Evolved states.
        U_target : qt.Qobj
            Target unitary.
        n_levels : int
            Number of levels.

        Returns
        -------
        tuple
            (fid_drag, fid_gauss) fidelity values.
        """
        psi0 = qt.basis(n_levels, 0)

        if n_levels > 2:
            # Project to computational subspace
            proj = qt.tensor([qt.basis(n_levels, i) for i in range(2)])
            U_target_full = proj * U_target * proj.dag()
        else:
            U_target_full = U_target

        fid_drag = qt.fidelity(U_target_full * psi0, U_drag) ** 2
        fid_gauss = qt.fidelity(U_target_full * psi0, U_gauss) ** 2

        return fid_drag, fid_gauss

    def _compute_leakage(
        self, U_drag: qt.Qobj, U_gauss: qt.Qobj, n_levels: int
    ) -> dict:
        """
        Compute leakage errors for multi-level systems.

        Parameters
        ----------
        U_drag, U_gauss : qt.Qobj
            Evolved states.
        n_levels : int
            Number of levels.

        Returns
        -------
        dict
            Leakage metrics.
        """
        leakage_drag = sum(np.abs(U_drag.full()[i, 0]) ** 2 for i in range(2, n_levels))
        leakage_gauss = sum(
            np.abs(U_gauss.full()[i, 0]) ** 2 for i in range(2, n_levels)
        )

        return {
            "drag_leakage": leakage_drag,
            "gaussian_leakage": leakage_gauss,
            "leakage_suppression": (
                (leakage_gauss - leakage_drag) / leakage_gauss
                if leakage_gauss > 0
                else 0.0
            ),
        }

    def compare_with_gaussian(
        self,
        times: np.ndarray,
        t_center: float,
        U_target: qt.Qobj,
        H_drift: qt.Qobj,
        n_levels: int = 2,
    ) -> Dict[str, Any]:
        """
        Compare DRAG pulse with standard Gaussian (β=0).

        Simulates both pulses and compares gate fidelity, leakage (if n_levels>=3),
        and computational subspace fidelity.

        Returns dict with: drag_fidelity, gaussian_fidelity, improvement,
        drag_leakage, gaussian_leakage (if n_levels>=3), leakage_suppression.
        """
        # Create control Hamiltonians
        H_x, H_y = self._create_control_hamiltonians(n_levels)

        # Create pulse coefficient functions
        coeff_I, coeff_Q = self._create_pulse_coefficients(times, t_center)

        # Simulate both pulses
        U_drag, U_gauss = self._simulate_drag_and_gaussian(
            H_drift, H_x, H_y, coeff_I, coeff_Q, times, n_levels
        )

        # Compute fidelities
        fid_drag, fid_gauss = self._compute_fidelities(
            U_drag, U_gauss, U_target, n_levels
        )

        # Assemble results
        comparison = {
            "drag_fidelity": fid_drag,
            "gaussian_fidelity": fid_gauss,
            "improvement": (fid_drag - fid_gauss) / (1 - fid_gauss)
            if fid_gauss < 1
            else 0.0,
        }

        # Add leakage metrics if applicable
        if n_levels >= 3:
            comparison.update(self._compute_leakage(U_drag, U_gauss, n_levels))

        return comparison


def _get_gate_angle(gate_type: str) -> float:
    """
    Get target rotation angle for gate type.

    Parameters
    ----------
    gate_type : str
        Type of gate

    Returns
    -------
    float
        Target rotation angle (radians)

    Raises
    ------
    ValueError
        If gate type is unknown
    """
    gate_angles = {
        "X": np.pi,
        "Y": np.pi,
        "X/2": np.pi / 2,
        "Y/2": np.pi / 2,
        "H": np.pi,  # Hadamard requires composite construction
    }

    if gate_type not in gate_angles:
        raise ValueError(
            f"Unknown gate type: {gate_type}. Must be one of {list(gate_angles.keys())}"
        )

    return gate_angles[gate_type]


def _compute_drag_pulse_params(
    target_angle: float,
    gate_time: float,
    anharmonicity: Optional[float],
    optimize_beta: bool,
) -> Tuple[float, float, float]:
    """
    Compute DRAG pulse parameters for target angle.

    Parameters
    ----------
    target_angle : float
        Target rotation angle
    gate_time : float
        Gate duration
    anharmonicity : float, optional
        Qubit anharmonicity
    optimize_beta : bool
        Whether to optimize beta

    Returns
    -------
    amplitude : float
        Pulse amplitude
    sigma : float
        Gaussian width
    beta : float
        DRAG correction parameter
    """
    # Pulse width (4σ truncation gives total width of 8σ)
    sigma = gate_time / 8.0  # Pulse occupies central 50% of gate time

    # Calculate amplitude for target rotation angle
    # For Gaussian: ∫ A*exp(-(t-tc)²/(2σ²)) dt ≈ A*σ*√(2π)
    amplitude = target_angle / (sigma * np.sqrt(2 * np.pi))

    # Set initial β (will be optimized if requested)
    beta = 0.0
    if optimize_beta and anharmonicity is not None:
        # β = -1/(2α), Motzoi et al. PRL 103, 110501 (2009)
        beta = -1.0 / (2.0 * anharmonicity)

    return amplitude, sigma, beta


def create_drag_pulse_for_gate(
    gate_type: str,
    gate_time: float,
    n_points: int = 500,
    anharmonicity: Optional[float] = None,
    optimize_beta: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create optimized DRAG pulse for standard gates with automatic β optimization.

    Parameters: gate_type ('X', 'Y', 'X/2', 'Y/2', 'H'), gate_time (ns),
    n_points, anharmonicity (MHz, optional), optimize_beta.
    Returns: (times, omega_I, omega_Q).
    """
    # Determine target rotation angle
    target_angle = _get_gate_angle(gate_type)

    # Compute pulse parameters
    amplitude, sigma, beta = _compute_drag_pulse_params(
        target_angle, gate_time, anharmonicity, optimize_beta
    )

    # Time array
    times = np.linspace(0, gate_time, n_points)
    t_center = gate_time / 2.0

    # Create DRAG pulse
    params = DRAGParameters(
        amplitude=amplitude, sigma=sigma, beta=beta, anharmonicity=anharmonicity
    )

    drag = DRAGPulse(params)
    omega_I, omega_Q = drag.envelope(times, t_center)

    return times, omega_I, omega_Q


def leakage_error_estimate(
    amplitude: float, sigma: float, anharmonicity: float, beta: float = 0.0
) -> float:
    """
    Estimate leakage error to |2⟩ state for given pulse parameters.

    This uses first-order perturbation theory to estimate the leakage
    probability without full numerical simulation.

    Parameters
    ----------
    amplitude : float
        Peak Rabi frequency (MHz)
    sigma : float
        Gaussian width (ns)
    anharmonicity : float
        Qubit anharmonicity α = ω₁₂ - ω₀₁ (MHz)
    beta : float
        DRAG coefficient (0 for standard Gaussian)

    Returns
    -------
    leakage : float
        Estimated leakage probability to |2⟩

    Notes
    -----
    For optimal DRAG (β = -1/(2α)), leakage is suppressed to second order.
    For Gaussian (β=0), leakage scales as (Ω/α)².

    References
    ----------
    Gambetta et al., PRA 83, 012308 (2011), Eqs. 10-12
    """
    if anharmonicity == 0:
        raise ValueError("Anharmonicity cannot be zero")

    # Optimal β = -1/(2α), Motzoi et al. PRL 103, 110501 (2009)
    beta_opt = -1.0 / (2.0 * anharmonicity)

    # Deviation from optimal
    delta_beta = beta - beta_opt

    # Leakage scales as (Ω/α)² for Gaussian, with correction term for β
    leakage_baseline = (amplitude / anharmonicity) ** 2

    # DRAG correction factor (reduces leakage when β ≈ β_opt)
    correction = 1.0 + (2.0 * amplitude * delta_beta / anharmonicity) ** 2

    leakage = leakage_baseline * correction

    # Approximate scaling with pulse duration (longer = more leakage)
    duration_factor = sigma * np.sqrt(2 * np.pi)  # Effective pulse duration
    leakage *= duration_factor

    return leakage


def _setup_scan_hamiltonians(n_levels: int) -> Tuple[qt.Qobj, qt.Qobj, qt.Qobj]:
    """Setup Hamiltonians and initial state for beta scan."""
    H_x = qt.create(n_levels) + qt.destroy(n_levels)
    H_y = 1j * (qt.create(n_levels) - qt.destroy(n_levels))
    psi0 = qt.basis(n_levels, 0)
    return H_x, H_y, psi0


def _create_drag_coefficients(
    times: np.ndarray, omega_I: np.ndarray, omega_Q: np.ndarray
) -> Tuple:
    """Create interpolated coefficient functions for QuTiP."""
    from scipy.interpolate import interp1d

    coeff_I_interp = interp1d(
        times, omega_I, kind="linear", fill_value=0.0, bounds_error=False
    )
    coeff_Q_interp = interp1d(
        times, omega_Q, kind="linear", fill_value=0.0, bounds_error=False
    )

    def coeff_I(t, args=None):
        return float(coeff_I_interp(t))

    def coeff_Q(t, args=None):
        return float(coeff_Q_interp(t))

    return coeff_I, coeff_Q


def _embed_target_unitary(U_target: qt.Qobj, n_levels: int) -> qt.Qobj:
    """Embed 2x2 target unitary into n_levels space."""
    U_full_array = np.eye(n_levels, dtype=complex)
    U_full_array[0:2, 0:2] = U_target.full()
    return qt.Qobj(U_full_array, dims=[[n_levels], [n_levels]])


def _evaluate_beta_value(
    beta: float,
    times: np.ndarray,
    t_center: float,
    amplitude: float,
    sigma: float,
    U_target: qt.Qobj,
    H_drift: qt.Qobj,
    H_x: qt.Qobj,
    H_y: qt.Qobj,
    psi0: qt.Qobj,
    n_levels: int,
) -> Tuple[float, float]:
    """Evaluate fidelity and leakage for a single beta value."""
    from qutip import mesolve

    # Generate DRAG pulse
    params = DRAGParameters(amplitude=amplitude, sigma=sigma, beta=beta)
    drag = DRAGPulse(params)
    omega_I, omega_Q = drag.envelope(times, t_center)

    # Create coefficient functions
    coeff_I, coeff_Q = _create_drag_coefficients(times, omega_I, omega_Q)

    # Evolve
    H = [H_drift, [H_x, coeff_I], [H_y, coeff_Q]]
    result = mesolve(H, psi0, times, [], [])
    psi_final = result.states[-1]

    # Compute fidelity
    U_target_full = _embed_target_unitary(U_target, n_levels)
    psi_target = U_target_full * psi0
    fidelity = qt.fidelity(psi_target, psi_final) ** 2

    # Compute leakage
    leakage = sum(np.abs(psi_final.full()[j, 0]) ** 2 for j in range(2, n_levels))

    return fidelity, leakage


def scan_beta_parameter(
    times: np.ndarray,
    t_center: float,
    amplitude: float,
    sigma: float,
    beta_range: np.ndarray,
    U_target: qt.Qobj,
    H_drift: qt.Qobj,
    n_levels: int = 3,
) -> Dict[str, np.ndarray]:
    """
    Scan β parameter to find optimal value for leakage suppression.

    Performs parameter sweep over β values and computes fidelity and leakage.
    """
    if n_levels < 3:
        raise ValueError("n_levels must be >= 3 for leakage analysis")

    fidelities = np.zeros_like(beta_range)
    leakages = np.zeros_like(beta_range)

    # Setup Hamiltonians and initial state
    H_x, H_y, psi0 = _setup_scan_hamiltonians(n_levels)

    # Scan beta values
    for i, beta in enumerate(beta_range):
        fidelities[i], leakages[i] = _evaluate_beta_value(
            beta,
            times,
            t_center,
            amplitude,
            sigma,
            U_target,
            H_drift,
            H_x,
            H_y,
            psi0,
            n_levels,
        )

    # Find optimal β
    optimal_idx = np.argmax(fidelities)
    optimal_beta = beta_range[optimal_idx]

    return {
        "beta_values": beta_range,
        "fidelities": fidelities,
        "leakages": leakages,
        "optimal_beta": optimal_beta,
        "optimal_fidelity": fidelities[optimal_idx],
        "optimal_leakage": leakages[optimal_idx],
    }
