"""
Control Hamiltonian for time-dependent quantum driving.

This module implements the control Hamiltonian H_c(t) which represents
externally applied electromagnetic fields that drive transitions between
qubit states. Combined with the drift Hamiltonian H_0, the total Hamiltonian is:
    H_total(t) = H_0 + H_c(t)

Physical Context:
-----------------
In quantum control experiments, we apply resonant or near-resonant microwave
pulses to manipulate qubit states. The control Hamiltonian in the lab frame is:
    H_c(t) = Ω(t) cos(ω_d*t + φ) σ_x

Where:
- Ω(t) is the pulse envelope (time-dependent Rabi frequency)
- ω_d is the drive frequency
- φ is the pulse phase
- σ_x is the Pauli-X operator (transverse driving)

Rotating Frame Transformation:
------------------------------
For near-resonant driving (ω_d ≈ ω_0), it's convenient to work in the rotating
frame where fast-oscillating terms are removed. The rotating-wave approximation
(RWA) gives:
    H_RWA(t) = Δ/2 σ_z + Ω(t)/2 (cos(φ) σ_x + sin(φ) σ_y)

Where Δ = ω_0 - ω_d is the detuning.

For on-resonance driving (Δ = 0):
    H_RWA(t) = Ω(t)/2 σ_x  (for φ = 0)
             = Ω(t)/2 σ_y  (for φ = π/2)

Rabi Oscillations:
------------------
Under constant driving (Ω(t) = Ω_0), the state oscillates between |0⟩ and |1⟩
at the Rabi frequency Ω_0. For a π-pulse (|0⟩ → |1⟩):
    duration = π / Ω_0

References:
-----------
- Nielsen & Chuang, "Quantum Computation and Quantum Information" (2010)
- Slichter, "Principles of Magnetic Resonance" (1990)
- Wiseman & Milburn, "Quantum Measurement and Control" (2010)
"""

import numpy as np
import qutip as qt
from typing import Union, Callable, Optional, Tuple

# Power of 10 compliance: Import assertion helpers
from ..constants import (
    MIN_PULSE_AMPLITUDE,
    MAX_PULSE_AMPLITUDE,
    MIN_ENERGY,
    MAX_ENERGY,
)


class ControlHamiltonian:
    """
    Represents the control Hamiltonian H_c(t) for time-dependent qubit driving.

    This class handles both lab-frame and rotating-frame representations,
    supports arbitrary pulse shapes, and provides utilities for common
    gate operations (X, Y, Hadamard gates via shaped pulses).

    Attributes
    ----------
    pulse_func : callable
        Function Ω(t) that returns the pulse amplitude at time t.
    drive_axis : str
        Control axis: 'x', 'y', or 'xy' (for DRAG-style pulses).
    phase : float
        Pulse phase in radians (for rotating frame).
    detuning : float
        Detuning Δ = ω_0 - ω_d from resonance.
    rotating_frame : bool
        If True, use rotating-wave approximation.
    """

    @staticmethod
    def _validate_pulse_func(pulse_func) -> None:
        """Validate pulse function."""
        if pulse_func is None:
            raise ValueError("pulse_func cannot be None")
        if not callable(pulse_func):
            raise TypeError(f"pulse_func must be callable, got {type(pulse_func)}")

        # Test pulse function with a sample value
        try:
            test_val = pulse_func(0.0)
            assert (
                np.isfinite(test_val)
                if np.isscalar(test_val)
                else np.all(np.isfinite(test_val))
            ), "pulse_func must return finite values"
        except Exception as e:
            raise ValueError(f"pulse_func failed at t=0: {e}")

    @staticmethod
    def _validate_drive_axis(drive_axis: str) -> None:
        """Validate drive axis parameter."""
        if drive_axis is None:
            raise ValueError("drive_axis cannot be None")
        if not isinstance(drive_axis, str):
            raise TypeError(f"drive_axis must be string, got {type(drive_axis)}")
        if drive_axis.lower() not in ["x", "y", "xy"]:
            raise ValueError(
                f"Invalid drive_axis '{drive_axis}'. Must be 'x', 'y', or 'xy'."
            )

    @staticmethod
    def _validate_phase(phase: float) -> None:
        """Validate phase parameter."""
        if not isinstance(phase, (int, float)):
            raise TypeError(f"phase must be numeric, got {type(phase)}")
        if not np.isfinite(phase):
            raise ValueError(f"phase must be finite, got {phase}")
        if not (-2 * np.pi <= phase <= 2 * np.pi):
            raise ValueError(f"phase {phase} outside reasonable range [-2π, 2π]")

    @staticmethod
    def _validate_detuning(detuning: float) -> None:
        """Validate detuning parameter."""
        if not isinstance(detuning, (int, float)):
            raise TypeError(f"detuning must be numeric, got {type(detuning)}")
        if not np.isfinite(detuning):
            raise ValueError(f"detuning must be finite, got {detuning}")
        if not (MIN_ENERGY <= detuning <= MAX_ENERGY):
            raise ValueError(
                f"detuning {detuning} outside reasonable bounds [{MIN_ENERGY}, {MAX_ENERGY}]"
            )

    @staticmethod
    def _validate_rotating_frame(rotating_frame: bool) -> None:
        """Validate rotating_frame parameter."""
        if not isinstance(rotating_frame, bool):
            raise TypeError(f"rotating_frame must be bool, got {type(rotating_frame)}")

    def __init__(
        self,
        pulse_func: Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]],
        drive_axis: str = "x",
        phase: float = 0.0,
        detuning: float = 0.0,
        rotating_frame: bool = True,
    ):
        """
        Initialize control Hamiltonian.

        Validates parameters and sets up Pauli operator cache for efficient
        time-dependent Hamiltonian evaluation.
        """
        # Validate all parameters
        self._validate_pulse_func(pulse_func)
        self._validate_drive_axis(drive_axis)
        self._validate_phase(phase)
        self._validate_detuning(detuning)
        self._validate_rotating_frame(rotating_frame)

        # Set attributes
        self.pulse_func = pulse_func
        self.drive_axis = drive_axis.lower()
        self.phase = phase
        self.detuning = detuning
        self.rotating_frame = rotating_frame

        # Cache Pauli operators
        self._sigma_x = qt.sigmax()
        self._sigma_y = qt.sigmay()
        self._sigma_z = qt.sigmaz()

        # Rule 5: Post-initialization invariant checks
        assert self._sigma_x.isherm, "sigma_x must be Hermitian"
        assert self._sigma_y.isherm, "sigma_y must be Hermitian"
        assert self._sigma_z.isherm, "sigma_z must be Hermitian"

    def _validate_time_parameter(self, t: float) -> None:
        """
        Validate time parameter.

        Parameters
        ----------
        t : float
            Time value to validate

        Raises
        ------
        ValueError
            If time is invalid
        """
        if t is None:
            raise ValueError("Time t cannot be None")
        if not isinstance(t, (int, float)):
            raise ValueError(f"Time must be numeric, got {type(t)}")
        if not np.isfinite(t):
            raise ValueError(f"Time must be finite, got {t}")
        if t < 0:
            raise ValueError(f"Time must be non-negative, got {t}")

    def _build_hamiltonian_operator(self, amplitude: float) -> qt.Qobj:
        """
        Build control Hamiltonian operator from amplitude.

        Parameters
        ----------
        amplitude : float
            Pulse amplitude

        Returns
        -------
        qt.Qobj
            Control Hamiltonian operator
        """
        # Apply phase rotation: H = Ω(t)/2 * [cos(φ)σ_x + sin(φ)σ_y]
        if self.drive_axis == "x":
            # X-axis drive with phase rotation
            H_c = (
                0.5
                * amplitude
                * (
                    np.cos(self.phase) * self._sigma_x
                    + np.sin(self.phase) * self._sigma_y
                )
            )
        elif self.drive_axis == "y":
            # Y-axis drive with phase rotation (phase shifted by π/2)
            H_c = (
                0.5
                * amplitude
                * (
                    -np.sin(self.phase) * self._sigma_x
                    + np.cos(self.phase) * self._sigma_y
                )
            )
        else:  # 'xy'
            # XY drive with phase rotation
            H_c = (
                0.5
                * amplitude
                * (
                    np.cos(self.phase) * self._sigma_x
                    + np.sin(self.phase) * self._sigma_y
                )
            )
        return H_c

    def hamiltonian(self, t: float) -> qt.Qobj:
        """
        Return control Hamiltonian H_c(t) at time t.

        For rotating frame (default):
            H_c(t) = Ω(t)/2 * (cos(φ) σ_x + sin(φ) σ_y)

        For lab frame:
            H_c(t) = Ω(t) * cos(ω_d*t + φ) σ_x

        Parameters
        ----------
        t : float
            Time at which to evaluate Hamiltonian.

        Returns
        -------
        qutip.Qobj
            Control Hamiltonian operator.
        """
        # Validate time parameter
        self._validate_time_parameter(t)

        # Evaluate pulse amplitude
        amplitude = self.pulse_func(t)
        assert np.isfinite(amplitude), (
            f"Pulse amplitude not finite at t={t}: {amplitude}"
        )

        # Build Hamiltonian operator
        H_c = self._build_hamiltonian_operator(amplitude)

        # Validate output
        assert H_c is not None, "Hamiltonian construction failed"
        assert isinstance(H_c, qt.Qobj), "H_c must be Qobj"
        assert H_c.isherm, f"Control Hamiltonian must be Hermitian at t={t}"

        return H_c

    def get_operator_at_time(self, t: float) -> qt.Qobj:
        """
        Alias for hamiltonian() method for backwards compatibility.

        Parameters
        ----------
        t : float
            Time at which to evaluate.

        Returns
        -------
        qt.Qobj
            Control Hamiltonian at time t.
        """
        return self.hamiltonian(t)

    def pulse_area(self, t_start: float, t_end: float, num_points: int = 1000) -> float:
        """
        Calculate integrated pulse area (useful for rotation angle).

        Parameters
        ----------
        t_start : float
            Start time for integration.
        t_end : float
            End time for integration.
        num_points : int
            Number of points for numerical integration.

        Returns
        -------
        float
            Integrated pulse area.
        """
        # Rule 5: Parameter validation
        if t_start < 0:
            raise ValueError(f"t_start must be non-negative, got {t_start}")
        if t_end <= t_start:
            raise ValueError(f"t_end {t_end} must be > t_start {t_start}")
        if num_points <= 0:
            raise ValueError(f"num_points must be positive, got {num_points}")

        times = np.linspace(t_start, t_end, num_points)
        amplitudes = np.array([self.pulse_func(t) for t in times])
        area = np.trapz(amplitudes, times)

        # Rule 5: Validate result
        assert np.isfinite(area), f"Pulse area not finite: {area}"

        return area

    def hamiltonian_coeff_form(self) -> list:
        """
        Return control Hamiltonian in QuTiP time-dependent coefficient format.

        This format is used by QuTiP's sesolve for efficient time-dependent
        Hamiltonian evolution: H(t) = H_0 + Σ f_i(t) * H_i

        Returns
        -------
        list
            List of [operator, coefficient_function] pairs for QuTiP.
            Format: [[σ_x, f_x(t, args)], [σ_y, f_y(t, args)]]

        Examples
        --------
        >>> H_ctrl = ControlHamiltonian(pulse_func)
        >>> H0 = drift_hamiltonian()
        >>> H_total = [H0] + H_ctrl.hamiltonian_coeff_form()
        >>> result = qt.sesolve(H_total, psi0, times)
        """
        if self.drive_axis == "x":
            coeff_x = lambda t, args: self.pulse_func(t) * np.cos(self.phase) / 2.0
            coeff_y = lambda t, args: self.pulse_func(t) * np.sin(self.phase) / 2.0
            return [[self._sigma_x, coeff_x], [self._sigma_y, coeff_y]]
        elif self.drive_axis == "y":
            coeff_x = lambda t, args: -self.pulse_func(t) * np.sin(self.phase) / 2.0
            coeff_y = lambda t, args: self.pulse_func(t) * np.cos(self.phase) / 2.0
            return [[self._sigma_x, coeff_x], [self._sigma_y, coeff_y]]
        else:  # 'xy' for DRAG
            # Assume pulse_func returns (omega_x, omega_y)
            coeff_x = lambda t, args: self.pulse_func(t)[0] / 2.0
            coeff_y = lambda t, args: self.pulse_func(t)[1] / 2.0
            return [[self._sigma_x, coeff_x], [self._sigma_y, coeff_y]]

    def evolve_state(
        self,
        psi0: qt.Qobj,
        times: np.ndarray,
        H_drift: Optional[qt.Qobj] = None,
    ) -> qt.solver.Result:
        """
        Evolve state under control Hamiltonian (and optional drift).

        Solves the Schrödinger equation:
            iℏ dψ/dt = (H_drift + H_control(t)) ψ

        Parameters
        ----------
        psi0 : qutip.Qobj
            Initial state |ψ(0)⟩.
        times : np.ndarray
            Array of times at which to compute the state.
        H_drift : qutip.Qobj, optional
            Static drift Hamiltonian H_0 (e.g., detuning term).
            If None, only control Hamiltonian is applied.

        Returns
        -------
        qutip.solver.Result
            Result object with .states (list of states) and .times.

        Examples
        --------
        >>> psi0 = qt.basis(2, 0)  # |0⟩
        >>> times = np.linspace(0, 100, 1000)
        >>> H_ctrl = ControlHamiltonian(pulse_func)
        >>> result = H_ctrl.evolve_state(psi0, times)
        >>> psi_final = result.states[-1]
        """
        # Build time-dependent Hamiltonian
        if H_drift is not None:
            # Drift + control
            if self.detuning != 0.0:
                # Add detuning term to drift
                H_drift_total = H_drift + (self.detuning / 2.0) * self._sigma_z
            else:
                H_drift_total = H_drift

            H_total = [H_drift_total] + self.hamiltonian_coeff_form()
        else:
            # Control only
            if self.detuning != 0.0:
                H_drift_detuning = (self.detuning / 2.0) * self._sigma_z
                H_total = [H_drift_detuning] + self.hamiltonian_coeff_form()
            else:
                H_total = self.hamiltonian_coeff_form()

        # Evolve using QuTiP
        result = qt.sesolve(H_total, psi0, times)
        return result

    def rabi_frequency(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Return effective Rabi frequency Ω_eff(t) including detuning.

        For detuned driving:
            Ω_eff = √(Ω² + Δ²)

        Parameters
        ----------
        t : float or np.ndarray
            Time(s) at which to evaluate Rabi frequency.

        Returns
        -------
        float or np.ndarray
            Effective Rabi frequency.
        """
        omega = self.pulse_func(t)
        if self.detuning == 0.0:
            return np.abs(omega)
        else:
            return np.sqrt(omega**2 + self.detuning**2)

    def gate_fidelity(
        self,
        psi0: qt.Qobj,
        psi_target: qt.Qobj,
        times: np.ndarray,
        H_drift: Optional[qt.Qobj] = None,
    ) -> float:
        """
        Compute gate fidelity F = |⟨ψ_target|ψ_final⟩|².

        Evolves psi0 under the control Hamiltonian and compares final state
        to the target state.

        Parameters
        ----------
        psi0 : qutip.Qobj
            Initial state.
        psi_target : qutip.Qobj
            Target state (desired outcome).
        times : np.ndarray
            Time evolution array (final time determines gate duration).
        H_drift : qutip.Qobj, optional
            Drift Hamiltonian (default None).

        Returns
        -------
        float
            Fidelity F ∈ [0, 1], where F=1 means perfect gate.

        Examples
        --------
        >>> psi0 = qt.basis(2, 0)
        >>> psi_target = qt.basis(2, 1)  # X-gate target
        >>> fidelity = H_ctrl.gate_fidelity(psi0, psi_target, times)
        >>> print(f"Gate fidelity: {fidelity:.6f}")
        """
        result = self.evolve_state(psi0, times, H_drift)
        psi_final = result.states[-1]
        return qt.fidelity(psi_final, psi_target) ** 2

    @staticmethod
    def pi_pulse_duration(rabi_frequency: float) -> float:
        """
        Calculate duration required for a π-pulse (X-gate).

        For constant driving at Rabi frequency Ω, the π-pulse duration is:
            T_π = π / Ω

        Parameters
        ----------
        rabi_frequency : float
            Constant Rabi frequency Ω (rad/s or 2π×Hz).

        Returns
        -------
        float
            Duration for π-pulse (same units as 1/rabi_frequency).

        Examples
        --------
        >>> omega = 2*np.pi*10  # 10 MHz
        >>> T_pi = ControlHamiltonian.pi_pulse_duration(omega)
        >>> # T_pi = 50 ns for 10 MHz driving
        """
        return np.pi / rabi_frequency

    @staticmethod
    def pi_half_pulse_duration(rabi_frequency: float) -> float:
        """
        Calculate duration required for a π/2-pulse (Hadamard-like gate).

        For constant driving:
            T_π/2 = π / (2Ω)

        Parameters
        ----------
        rabi_frequency : float
            Constant Rabi frequency Ω.

        Returns
        -------
        float
            Duration for π/2-pulse.
        """
        return np.pi / (2.0 * rabi_frequency)

    def __repr__(self) -> str:
        """String representation of ControlHamiltonian."""
        return (
            f"ControlHamiltonian(drive_axis='{self.drive_axis}', "
            f"phase={self.phase:.3f}, detuning={self.detuning:.3f}, "
            f"rotating_frame={self.rotating_frame})"
        )
