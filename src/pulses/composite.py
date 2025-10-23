"""
Composite Pulse Module for Error-Robust Quantum Gates.

This module implements composite pulse sequences designed to cancel systematic
errors (detuning, amplitude miscalibration) to first or higher order. These
sequences improve gate robustness without requiring additional calibration.

Composite pulses work by applying a carefully designed sequence of rotations
that, when combined, produce the desired gate while canceling certain error
terms to first order in perturbation theory.

Supported Sequences:
-------------------
1. BB1 (Broadband): Cancels detuning errors to first order
2. CORPSE: Cancels both detuning and amplitude errors
3. SK1 (Solovay-Kitaev): Efficient universal gate decomposition
4. Custom sequences via optimization

Physical Background:
-------------------
In practice, quantum gates suffer from systematic errors:
- Detuning errors: ω_applied ≠ ω_qubit (calibration drift)
- Amplitude errors: Ω_applied ≠ Ω_desired (DAC errors, attenuation)
- Phase errors: φ_applied ≠ φ_desired (clock jitter)

Composite pulses construct error-robust gates by combining multiple rotations
such that errors cancel to first (or higher) order, dramatically improving
fidelity in the presence of these systematic errors.

References:
-----------
[1] Wimperis, S., "Broadband, narrowband, and passband composite pulses for
    use in advanced NMR experiments," J. Magn. Reson. A 109, 221 (1994).
[2] Cummins, H. K. et al., "Tackling systematic errors in quantum logic gates
    with composite rotations," Phys. Rev. A 67, 042308 (2003).
[3] Brown, K. R. et al., "Single-qubit-gate error below 10⁻⁴ in a trapped ion,"
    Phys. Rev. A 84, 030303(R) (2011).
[4] Kabytayev, C. et al., "Robustness of composite pulses to time-dependent
    control noise," Phys. Rev. A 90, 012316 (2014).
"""

import numpy as np
import qutip as qt
from typing import List, Tuple, Dict, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum


class RotationAxis(Enum):
    """Enumeration of rotation axes."""

    X = "X"
    Y = "Y"
    Z = "Z"


@dataclass
class PulseSegment:
    """
    Single segment of a composite pulse sequence.

    Attributes
    ----------
    axis : RotationAxis
        Rotation axis (X, Y, or Z)
    angle : float
        Rotation angle in radians
    phase : float
        Phase offset (for arbitrary axis in XY plane)
    duration : float, optional
        Pulse duration (if None, computed from angle and Rabi frequency)
    """

    axis: RotationAxis
    angle: float
    phase: float = 0.0
    duration: Optional[float] = None

    def __post_init__(self):
        """Validate parameters."""
        if not isinstance(self.axis, RotationAxis):
            if isinstance(self.axis, str):
                self.axis = RotationAxis(self.axis.upper())
            else:
                raise ValueError(f"Invalid axis: {self.axis}")


@dataclass
class CompositeSequence:
    """
    Complete composite pulse sequence.

    Attributes
    ----------
    name : str
        Name of the sequence (e.g., "BB1", "CORPSE")
    segments : List[PulseSegment]
        List of pulse segments
    target_gate : str
        Target gate this sequence implements (e.g., "X", "Y")
    error_order : int
        Order to which errors are suppressed (1 or 2)
    error_types : List[str]
        Types of errors suppressed (e.g., ["detuning", "amplitude"])
    """

    name: str
    segments: List[PulseSegment]
    target_gate: str
    error_order: int = 1
    error_types: List[str] = None

    def __post_init__(self):
        if self.error_types is None:
            self.error_types = []


class CompositePulse:
    """
    Composite pulse sequence generator and analyzer.

    This class provides methods to:
    - Generate standard composite sequences (BB1, CORPSE, SK1)
    - Analyze robustness to systematic errors
    - Optimize custom composite sequences
    - Convert sequences to time-dependent pulses

    Examples
    --------
    >>> # Create BB1 sequence for X-gate
    >>> composite = CompositePulse()
    >>> bb1_seq = composite.bb1_xgate(rabi_frequency=10.0)
    >>> print(bb1_seq.name, len(bb1_seq.segments))
    BB1-X 5

    >>> # Analyze robustness to detuning
    >>> detunings = np.linspace(-2, 2, 41)
    >>> fidelities = composite.sweep_detuning(bb1_seq, detunings)
    """

    def __init__(self, rabi_frequency: float = 1.0):
        """
        Initialize composite pulse generator.

        Parameters
        ----------
        rabi_frequency : float
            Default Rabi frequency in MHz (or rad/s depending on convention)
        """
        self.rabi_frequency = rabi_frequency

    # ==================== BB1 Sequences ====================

    def bb1_xgate(self, rabi_frequency: Optional[float] = None) -> CompositeSequence:
        """
        Generate BB1 sequence for X-gate.

        The BB1 (Broadband 1) sequence cancels detuning errors to first order:
            X_BB1(π) = X(φ) Y(π) X(2π - 2φ) Y(π) X(φ)
        where φ = arccos(-1/4) ≈ 104.5°

        Parameters
        ----------
        rabi_frequency : float, optional
            Rabi frequency for pulse duration calculation

        Returns
        -------
        CompositeSequence
            BB1 sequence for X-gate

        References
        ----------
        Wimperis, J. Magn. Reson. A 109, 221 (1994)
        """
        if rabi_frequency is None:
            rabi_frequency = self.rabi_frequency

        # BB1 angle: cos(φ) = -1/4
        phi = np.arccos(-1.0 / 4.0)

        segments = [
            PulseSegment(RotationAxis.X, phi),
            PulseSegment(RotationAxis.Y, np.pi),
            PulseSegment(RotationAxis.X, 2 * np.pi - 2 * phi),
            PulseSegment(RotationAxis.Y, np.pi),
            PulseSegment(RotationAxis.X, phi),
        ]

        return CompositeSequence(
            name="BB1-X",
            segments=segments,
            target_gate="X",
            error_order=1,
            error_types=["detuning"],
        )

    def bb1_ygate(self, rabi_frequency: Optional[float] = None) -> CompositeSequence:
        """
        Generate BB1 sequence for Y-gate.

        BB1 for Y is constructed by conjugating the X sequence.

        Parameters
        ----------
        rabi_frequency : float, optional
            Rabi frequency for pulse duration calculation

        Returns
        -------
        CompositeSequence
            BB1 sequence for Y-gate
        """
        if rabi_frequency is None:
            rabi_frequency = self.rabi_frequency

        phi = np.arccos(-1.0 / 4.0)

        segments = [
            PulseSegment(RotationAxis.Y, phi),
            PulseSegment(RotationAxis.X, np.pi),
            PulseSegment(RotationAxis.Y, 2 * np.pi - 2 * phi),
            PulseSegment(RotationAxis.X, np.pi),
            PulseSegment(RotationAxis.Y, phi),
        ]

        return CompositeSequence(
            name="BB1-Y",
            segments=segments,
            target_gate="Y",
            error_order=1,
            error_types=["detuning"],
        )

    # ==================== CORPSE Sequences ====================

    def corpse_xgate(
        self, theta: Optional[float] = None, rabi_frequency: Optional[float] = None
    ) -> CompositeSequence:
        """
        Generate CORPSE sequence for X-gate.

        CORPSE (Compensation for Off-Resonance with a Pulse SEquence) cancels
        both detuning and amplitude errors to first order:
            X_CORPSE(π) = X(θ) X̄(2θ + π) X(θ)
        where X̄ denotes rotation in opposite direction.

        Optimal θ for detuning compensation: θ = π/2

        Parameters
        ----------
        theta : float, optional
            CORPSE parameter (default π/2 for optimal detuning compensation)
        rabi_frequency : float, optional
            Rabi frequency for pulse duration calculation

        Returns
        -------
        CompositeSequence
            CORPSE sequence for X-gate

        References
        ----------
        Cummins et al., Phys. Rev. A 67, 042308 (2003)
        """
        if theta is None:
            theta = np.pi / 2.0  # Optimal for detuning
        if rabi_frequency is None:
            rabi_frequency = self.rabi_frequency

        segments = [
            PulseSegment(RotationAxis.X, theta),
            PulseSegment(
                RotationAxis.X, -(2 * theta + np.pi)
            ),  # Negative = opposite direction
            PulseSegment(RotationAxis.X, theta),
        ]

        return CompositeSequence(
            name="CORPSE-X",
            segments=segments,
            target_gate="X",
            error_order=1,
            error_types=["detuning", "amplitude"],
        )

    def corpse_ygate(
        self, theta: Optional[float] = None, rabi_frequency: Optional[float] = None
    ) -> CompositeSequence:
        """
        Generate CORPSE sequence for Y-gate.

        Parameters
        ----------
        theta : float, optional
            CORPSE parameter (default π/2)
        rabi_frequency : float, optional
            Rabi frequency for pulse duration calculation

        Returns
        -------
        CompositeSequence
            CORPSE sequence for Y-gate
        """
        if theta is None:
            theta = np.pi / 2.0
        if rabi_frequency is None:
            rabi_frequency = self.rabi_frequency

        segments = [
            PulseSegment(RotationAxis.Y, theta),
            PulseSegment(RotationAxis.Y, -(2 * theta + np.pi)),
            PulseSegment(RotationAxis.Y, theta),
        ]

        return CompositeSequence(
            name="CORPSE-Y",
            segments=segments,
            target_gate="Y",
            error_order=1,
            error_types=["detuning", "amplitude"],
        )

    def short_corpse_xgate(
        self, rabi_frequency: Optional[float] = None
    ) -> CompositeSequence:
        """
        Generate Short CORPSE (SCORPSE) sequence for X-gate.

        SCORPSE uses θ = π/3 for a more compact sequence:
            Total rotation = π + 2π/3 + 2π/3 = 7π/3 ≈ 2.33π
        Compared to BB1's total rotation of 5π.

        Parameters
        ----------
        rabi_frequency : float, optional
            Rabi frequency for pulse duration calculation

        Returns
        -------
        CompositeSequence
            SCORPSE sequence for X-gate
        """
        if rabi_frequency is None:
            rabi_frequency = self.rabi_frequency

        theta = np.pi / 3.0

        segments = [
            PulseSegment(RotationAxis.X, theta),
            PulseSegment(RotationAxis.X, -(2 * theta + np.pi)),
            PulseSegment(RotationAxis.X, theta),
        ]

        return CompositeSequence(
            name="SCORPSE-X",
            segments=segments,
            target_gate="X",
            error_order=1,
            error_types=["detuning"],
        )

    # ==================== SK1 Sequences ====================

    def sk1_sequence(
        self, target_unitary: qt.Qobj, tolerance: float = 1e-4
    ) -> CompositeSequence:
        """
        Generate Solovay-Kitaev decomposition for arbitrary single-qubit gate.

        The SK algorithm decomposes an arbitrary unitary into a sequence of
        gates from a universal set {H, T} with logarithmic overhead.

        This is a simplified implementation using Euler angle decomposition.

        Parameters
        ----------
        target_unitary : qt.Qobj
            Target 2×2 unitary matrix
        tolerance : float
            Approximation tolerance

        Returns
        -------
        CompositeSequence
            SK1 decomposition sequence

        Notes
        -----
        This implementation uses Euler angle decomposition rather than full
        Solovay-Kitaev algorithm. For production use, consider more sophisticated
        methods.
        """
        # Decompose U = R_z(α) R_y(β) R_z(γ)
        alpha, beta, gamma = self._euler_angles(target_unitary)

        segments = [
            PulseSegment(RotationAxis.Z, alpha, phase=0.0),
            PulseSegment(RotationAxis.Y, beta, phase=0.0),
            PulseSegment(RotationAxis.Z, gamma, phase=0.0),
        ]

        return CompositeSequence(
            name="SK1",
            segments=segments,
            target_gate="Custom",
            error_order=0,  # No error suppression, just decomposition
            error_types=[],
        )

    # ==================== Custom Sequences ====================

    def knill_sequence(
        self, rabi_frequency: Optional[float] = None
    ) -> CompositeSequence:
        """
        Generate Knill's error-correcting sequence.

        A 5-pulse sequence that corrects both detuning and amplitude errors
        to first order with shorter total duration than BB1.

        Parameters
        ----------
        rabi_frequency : float, optional
            Rabi frequency for pulse duration calculation

        Returns
        -------
        CompositeSequence
            Knill sequence for X-gate

        References
        ----------
        Knill, Nature 434, 39 (2005)
        """
        if rabi_frequency is None:
            rabi_frequency = self.rabi_frequency

        # Knill angles (optimized numerically)
        angles = [np.pi / 4, 3 * np.pi / 4, np.pi, 3 * np.pi / 4, np.pi / 4]
        axes = [
            RotationAxis.X,
            RotationAxis.Y,
            RotationAxis.X,
            RotationAxis.Y,
            RotationAxis.X,
        ]

        segments = [PulseSegment(axis, angle) for axis, angle in zip(axes, angles)]

        return CompositeSequence(
            name="Knill",
            segments=segments,
            target_gate="X",
            error_order=1,
            error_types=["detuning", "amplitude"],
        )

    def arbitrary_rotation(self, axis: np.ndarray, angle: float) -> CompositeSequence:
        """
        Generate sequence for arbitrary axis rotation.

        Decomposes rotation around axis n̂ by angle θ into standard gates.

        Parameters
        ----------
        axis : np.ndarray
            Rotation axis as 3D unit vector [nx, ny, nz]
        angle : float
            Rotation angle in radians

        Returns
        -------
        CompositeSequence
            Sequence implementing R_n(θ)

        Notes
        -----
        Uses decomposition: R_n(θ) = R_z(φ) R_y(θ) R_z(-φ)
        where φ = atan2(ny, nx)
        """
        # Normalize axis
        axis = np.array(axis)
        axis = axis / np.linalg.norm(axis)

        nx, ny, nz = axis

        # Decompose into ZYZ
        phi = np.arctan2(ny, nx)
        theta_eff = np.arccos(nz)

        segments = [
            PulseSegment(RotationAxis.Z, phi),
            PulseSegment(RotationAxis.Y, angle),
            PulseSegment(RotationAxis.Z, -phi),
        ]

        return CompositeSequence(
            name="ArbitraryRotation",
            segments=segments,
            target_gate="Custom",
            error_order=0,
            error_types=[],
        )

    # ==================== Analysis Methods ====================

    def simulate_sequence(
        self,
        sequence: CompositeSequence,
        detuning: float = 0.0,
        amplitude_error: float = 0.0,
        phase_error: float = 0.0,
    ) -> qt.Qobj:
        """
        Simulate composite pulse sequence with errors.

        Parameters
        ----------
        sequence : CompositeSequence
            Composite sequence to simulate
        detuning : float
            Detuning error in MHz (or rad/s)
        amplitude_error : float
            Relative amplitude error (e.g., 0.1 = 10% error)
        phase_error : float
            Phase error in radians

        Returns
        -------
        qt.Qobj
            Resulting unitary operator

        Examples
        --------
        >>> composite = CompositePulse()
        >>> bb1 = composite.bb1_xgate()
        >>> U_ideal = composite.simulate_sequence(bb1)
        >>> U_noisy = composite.simulate_sequence(bb1, detuning=1.0)
        """
        U_total = qt.qeye(2)

        for i, segment in enumerate(sequence.segments):
            # Get rotation operator for this segment
            angle_with_error = segment.angle * (1.0 + amplitude_error)
            phase_with_error = segment.phase + phase_error

            # Build rotation operator
            if segment.axis == RotationAxis.X:
                H_control = qt.sigmax()
            elif segment.axis == RotationAxis.Y:
                H_control = qt.sigmay()
            elif segment.axis == RotationAxis.Z:
                H_control = qt.sigmaz()
            else:
                raise ValueError(f"Unknown axis: {segment.axis}")

            # Duration of this segment
            if segment.duration is not None:
                dt = segment.duration
            else:
                # Duration based on angle and Rabi frequency
                dt = abs(angle_with_error) / self.rabi_frequency

            # Rotation from control pulse (angle/2 because of Pauli matrices)
            U_pulse = (-1j * angle_with_error / 2 * H_control).expm()

            # Detuning evolution during pulse (free precession)
            # Detuning acts as additional Z rotation
            U_detuning = (-1j * detuning * dt / 2 * qt.sigmaz()).expm()

            # Combined evolution: pulse and detuning
            # Order matters: detuning is always-on, pulse is applied
            U_segment = U_detuning * U_pulse

            U_total = U_segment * U_total

        return U_total

    def gate_fidelity(
        self,
        sequence: CompositeSequence,
        target_unitary: Optional[qt.Qobj] = None,
        detuning: float = 0.0,
        amplitude_error: float = 0.0,
        phase_error: float = 0.0,
    ) -> float:
        """
        Calculate gate fidelity for composite sequence with errors.

        Parameters
        ----------
        sequence : CompositeSequence
            Composite sequence to analyze
        target_unitary : qt.Qobj, optional
            Target unitary (if None, inferred from sequence.target_gate)
        detuning : float
            Detuning error in MHz
        amplitude_error : float
            Relative amplitude error
        phase_error : float
            Phase error in radians

        Returns
        -------
        float
            Average gate fidelity (0 to 1)

        Notes
        -----
        Uses average gate fidelity:
            F_avg = (|Tr(U_target† U_actual)|² + d) / (d(d+1))
        where d=2 for single-qubit gates.
        """
        # Get target unitary
        if target_unitary is None:
            target_unitary = self._standard_gate(sequence.target_gate)

        # Simulate sequence
        U_actual = self.simulate_sequence(
            sequence, detuning, amplitude_error, phase_error
        )

        # Calculate average gate fidelity
        d = 2  # Dimension for single-qubit
        overlap = (target_unitary.dag() * U_actual).tr()

        # Average gate fidelity formula
        F_avg = (np.abs(overlap) ** 2 + d) / (d * (d + 1))

        # Ensure real and in [0,1]
        F_avg = float(np.real(F_avg))
        F_avg = np.clip(F_avg, 0.0, 1.0)

        return F_avg

    def sweep_detuning(
        self,
        sequence: CompositeSequence,
        detuning_range: np.ndarray,
        target_unitary: Optional[qt.Qobj] = None,
    ) -> np.ndarray:
        """
        Sweep detuning and compute fidelity vs. detuning.

        Parameters
        ----------
        sequence : CompositeSequence
            Composite sequence to analyze
        detuning_range : np.ndarray
            Array of detuning values to test
        target_unitary : qt.Qobj, optional
            Target unitary

        Returns
        -------
        np.ndarray
            Fidelity for each detuning value
        """
        fidelities = np.zeros_like(detuning_range, dtype=float)

        for i, detuning in enumerate(detuning_range):
            fidelities[i] = self.gate_fidelity(
                sequence, target_unitary, detuning=detuning
            )

        return fidelities

    def sweep_amplitude_error(
        self,
        sequence: CompositeSequence,
        amplitude_error_range: np.ndarray,
        target_unitary: Optional[qt.Qobj] = None,
    ) -> np.ndarray:
        """
        Sweep amplitude error and compute fidelity.

        Parameters
        ----------
        sequence : CompositeSequence
            Composite sequence to analyze
        amplitude_error_range : np.ndarray
            Array of relative amplitude errors (e.g., [-0.1, 0, 0.1])
        target_unitary : qt.Qobj, optional
            Target unitary

        Returns
        -------
        np.ndarray
            Fidelity for each amplitude error value
        """
        fidelities = np.zeros_like(amplitude_error_range, dtype=float)

        for i, amp_err in enumerate(amplitude_error_range):
            fidelities[i] = self.gate_fidelity(
                sequence, target_unitary, amplitude_error=amp_err
            )

        return fidelities

    def robustness_radius(
        self,
        sequence: CompositeSequence,
        error_type: str = "detuning",
        fidelity_threshold: float = 0.99,
        max_error: float = 5.0,
        n_points: int = 100,
    ) -> float:
        """
        Compute robustness radius (maximum error maintaining fidelity threshold).

        Parameters
        ----------
        sequence : CompositeSequence
            Composite sequence to analyze
        error_type : str
            Type of error: 'detuning' or 'amplitude'
        fidelity_threshold : float
            Minimum acceptable fidelity
        max_error : float
            Maximum error value to search
        n_points : int
            Number of points in search

        Returns
        -------
        float
            Maximum error value maintaining fidelity > threshold

        Examples
        --------
        >>> composite = CompositePulse()
        >>> bb1 = composite.bb1_xgate()
        >>> radius = composite.robustness_radius(bb1, 'detuning', 0.99)
        >>> print(f"Robust to ±{radius:.2f} MHz detuning")
        """
        error_range = np.linspace(-max_error, max_error, n_points)

        if error_type == "detuning":
            fidelities = self.sweep_detuning(sequence, error_range)
        elif error_type == "amplitude":
            fidelities = self.sweep_amplitude_error(sequence, error_range)
        else:
            raise ValueError(f"Unknown error type: {error_type}")

        # Find points where fidelity exceeds threshold
        above_threshold = fidelities >= fidelity_threshold

        if not np.any(above_threshold):
            return 0.0

        # Find largest error magnitude with acceptable fidelity
        indices = np.where(above_threshold)[0]
        max_idx = np.max(indices)
        min_idx = np.min(indices)

        radius = min(abs(error_range[max_idx]), abs(error_range[min_idx]))

        return float(radius)

    def compare_sequences(
        self,
        sequences: List[CompositeSequence],
        error_type: str = "detuning",
        error_range: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Compare multiple composite sequences.

        Parameters
        ----------
        sequences : List[CompositeSequence]
            List of sequences to compare
        error_type : str
            Type of error to sweep: 'detuning' or 'amplitude'
        error_range : np.ndarray, optional
            Range of error values (default: -2 to 2 MHz or -0.2 to 0.2)

        Returns
        -------
        dict
            Dictionary mapping sequence names to fidelity arrays
        """
        if error_range is None:
            if error_type == "detuning":
                error_range = np.linspace(-2.0, 2.0, 41)
            else:
                error_range = np.linspace(-0.2, 0.2, 41)

        results = {}

        for seq in sequences:
            if error_type == "detuning":
                fidelities = self.sweep_detuning(seq, error_range)
            else:
                fidelities = self.sweep_amplitude_error(seq, error_range)

            results[seq.name] = fidelities

        return results

    # ==================== Utility Methods ====================

    def total_duration(self, sequence: CompositeSequence) -> float:
        """
        Calculate total duration of composite sequence.

        Parameters
        ----------
        sequence : CompositeSequence
            Composite sequence

        Returns
        -------
        float
            Total duration in units of 1/Ω (or ns if Ω in MHz)
        """
        total_time = 0.0

        for segment in sequence.segments:
            if segment.duration is not None:
                total_time += segment.duration
            else:
                total_time += abs(segment.angle) / self.rabi_frequency

        return total_time

    def total_rotation_angle(self, sequence: CompositeSequence) -> float:
        """
        Calculate total rotation angle (sum of absolute values).

        Parameters
        ----------
        sequence : CompositeSequence
            Composite sequence

        Returns
        -------
        float
            Sum of |θ_i| for all segments
        """
        return sum(abs(seg.angle) for seg in sequence.segments)

    def _standard_gate(self, gate_name: str) -> qt.Qobj:
        """Get standard gate unitary by name."""
        gates = {
            "X": qt.sigmax(),
            "Y": qt.sigmay(),
            "Z": qt.sigmaz(),
            "H": (qt.sigmax() + qt.sigmaz()) / np.sqrt(2),  # Hadamard gate
            "I": qt.qeye(2),
        }

        if gate_name.upper() not in gates:
            raise ValueError(f"Unknown gate: {gate_name}")

        return gates[gate_name.upper()]

    def _euler_angles(self, U: qt.Qobj) -> Tuple[float, float, float]:
        """
        Decompose single-qubit unitary into Euler angles (ZYZ convention).

        Parameters
        ----------
        U : qt.Qobj
            2×2 unitary matrix

        Returns
        -------
        alpha, beta, gamma : float
            Euler angles such that U = R_z(α) R_y(β) R_z(γ)
        """
        U_arr = U.full()

        # Extract angles from matrix elements
        # U = [[U00, U01], [U10, U11]]
        U00 = U_arr[0, 0]
        U01 = U_arr[0, 1]
        U10 = U_arr[1, 0]
        U11 = U_arr[1, 1]

        # beta from |U01|
        beta = 2 * np.arccos(min(1.0, abs(U00)))

        if abs(np.sin(beta / 2)) < 1e-10:
            # Special case: beta = 0 or 2π
            alpha = 0.0
            gamma = 2 * np.angle(U00)
        else:
            # General case
            alpha = np.angle(-U01) + np.angle(U00)
            gamma = np.angle(U10) + np.angle(U00)

        return alpha, beta, gamma

    def sequence_to_pulses(
        self,
        sequence: CompositeSequence,
        times: np.ndarray,
        pulse_shape: str = "square",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert composite sequence to time-dependent pulse envelopes.

        Parameters
        ----------
        sequence : CompositeSequence
            Composite sequence to convert
        times : np.ndarray
            Time array for pulse definition
        pulse_shape : str
            Pulse shape for each segment: 'square' or 'gaussian'

        Returns
        -------
        omega_x : np.ndarray
            X-axis control amplitude vs time
        omega_y : np.ndarray
            Y-axis control amplitude vs time

        Notes
        -----
        Z rotations are implemented as frame changes, not actual pulses.
        """
        omega_x = np.zeros_like(times)
        omega_y = np.zeros_like(times)

        t_current = times[0]

        for segment in sequence.segments:
            if segment.duration is not None:
                duration = segment.duration
            else:
                duration = abs(segment.angle) / self.rabi_frequency

            t_end = t_current + duration
            mask = (times >= t_current) & (times < t_end)

            if segment.axis == RotationAxis.X:
                omega_x[mask] = np.sign(segment.angle) * self.rabi_frequency
            elif segment.axis == RotationAxis.Y:
                omega_y[mask] = np.sign(segment.angle) * self.rabi_frequency
            # Z rotations are frame changes, no actual pulse

            t_current = t_end

        return omega_x, omega_y
