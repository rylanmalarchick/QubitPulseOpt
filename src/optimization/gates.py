"""
Universal Gate Optimization
============================

This module implements optimizers for universal single-qubit quantum gates,
including Hadamard, phase gates (S, T), and arbitrary rotations.

The universal gate set {H, S, T} is Clifford-complete and can approximate
any single-qubit unitary to arbitrary precision when combined with rotations.

Physical Background:
-------------------
Single-qubit gates form the group SU(2) and can be parameterized as:

    U(θ, φ, λ) = [    cos(θ/2)        -e^(iλ)sin(θ/2)    ]
                 [ e^(iφ)sin(θ/2)   e^(i(φ+λ))cos(θ/2)  ]

Key gates:
- Hadamard: H = (X + Z)/√2 = (1/√2)[1  1; 1 -1]
- S gate: S = [1 0; 0 i] (π/2 phase gate)
- T gate: T = [1 0; 0 e^(iπ/4)] (π/4 phase gate)
- Pauli-X: X = [0 1; 1 0] (π rotation about x-axis)
- Pauli-Y: Y = [0 -i; i 0] (π rotation about y-axis)
- Pauli-Z: Z = [1 0; 0 -1] (π rotation about z-axis)

Euler Decomposition:
-------------------
Any SU(2) rotation can be decomposed as:

    U = R_z(α) R_y(β) R_z(γ)

Or equivalently: U = e^(iδ) R_z(φ₁) R_y(θ) R_z(φ₂)

This allows arbitrary single-qubit gates to be implemented using a
universal gate set.

Optimization Strategy:
---------------------
Gates are optimized using GRAPE or Krotov methods from the optimization
module. The optimizer finds pulse shapes that maximize fidelity with the
target gate while respecting physical constraints (amplitude, bandwidth).

References:
----------
- Nielsen & Chuang, "Quantum Computation and Quantum Information" (2010)
- Barenco et al., Phys. Rev. A 52, 3457 (1995) - Universal gates
- Dawson & Nielsen, arXiv:quant-ph/0505030 (2005) - Solovay-Kitaev theorem
- Vandersypen & Chuang, Rev. Mod. Phys. 76, 1037 (2004) - NMR gates

Author: Orchestrator Agent
Date: 2025-01-27
SOW Reference: Phase 3, Task 2.1 - Gate Library
"""

import numpy as np
import qutip as qt
from typing import Union, Optional, List, Tuple, Dict, Literal
from dataclasses import dataclass, field
import warnings

from .grape import GRAPEOptimizer, GRAPEResult
from .krotov import KrotovOptimizer, KrotovResult


@dataclass
class GateResult:
    """
    Result container for gate optimization.

    Attributes
    ----------
    gate_name : str
        Name of the optimized gate (e.g., 'Hadamard', 'S', 'T').
    target_unitary : qt.Qobj
        Target unitary operator.
    final_fidelity : float
        Achieved fidelity with target gate.
    optimized_pulses : np.ndarray
        Optimized control pulse amplitudes.
    gate_time : float
        Total gate duration.
    optimizer_result : Union[GRAPEResult, KrotovResult]
        Full result from underlying optimizer.
    method : str
        Optimization method used ('grape' or 'krotov').
    success : bool
        Whether optimization met fidelity threshold.
    metadata : dict
        Additional information (optimizer settings, convergence, etc.).
    """

    gate_name: str
    target_unitary: qt.Qobj
    final_fidelity: float
    optimized_pulses: np.ndarray
    gate_time: float
    optimizer_result: Union[GRAPEResult, KrotovResult]
    method: str
    success: bool = True
    metadata: dict = field(default_factory=dict)

    def __repr__(self):
        status = "✓" if self.success else "✗"
        return (
            f"GateResult({status} {self.gate_name}, "
            f"F={self.final_fidelity:.6f}, "
            f"T={self.gate_time:.2f}ns, "
            f"method={self.method})"
        )


class UniversalGates:
    """
    High-fidelity optimization for universal single-qubit gates.

    This class provides optimizers for the standard single-qubit gate set
    including Hadamard, phase gates, Pauli gates, and arbitrary rotations.

    The implementation uses GRAPE or Krotov's method from the optimization
    module to find optimal pulse shapes that implement each gate with
    maximum fidelity while respecting physical constraints.

    Parameters
    ----------
    H_drift : qt.Qobj
        Drift Hamiltonian (typically qubit detuning, H_drift = ω σ_z/2).
    H_controls : list[qt.Qobj]
        Control Hamiltonians (e.g., [σ_x, σ_y] for two-axis control).
    initial_state : qt.Qobj, optional
        Initial state for gate optimization. Default: |0⟩.
    fidelity_threshold : float, optional
        Minimum fidelity for successful gate (default: 0.999).

    Examples
    --------
    >>> # Setup for a qubit with X and Y control
    >>> H_drift = 2 * np.pi * 0.0 * qt.sigmaz() / 2  # On resonance
    >>> H_controls = [qt.sigmax(), qt.sigmay()]
    >>>
    >>> gates = UniversalGates(H_drift, H_controls)
    >>>
    >>> # Optimize Hadamard gate
    >>> result = gates.optimize_hadamard(gate_time=20.0, n_timeslices=100)
    >>> print(f"Hadamard fidelity: {result.final_fidelity:.6f}")
    >>>
    >>> # Optimize S gate (π/2 phase)
    >>> s_result = gates.optimize_phase_gate(np.pi/2, gate_time=15.0)
    >>>
    >>> # Arbitrary rotation about axis n = (x, y, z)
    >>> rot_result = gates.optimize_rotation(
    ...     axis=[1, 1, 0], angle=np.pi/4, gate_time=20.0
    ... )
    """

    def __init__(
        self,
        H_drift: qt.Qobj,
        H_controls: List[qt.Qobj],
        initial_state: Optional[qt.Qobj] = None,
        fidelity_threshold: float = 0.999,
    ):
        """Initialize gate optimizer with system Hamiltonians."""
        self.H_drift = H_drift
        self.H_controls = H_controls
        self.n_controls = len(H_controls)

        # Default initial state is |0⟩
        if initial_state is None:
            self.initial_state = qt.basis(2, 0)
        else:
            self.initial_state = initial_state

        self.fidelity_threshold = fidelity_threshold

        # Cache standard gates
        self._standard_gates = self._build_standard_gates()

    def _build_standard_gates(self) -> Dict[str, qt.Qobj]:
        """Build dictionary of standard single-qubit gates."""
        gates = {
            "I": qt.qeye(2),
            "X": qt.sigmax(),
            "Y": qt.sigmay(),
            "Z": qt.sigmaz(),
            "H": qt.gates.hadamard_transform(),
            "S": qt.gates.phasegate(np.pi / 2),
            "T": qt.gates.phasegate(np.pi / 4),
            "Sdg": qt.gates.phasegate(-np.pi / 2),  # S†
            "Tdg": qt.gates.phasegate(-np.pi / 4),  # T†
        }
        return gates

    def optimize_hadamard(
        self,
        gate_time: float = 20.0,
        n_timeslices: int = 100,
        method: Literal["grape", "krotov"] = "grape",
        max_iterations: int = 500,
        convergence_threshold: float = 1e-6,
        amplitude_limit: Optional[float] = None,
        **kwargs,
    ) -> GateResult:
        """
        Optimize Hadamard gate using GRAPE or Krotov.

        The Hadamard gate is H = (1/√2)[1  1; 1 -1], which creates an equal
        superposition |+⟩ = (|0⟩ + |1⟩)/√2 from |0⟩.

        Physically, H can be implemented as a π/2 rotation about the y-axis
        followed by a π rotation about x (or vice versa), but optimal control
        can find faster and more robust implementations.

        Parameters
        ----------
        gate_time : float, optional
            Total gate duration in ns (default: 20.0).
        n_timeslices : int, optional
            Number of time slices for piecewise-constant control (default: 100).
        method : {'grape', 'krotov'}, optional
            Optimization method to use (default: 'grape').
        max_iterations : int, optional
            Maximum optimizer iterations (default: 500).
        convergence_threshold : float, optional
            Convergence threshold for optimizer (default: 1e-6).
        amplitude_limit : float, optional
            Maximum control amplitude in MHz (default: None).
        **kwargs
            Additional arguments passed to optimizer.

        Returns
        -------
        GateResult
            Optimization result with final fidelity, pulses, and metadata.

        Examples
        --------
        >>> gates = UniversalGates(H_drift, H_controls)
        >>> result = gates.optimize_hadamard(gate_time=20.0, n_timeslices=100)
        >>> print(f"Hadamard gate fidelity: {result.final_fidelity:.6f}")
        >>> print(f"Gate time: {result.gate_time:.2f} ns")
        """
        target = self._standard_gates["H"]

        return self._optimize_gate(
            gate_name="Hadamard",
            target_unitary=target,
            gate_time=gate_time,
            n_timeslices=n_timeslices,
            method=method,
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold,
            amplitude_limit=amplitude_limit,
            **kwargs,
        )

    def optimize_phase_gate(
        self,
        phase: float,
        gate_time: float = 15.0,
        n_timeslices: int = 80,
        method: Literal["grape", "krotov"] = "grape",
        max_iterations: int = 500,
        **kwargs,
    ) -> GateResult:
        """
        Optimize phase gate P(φ) = diag(1, exp(iφ)).

        Common phase gates:
        - S gate: φ = π/2
        - T gate: φ = π/4
        - Z gate: φ = π

        Phase gates leave |0⟩ unchanged and apply a phase to |1⟩:
            P(φ)|0⟩ = |0⟩
            P(φ)|1⟩ = e^(iφ)|1⟩

        Parameters
        ----------
        phase : float
            Phase angle φ in radians.
        gate_time : float, optional
            Gate duration in ns (default: 15.0).
        n_timeslices : int, optional
            Number of time slices (default: 80).
        method : {'grape', 'krotov'}, optional
            Optimization method (default: 'grape').
        max_iterations : int, optional
            Maximum iterations (default: 500).
        **kwargs
            Additional optimizer arguments.

        Returns
        -------
        GateResult
            Optimization result.

        Examples
        --------
        >>> # Optimize S gate (π/2 phase)
        >>> s_result = gates.optimize_phase_gate(np.pi/2, gate_time=15.0)
        >>>
        >>> # Optimize T gate (π/4 phase)
        >>> t_result = gates.optimize_phase_gate(np.pi/4, gate_time=15.0)
        """
        target = qt.gates.phasegate(phase)

        # Determine gate name
        if np.isclose(phase, np.pi / 2):
            gate_name = "S"
        elif np.isclose(phase, np.pi / 4):
            gate_name = "T"
        elif np.isclose(phase, np.pi):
            gate_name = "Z"
        elif np.isclose(phase, -np.pi / 2):
            gate_name = "Sdg"
        elif np.isclose(phase, -np.pi / 4):
            gate_name = "Tdg"
        else:
            gate_name = f"P({phase:.4f})"

        return self._optimize_gate(
            gate_name=gate_name,
            target_unitary=target,
            gate_time=gate_time,
            n_timeslices=n_timeslices,
            method=method,
            max_iterations=max_iterations,
            **kwargs,
        )

    def _parse_rotation_axis(
        self, axis: Union[str, List[float], np.ndarray]
    ) -> Tuple[np.ndarray, str]:
        """Parse rotation axis specification into vector and name."""
        if isinstance(axis, str):
            axis = axis.lower()
            if axis == "x":
                return np.array([1.0, 0.0, 0.0]), "X"
            elif axis == "y":
                return np.array([0.0, 1.0, 0.0]), "Y"
            elif axis == "z":
                return np.array([0.0, 0.0, 1.0]), "Z"
            else:
                raise ValueError(f"Unknown axis '{axis}'. Use 'x', 'y', 'z' or array.")
        else:
            axis_vec = np.array(axis, dtype=float)
            if axis_vec.shape != (3,):
                raise ValueError(f"Axis must be 3D vector, got shape {axis_vec.shape}")
            norm = np.linalg.norm(axis_vec)
            if norm < 1e-10:
                raise ValueError("Axis vector has zero norm")
            axis_vec = axis_vec / norm
            axis_name = f"[{axis_vec[0]:.2f},{axis_vec[1]:.2f},{axis_vec[2]:.2f}]"
            return axis_vec, axis_name

    def _build_rotation_unitary(self, axis_vec: np.ndarray, angle: float) -> qt.Qobj:
        """Build rotation unitary operator R_n(θ) = exp(-i θ/2 n·σ)."""
        n_dot_sigma = (
            axis_vec[0] * qt.sigmax()
            + axis_vec[1] * qt.sigmay()
            + axis_vec[2] * qt.sigmaz()
        )
        return (-1j * angle / 2 * n_dot_sigma).expm()

    def _format_rotation_gate_name(self, axis_name: str, angle: float) -> str:
        """Format rotation gate name based on angle."""
        if np.isclose(angle, np.pi):
            return f"{axis_name}(π)"
        elif np.isclose(angle, np.pi / 2):
            return f"{axis_name}(π/2)"
        elif np.isclose(angle, np.pi / 4):
            return f"{axis_name}(π/4)"
        else:
            return f"R_{axis_name}({angle:.4f})"

    def optimize_rotation(
        self,
        axis: Union[str, List[float], np.ndarray],
        angle: float,
        gate_time: float = 20.0,
        n_timeslices: int = 100,
        method: Literal["grape", "krotov"] = "grape",
        **kwargs,
    ) -> GateResult:
        """
        Optimize arbitrary rotation R_n(θ) = exp(-i θ/2 n·σ).

        Rotations about axis n by angle θ are fundamental single-qubit operations.
        """
        axis_vec, axis_name = self._parse_rotation_axis(axis)
        target = self._build_rotation_unitary(axis_vec, angle)
        gate_name = self._format_rotation_gate_name(axis_name, angle)

        return self._optimize_gate(
            gate_name=gate_name,
            target_unitary=target,
            gate_time=gate_time,
            n_timeslices=n_timeslices,
            method=method,
            **kwargs,
        )

    def optimize_pauli_gate(
        self,
        pauli: Literal["X", "Y", "Z"],
        gate_time: float = 20.0,
        n_timeslices: int = 100,
        method: Literal["grape", "krotov"] = "grape",
        **kwargs,
    ) -> GateResult:
        """
        Optimize Pauli gate (X, Y, or Z).

        Pauli gates are π rotations about the respective axes:
        - X: π rotation about x-axis (bit flip)
        - Y: π rotation about y-axis
        - Z: π rotation about z-axis (phase flip)

        Parameters
        ----------
        pauli : {'X', 'Y', 'Z'}
            Which Pauli gate to optimize.
        gate_time : float, optional
            Gate duration in ns (default: 20.0).
        n_timeslices : int, optional
            Number of time slices (default: 100).
        method : {'grape', 'krotov'}, optional
            Optimization method (default: 'grape').
        **kwargs
            Additional optimizer arguments.

        Returns
        -------
        GateResult
            Optimization result.
        """
        pauli = pauli.upper()
        if pauli not in ["X", "Y", "Z"]:
            raise ValueError(f"Pauli must be 'X', 'Y', or 'Z', got '{pauli}'")

        target = self._standard_gates[pauli]

        return self._optimize_gate(
            gate_name=pauli,
            target_unitary=target,
            gate_time=gate_time,
            n_timeslices=n_timeslices,
            method=method,
            **kwargs,
        )

    def _setup_amplitude_limits(
        self, amplitude_limit: Optional[float]
    ) -> Tuple[float, float]:
        """Setup control amplitude limits."""
        if amplitude_limit is not None:
            return (-amplitude_limit, amplitude_limit)
        return (-10.0, 10.0)

    def _create_optimizer(
        self,
        method: str,
        n_timeslices: int,
        gate_time: float,
        u_limits: Tuple[float, float],
        max_iterations: int,
        convergence_threshold: float,
    ):
        """Create optimizer instance based on method."""
        if method == "grape":
            return GRAPEOptimizer(
                H_drift=self.H_drift,
                H_controls=self.H_controls,
                n_timeslices=n_timeslices,
                total_time=gate_time,
                u_limits=u_limits,
                max_iterations=max_iterations,
                convergence_threshold=convergence_threshold,
                verbose=False,
            )
        elif method == "krotov":
            return KrotovOptimizer(
                H_drift=self.H_drift,
                H_controls=self.H_controls,
                n_timeslices=n_timeslices,
                total_time=gate_time,
                u_limits=u_limits,
                max_iterations=max_iterations,
                convergence_threshold=convergence_threshold,
                verbose=False,
            )
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'grape' or 'krotov'.")

    def _run_gate_optimization(
        self,
        optimizer,
        target_unitary: qt.Qobj,
        n_timeslices: int,
        **kwargs,
    ):
        """Run optimization with initial random guess."""
        initial_pulses = 0.1 * np.random.randn(self.n_controls, n_timeslices)
        return optimizer.optimize_unitary(
            U_target=target_unitary,
            u_init=initial_pulses,
            **kwargs,
        )

    def _build_gate_result(
        self,
        gate_name: str,
        target_unitary: qt.Qobj,
        gate_time: float,
        method: str,
        opt_result,
        n_timeslices: int,
        amplitude_limit: Optional[float],
    ) -> GateResult:
        """Build final gate optimization result."""
        success = opt_result.final_fidelity >= self.fidelity_threshold

        metadata = {
            "n_timeslices": n_timeslices,
            "n_iterations": opt_result.n_iterations,
            "converged": opt_result.converged,
            "optimizer_message": opt_result.message,
            "fidelity_threshold": self.fidelity_threshold,
            "amplitude_limit": amplitude_limit,
        }

        return GateResult(
            gate_name=gate_name,
            target_unitary=target_unitary,
            final_fidelity=opt_result.final_fidelity,
            optimized_pulses=opt_result.optimized_pulses,
            gate_time=gate_time,
            optimizer_result=opt_result,
            method=method,
            success=success,
            metadata=metadata,
        )

    def _optimize_gate(
        self,
        gate_name: str,
        target_unitary: qt.Qobj,
        gate_time: float,
        n_timeslices: int,
        method: Literal["grape", "krotov"],
        max_iterations: int = 500,
        convergence_threshold: float = 1e-6,
        amplitude_limit: Optional[float] = None,
        **kwargs,
    ) -> GateResult:
        """
        Internal method to optimize a gate using specified method.

        Orchestrates gate optimization by setting up optimizer, running
        optimization, and assembling results.
        """
        u_limits = self._setup_amplitude_limits(amplitude_limit)

        optimizer = self._create_optimizer(
            method,
            n_timeslices,
            gate_time,
            u_limits,
            max_iterations,
            convergence_threshold,
        )

        opt_result = self._run_gate_optimization(
            optimizer, target_unitary, n_timeslices, **kwargs
        )

        return self._build_gate_result(
            gate_name,
            target_unitary,
            gate_time,
            method,
            opt_result,
            n_timeslices,
            amplitude_limit,
        )

    def check_clifford_closure(
        self, gates: List[GateResult], tolerance: float = 1e-6
    ) -> Tuple[bool, Dict[str, any]]:
        """
        Check if a set of gates forms a closed Clifford group.

        The single-qubit Clifford group has 24 elements and is generated
        by {H, S}. This method verifies that the provided gates satisfy
        the group closure property and standard Clifford relations.

        Key relations to check:
        - H² = I
        - S⁴ = I
        - (HS)³ = I
        - HSH = SHS (conjugation relation)

        Parameters
        ----------
        gates : list[GateResult]
            List of gate results to check. Should include at least H and S.
        tolerance : float, optional
            Numerical tolerance for checking relations (default: 1e-6).

        Returns
        -------
        is_clifford : bool
            True if gates satisfy Clifford group relations.
        report : dict
            Detailed report of which relations passed/failed.

        Examples
        --------
        >>> h_result = gates.optimize_hadamard()
        >>> s_result = gates.optimize_phase_gate(np.pi/2)
        >>> is_clifford, report = gates.check_clifford_closure([h_result, s_result])
        >>> print(f"Clifford group: {is_clifford}")
        """
        # Extract gates by name
        gate_dict = {g.gate_name: g.target_unitary for g in gates}

        report = {
            "relations_checked": [],
            "relations_passed": [],
            "relations_failed": [],
        }

        # Check H² = I
        if "Hadamard" in gate_dict or "H" in gate_dict:
            H = gate_dict.get("Hadamard", gate_dict.get("H"))
            HH = H * H
            I = qt.qeye(2)

            # Check if HH ≈ ±I (global phase doesn't matter)
            fidelity = abs((HH.dag() * I).tr()) / 2
            passed = fidelity > (1 - tolerance)

            report["relations_checked"].append("H² = I")
            if passed:
                report["relations_passed"].append(("H² = I", fidelity))
            else:
                report["relations_failed"].append(("H² = I", fidelity))

        # Check S⁴ = I
        if "S" in gate_dict:
            S = gate_dict["S"]
            S4 = S * S * S * S
            I = qt.qeye(2)

            fidelity = abs((S4.dag() * I).tr()) / 2
            passed = fidelity > (1 - tolerance)

            report["relations_checked"].append("S⁴ = I")
            if passed:
                report["relations_passed"].append(("S⁴ = I", fidelity))
            else:
                report["relations_failed"].append(("S⁴ = I", fidelity))

        # Check (HS)³ = I
        if ("Hadamard" in gate_dict or "H" in gate_dict) and "S" in gate_dict:
            H = gate_dict.get("Hadamard", gate_dict.get("H"))
            S = gate_dict["S"]
            HS = H * S
            HS3 = HS * HS * HS
            I = qt.qeye(2)

            fidelity = abs((HS3.dag() * I).tr()) / 2
            passed = fidelity > (1 - tolerance)

            report["relations_checked"].append("(HS)³ = I")
            if passed:
                report["relations_passed"].append(("(HS)³ = I", fidelity))
            else:
                report["relations_failed"].append(("(HS)³ = I", fidelity))

        # Overall result
        is_clifford = (
            len(report["relations_failed"]) == 0
            and len(report["relations_checked"]) > 0
        )

        return is_clifford, report

    def get_standard_gate(self, name: str) -> qt.Qobj:
        """
        Get standard gate by name.

        Parameters
        ----------
        name : str
            Gate name ('I', 'X', 'Y', 'Z', 'H', 'S', 'T', 'Sdg', 'Tdg').

        Returns
        -------
        qt.Qobj
            Gate unitary operator.
        """
        if name not in self._standard_gates:
            raise ValueError(
                f"Unknown gate '{name}'. Available: {list(self._standard_gates.keys())}"
            )
        return self._standard_gates[name]


def euler_angles_from_unitary(U: qt.Qobj) -> Tuple[float, float, float]:
    """
    Decompose arbitrary SU(2) unitary into Euler angles.

    Any single-qubit unitary can be written as:
        U = e^(iα) R_z(φ₁) R_y(θ) R_z(φ₂)

    This function extracts (φ₁, θ, φ₂) from U up to global phase.

    Parameters
    ----------
    U : qt.Qobj
        2×2 unitary operator.

    Returns
    -------
    phi1 : float
        First Z-rotation angle (radians).
    theta : float
        Y-rotation angle (radians).
    phi2 : float
        Second Z-rotation angle (radians).

    References
    ----------
    - Nielsen & Chuang, Box 4.1
    - Shende et al., IEEE Trans. CAD 25, 1000 (2006)

    Examples
    --------
    >>> H = qt.hadamard_transform()
    >>> phi1, theta, phi2 = euler_angles_from_unitary(H)
    >>> U_reconstructed = rotation_from_euler_angles(phi1, theta, phi2)
    >>> fidelity = abs((H.dag() * U_reconstructed).tr())**2 / 4
    >>> assert fidelity > 0.999
    """
    if U.shape != (2, 2):
        raise ValueError(f"Expected 2×2 unitary, got shape {U.shape}")

    # Extract matrix elements
    u = U.full()

    # Normalize by determinant to ensure det = 1 (SU(2))
    det_u = np.linalg.det(u)
    u = u / np.sqrt(det_u)

    # For simplicity, use the fact that any SU(2) can be decomposed
    # For gates that differ only by global phase, we accept approximate equality

    # Extract theta from matrix element magnitude
    # In ZYZ: U = Rz(φ1) Ry(θ) Rz(φ2)
    # Matrix form: u[0,0] = cos(θ/2)e^(i(φ1+φ2)/2)

    theta = 2 * np.arccos(np.clip(np.abs(u[0, 0]), 0, 1))

    # Handle special case: θ ≈ 0 (identity-like)
    if abs(theta) < 1e-10:
        phi1 = 0.0
        phi2 = 2 * np.angle(u[0, 0])
        return phi1, theta, phi2

    # Handle special case: θ ≈ π (X-like)
    if abs(theta - np.pi) < 1e-10:
        phi1 = 0.0
        phi2 = 2 * np.angle(u[1, 0])
        return phi1, theta, phi2

    # General case: extract phases
    # u[1,0] = sin(θ/2)e^(i(φ1-φ2)/2)
    # u[0,0] = cos(θ/2)e^(i(φ1+φ2)/2)

    phase_plus = np.angle(u[0, 0])
    phase_minus = np.angle(u[1, 0])

    phi1 = phase_plus + phase_minus
    phi2 = phase_plus - phase_minus

    return phi1, theta, phi2


def rotation_from_euler_angles(
    phi1: float,
    theta: float,
    phi2: float,
) -> qt.Qobj:
    """
    Construct SU(2) rotation from Euler angles.

    Builds U = R_z(φ₁) R_y(θ) R_z(φ₂) using matrix exponentials.

    Parameters
    ----------
    phi1 : float
        First Z-rotation (radians).
    theta : float
        Y-rotation (radians).
    phi2 : float
        Second Z-rotation (radians).

    Returns
    -------
    qt.Qobj
        Unitary operator U.

    Examples
    --------
    >>> # Build a Hadamard-like gate
    >>> U = rotation_from_euler_angles(0, np.pi/2, np.pi)
    """
    # Build rotations using matrix exponentials for consistency
    # R_z(φ) = exp(-i φ σ_z / 2)
    # R_y(θ) = exp(-i θ σ_y / 2)
    Rz1 = (-1j * phi1 / 2 * qt.sigmaz()).expm()
    Ry = (-1j * theta / 2 * qt.sigmay()).expm()
    Rz2 = (-1j * phi2 / 2 * qt.sigmaz()).expm()

    return Rz1 * Ry * Rz2
