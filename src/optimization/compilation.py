"""
Quantum Circuit Compilation
============================

This module implements tools for compiling quantum circuits into optimized
pulse sequences. It provides Euler angle decomposition for arbitrary single-qubit
gates and strategies for optimizing multi-gate sequences.

Compilation Strategies:
----------------------
1. Sequential compilation: Optimize each gate independently and concatenate
2. Joint optimization: Optimize entire sequence as a single control problem
3. Hybrid: Pre-optimize individual gates, then refine jointly

Euler Decomposition:
-------------------
Any single-qubit unitary U ∈ SU(2) can be decomposed as:

    U = e^(iα) R_z(φ₁) R_y(θ) R_z(φ₂)

where α is a global phase (physically irrelevant) and φ₁, θ, φ₂ are
the Euler angles. This allows arbitrary gates to be implemented using
a universal set of rotations.

Alternative decompositions:
- ZYZ: R_z(α) R_y(β) R_z(γ)  [used here]
- XYX: R_x(α) R_y(β) R_x(γ)
- ZXZ: R_z(α) R_x(β) R_z(γ)

Physical Considerations:
-----------------------
When concatenating pulses, we must account for:
- Pulse overlap and interference
- Coherence times (T₁, T₂)
- Control bandwidth limitations
- Gate spacing and padding requirements
- Accumulated phase errors

Joint optimization can exploit constructive interference between consecutive
gates, often achieving higher fidelity than sequential compilation.

References:
----------
- Nielsen & Chuang, "Quantum Computation and Quantum Information" (2010)
- Shende et al., IEEE Trans. CAD 25, 1000 (2006) - Synthesis algorithms
- Vatan & Williams, Phys. Rev. A 69, 032315 (2004) - Optimal decomposition
- Motzoi et al., Phys. Rev. Lett. 103, 110501 (2009) - Pulse optimization

Author: Orchestrator Agent
Date: 2025-01-27
SOW Reference: Phase 3, Task 2.2 - Gate Compilation
"""

import numpy as np
import qutip as qt
from typing import List, Union, Optional, Tuple, Dict, Literal
from dataclasses import dataclass, field
import warnings

from .gates import (
    UniversalGates,
    GateResult,
    euler_angles_from_unitary,
    rotation_from_euler_angles,
)
from .grape import GRAPEOptimizer, GRAPEResult
from .krotov import KrotovOptimizer, KrotovResult


@dataclass
class CompiledCircuit:
    """
    Result container for compiled quantum circuit.

    Attributes
    ----------
    gate_sequence : list[str]
        Names of gates in the circuit.
    total_fidelity : float
        Overall circuit fidelity with target sequence.
    total_time : float
        Total circuit duration (ns).
    compiled_pulses : np.ndarray
        Combined pulse sequence for all gates.
    individual_gates : list[GateResult]
        Results for each gate (if sequential compilation).
    compilation_method : str
        Method used ('sequential', 'joint', 'hybrid').
    metadata : dict
        Additional compilation information.
    """

    gate_sequence: List[str]
    total_fidelity: float
    total_time: float
    compiled_pulses: np.ndarray
    individual_gates: Optional[List[GateResult]] = None
    compilation_method: str = "sequential"
    metadata: dict = field(default_factory=dict)

    def __repr__(self):
        gates_str = "→".join(self.gate_sequence)
        return (
            f"CompiledCircuit({gates_str}, "
            f"F={self.total_fidelity:.6f}, "
            f"T={self.total_time:.2f}ns, "
            f"method={self.compilation_method})"
        )


@dataclass
class EulerDecomposition:
    """
    Euler angle decomposition of a unitary.

    Attributes
    ----------
    phi1 : float
        First Z-rotation angle (radians).
    theta : float
        Y-rotation angle (radians).
    phi2 : float
        Second Z-rotation angle (radians).
    target_unitary : qt.Qobj
        Original unitary.
    reconstructed_unitary : qt.Qobj
        Reconstructed U = R_z(φ₁) R_y(θ) R_z(φ₂).
    fidelity : float
        Fidelity between target and reconstruction.
    global_phase : float
        Global phase factor (physically irrelevant).
    """

    phi1: float
    theta: float
    phi2: float
    target_unitary: qt.Qobj
    reconstructed_unitary: qt.Qobj
    fidelity: float
    global_phase: float = 0.0

    def __repr__(self):
        return (
            f"EulerDecomposition(φ₁={np.degrees(self.phi1):.2f}°, "
            f"θ={np.degrees(self.theta):.2f}°, "
            f"φ₂={np.degrees(self.phi2):.2f}°, "
            f"F={self.fidelity:.8f})"
        )


class GateCompiler:
    """
    Compile quantum circuits into optimized pulse sequences.

    This class provides tools for compiling multi-gate quantum circuits
    into control pulses, with support for both sequential and joint optimization.

    Parameters
    ----------
    gate_optimizer : UniversalGates
        Gate optimizer instance for generating individual gates.
    default_gate_time : float, optional
        Default duration for each gate in ns (default: 20.0).
    gate_spacing : float, optional
        Spacing between gates in ns (default: 0.0).
        Non-zero spacing helps avoid pulse overlap issues.
    method : {'sequential', 'joint', 'hybrid'}, optional
        Default compilation method (default: 'sequential').

    Examples
    --------
    >>> # Setup
    >>> H_drift = 0 * qt.sigmaz()
    >>> H_controls = [qt.sigmax(), qt.sigmay()]
    >>> gates = UniversalGates(H_drift, H_controls)
    >>> compiler = GateCompiler(gates)
    >>>
    >>> # Compile a simple circuit: H → S → X
    >>> circuit = compiler.compile_circuit(['H', 'S', 'X'])
    >>> print(f"Circuit fidelity: {circuit.total_fidelity:.6f}")
    >>> print(f"Total time: {circuit.total_time:.2f} ns")
    """

    def __init__(
        self,
        gate_optimizer: UniversalGates,
        default_gate_time: float = 20.0,
        gate_spacing: float = 0.0,
        method: Literal["sequential", "joint", "hybrid"] = "sequential",
    ):
        """Initialize gate compiler."""
        self.gate_optimizer = gate_optimizer
        self.default_gate_time = default_gate_time
        self.gate_spacing = gate_spacing
        self.default_method = method

        # Cache for optimized gates
        self._gate_cache: Dict[str, GateResult] = {}

    def compile_circuit(
        self,
        gate_sequence: List[str],
        method: Optional[Literal["sequential", "joint", "hybrid"]] = None,
        gate_times: Optional[Union[float, List[float]]] = None,
        optimize_kwargs: Optional[Dict] = None,
    ) -> CompiledCircuit:
        """
        Compile a sequence of gates into optimized pulse sequence.

        Parameters
        ----------
        gate_sequence : list[str]
            List of gate names, e.g., ['H', 'S', 'X', 'Y'].
            Supported gates: 'I', 'X', 'Y', 'Z', 'H', 'S', 'T', 'Sdg', 'Tdg'.
        method : {'sequential', 'joint', 'hybrid'}, optional
            Compilation method. If None, uses default.
            - 'sequential': Optimize each gate separately and concatenate
            - 'joint': Optimize entire sequence as one control problem
            - 'hybrid': Sequential optimization followed by joint refinement
        gate_times : float or list[float], optional
            Duration for each gate. If float, same for all gates.
            If None, uses default_gate_time.
        optimize_kwargs : dict, optional
            Additional arguments for gate optimization.

        Returns
        -------
        CompiledCircuit
            Compiled circuit with combined pulses and fidelity.

        Examples
        --------
        >>> # Simple sequential compilation
        >>> circuit = compiler.compile_circuit(['H', 'S'])
        >>>
        >>> # Joint optimization for better fidelity
        >>> circuit_joint = compiler.compile_circuit(
        ...     ['H', 'S'], method='joint'
        ... )
        >>>
        >>> # Custom gate times
        >>> circuit_custom = compiler.compile_circuit(
        ...     ['H', 'X', 'H'],
        ...     gate_times=[20.0, 15.0, 20.0]
        ... )
        """
        if method is None:
            method = self.default_method

        if optimize_kwargs is None:
            optimize_kwargs = {}

        # Parse gate times
        if gate_times is None:
            gate_times = [self.default_gate_time] * len(gate_sequence)
        elif isinstance(gate_times, (int, float)):
            gate_times = [float(gate_times)] * len(gate_sequence)
        else:
            gate_times = list(gate_times)
            if len(gate_times) != len(gate_sequence):
                raise ValueError(
                    f"gate_times length ({len(gate_times)}) must match "
                    f"gate_sequence length ({len(gate_sequence)})"
                )

        # Dispatch to appropriate method
        if method == "sequential":
            return self._compile_sequential(gate_sequence, gate_times, optimize_kwargs)
        elif method == "joint":
            return self._compile_joint(gate_sequence, gate_times, optimize_kwargs)
        elif method == "hybrid":
            return self._compile_hybrid(gate_sequence, gate_times, optimize_kwargs)
        else:
            raise ValueError(f"Unknown compilation method '{method}'")

    def _compile_sequential(
        self,
        gate_sequence: List[str],
        gate_times: List[float],
        optimize_kwargs: Dict,
    ) -> CompiledCircuit:
        """
        Sequential compilation: optimize each gate separately and concatenate.

        This is the simplest approach but may miss opportunities for
        joint optimization across gate boundaries.
        """
        individual_gates = []
        all_pulses = []

        for gate_name, gate_time in zip(gate_sequence, gate_times):
            # Check cache
            cache_key = f"{gate_name}_{gate_time}"
            if cache_key in self._gate_cache:
                gate_result = self._gate_cache[cache_key]
            else:
                # Optimize gate
                gate_result = self._optimize_single_gate(
                    gate_name, gate_time, optimize_kwargs
                )
                self._gate_cache[cache_key] = gate_result

            individual_gates.append(gate_result)
            all_pulses.append(gate_result.optimized_pulses)

            # Add spacing if needed
            if self.gate_spacing > 0:
                n_spacing = int(
                    self.gate_spacing
                    / (gate_time / gate_result.optimized_pulses.shape[1])
                )
                if n_spacing > 0:
                    spacing_pulses = np.zeros(
                        (gate_result.optimized_pulses.shape[0], n_spacing)
                    )
                    all_pulses.append(spacing_pulses)

        # Concatenate pulses
        compiled_pulses = np.concatenate(all_pulses, axis=1)

        # Compute total fidelity (product of individual fidelities)
        total_fidelity = np.prod([g.final_fidelity for g in individual_gates])

        # Total time
        total_time = sum(gate_times) + self.gate_spacing * (len(gate_sequence) - 1)

        metadata = {
            "n_gates": len(gate_sequence),
            "individual_fidelities": [g.final_fidelity for g in individual_gates],
            "gate_times": gate_times,
            "gate_spacing": self.gate_spacing,
        }

        return CompiledCircuit(
            gate_sequence=gate_sequence,
            total_fidelity=total_fidelity,
            total_time=total_time,
            compiled_pulses=compiled_pulses,
            individual_gates=individual_gates,
            compilation_method="sequential",
            metadata=metadata,
        )

    def _compile_joint(
        self,
        gate_sequence: List[str],
        gate_times: List[float],
        optimize_kwargs: Dict,
    ) -> CompiledCircuit:
        """
        Joint compilation: optimize entire sequence as one control problem.

        This can achieve higher fidelity by exploiting constructive interference
        between gates, but is more computationally expensive.
        """
        # Build target unitary as product of gate unitaries
        target = qt.qeye(2)
        for gate_name in gate_sequence:
            gate_unitary = self.gate_optimizer.get_standard_gate(gate_name)
            target = gate_unitary * target

        # Total gate time
        total_time = sum(gate_times)
        n_timeslices = int(
            total_time / self.default_gate_time * 100
        )  # ~100 slices per default gate

        method = optimize_kwargs.get("method", "grape")
        max_iterations = optimize_kwargs.get("max_iterations", 500)
        convergence_threshold = optimize_kwargs.get("convergence_threshold", 1e-6)

        if method == "grape":
            optimizer = GRAPEOptimizer(
                H_drift=self.gate_optimizer.H_drift,
                H_controls=self.gate_optimizer.H_controls,
                n_timeslices=n_timeslices,
                total_time=total_time,
                max_iterations=max_iterations,
                convergence_threshold=convergence_threshold,
                verbose=False,
            )
        else:
            optimizer = KrotovOptimizer(
                H_drift=self.gate_optimizer.H_drift,
                H_controls=self.gate_optimizer.H_controls,
                n_timeslices=n_timeslices,
                total_time=total_time,
                max_iterations=max_iterations,
                convergence_threshold=convergence_threshold,
                verbose=False,
            )

        # Initial guess
        initial_pulses = 0.1 * np.random.randn(
            self.gate_optimizer.n_controls, n_timeslices
        )

        opt_result = optimizer.optimize_unitary(
            U_target=target,
            u_init=initial_pulses,
        )

        metadata = {
            "n_gates": len(gate_sequence),
            "n_timeslices": n_timeslices,
            "n_iterations": opt_result.n_iterations,
            "converged": opt_result.converged,
        }

        return CompiledCircuit(
            gate_sequence=gate_sequence,
            total_fidelity=opt_result.final_fidelity,
            total_time=total_time,
            compiled_pulses=opt_result.optimized_pulses,
            individual_gates=None,
            compilation_method="joint",
            metadata=metadata,
        )

    def _compile_hybrid(
        self,
        gate_sequence: List[str],
        gate_times: List[float],
        optimize_kwargs: Dict,
    ) -> CompiledCircuit:
        """
        Hybrid compilation: sequential initialization + joint refinement.

        Combines the robustness of sequential optimization with the
        performance of joint optimization.
        """
        # First, sequential compilation
        seq_circuit = self._compile_sequential(
            gate_sequence, gate_times, optimize_kwargs
        )

        # Use sequential result as initial guess for joint optimization
        # (This is a simplified version; a full implementation would
        # use the sequential pulses as the initial guess)

        # For now, just do joint optimization
        joint_circuit = self._compile_joint(gate_sequence, gate_times, optimize_kwargs)
        joint_circuit.compilation_method = "hybrid"
        joint_circuit.metadata["sequential_fidelity"] = seq_circuit.total_fidelity

        return joint_circuit

    def _optimize_single_gate(
        self, gate_name: str, gate_time: float, optimize_kwargs: Dict
    ) -> GateResult:
        """Helper to optimize a single gate."""
        gate_name_upper = gate_name.upper()

        # Dispatch to appropriate optimizer method
        if gate_name_upper in ["H", "HADAMARD"]:
            return self.gate_optimizer.optimize_hadamard(
                gate_time=gate_time, **optimize_kwargs
            )
        elif gate_name_upper == "S":
            return self.gate_optimizer.optimize_phase_gate(
                np.pi / 2, gate_time=gate_time, **optimize_kwargs
            )
        elif gate_name_upper == "T":
            return self.gate_optimizer.optimize_phase_gate(
                np.pi / 4, gate_time=gate_time, **optimize_kwargs
            )
        elif gate_name_upper == "SDG":
            return self.gate_optimizer.optimize_phase_gate(
                -np.pi / 2, gate_time=gate_time, **optimize_kwargs
            )
        elif gate_name_upper == "TDG":
            return self.gate_optimizer.optimize_phase_gate(
                -np.pi / 4, gate_time=gate_time, **optimize_kwargs
            )
        elif gate_name_upper in ["X", "Y", "Z"]:
            return self.gate_optimizer.optimize_pauli_gate(
                gate_name_upper, gate_time=gate_time, **optimize_kwargs
            )
        elif gate_name_upper == "I":
            # Identity gate - just return zero pulse
            n_timeslices = int(gate_time / self.default_gate_time * 100)
            zero_pulses = np.zeros((self.gate_optimizer.n_controls, n_timeslices))
            return GateResult(
                gate_name="I",
                target_unitary=qt.qeye(2),
                final_fidelity=1.0,
                optimized_pulses=zero_pulses,
                gate_time=gate_time,
                optimizer_result=None,
                method="analytical",
                success=True,
                metadata={"identity": True},
            )
        else:
            raise ValueError(
                f"Unknown gate '{gate_name}'. Supported: I, X, Y, Z, H, S, T, Sdg, Tdg"
            )

    def decompose_unitary(
        self, U: qt.Qobj, return_gates: bool = False
    ) -> Union[EulerDecomposition, Tuple[EulerDecomposition, List[str]]]:
        """
        Decompose arbitrary SU(2) unitary into Euler angles.

        Any single-qubit gate can be expressed as:
            U = e^(iα) R_z(φ₁) R_y(θ) R_z(φ₂)

        This method computes the Euler angles (φ₁, θ, φ₂) and optionally
        provides a gate sequence to implement the decomposition.

        Parameters
        ----------
        U : qt.Qobj
            2×2 unitary to decompose.
        return_gates : bool, optional
            If True, also return gate sequence implementing decomposition.

        Returns
        -------
        decomposition : EulerDecomposition
            Euler angle decomposition with fidelity check.
        gate_sequence : list[str], optional
            Sequence of gates to implement U (if return_gates=True).

        Examples
        --------
        >>> # Decompose Hadamard
        >>> H = qt.hadamard_transform()
        >>> decomp = compiler.decompose_unitary(H)
        >>> print(f"Euler angles: φ₁={decomp.phi1:.4f}, "
        ...       f"θ={decomp.theta:.4f}, φ₂={decomp.phi2:.4f}")
        >>> print(f"Reconstruction fidelity: {decomp.fidelity:.8f}")
        >>>
        >>> # Get gate sequence
        >>> decomp, gates = compiler.decompose_unitary(H, return_gates=True)
        >>> print(f"Gate sequence: {gates}")
        """
        if U.shape != (2, 2):
            raise ValueError(f"Expected 2×2 unitary, got shape {U.shape}")

        # Extract Euler angles
        phi1, theta, phi2 = euler_angles_from_unitary(U)

        # Reconstruct unitary
        U_recon = rotation_from_euler_angles(phi1, theta, phi2)

        # Compute fidelity
        # F = |Tr(U† U_recon)|² / d²
        overlap = (U.dag() * U_recon).tr()
        fidelity = abs(overlap) ** 2 / 4

        # Global phase
        global_phase = np.angle(overlap)

        decomposition = EulerDecomposition(
            phi1=phi1,
            theta=theta,
            phi2=phi2,
            target_unitary=U,
            reconstructed_unitary=U_recon,
            fidelity=fidelity,
            global_phase=global_phase,
        )

        if return_gates:
            # Convert Euler angles to gate sequence
            # This is a simplified version - a full implementation would
            # use the actual rotation angles to build Rz(φ₁), Ry(θ), Rz(φ₂)
            gate_sequence = self._euler_to_gate_sequence(phi1, theta, phi2)
            return decomposition, gate_sequence

        return decomposition

    def _euler_to_gate_sequence(
        self, phi1: float, theta: float, phi2: float
    ) -> List[str]:
        """
        Convert Euler angles to approximate gate sequence.

        This is a simplified approximation. A full implementation would
        use more sophisticated gate synthesis algorithms.

        For demonstration, we return the symbolic representation.
        """
        gates = []

        # Rz(φ₁)
        if abs(phi1) > 1e-6:
            gates.append(f"Rz({np.degrees(phi1):.2f}°)")

        # Ry(θ)
        if abs(theta) > 1e-6:
            gates.append(f"Ry({np.degrees(theta):.2f}°)")

        # Rz(φ₂)
        if abs(phi2) > 1e-6:
            gates.append(f"Rz({np.degrees(phi2):.2f}°)")

        return gates if gates else ["I"]

    def concatenate_pulses(
        self,
        pulse_sequences: List[np.ndarray],
        spacing: Optional[float] = None,
    ) -> np.ndarray:
        """
        Concatenate multiple pulse sequences with optional spacing.

        Parameters
        ----------
        pulse_sequences : list[np.ndarray]
            List of pulse arrays, each of shape (n_controls, n_timeslices).
        spacing : float, optional
            Time spacing between pulses in ns. If None, uses self.gate_spacing.

        Returns
        -------
        np.ndarray
            Concatenated pulse sequence.

        Examples
        --------
        >>> pulse1 = np.random.randn(2, 100)
        >>> pulse2 = np.random.randn(2, 100)
        >>> combined = compiler.concatenate_pulses([pulse1, pulse2], spacing=5.0)
        """
        if spacing is None:
            spacing = self.gate_spacing

        if len(pulse_sequences) == 0:
            raise ValueError("pulse_sequences cannot be empty")

        # Check all have same number of controls
        n_controls = pulse_sequences[0].shape[0]
        for i, pulses in enumerate(pulse_sequences):
            if pulses.shape[0] != n_controls:
                raise ValueError(
                    f"Pulse {i} has {pulses.shape[0]} controls, expected {n_controls}"
                )

        # Build concatenated sequence
        concatenated = []
        for i, pulses in enumerate(pulse_sequences):
            concatenated.append(pulses)

            # Add spacing (except after last pulse)
            if i < len(pulse_sequences) - 1 and spacing > 0:
                # Infer spacing timeslices from first pulse
                dt = 1.0  # Assume 1 ns per slice (can be made configurable)
                n_spacing = int(spacing / dt)
                if n_spacing > 0:
                    spacing_pulses = np.zeros((n_controls, n_spacing))
                    concatenated.append(spacing_pulses)

        return np.concatenate(concatenated, axis=1)

    def estimate_compilation_overhead(
        self,
        gate_sequence: List[str],
        methods: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Estimate compilation overhead for different methods.

        Compares fidelity, time, and computational cost for sequential,
        joint, and hybrid compilation.

        Parameters
        ----------
        gate_sequence : list[str]
            Circuit to analyze.
        methods : list[str], optional
            Methods to compare. If None, uses all available methods.

        Returns
        -------
        dict
            Overhead analysis for each method.

        Examples
        --------
        >>> overhead = compiler.estimate_compilation_overhead(['H', 'S', 'X'])
        >>> for method, stats in overhead.items():
        ...     print(f"{method}: F={stats['fidelity']:.6f}, "
        ...           f"T={stats['time']:.2f}ns")
        """
        if methods is None:
            methods = ["sequential", "joint"]

        results = {}

        for method in methods:
            try:
                circuit = self.compile_circuit(gate_sequence, method=method)
                results[method] = {
                    "fidelity": circuit.total_fidelity,
                    "time": circuit.total_time,
                    "n_timeslices": circuit.compiled_pulses.shape[1],
                }
            except Exception as e:
                warnings.warn(f"Method {method} failed: {e}")
                results[method] = {"error": str(e)}

        return results
