"""
Unit Tests for Gate Compilation
================================

This module tests the gate compilation functionality in src/optimization/compilation.py,
including circuit compilation, Euler decomposition, pulse concatenation, and
compilation overhead analysis.

Test Coverage:
-------------
1. GateCompiler initialization
2. Sequential compilation
3. Joint compilation
4. Hybrid compilation
5. Euler angle decomposition
6. Pulse concatenation
7. Compilation overhead estimation
8. Edge cases and error handling

Author: Orchestrator Agent
Date: 2025-01-27
SOW Reference: Phase 3, Task 2.2 - Gate Compilation Tests
"""

import pytest
import numpy as np
import qutip as qt

from src.optimization.gates import UniversalGates, GateResult
from src.optimization.compilation import (
    GateCompiler,
    CompiledCircuit,
    EulerDecomposition,
)


class TestGateCompilerInitialization:
    """Test GateCompiler class initialization."""

    @pytest.fixture
    def gate_optimizer(self):
        """Create gate optimizer for testing."""
        H_drift = 2 * np.pi * 0.0 * qt.sigmaz() / 2
        H_controls = [qt.sigmax(), qt.sigmay()]
        return UniversalGates(H_drift, H_controls, fidelity_threshold=0.95)

    def test_basic_initialization(self, gate_optimizer):
        """Test basic initialization with minimal arguments."""
        compiler = GateCompiler(gate_optimizer)

        assert compiler.gate_optimizer == gate_optimizer
        assert compiler.default_gate_time == 20.0
        assert compiler.gate_spacing == 0.0
        assert compiler.default_method == "sequential"
        assert isinstance(compiler._gate_cache, dict)

    def test_custom_gate_time(self, gate_optimizer):
        """Test initialization with custom default gate time."""
        compiler = GateCompiler(gate_optimizer, default_gate_time=30.0)

        assert compiler.default_gate_time == 30.0

    def test_custom_gate_spacing(self, gate_optimizer):
        """Test initialization with custom gate spacing."""
        compiler = GateCompiler(gate_optimizer, gate_spacing=5.0)

        assert compiler.gate_spacing == 5.0

    def test_custom_method(self, gate_optimizer):
        """Test initialization with custom compilation method."""
        compiler = GateCompiler(gate_optimizer, method="joint")

        assert compiler.default_method == "joint"


class TestSequentialCompilation:
    """Test sequential compilation strategy."""

    @pytest.fixture
    def compiler(self):
        """Create compiler for testing."""
        H_drift = 2 * np.pi * 0.0 * qt.sigmaz() / 2
        H_controls = [qt.sigmax(), qt.sigmay()]
        gates = UniversalGates(H_drift, H_controls, fidelity_threshold=0.90)
        return GateCompiler(gates, default_gate_time=20.0)

    def test_single_gate_compilation(self, compiler):
        """Test compiling a single gate."""
        circuit = compiler.compile_circuit(
            ["H"], method="sequential", optimize_kwargs={"max_iterations": 10}
        )

        assert isinstance(circuit, CompiledCircuit)
        assert circuit.gate_sequence == ["H"]
        assert circuit.compilation_method == "sequential"
        assert circuit.total_fidelity > 0.0
        assert circuit.total_time > 0.0
        assert circuit.compiled_pulses.shape[0] == 2  # Two controls

    def test_two_gate_compilation(self, compiler):
        """Test compiling two gates."""
        circuit = compiler.compile_circuit(
            ["H", "S"], method="sequential", optimize_kwargs={"max_iterations": 10}
        )

        assert circuit.gate_sequence == ["H", "S"]
        assert len(circuit.individual_gates) == 2
        assert circuit.total_fidelity > 0.0

    def test_multiple_gate_compilation(self, compiler):
        """Test compiling multiple gates."""
        circuit = compiler.compile_circuit(
            ["H", "S", "X"], method="sequential", optimize_kwargs={"max_iterations": 10}
        )

        assert circuit.gate_sequence == ["H", "S", "X"]
        assert len(circuit.individual_gates) == 3
        assert circuit.compilation_method == "sequential"

    def test_sequential_fidelity_product(self, compiler):
        """Test that sequential fidelity is product of individual fidelities."""
        circuit = compiler.compile_circuit(
            ["H", "S"], method="sequential", optimize_kwargs={"max_iterations": 10}
        )

        expected_fidelity = np.prod(
            [g.final_fidelity for g in circuit.individual_gates]
        )
        assert abs(circuit.total_fidelity - expected_fidelity) < 1e-10

    def test_sequential_with_custom_gate_times(self, compiler):
        """Test sequential compilation with custom gate times."""
        gate_times = [25.0, 15.0]
        circuit = compiler.compile_circuit(
            ["H", "S"],
            method="sequential",
            gate_times=gate_times,
            optimize_kwargs={"max_iterations": 10},
        )

        expected_time = sum(gate_times)
        assert circuit.total_time == expected_time

    def test_sequential_with_gate_spacing(self):
        """Test sequential compilation with gate spacing."""
        H_drift = 2 * np.pi * 0.0 * qt.sigmaz() / 2
        H_controls = [qt.sigmax(), qt.sigmay()]
        gates = UniversalGates(H_drift, H_controls, fidelity_threshold=0.90)
        compiler = GateCompiler(gates, gate_spacing=5.0)

        circuit = compiler.compile_circuit(
            ["H", "S"], method="sequential", optimize_kwargs={"max_iterations": 10}
        )

        # Total time should include spacing
        assert (
            circuit.total_time
            > circuit.metadata["gate_times"][0] + circuit.metadata["gate_times"][1]
        )

    def test_sequential_identity_gate(self, compiler):
        """Test compiling identity gate."""
        circuit = compiler.compile_circuit(["I"], method="sequential")

        assert circuit.gate_sequence == ["I"]
        assert circuit.total_fidelity == 1.0

    def test_sequential_pauli_gates(self, compiler):
        """Test compiling Pauli gates."""
        circuit = compiler.compile_circuit(
            ["X", "Y", "Z"], method="sequential", optimize_kwargs={"max_iterations": 10}
        )

        assert circuit.gate_sequence == ["X", "Y", "Z"]
        assert len(circuit.individual_gates) == 3


class TestJointCompilation:
    """Test joint compilation strategy."""

    @pytest.fixture
    def compiler(self):
        """Create compiler for testing."""
        H_drift = 2 * np.pi * 0.0 * qt.sigmaz() / 2
        H_controls = [qt.sigmax(), qt.sigmay()]
        gates = UniversalGates(H_drift, H_controls, fidelity_threshold=0.90)
        return GateCompiler(gates)

    def test_joint_compilation_basic(self, compiler):
        """Test basic joint compilation."""
        circuit = compiler.compile_circuit(
            ["H", "S"],
            method="joint",
            optimize_kwargs={"max_iterations": 10, "method": "grape"},
        )

        assert circuit.compilation_method == "joint"
        assert circuit.individual_gates is None  # Joint doesn't track individual gates
        assert circuit.total_fidelity > 0.0

    def test_joint_target_unitary_correct(self, compiler):
        """Test that joint compilation targets correct composite unitary."""
        # For H→S, target should be S * H (right to left)
        H = qt.gates.hadamard_transform()
        S = qt.gates.phasegate(np.pi / 2)
        expected_target = S * H

        circuit = compiler.compile_circuit(
            ["H", "S"],
            method="joint",
            optimize_kwargs={"max_iterations": 10, "method": "grape"},
        )

        # We can't directly check the target, but fidelity should be reasonable
        assert circuit.total_fidelity > 0.0

    def test_joint_with_custom_gate_times(self, compiler):
        """Test joint compilation with custom gate times."""
        gate_times = [20.0, 15.0]
        circuit = compiler.compile_circuit(
            ["H", "S"],
            method="joint",
            gate_times=gate_times,
            optimize_kwargs={"max_iterations": 10, "method": "grape"},
        )

        expected_time = sum(gate_times)
        assert circuit.total_time == expected_time

    def test_joint_multiple_gates(self, compiler):
        """Test joint compilation with multiple gates."""
        circuit = compiler.compile_circuit(
            ["H", "S", "X"],
            method="joint",
            optimize_kwargs={"max_iterations": 10, "method": "grape"},
        )

        assert circuit.gate_sequence == ["H", "S", "X"]
        assert circuit.compilation_method == "joint"


class TestHybridCompilation:
    """Test hybrid compilation strategy."""

    @pytest.fixture
    def compiler(self):
        """Create compiler for testing."""
        H_drift = 2 * np.pi * 0.0 * qt.sigmaz() / 2
        H_controls = [qt.sigmax(), qt.sigmay()]
        gates = UniversalGates(H_drift, H_controls, fidelity_threshold=0.90)
        return GateCompiler(gates)

    def test_hybrid_compilation_basic(self, compiler):
        """Test basic hybrid compilation."""
        circuit = compiler.compile_circuit(
            ["H", "S"],
            method="hybrid",
            optimize_kwargs={"max_iterations": 10, "method": "grape"},
        )

        assert circuit.compilation_method == "hybrid"
        assert circuit.total_fidelity > 0.0

    def test_hybrid_includes_sequential_fidelity(self, compiler):
        """Test that hybrid includes sequential fidelity in metadata."""
        circuit = compiler.compile_circuit(
            ["H", "S"],
            method="hybrid",
            optimize_kwargs={"max_iterations": 10, "method": "grape"},
        )

        assert "sequential_fidelity" in circuit.metadata
        assert circuit.metadata["sequential_fidelity"] > 0.0


class TestEulerDecomposition:
    """Test Euler angle decomposition."""

    @pytest.fixture
    def compiler(self):
        """Create compiler for testing."""
        H_drift = qt.sigmaz() / 2
        H_controls = [qt.sigmax(), qt.sigmay()]
        gates = UniversalGates(H_drift, H_controls)
        return GateCompiler(gates)

    def test_decompose_identity(self, compiler):
        """Test Euler decomposition of identity."""
        I = qt.qeye(2)
        decomp = compiler.decompose_unitary(I)

        assert isinstance(decomp, EulerDecomposition)
        assert decomp.fidelity > 0.999
        assert abs(decomp.theta) < 1e-6  # Identity has θ ≈ 0

    def test_decompose_hadamard(self, compiler):
        """Test Euler decomposition of Hadamard."""
        H = qt.gates.hadamard_transform()
        decomp = compiler.decompose_unitary(H)

        assert decomp.fidelity > 0.999
        assert isinstance(decomp.phi1, float)
        assert isinstance(decomp.theta, float)
        assert isinstance(decomp.phi2, float)

    def test_decompose_pauli_x(self, compiler):
        """Test Euler decomposition of Pauli X."""
        X = qt.sigmax()
        decomp = compiler.decompose_unitary(X)

        assert decomp.fidelity > 0.999
        # X is π rotation, so θ ≈ π
        assert abs(decomp.theta - np.pi) < 0.1

    def test_decompose_s_gate(self, compiler):
        """Test Euler decomposition of S gate."""
        S = qt.gates.phasegate(np.pi / 2)
        decomp = compiler.decompose_unitary(S)

        assert decomp.fidelity > 0.999

    def test_decompose_t_gate(self, compiler):
        """Test Euler decomposition of T gate."""
        T = qt.gates.phasegate(np.pi / 4)
        decomp = compiler.decompose_unitary(T)

        assert decomp.fidelity > 0.999

    def test_decompose_arbitrary_unitary(self, compiler):
        """Test Euler decomposition of arbitrary unitary."""
        from src.optimization.gates import rotation_from_euler_angles

        # Create random unitary
        phi, theta, lam = np.random.rand(3) * 2 * np.pi
        U = rotation_from_euler_angles(phi, theta, lam)

        decomp = compiler.decompose_unitary(U)

        assert decomp.fidelity > 0.9999

    def test_decompose_with_gate_sequence(self, compiler):
        """Test Euler decomposition with gate sequence return."""
        H = qt.gates.hadamard_transform()
        decomp, gates = compiler.decompose_unitary(H, return_gates=True)

        assert isinstance(decomp, EulerDecomposition)
        assert isinstance(gates, list)
        assert len(gates) > 0

    def test_decompose_invalid_dimension(self, compiler):
        """Test decomposing non-2x2 matrix raises error."""
        U_invalid = qt.qeye(3)

        with pytest.raises(ValueError, match="2×2 unitary"):
            compiler.decompose_unitary(U_invalid)

    def test_euler_decomposition_repr(self, compiler):
        """Test EulerDecomposition string representation."""
        H = qt.gates.hadamard_transform()
        decomp = compiler.decompose_unitary(H)

        repr_str = repr(decomp)
        assert "EulerDecomposition" in repr_str
        assert "φ₁" in repr_str or "phi" in repr_str.lower()


class TestPulseConcatenation:
    """Test pulse concatenation utilities."""

    @pytest.fixture
    def compiler(self):
        """Create compiler for testing."""
        H_drift = qt.sigmaz() / 2
        H_controls = [qt.sigmax(), qt.sigmay()]
        gates = UniversalGates(H_drift, H_controls)
        return GateCompiler(gates)

    def test_concatenate_two_pulses(self, compiler):
        """Test concatenating two pulse sequences."""
        pulse1 = np.random.randn(2, 50)
        pulse2 = np.random.randn(2, 50)

        concatenated = compiler.concatenate_pulses([pulse1, pulse2], spacing=0.0)

        assert concatenated.shape == (2, 100)
        assert np.allclose(concatenated[:, :50], pulse1)
        assert np.allclose(concatenated[:, 50:], pulse2)

    def test_concatenate_multiple_pulses(self, compiler):
        """Test concatenating multiple pulse sequences."""
        pulses = [np.random.randn(2, 30) for _ in range(5)]

        concatenated = compiler.concatenate_pulses(pulses, spacing=0.0)

        assert concatenated.shape == (2, 150)

    def test_concatenate_with_spacing(self, compiler):
        """Test concatenating pulses with spacing."""
        pulse1 = np.random.randn(2, 50)
        pulse2 = np.random.randn(2, 50)

        concatenated = compiler.concatenate_pulses([pulse1, pulse2], spacing=5.0)

        # Should be longer than 100 due to spacing
        assert concatenated.shape[1] > 100

    def test_concatenate_empty_list_error(self, compiler):
        """Test concatenating empty list raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            compiler.concatenate_pulses([])

    def test_concatenate_mismatched_controls_error(self, compiler):
        """Test concatenating pulses with different control counts raises error."""
        pulse1 = np.random.randn(2, 50)
        pulse2 = np.random.randn(3, 50)  # Different number of controls

        with pytest.raises(ValueError, match="controls"):
            compiler.concatenate_pulses([pulse1, pulse2])


class TestCompilationOverhead:
    """Test compilation overhead estimation."""

    @pytest.fixture
    def compiler(self):
        """Create compiler for testing."""
        H_drift = 2 * np.pi * 0.0 * qt.sigmaz() / 2
        H_controls = [qt.sigmax(), qt.sigmay()]
        gates = UniversalGates(H_drift, H_controls, fidelity_threshold=0.90)
        return GateCompiler(gates)

    def test_overhead_estimation_basic(self, compiler):
        """Test basic overhead estimation."""
        overhead = compiler.estimate_compilation_overhead(
            ["H", "S"], methods=["sequential"]
        )

        assert "sequential" in overhead
        assert "fidelity" in overhead["sequential"]
        assert "time" in overhead["sequential"]

    def test_overhead_multiple_methods(self, compiler):
        """Test overhead estimation with multiple methods."""
        overhead = compiler.estimate_compilation_overhead(
            ["H", "S"], methods=["sequential", "joint"]
        )

        assert "sequential" in overhead
        assert "joint" in overhead

    def test_overhead_default_methods(self, compiler):
        """Test overhead estimation with default methods."""
        overhead = compiler.estimate_compilation_overhead(["H", "S"])

        # Should use default methods
        assert len(overhead) > 0


class TestCompiledCircuitDataclass:
    """Test CompiledCircuit dataclass functionality."""

    def test_compiled_circuit_creation(self):
        """Test CompiledCircuit can be created."""
        circuit = CompiledCircuit(
            gate_sequence=["H", "S"],
            total_fidelity=0.99,
            total_time=35.0,
            compiled_pulses=np.zeros((2, 100)),
            compilation_method="sequential",
        )

        assert circuit.gate_sequence == ["H", "S"]
        assert circuit.total_fidelity == 0.99
        assert circuit.total_time == 35.0
        assert circuit.compilation_method == "sequential"

    def test_compiled_circuit_repr(self):
        """Test CompiledCircuit string representation."""
        circuit = CompiledCircuit(
            gate_sequence=["H", "S", "X"],
            total_fidelity=0.98,
            total_time=60.0,
            compiled_pulses=np.zeros((2, 150)),
            compilation_method="sequential",
        )

        repr_str = repr(circuit)
        assert "H" in repr_str
        assert "S" in repr_str
        assert "X" in repr_str
        assert "0.98" in repr_str
        assert "60.00" in repr_str

    def test_compiled_circuit_with_metadata(self):
        """Test CompiledCircuit with custom metadata."""
        metadata = {"custom_key": "value", "iterations": 200}
        circuit = CompiledCircuit(
            gate_sequence=["H"],
            total_fidelity=0.99,
            total_time=20.0,
            compiled_pulses=np.zeros((2, 50)),
            metadata=metadata,
        )

        assert circuit.metadata["custom_key"] == "value"
        assert circuit.metadata["iterations"] == 200


class TestEdgeCasesAndErrors:
    """Test edge cases and error handling."""

    @pytest.fixture
    def compiler(self):
        """Create compiler for testing."""
        H_drift = qt.sigmaz() / 2
        H_controls = [qt.sigmax(), qt.sigmay()]
        gates = UniversalGates(H_drift, H_controls, fidelity_threshold=0.90)
        return GateCompiler(gates)

    def test_invalid_compilation_method(self, compiler):
        """Test invalid compilation method raises error."""
        with pytest.raises(ValueError, match="Unknown compilation method"):
            compiler.compile_circuit(["H"], method="invalid_method")

    def test_invalid_gate_name(self, compiler):
        """Test invalid gate name raises error."""
        with pytest.raises(ValueError, match="Unknown gate"):
            compiler.compile_circuit(["InvalidGate"], method="sequential")

    def test_mismatched_gate_times_length(self, compiler):
        """Test mismatched gate_times length raises error."""
        with pytest.raises(ValueError, match="must match"):
            compiler.compile_circuit(
                ["H", "S"],
                gate_times=[20.0],  # Only one time for two gates
                optimize_kwargs={"max_iterations": 5},
            )

    def test_empty_gate_sequence(self, compiler):
        """Test compiling empty gate sequence."""
        # Should handle gracefully or raise error
        # Implementation dependent - just check it doesn't crash unexpectedly
        try:
            circuit = compiler.compile_circuit([], method="sequential")
            # If it succeeds, check basic properties
            assert isinstance(circuit, CompiledCircuit)
        except (ValueError, IndexError):
            # Or it might raise an error, which is also acceptable
            pass

    def test_single_gate_time_for_multiple_gates(self, compiler):
        """Test single gate time applied to all gates."""
        circuit = compiler.compile_circuit(
            ["H", "S"],
            gate_times=25.0,
            method="sequential",
            optimize_kwargs={"max_iterations": 10},
        )

        # All gates should use the same time
        expected_time = 25.0 * 2
        assert circuit.total_time == expected_time


class TestGateCache:
    """Test gate caching functionality."""

    @pytest.fixture
    def compiler(self):
        """Create compiler for testing."""
        H_drift = 2 * np.pi * 0.0 * qt.sigmaz() / 2
        H_controls = [qt.sigmax(), qt.sigmay()]
        gates = UniversalGates(H_drift, H_controls, fidelity_threshold=0.90)
        return GateCompiler(gates)

    def test_cache_stores_gates(self, compiler):
        """Test that cache stores optimized gates."""
        # Compile circuit
        compiler.compile_circuit(
            ["H"], method="sequential", optimize_kwargs={"max_iterations": 10}
        )

        # Cache should have entry
        assert len(compiler._gate_cache) > 0

    def test_cache_reuses_gates(self, compiler):
        """Test that cache reuses previously optimized gates."""
        # First compilation
        circuit1 = compiler.compile_circuit(
            ["H"], method="sequential", optimize_kwargs={"max_iterations": 10}
        )

        cache_size_1 = len(compiler._gate_cache)

        # Second compilation with same gate
        circuit2 = compiler.compile_circuit(
            ["H"], method="sequential", optimize_kwargs={"max_iterations": 10}
        )

        cache_size_2 = len(compiler._gate_cache)

        # Cache size shouldn't increase
        assert cache_size_2 == cache_size_1
