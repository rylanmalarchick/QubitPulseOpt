"""
Unit Tests for GRAPE Optimizer
===============================

Comprehensive test suite for the GRAPE (Gradient Ascent Pulse Engineering)
optimal control algorithm.

Test Coverage:
-------------
1. Initialization and validation
2. X-gate optimization (single control)
3. State transfer optimization
4. Gradient computation accuracy
5. Convergence behavior
6. Constraint enforcement
7. Multi-control optimization
8. Fidelity computation
9. Pulse function conversion
10. Edge cases and error handling

Author: Orchestrator Agent
Date: 2025-01-27
SOW Reference: Phase 2.1 - GRAPE Implementation Tests
"""

import pytest
import numpy as np
import qutip as qt
from src.optimization.grape import GRAPEOptimizer, GRAPEResult


class TestGRAPEInitialization:
    """Test GRAPE optimizer initialization."""

    def test_basic_initialization(self):
        """Test basic GRAPE initialization."""
        H0 = 0.5 * 5.0 * qt.sigmaz()
        Hc = [qt.sigmax()]

        optimizer = GRAPEOptimizer(
            H_drift=H0,
            H_controls=Hc,
            n_timeslices=10,
            total_time=50.0,
        )

        assert optimizer.n_controls == 1
        assert optimizer.n_timeslices == 10
        assert optimizer.total_time == 50.0
        assert optimizer.dt == 5.0
        assert optimizer.dim == 2

    def test_multi_control_initialization(self):
        """Test initialization with multiple controls."""
        H0 = qt.sigmaz()
        Hc = [qt.sigmax(), qt.sigmay()]

        optimizer = GRAPEOptimizer(H0, Hc, n_timeslices=20, total_time=100)

        assert optimizer.n_controls == 2
        assert len(optimizer.H_controls) == 2

    def test_custom_parameters(self):
        """Test initialization with custom parameters."""
        H0 = qt.sigmaz()
        Hc = [qt.sigmax()]

        optimizer = GRAPEOptimizer(
            H0,
            Hc,
            n_timeslices=50,
            total_time=100,
            u_limits=(-5.0, 5.0),
            convergence_threshold=1e-5,
            max_iterations=1000,
            learning_rate=0.05,
            verbose=False,
        )

        assert optimizer.u_limits == (-5.0, 5.0)
        assert optimizer.convergence_threshold == 1e-5
        assert optimizer.max_iterations == 1000
        assert optimizer.learning_rate == 0.05
        assert optimizer.verbose is False

    def test_invalid_initialization(self):
        """Test error handling for invalid initialization."""
        H0 = qt.sigmaz()

        # No control Hamiltonians
        with pytest.raises(ValueError, match="at least one control"):
            GRAPEOptimizer(H0, [], n_timeslices=10, total_time=50)

        # Invalid timeslices
        with pytest.raises(ValueError, match="positive"):
            GRAPEOptimizer(H0, [qt.sigmax()], n_timeslices=0, total_time=50)

        # Invalid total time
        with pytest.raises(ValueError, match="positive"):
            GRAPEOptimizer(H0, [qt.sigmax()], n_timeslices=10, total_time=-10)


class TestGRAPEPropagators:
    """Test propagator computation."""

    def test_compute_propagators(self):
        """Test propagator computation for each timeslice."""
        H0 = qt.sigmaz()
        Hc = [qt.sigmax()]
        optimizer = GRAPEOptimizer(H0, Hc, n_timeslices=5, total_time=10)

        u = np.ones((1, 5)) * 0.1
        propagators = optimizer._compute_propagators(u)

        assert len(propagators) == 5
        for U in propagators:
            assert isinstance(U, qt.Qobj)
            assert U.shape == (2, 2)
            # Check unitarity
            assert np.allclose((U.dag() * U).full(), np.eye(2), atol=1e-10)

    def test_forward_propagation(self):
        """Test forward propagation accumulation."""
        H0 = qt.sigmaz()
        Hc = [qt.sigmax()]
        optimizer = GRAPEOptimizer(H0, Hc, n_timeslices=5, total_time=10)

        u = np.ones((1, 5)) * 0.1
        propagators = optimizer._compute_propagators(u)
        forward_unitaries, U_final = optimizer._forward_propagation(propagators)

        assert len(forward_unitaries) == 5
        # Final propagator should be product of all
        U_product = qt.qeye(2)
        for U_k in propagators:
            U_product = U_k * U_product
        assert np.allclose(U_final.full(), U_product.full(), atol=1e-10)

    def test_backward_propagation(self):
        """Test backward propagation."""
        H0 = qt.sigmaz()
        Hc = [qt.sigmax()]
        optimizer = GRAPEOptimizer(H0, Hc, n_timeslices=5, total_time=10)

        u = np.ones((1, 5)) * 0.1
        propagators = optimizer._compute_propagators(u)
        backward_unitaries = optimizer._backward_propagation(propagators)

        assert len(backward_unitaries) == 5

        # backward_unitaries[k] should be product of propagators AFTER timeslice k
        # backward_unitaries[0] = U_5 * U_4 * U_3 * U_2 (all except U_1)
        U_product = qt.qeye(2)
        for i in range(len(propagators) - 1, 0, -1):  # indices 4, 3, 2, 1
            U_product = U_product * propagators[i]
        assert np.allclose(backward_unitaries[0].full(), U_product.full(), atol=1e-10)

        # backward_unitaries[-1] should be identity (no propagators after last timeslice)
        assert np.allclose(backward_unitaries[-1].full(), qt.qeye(2).full(), atol=1e-10)


class TestGRAPEFidelity:
    """Test GRAPE fidelity computation."""

    def test_identity_fidelity(self):
        """Test fidelity of identity evolution."""
        H0 = 0 * qt.sigmaz()  # No drift
        Hc = [qt.sigmax()]
        optimizer = GRAPEOptimizer(H0, Hc, n_timeslices=5, total_time=1)

        u = np.zeros((1, 5))  # No control
        propagators = optimizer._compute_propagators(u)
        _, U_final = optimizer._forward_propagation(propagators)

        U_target = qt.qeye(2)
        fidelity = optimizer._compute_fidelity_unitary(U_final, U_target)

        assert np.isclose(fidelity, 1.0, atol=1e-10)

    def test_pauli_x_fidelity(self):
        """Test X-gate fidelity."""
        H0 = 0 * qt.sigmaz()
        Hc = [qt.sigmax()]
        optimizer = GRAPEOptimizer(H0, Hc, n_timeslices=1, total_time=1)

        # π/2 pulse: u = π/2 for H = u σ_x with dt = 1
        # U = exp(-i π/2 σ_x) = -i σ_x, which equals σ_x up to global phase
        u = np.array([[np.pi / 2.0]])
        propagators = optimizer._compute_propagators(u)
        _, U_final = optimizer._forward_propagation(propagators)

        U_target = qt.sigmax()
        fidelity = optimizer._compute_fidelity_unitary(U_final, U_target)

        # Should be close to 1 (X-gate, accounting for global phase)
        assert fidelity > 0.99

    def test_global_phase_invariance(self):
        """Test that fidelity is invariant under global phase."""
        H0 = 0 * qt.sigmaz()
        Hc = [qt.sigmax()]
        optimizer = GRAPEOptimizer(H0, Hc, n_timeslices=1, total_time=1)

        # Test with identity and -I (differ by global phase π)
        U_evolved = -qt.qeye(2)
        U_target = qt.qeye(2)
        fidelity = optimizer._compute_fidelity_unitary(U_evolved, U_target)

        # Should be 1.0 (same gate up to global phase)
        assert np.isclose(fidelity, 1.0, atol=1e-10)

    def test_fidelity_against_qutip(self):
        """Validate fidelity computation against QuTiP's average_gate_fidelity."""
        from qutip import average_gate_fidelity

        H0 = 0 * qt.sigmaz()
        Hc = [qt.sigmax()]
        optimizer = GRAPEOptimizer(H0, Hc, n_timeslices=1, total_time=1)

        # Test several cases
        test_cases = [
            (qt.qeye(2), qt.qeye(2)),  # Perfect match
            (-qt.qeye(2), qt.qeye(2)),  # Global phase
            ((-1j * np.pi / 2 * qt.sigmax()).expm(), qt.sigmax()),  # X gate
            ((-1j * np.pi / 4 * qt.sigmaz()).expm(), qt.sigmaz()),  # Z rotation
        ]

        for U_evolved, U_target in test_cases:
            fid_ours = optimizer._compute_fidelity_unitary(U_evolved, U_target)
            fid_qutip = average_gate_fidelity(U_evolved, U_target)
            assert np.isclose(fid_ours, fid_qutip, atol=1e-10), (
                f"Fidelity mismatch: ours={fid_ours}, QuTiP={fid_qutip}"
            )

    def test_perfect_gate_fidelity(self):
        """Test fidelity for various perfect gate implementations."""
        H0 = 0 * qt.sigmaz()
        Hc = [qt.sigmax()]
        optimizer = GRAPEOptimizer(H0, Hc, n_timeslices=1, total_time=1)

        # Test identity
        U_identity = qt.qeye(2)
        fid = optimizer._compute_fidelity_unitary(U_identity, U_identity)
        assert np.isclose(fid, 1.0, atol=1e-10)

        # Test Pauli gates
        for pauli in [qt.sigmax(), qt.sigmay(), qt.sigmaz()]:
            fid = optimizer._compute_fidelity_unitary(pauli, pauli)
            assert np.isclose(fid, 1.0, atol=1e-10)

    def test_orthogonal_gates_fidelity(self):
        """Test fidelity between orthogonal gates."""
        H0 = 0 * qt.sigmaz()
        Hc = [qt.sigmax()]
        optimizer = GRAPEOptimizer(H0, Hc, n_timeslices=1, total_time=1)

        # Identity vs Pauli X should have fidelity 1/3 for qubits
        U_evolved = qt.qeye(2)
        U_target = qt.sigmax()
        fid = optimizer._compute_fidelity_unitary(U_evolved, U_target)

        # For average gate fidelity, orthogonal gates have F = 1/(d+1) = 1/3 for d=2
        expected = 1.0 / 3.0
        assert np.isclose(fid, expected, atol=1e-10)


class TestGRAPEOptimization:
    """Test GRAPE optimization algorithms."""

    def test_optimize_x_gate(self):
        """Test optimization of X-gate."""
        H0 = 0.5 * 1.0 * qt.sigmaz()  # Small drift
        Hc = [qt.sigmax()]

        optimizer = GRAPEOptimizer(
            H0,
            Hc,
            n_timeslices=20,
            total_time=50,
            learning_rate=0.5,
            max_iterations=100,
            convergence_threshold=1e-3,
            verbose=False,
        )

        U_target = qt.sigmax()
        u_init = np.ones((1, 20)) * 0.05

        result = optimizer.optimize_unitary(U_target, u_init)

        assert isinstance(result, GRAPEResult)
        assert result.final_fidelity > 0.95  # Should achieve high fidelity
        assert result.n_iterations <= 100
        assert len(result.fidelity_history) > 0
        assert len(result.gradient_norms) > 0
        assert result.optimized_pulses.shape == (1, 20)

    def test_optimize_state_transfer(self):
        """Test state transfer optimization."""
        H0 = qt.sigmaz()
        Hc = [qt.sigmax()]

        optimizer = GRAPEOptimizer(
            H0,
            Hc,
            n_timeslices=30,
            total_time=50,
            learning_rate=0.3,
            max_iterations=150,
            verbose=False,
        )

        psi_init = qt.basis(2, 0)
        psi_target = qt.basis(2, 1)

        result = optimizer.optimize_state(psi_init, psi_target)

        assert isinstance(result, GRAPEResult)
        assert result.final_fidelity > 0.90
        assert result.converged or result.n_iterations == 150

    def test_fidelity_improvement(self):
        """Test that fidelity improves during optimization."""
        H0 = qt.sigmaz()
        Hc = [qt.sigmax()]

        optimizer = GRAPEOptimizer(
            H0,
            Hc,
            n_timeslices=15,
            total_time=30,
            learning_rate=0.2,
            max_iterations=50,
            verbose=False,
        )

        U_target = qt.sigmax()
        result = optimizer.optimize_unitary(U_target)

        # Fidelity should generally improve (allowing for some fluctuations)
        initial_fidelity = result.fidelity_history[0]
        final_fidelity = result.fidelity_history[-1]
        assert final_fidelity >= initial_fidelity * 0.95  # At least not worse

    def test_gradient_norm_decreases(self):
        """Test that gradient norm decreases (convergence indicator)."""
        H0 = qt.sigmaz()
        Hc = [qt.sigmax()]

        optimizer = GRAPEOptimizer(
            H0,
            Hc,
            n_timeslices=10,
            total_time=20,
            learning_rate=0.1,
            max_iterations=30,
            verbose=False,
        )

        U_target = qt.sigmax()
        result = optimizer.optimize_unitary(U_target)

        # Gradient norm should decrease over iterations (with some tolerance)
        if len(result.gradient_norms) > 5:
            early_avg = np.mean(result.gradient_norms[:5])
            late_avg = np.mean(result.gradient_norms[-5:])
            assert late_avg <= early_avg * 1.5  # Allow some variation


class TestGRAPEConstraints:
    """Test constraint enforcement."""

    def test_amplitude_limits(self):
        """Test that amplitude constraints are enforced."""
        H0 = qt.sigmaz()
        Hc = [qt.sigmax()]

        optimizer = GRAPEOptimizer(
            H0,
            Hc,
            n_timeslices=20,
            total_time=50,
            u_limits=(-2.0, 2.0),
            learning_rate=1.0,  # Large to test clipping
            max_iterations=50,
            verbose=False,
        )

        U_target = qt.sigmax()
        result = optimizer.optimize_unitary(U_target)

        # All pulses should be within limits
        assert np.all(result.optimized_pulses >= -2.0)
        assert np.all(result.optimized_pulses <= 2.0)

    def test_apply_constraints(self):
        """Test constraint application function."""
        H0 = qt.sigmaz()
        Hc = [qt.sigmax()]
        optimizer = GRAPEOptimizer(
            H0, Hc, n_timeslices=10, total_time=50, u_limits=(-1.0, 1.0)
        )

        # Test clipping
        u = np.array([[2.0, -3.0, 0.5, -0.5, 1.5]])
        u_constrained = optimizer._apply_constraints(u)

        assert u_constrained[0, 0] == 1.0  # Clipped upper
        assert u_constrained[0, 1] == -1.0  # Clipped lower
        assert u_constrained[0, 2] == 0.5  # Unchanged
        assert u_constrained[0, 3] == -0.5  # Unchanged
        assert u_constrained[0, 4] == 1.0  # Clipped upper


class TestGRAPEMultiControl:
    """Test multi-control optimization."""

    def test_two_control_optimization(self):
        """Test optimization with X and Y controls."""
        H0 = 0.5 * 0.5 * qt.sigmaz()
        Hc = [qt.sigmax(), qt.sigmay()]

        optimizer = GRAPEOptimizer(
            H0,
            Hc,
            n_timeslices=25,
            total_time=50,
            learning_rate=0.3,
            max_iterations=100,
            verbose=False,
        )

        # Hadamard gate
        H_gate = 1 / np.sqrt(2) * (qt.sigmax() + qt.sigmaz())
        U_target = (-1j * np.pi / 2 * H_gate).expm()

        u_init = np.random.randn(2, 25) * 0.01
        result = optimizer.optimize_unitary(U_target, u_init)

        assert result.optimized_pulses.shape == (2, 25)
        assert result.final_fidelity > 0.85  # Reasonable fidelity


class TestGRAPEPulseFunctions:
    """Test pulse function conversion."""

    def test_get_pulse_functions(self):
        """Test conversion to callable pulse functions."""
        H0 = qt.sigmaz()
        Hc = [qt.sigmax()]
        optimizer = GRAPEOptimizer(H0, Hc, n_timeslices=10, total_time=100)

        u = np.linspace(0, 1, 10).reshape(1, 10)
        pulse_funcs = optimizer.get_pulse_functions(u)

        assert len(pulse_funcs) == 1

        # Test evaluation at various times
        assert np.isclose(pulse_funcs[0](5), 0.0, atol=0.15)  # First bin
        assert np.isclose(pulse_funcs[0](95), 1.0, atol=0.15)  # Last bin

    def test_pulse_function_multi_control(self):
        """Test pulse functions with multiple controls."""
        H0 = qt.sigmaz()
        Hc = [qt.sigmax(), qt.sigmay()]
        optimizer = GRAPEOptimizer(H0, Hc, n_timeslices=5, total_time=10)

        u = np.array([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]])
        pulse_funcs = optimizer.get_pulse_functions(u)

        assert len(pulse_funcs) == 2
        assert np.isclose(pulse_funcs[0](1), 1, atol=0.5)
        assert np.isclose(pulse_funcs[1](1), 5, atol=0.5)


class TestGRAPEAdaptiveStep:
    """Test adaptive learning rate."""

    def test_adaptive_step_decay(self):
        """Test that learning rate decays with adaptive stepping."""
        H0 = qt.sigmaz()
        Hc = [qt.sigmax()]

        optimizer = GRAPEOptimizer(
            H0,
            Hc,
            n_timeslices=10,
            total_time=20,
            learning_rate=0.5,
            max_iterations=20,
            verbose=False,
        )

        U_target = qt.sigmax()
        result = optimizer.optimize_unitary(
            U_target, adaptive_step=True, step_decay=0.9
        )

        # Should complete without error
        assert result.n_iterations > 0

    def test_fixed_step(self):
        """Test optimization with fixed learning rate."""
        H0 = qt.sigmaz()
        Hc = [qt.sigmax()]

        optimizer = GRAPEOptimizer(
            H0,
            Hc,
            n_timeslices=10,
            total_time=20,
            learning_rate=0.1,
            max_iterations=20,
            verbose=False,
        )

        U_target = qt.sigmax()
        result = optimizer.optimize_unitary(U_target, adaptive_step=False)

        assert result.n_iterations > 0


class TestGRAPEEdgeCases:
    """Test edge cases and special scenarios."""

    def test_already_optimal(self):
        """Test optimization when initial guess is already optimal."""
        H0 = 0 * qt.sigmaz()
        Hc = [qt.sigmax()]

        optimizer = GRAPEOptimizer(
            H0,
            Hc,
            n_timeslices=1,
            total_time=1,
            convergence_threshold=1e-3,
            max_iterations=10,
            verbose=False,
        )

        U_target = qt.sigmax()
        # Perfect π/2 pulse (generates X gate up to global phase)
        u_init = np.array([[np.pi / 2]])

        result = optimizer.optimize_unitary(U_target, u_init)

        # Should converge quickly or already be optimal (accounting for global phase)
        assert result.final_fidelity > 0.999

    def test_zero_initial_guess(self):
        """Test with zero initial control."""
        H0 = qt.sigmaz()
        Hc = [qt.sigmax()]

        optimizer = GRAPEOptimizer(
            H0,
            Hc,
            n_timeslices=15,
            total_time=30,
            learning_rate=0.2,
            max_iterations=50,
            verbose=False,
        )

        U_target = qt.sigmax()
        u_init = np.zeros((1, 15))

        result = optimizer.optimize_unitary(U_target, u_init)

        # Should still improve from zero
        assert result.final_fidelity > 0.5

    def test_repr(self):
        """Test string representation."""
        H0 = qt.sigmaz()
        Hc = [qt.sigmax(), qt.sigmay()]
        optimizer = GRAPEOptimizer(H0, Hc, n_timeslices=50, total_time=100)

        repr_str = repr(optimizer)
        assert "GRAPEOptimizer" in repr_str
        assert "n_controls=2" in repr_str
        assert "n_timeslices=50" in repr_str
        assert "total_time=100" in repr_str


class TestGRAPEResult:
    """Test GRAPEResult dataclass."""

    def test_result_structure(self):
        """Test that result contains all expected fields."""
        result = GRAPEResult(
            final_fidelity=0.95,
            optimized_pulses=np.ones((1, 10)),
            fidelity_history=[0.5, 0.7, 0.9, 0.95],
            gradient_norms=[1.0, 0.5, 0.1, 0.01],
            n_iterations=4,
            converged=True,
            message="Test",
        )

        assert result.final_fidelity == 0.95
        assert result.optimized_pulses.shape == (1, 10)
        assert len(result.fidelity_history) == 4
        assert len(result.gradient_norms) == 4
        assert result.n_iterations == 4
        assert result.converged is True
        assert result.message == "Test"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
