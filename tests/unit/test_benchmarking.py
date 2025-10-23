"""
Unit tests for randomized benchmarking module.

Tests cover:
- Clifford group generation and sampling
- RB sequence generation
- Decay curve fitting and fidelity extraction
- Interleaved RB
- Noise models
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_less
import qutip as qt

from src.optimization.benchmarking import (
    CliffordGroup,
    RBSequenceGenerator,
    RBExperiment,
    InterleavedRB,
    RBResult,
    depolarizing_noise,
    amplitude_damping_noise,
    visualize_rb_decay,
)


# Fixtures
@pytest.fixture
def clifford_group():
    """Clifford group instance."""
    return CliffordGroup()


@pytest.fixture
def rb_sequence_gen(clifford_group):
    """RB sequence generator instance."""
    return RBSequenceGenerator(clifford_group)


@pytest.fixture
def rb_experiment():
    """RB experiment instance."""
    return RBExperiment()


@pytest.fixture
def interleaved_rb():
    """Interleaved RB instance."""
    return InterleavedRB()


# Test CliffordGroup
class TestCliffordGroup:
    """Tests for CliffordGroup class."""

    def test_initialization(self, clifford_group):
        """Test Clifford group initialization."""
        assert clifford_group.num_cliffords == 24
        assert len(clifford_group._clifford_gates) == 24

    def test_pauli_operators(self, clifford_group):
        """Test Pauli operators are correct."""
        # Check dimensions
        assert clifford_group.I.dims == [[2], [2]]
        assert clifford_group.X.dims == [[2], [2]]
        assert clifford_group.Y.dims == [[2], [2]]
        assert clifford_group.Z.dims == [[2], [2]]

        # Check Pauli algebra
        # X^2 = Y^2 = Z^2 = I
        assert (clifford_group.X * clifford_group.X - clifford_group.I).norm() < 1e-10
        assert (clifford_group.Y * clifford_group.Y - clifford_group.I).norm() < 1e-10
        assert (clifford_group.Z * clifford_group.Z - clifford_group.I).norm() < 1e-10

    def test_clifford_generators(self, clifford_group):
        """Test Hadamard and S gate."""
        # H^2 = I
        H = clifford_group.H
        assert (H * H - clifford_group.I).norm() < 1e-10

        # S^4 = I
        S = clifford_group.S
        S4 = S * S * S * S
        assert (S4 - clifford_group.I).norm() < 1e-10

    def test_all_cliffords_are_unitary(self, clifford_group):
        """Test that all Clifford gates are unitary."""
        for gate in clifford_group._clifford_gates:
            # U·U† = I
            product = gate * gate.dag()
            diff = (product - clifford_group.I).norm()
            assert diff < 1e-10, f"Gate is not unitary: {diff}"

    def test_get_random_clifford(self, clifford_group):
        """Test random Clifford sampling."""
        gate = clifford_group.get_random_clifford()
        assert gate in clifford_group._clifford_gates

    def test_get_clifford_by_index(self, clifford_group):
        """Test getting Clifford by index."""
        gate = clifford_group.get_clifford(0)
        assert gate is not None

        gate = clifford_group.get_clifford(23)
        assert gate is not None

    def test_invalid_clifford_index(self, clifford_group):
        """Test error on invalid index."""
        with pytest.raises(ValueError, match="Clifford index must be 0-23"):
            clifford_group.get_clifford(24)

        with pytest.raises(ValueError):
            clifford_group.get_clifford(-1)

    def test_find_inverse(self, clifford_group):
        """Test finding inverse Clifford."""
        gate = clifford_group.get_clifford(5)
        inverse = clifford_group.find_inverse(gate)

        # Product should be identity (up to global phase)
        product = gate * inverse
        overlap = np.abs((product.dag() * clifford_group.I).tr())
        assert np.abs(overlap - 2) < 1e-10

    def test_compose_cliffords(self, clifford_group):
        """Test composing multiple Cliffords."""
        gates = [clifford_group.get_clifford(i) for i in range(3)]
        composed = clifford_group.compose_cliffords(gates)

        # Manual composition
        manual = gates[2] * gates[1] * gates[0]

        assert (composed - manual).norm() < 1e-10

    def test_clifford_closure(self, clifford_group):
        """Test that Clifford group is closed under composition."""
        # Sample a few random products
        for _ in range(10):
            g1 = clifford_group.get_random_clifford()
            g2 = clifford_group.get_random_clifford()
            product = g2 * g1

            # Check if product is in the group (up to phase)
            found = False
            for c in clifford_group._clifford_gates:
                overlap = np.abs((product.dag() * c).tr())
                # More lenient tolerance for numerical precision
                if np.abs(overlap - 2) < 1e-8:
                    found = True
                    break

            assert found, "Clifford group not closed under composition"


# Test RBSequenceGenerator
class TestRBSequenceGenerator:
    """Tests for RB sequence generator."""

    def test_initialization(self, rb_sequence_gen):
        """Test sequence generator initialization."""
        assert rb_sequence_gen.clifford_group is not None

    def test_generate_sequence_basic(self, rb_sequence_gen):
        """Test basic sequence generation."""
        length = 10
        sequence, recovery = rb_sequence_gen.generate_sequence(length)

        assert len(sequence) == length
        assert recovery is not None
        assert all(isinstance(gate, qt.Qobj) for gate in sequence)

    def test_sequence_implements_identity(self, rb_sequence_gen):
        """Test that sequence + recovery = identity."""
        length = 5
        sequence, recovery = rb_sequence_gen.generate_sequence(length)

        # Compose entire sequence including recovery
        full_sequence = sequence + [recovery]
        product = rb_sequence_gen.clifford_group.compose_cliffords(full_sequence)

        # Should be identity (up to global phase)
        I = rb_sequence_gen.clifford_group.I
        overlap = np.abs((product.dag() * I).tr())
        assert np.abs(overlap - 2) < 1e-9

    def test_generate_sequence_no_recovery(self, rb_sequence_gen):
        """Test sequence generation without recovery gate."""
        length = 10
        sequence, recovery = rb_sequence_gen.generate_sequence(
            length, return_recovery=False
        )

        assert len(sequence) == length
        assert recovery is None

    def test_generate_rb_sequences(self, rb_sequence_gen):
        """Test generating multiple sequences."""
        lengths = [5, 10, 15]
        num_samples = 10

        rb_sequences = rb_sequence_gen.generate_rb_sequences(
            lengths, num_samples=num_samples
        )

        assert len(rb_sequences) == len(lengths)
        for length in lengths:
            assert len(rb_sequences[length]) == num_samples
            for seq, recovery in rb_sequences[length]:
                assert len(seq) == length
                assert recovery is not None

    def test_zero_length_sequence(self, rb_sequence_gen):
        """Test zero-length sequence."""
        sequence, recovery = rb_sequence_gen.generate_sequence(0)

        assert len(sequence) == 0
        # Recovery should be identity or close to it
        I = rb_sequence_gen.clifford_group.I
        overlap = np.abs((recovery.dag() * I).tr())
        assert np.abs(overlap - 2) < 1e-9

    def test_invalid_length(self, rb_sequence_gen):
        """Test error on invalid sequence length."""
        with pytest.raises(ValueError, match="Sequence length must be non-negative"):
            rb_sequence_gen.generate_sequence(-1)


# Test RBExperiment
class TestRBExperiment:
    """Tests for RB experiment simulation."""

    def test_initialization(self, rb_experiment):
        """Test RB experiment initialization."""
        assert rb_experiment.initial_state is not None
        assert rb_experiment.measurement_basis is not None
        assert rb_experiment.clifford_group is not None

    def test_simulate_sequence_ideal(self, rb_experiment):
        """Test simulating ideal (noiseless) sequence."""
        # Generate a sequence
        length = 5
        sequence, recovery = rb_experiment.sequence_gen.generate_sequence(length)

        # Simulate without noise
        prob = rb_experiment.simulate_sequence(sequence, recovery, noise_model=None)

        # Should return to initial state (probability ≈ 1)
        assert prob > 0.99

    def test_simulate_sequence_with_noise(self, rb_experiment):
        """Test simulating sequence with noise."""
        length = 10
        sequence, recovery = rb_experiment.sequence_gen.generate_sequence(length)

        # Simple noise model
        def noise(gate):
            return depolarizing_noise(gate, error_rate=0.01)

        prob = rb_experiment.simulate_sequence(sequence, recovery, noise_model=noise)

        # With noise, survival probability should be less than ideal
        assert 0 <= prob <= 1

    def test_run_rb_experiment_ideal(self, rb_experiment):
        """Test running full RB experiment without noise."""
        sequence_lengths = [1, 5, 10]
        num_samples = 10

        result = rb_experiment.run_rb_experiment(
            sequence_lengths, num_samples=num_samples, noise_model=None
        )

        assert isinstance(result, RBResult)
        assert len(result.sequence_lengths) == len(sequence_lengths)
        assert len(result.survival_probabilities) == len(sequence_lengths)
        # Ideal case should have high fidelity
        assert result.average_fidelity > 0.95

    def test_run_rb_experiment_with_noise(self, rb_experiment):
        """Test RB experiment with depolarizing noise."""
        sequence_lengths = [1, 5, 10, 15]
        num_samples = 15
        error_rate = 0.01

        def noise(gate):
            return depolarizing_noise(gate, error_rate=error_rate)

        result = rb_experiment.run_rb_experiment(
            sequence_lengths, num_samples=num_samples, noise_model=noise
        )

        assert isinstance(result, RBResult)
        # With noise, fidelity should be lower
        assert result.average_fidelity < 1.0
        assert result.gate_infidelity > 0

    def test_fit_rb_decay(self, rb_experiment):
        """Test RB decay curve fitting."""
        # Synthetic decay data
        sequence_lengths = np.array([1, 5, 10, 20, 30])
        A, p, B = 0.5, 0.98, 0.5
        survival_probs = A * p**sequence_lengths + B

        result = rb_experiment.fit_rb_decay(sequence_lengths, survival_probs)

        # Check fitted parameters are close to true values
        assert_allclose(result.fit_parameters["A"], A, rtol=0.1)
        assert_allclose(result.fit_parameters["p"], p, rtol=0.05)
        assert_allclose(result.fit_parameters["B"], B, rtol=0.1)

    def test_rb_result_attributes(self, rb_experiment):
        """Test that RBResult has all required attributes."""
        sequence_lengths = [1, 5, 10]
        num_samples = 10

        result = rb_experiment.run_rb_experiment(sequence_lengths, num_samples)

        assert hasattr(result, "sequence_lengths")
        assert hasattr(result, "survival_probabilities")
        assert hasattr(result, "fit_parameters")
        assert hasattr(result, "average_fidelity")
        assert hasattr(result, "gate_infidelity")
        assert hasattr(result, "std_error")

        # Check fit parameters
        assert "A" in result.fit_parameters
        assert "p" in result.fit_parameters
        assert "B" in result.fit_parameters


# Test InterleavedRB
class TestInterleavedRB:
    """Tests for interleaved randomized benchmarking."""

    def test_initialization(self, interleaved_rb):
        """Test interleaved RB initialization."""
        assert interleaved_rb.clifford_group is not None
        assert interleaved_rb.rb_experiment is not None

    def test_generate_interleaved_sequence(self, interleaved_rb):
        """Test generating interleaved sequence."""
        target_gate = interleaved_rb.clifford_group.X
        length = 5

        sequence, recovery = interleaved_rb.generate_interleaved_sequence(
            length, target_gate
        )

        # Sequence should have length 2*length (Clifford + target alternating)
        # Plus recovery gate
        assert len(sequence) == 2 * length

    def test_interleaved_sequence_identity(self, interleaved_rb):
        """Test that interleaved sequence + recovery = identity."""
        target_gate = interleaved_rb.clifford_group.H
        length = 3

        sequence, recovery = interleaved_rb.generate_interleaved_sequence(
            length, target_gate
        )

        # Compose full sequence
        full_sequence = sequence + [recovery]
        product = interleaved_rb.clifford_group.compose_cliffords(full_sequence)

        # Should be identity (up to global phase)
        I = interleaved_rb.clifford_group.I
        overlap = np.abs((product.dag() * I).tr())
        assert np.abs(overlap - 2) < 1e-8

    def test_run_interleaved_rb(self, interleaved_rb):
        """Test running interleaved RB experiment."""
        target_gate = interleaved_rb.clifford_group.X
        sequence_lengths = [1, 3, 5]
        num_samples = 10

        standard, interleaved, F_gate = interleaved_rb.run_interleaved_rb(
            target_gate, sequence_lengths, num_samples, noise_model=None
        )

        assert isinstance(standard, RBResult)
        assert isinstance(interleaved, RBResult)
        # Ideal case: target gate fidelity should be high
        assert F_gate > 0.9

    def test_interleaved_rb_with_noise(self, interleaved_rb):
        """Test interleaved RB with noisy target gate."""
        # Perfect Cliffords, noisy target gate
        target_gate = interleaved_rb.clifford_group.X
        sequence_lengths = [1, 3, 5]
        num_samples = 10

        # Create noise model that affects target gate more
        def noise(gate):
            # Check if gate is approximately X
            overlap = np.abs((gate.dag() * target_gate).tr())
            if np.abs(overlap - 2) < 0.1:
                # Target gate: add more noise
                return depolarizing_noise(gate, error_rate=0.05)
            else:
                # Other gates: less noise
                return depolarizing_noise(gate, error_rate=0.001)

        standard, interleaved, F_gate = interleaved_rb.run_interleaved_rb(
            target_gate, sequence_lengths, num_samples, noise_model=noise
        )

        # Interleaved should have lower fidelity than standard
        assert interleaved.average_fidelity <= standard.average_fidelity
        # Target gate fidelity should reflect added noise
        assert F_gate < 1.0


# Test noise models
class TestNoiseModels:
    """Tests for noise model functions."""

    def test_depolarizing_noise_zero_rate(self):
        """Test depolarizing noise with zero error rate."""
        gate = qt.sigmax()
        noisy = depolarizing_noise(gate, error_rate=0.0)

        # Should be unchanged
        assert (noisy - gate).norm() < 1e-10

    def test_depolarizing_noise_nonzero_rate(self):
        """Test depolarizing noise with nonzero error rate."""
        gate = qt.sigmax()
        # Run multiple times due to randomness
        results = []
        for _ in range(20):
            noisy = depolarizing_noise(gate, error_rate=0.5)
            results.append(noisy)

        # At least some should be different from original (stochastic)
        different_count = sum(1 for r in results if (r - gate).norm() > 1e-10)
        assert different_count > 0

    def test_amplitude_damping_noise(self):
        """Test amplitude damping noise model."""
        gate = qt.gates.hadamard_transform()
        T1 = 1e-5
        gate_time = 1e-6

        noisy = amplitude_damping_noise(gate, T1, gate_time)

        # Should return a valid Qobj
        assert isinstance(noisy, qt.Qobj)


# Test visualization
class TestVisualization:
    """Tests for visualization functions."""

    def test_visualize_rb_decay_basic(self, rb_experiment):
        """Test basic RB decay visualization."""
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        # Create synthetic result
        sequence_lengths = np.array([1, 5, 10, 20])
        survival_probs = 0.5 * 0.98**sequence_lengths + 0.5
        result = rb_experiment.fit_rb_decay(sequence_lengths, survival_probs)

        fig, ax = plt.subplots()
        returned_ax = visualize_rb_decay(result, ax=ax)

        assert returned_ax is ax
        plt.close(fig)

    def test_visualize_rb_decay_no_fit(self, rb_experiment):
        """Test visualization without fit curve."""
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        sequence_lengths = np.array([1, 5, 10])
        survival_probs = np.array([0.95, 0.85, 0.75])
        result = rb_experiment.fit_rb_decay(sequence_lengths, survival_probs)

        fig, ax = plt.subplots()
        visualize_rb_decay(result, ax=ax, show_fit=False)
        plt.close(fig)


# Integration tests
class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_rb_pipeline(self):
        """Test complete RB pipeline from sequence generation to fidelity extraction."""
        # Setup
        clifford_group = CliffordGroup()
        rb_exp = RBExperiment()

        # Define noise
        error_rate = 0.005

        def noise(gate):
            return depolarizing_noise(gate, error_rate=error_rate)

        # Run RB
        sequence_lengths = [1, 5, 10, 15, 20]
        num_samples = 20

        result = rb_exp.run_rb_experiment(
            sequence_lengths, num_samples=num_samples, noise_model=noise
        )

        # Verify result
        assert isinstance(result, RBResult)
        assert result.average_fidelity < 1.0
        assert result.gate_infidelity > 0
        # Rough check: fidelity should be reasonable for this error rate
        assert 0.8 < result.average_fidelity < 1.0

    def test_compare_standard_vs_interleaved(self):
        """Test comparing standard and interleaved RB."""
        interleaved_rb = InterleavedRB()
        target_gate = interleaved_rb.clifford_group.H

        sequence_lengths = [1, 3, 5, 7]
        num_samples = 15

        # Ideal case
        standard, interleaved, F_gate = interleaved_rb.run_interleaved_rb(
            target_gate, sequence_lengths, num_samples, noise_model=None
        )

        # Both should have high fidelity in ideal case
        assert standard.average_fidelity > 0.95
        assert interleaved.average_fidelity > 0.95
        assert F_gate > 0.95

    def test_rb_with_multiple_noise_levels(self):
        """Test RB at different noise levels."""
        rb_exp = RBExperiment()
        sequence_lengths = [1, 5, 10]
        num_samples = 15

        error_rates = [0.001, 0.01, 0.05]
        fidelities = []

        for rate in error_rates:

            def noise(gate):
                return depolarizing_noise(gate, error_rate=rate)

            result = rb_exp.run_rb_experiment(
                sequence_lengths, num_samples=num_samples, noise_model=noise
            )
            fidelities.append(result.average_fidelity)

        # Fidelity should decrease with increasing error rate (allow for small numerical errors)
        assert fidelities[0] >= fidelities[1] >= fidelities[2]


# Performance and edge cases
class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_gate_sequence(self, rb_sequence_gen):
        """Test sequence of length 1."""
        sequence, recovery = rb_sequence_gen.generate_sequence(1)

        assert len(sequence) == 1

        # Should still implement identity with recovery
        full = sequence + [recovery]
        product = rb_sequence_gen.clifford_group.compose_cliffords(full)
        I = rb_sequence_gen.clifford_group.I
        overlap = np.abs((product.dag() * I).tr())
        assert np.abs(overlap - 2) < 1e-9

    def test_very_long_sequence(self, rb_sequence_gen):
        """Test generating very long sequence."""
        length = 100
        sequence, recovery = rb_sequence_gen.generate_sequence(length)

        assert len(sequence) == length

    def test_rb_with_short_sequences_only(self, rb_experiment):
        """Test RB with only short sequences."""
        sequence_lengths = [1, 2, 3]
        num_samples = 10

        result = rb_experiment.run_rb_experiment(sequence_lengths, num_samples)

        # Should still produce valid result
        assert isinstance(result, RBResult)
        assert 0 <= result.average_fidelity <= 1

    def test_rb_result_repr(self):
        """Test RBResult string representation."""
        result = RBResult(
            sequence_lengths=np.array([1, 5, 10]),
            survival_probabilities=np.array([0.95, 0.85, 0.75]),
            fit_parameters={"A": 0.5, "p": 0.95, "B": 0.5},
            average_fidelity=0.975,
            gate_infidelity=0.025,
        )

        repr_str = repr(result)
        assert "F_avg" in repr_str
        assert "0.975" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
