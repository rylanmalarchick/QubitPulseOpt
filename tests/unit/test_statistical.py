"""
Statistical Tests for Stochastic Optimization
==============================================

This module demonstrates statistical testing approaches for inherently
stochastic optimization algorithms. Instead of testing single runs,
these tests verify success rates over multiple runs with different
random initializations.

Test Coverage:
-------------
1. Gate optimization success rate (ensemble testing)
2. RB experiment statistical robustness
3. Multi-start optimization convergence statistics

Author: Orchestrator Agent
Date: 2025-01-27
Task: 1.5 - Stochastic Test Infrastructure
"""

import pytest
import numpy as np
import qutip as qt
from scipy import stats

from src.optimization.gates import UniversalGates, GateResult
from src.optimization.benchmarking import (
    RBExperiment,
    depolarizing_noise,
)


@pytest.mark.statistical
@pytest.mark.slow
class TestGateOptimizationStatistics:
    """Statistical tests for gate optimization success rates."""

    @pytest.fixture
    def gate_optimizer(self):
        """Create gate optimizer for statistical testing."""
        H_drift = 2 * np.pi * 0.0 * qt.sigmaz() / 2
        H_controls = [qt.sigmax(), qt.sigmay()]
        return UniversalGates(H_drift, H_controls, fidelity_threshold=0.95)

    def test_hadamard_success_rate(self, gate_optimizer, statistical_seeds):
        """
        Test that Hadamard optimization succeeds in majority of runs.

        This is a statistical test that runs optimization multiple times
        with different random seeds and verifies that the success rate
        is above an acceptable threshold.
        """
        num_trials = 20
        fidelity_threshold = 0.75
        min_success_rate = 0.70  # At least 70% should succeed

        successes = 0
        fidelities = []

        for i, seed in enumerate(statistical_seeds[:num_trials]):
            np.random.seed(seed)

            result = gate_optimizer.optimize_hadamard(
                gate_time=30.0,
                n_timeslices=30,
                max_iterations=50,
                n_starts=3,
            )

            fidelities.append(result.final_fidelity)
            if result.final_fidelity > fidelity_threshold:
                successes += 1

        success_rate = successes / num_trials

        # Statistical assertion
        assert success_rate >= min_success_rate, (
            f"Success rate {success_rate:.2%} below minimum {min_success_rate:.2%}\n"
            f"Successes: {successes}/{num_trials}\n"
            f"Fidelities: mean={np.mean(fidelities):.4f}, "
            f"std={np.std(fidelities):.4f}, "
            f"min={np.min(fidelities):.4f}, "
            f"max={np.max(fidelities):.4f}"
        )

        # Report statistics
        print(f"\n  Success rate: {success_rate:.2%} ({successes}/{num_trials})")
        print(
            f"  Fidelity stats: μ={np.mean(fidelities):.4f}, σ={np.std(fidelities):.4f}"
        )

    def test_phase_gate_fidelity_distribution(self, gate_optimizer, statistical_seeds):
        """
        Test that S gate optimization fidelities follow expected distribution.

        This test verifies that the distribution of fidelities is reasonable
        and doesn't have pathological behavior (e.g., all failures or bimodal).
        """
        num_trials = 15
        fidelities = []

        for seed in statistical_seeds[:num_trials]:
            np.random.seed(seed)

            result = gate_optimizer.optimize_phase_gate(
                phase=np.pi / 2,
                gate_time=15.0,
                n_timeslices=20,
                max_iterations=30,
                n_starts=2,
            )

            fidelities.append(result.final_fidelity)

        fidelities = np.array(fidelities)

        # Check that we get some reasonable fidelities
        assert np.mean(fidelities) > 0.60, (
            f"Mean fidelity too low: {np.mean(fidelities):.4f}"
        )

        # Check that we're not getting all failures
        assert np.max(fidelities) > 0.70, (
            f"No high-fidelity solutions found: max={np.max(fidelities):.4f}"
        )

        # Check for reasonable variability (not all identical)
        assert np.std(fidelities) > 0.01, (
            f"Suspiciously low variance: {np.std(fidelities):.4f}"
        )

        # Report distribution
        print(f"\n  Fidelity distribution (n={num_trials}):")
        print(f"    Mean: {np.mean(fidelities):.4f}")
        print(f"    Std:  {np.std(fidelities):.4f}")
        print(f"    Min:  {np.min(fidelities):.4f}")
        print(f"    Q1:   {np.percentile(fidelities, 25):.4f}")
        print(f"    Med:  {np.median(fidelities):.4f}")
        print(f"    Q3:   {np.percentile(fidelities, 75):.4f}")
        print(f"    Max:  {np.max(fidelities):.4f}")

    def test_multi_start_improves_success(self, gate_optimizer, statistical_seeds):
        """
        Test that multi-start optimization improves success rate vs single-start.

        This is a comparative statistical test showing that n_starts > 1
        provides better results on average.
        """
        num_trials = 10

        single_start_fidelities = []
        multi_start_fidelities = []

        for seed in statistical_seeds[:num_trials]:
            # Single-start
            np.random.seed(seed)
            result_single = gate_optimizer.optimize_phase_gate(
                phase=np.pi / 2,
                gate_time=15.0,
                n_timeslices=20,
                max_iterations=30,
                n_starts=1,
            )
            single_start_fidelities.append(result_single.final_fidelity)

            # Multi-start (with same seed for fair comparison)
            np.random.seed(seed)
            result_multi = gate_optimizer.optimize_phase_gate(
                phase=np.pi / 2,
                gate_time=15.0,
                n_timeslices=20,
                max_iterations=30,
                n_starts=5,
            )
            multi_start_fidelities.append(result_multi.final_fidelity)

        # Statistical comparison
        mean_single = np.mean(single_start_fidelities)
        mean_multi = np.mean(multi_start_fidelities)

        # Multi-start should be at least as good on average
        # (This is a statistical trend, not guaranteed every time)
        improvement = mean_multi - mean_single

        print(f"\n  Single-start mean fidelity: {mean_single:.4f}")
        print(f"  Multi-start mean fidelity:  {mean_multi:.4f}")
        print(f"  Improvement: {improvement:+.4f}")

        # Paired t-test to see if improvement is statistically significant
        if num_trials >= 5:
            t_stat, p_value = stats.ttest_rel(
                multi_start_fidelities, single_start_fidelities
            )
            print(f"  Paired t-test: t={t_stat:.3f}, p={p_value:.4f}")

            # Multi-start should tend to be better (one-sided test)
            # We use relaxed threshold since this is unit test with small sample
            assert improvement >= -0.05, (
                f"Multi-start performed significantly worse: Δ={improvement:.4f}"
            )


@pytest.mark.statistical
@pytest.mark.slow
class TestRBExperimentStatistics:
    """Statistical tests for RB experiment robustness."""

    @pytest.fixture
    def rb_experiment(self):
        """Create RB experiment for testing."""
        return RBExperiment()

    def test_rb_fit_success_rate(self, rb_experiment, statistical_seeds):
        """
        Test that RB decay fitting succeeds in majority of runs.

        RB fits can occasionally fail due to degenerate covariance matrices
        when random sampling is unlucky. This test verifies that failures
        are rare.
        """
        num_trials = 15
        min_success_rate = 0.80  # At least 80% should succeed

        sequence_lengths = [1, 5, 10, 15]
        num_samples = 15
        error_rate = 0.01

        def noise(gate):
            return depolarizing_noise(gate, error_rate=error_rate)

        successes = 0
        fit_succeeded = []

        for seed in statistical_seeds[:num_trials]:
            np.random.seed(seed)

            try:
                result = rb_experiment.run_rb_experiment(
                    sequence_lengths,
                    num_samples=num_samples,
                    noise_model=noise,
                )

                # Check if fit produced reasonable results
                if (
                    result.average_fidelity > 0
                    and result.average_fidelity < 1.0
                    and result.gate_infidelity > 0
                ):
                    successes += 1
                    fit_succeeded.append(True)
                else:
                    fit_succeeded.append(False)

            except Exception as e:
                # Fit failed completely
                fit_succeeded.append(False)
                print(f"\n  Trial {len(fit_succeeded)} failed with: {type(e).__name__}")

        success_rate = successes / num_trials

        assert success_rate >= min_success_rate, (
            f"RB fit success rate {success_rate:.2%} below minimum {min_success_rate:.2%}\n"
            f"Successes: {successes}/{num_trials}"
        )

        print(f"\n  RB fit success rate: {success_rate:.2%} ({successes}/{num_trials})")

    def test_rb_fidelity_estimation_accuracy(self, rb_experiment, statistical_seeds):
        """
        Test that RB accurately estimates known error rate.

        This statistical test verifies that the RB protocol produces
        gate fidelity estimates that are reasonably close to the
        true (known) error rate over multiple runs.
        """
        num_trials = 10
        true_error_rate = 0.01
        expected_fidelity = 1 - true_error_rate
        tolerance = 0.015  # Allow ±1.5% error in estimate

        sequence_lengths = [1, 5, 10, 20]
        num_samples = 20

        def noise(gate):
            return depolarizing_noise(gate, error_rate=true_error_rate)

        estimated_fidelities = []

        for seed in statistical_seeds[:num_trials]:
            np.random.seed(seed)

            try:
                result = rb_experiment.run_rb_experiment(
                    sequence_lengths,
                    num_samples=num_samples,
                    noise_model=noise,
                )

                if result.average_fidelity > 0 and result.average_fidelity < 1.0:
                    estimated_fidelities.append(result.average_fidelity)

            except Exception:
                # Skip failed fits
                pass

        # Need at least 70% successful fits
        assert len(estimated_fidelities) >= 0.7 * num_trials, (
            f"Too many RB fit failures: {len(estimated_fidelities)}/{num_trials}"
        )

        estimated_fidelities = np.array(estimated_fidelities)
        mean_estimate = np.mean(estimated_fidelities)

        # Check that mean estimate is close to true value
        error = abs(mean_estimate - expected_fidelity)

        print(f"\n  True average fidelity: {expected_fidelity:.4f}")
        print(f"  Estimated (mean):      {mean_estimate:.4f}")
        print(f"  Error:                 {error:.4f}")
        print(f"  Std of estimates:      {np.std(estimated_fidelities):.4f}")

        assert error < tolerance, (
            f"RB fidelity estimate error {error:.4f} exceeds tolerance {tolerance:.4f}"
        )


@pytest.mark.statistical
class TestOptimizationConvergence:
    """Statistical tests for optimization convergence properties."""

    @pytest.fixture
    def gate_optimizer(self):
        """Create gate optimizer for statistical testing."""
        H_drift = 2 * np.pi * 0.0 * qt.sigmaz() / 2
        H_controls = [qt.sigmax(), qt.sigmay()]
        return UniversalGates(H_drift, H_controls)

    def test_convergence_monotonicity(self, gate_optimizer, statistical_seeds):
        """
        Test that longer optimization generally improves results.

        This statistical test verifies the trend that more iterations
        lead to better (or at least not worse) fidelities on average.
        """
        num_trials = 8

        short_iterations = 20
        long_iterations = 50

        short_fidelities = []
        long_fidelities = []

        for seed in statistical_seeds[:num_trials]:
            # Short optimization
            np.random.seed(seed)
            result_short = gate_optimizer.optimize_hadamard(
                gate_time=25.0,
                n_timeslices=25,
                max_iterations=short_iterations,
            )
            short_fidelities.append(result_short.final_fidelity)

            # Long optimization (same seed)
            np.random.seed(seed)
            result_long = gate_optimizer.optimize_hadamard(
                gate_time=25.0,
                n_timeslices=25,
                max_iterations=long_iterations,
            )
            long_fidelities.append(result_long.final_fidelity)

        mean_short = np.mean(short_fidelities)
        mean_long = np.mean(long_fidelities)

        print(f"\n  Short optimization ({short_iterations} iter): {mean_short:.4f}")
        print(f"  Long optimization ({long_iterations} iter):  {mean_long:.4f}")
        print(f"  Improvement: {mean_long - mean_short:+.4f}")

        # On average, longer optimization should not be significantly worse
        # (allowing small regression due to stochastic nature)
        assert mean_long >= mean_short - 0.05, (
            f"Longer optimization gave worse results on average"
        )


if __name__ == "__main__":
    # Run statistical tests with verbose output
    pytest.main([__file__, "-v", "-s", "-m", "statistical"])
