"""
Task 3 Demonstration: Filter Functions & Randomized Benchmarking

This script demonstrates the key features implemented in Task 3:
1. Filter function analysis for noise spectroscopy
2. Noise-tailored pulse optimization
3. Randomized benchmarking for gate fidelity extraction
4. Fisher information and worst-case analysis

Author: Orchestrator Agent
Date: 2025-01-28
"""

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

from src.optimization.filter_functions import (
    FilterFunctionCalculator,
    NoisePSD,
    NoiseInfidelityCalculator,
    NoiseTailoredOptimizer,
    analyze_pulse_noise_sensitivity,
    visualize_filter_function,
)
from src.optimization.benchmarking import (
    RBExperiment,
    InterleavedRB,
    depolarizing_noise,
    visualize_rb_decay,
)
from src.optimization.robustness import RobustnessTester


def demo_filter_functions():
    """Demonstrate filter function analysis."""
    print("\n" + "=" * 70)
    print("DEMO 1: Filter Function Analysis")
    print("=" * 70)

    # Create a Gaussian pulse
    times = np.linspace(0, 10e-6, 200)
    t0 = 5e-6
    sigma = 1e-6
    amplitude = 1e6
    amplitudes = amplitude * np.exp(-((times - t0) ** 2) / (2 * sigma**2))

    print(f"\nPulse parameters:")
    print(f"  Duration: {times[-1] * 1e6:.2f} μs")
    print(f"  Peak amplitude: {amplitude / 1e6:.2f} MHz")
    print(f"  Pulse width (σ): {sigma * 1e9:.2f} ns")

    # Analyze under different noise models
    print("\nAnalyzing noise sensitivity...")
    results = analyze_pulse_noise_sensitivity(times, amplitudes, noise_type="amplitude")

    print("\nNoise infidelity results:")
    for name, result in results.items():
        print(f"  {name:12s}: χ = {result.noise_infidelity:.4e}")

    # Visualize filter function
    ff_calc = FilterFunctionCalculator(n_freq=200)
    ff_result = ff_calc.compute_from_pulse(times, amplitudes, noise_type="amplitude")

    print(f"\nFilter function computed:")
    print(
        f"  Frequency range: [{ff_result.frequencies[0] / 1e6:.2f}, "
        f"{ff_result.frequencies[-1] / 1e6:.2f}] MHz"
    )
    print(f"  Max F(ω): {np.max(ff_result.filter_function):.4e}")

    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    psd = NoisePSD.one_over_f(amplitude=1e-8, alpha=1.0)
    visualize_filter_function(ff_result, noise_psd=psd, log_scale=True, ax=ax)
    plt.tight_layout()
    plt.savefig("examples/filter_function_demo.png", dpi=150)
    print(f"\n✅ Filter function plot saved to examples/filter_function_demo.png")


def demo_noise_tailored_optimization():
    """Demonstrate noise-tailored pulse optimization."""
    print("\n" + "=" * 70)
    print("DEMO 2: Noise-Tailored Optimization")
    print("=" * 70)

    # Start with a simple square pulse
    times = np.linspace(0, 10e-6, 50)
    initial_amps = np.ones(50) * 5e5

    # Define target noise environment (1/f noise)
    psd = NoisePSD.one_over_f(amplitude=1e-8, alpha=1.0)

    print(f"\nOptimizing pulse for 1/f noise...")
    print(f"  Initial pulse: square, amplitude = {initial_amps[0] / 1e6:.2f} MHz")

    # Compute initial infidelity
    infid_calc = NoiseInfidelityCalculator()
    initial_result = infid_calc.compute_from_pulse(
        times, initial_amps, psd, noise_type="amplitude"
    )
    print(f"  Initial infidelity: χ = {initial_result.noise_infidelity:.4e}")

    # Optimize
    optimizer = NoiseTailoredOptimizer()
    opt_result = optimizer.optimize_pulse_shape(
        times,
        initial_amps,
        psd,
        noise_type="amplitude",
        constraints={"max_amplitude": 1e6},
        max_iter=50,
    )

    print(f"\n✅ Optimization complete:")
    print(f"  Final infidelity: χ = {opt_result['infidelity']:.4e}")
    print(
        f"  Improvement: {(1 - opt_result['infidelity'] / initial_result.noise_infidelity) * 100:.1f}%"
    )
    print(f"  Iterations: {opt_result['iterations']}")


def demo_randomized_benchmarking():
    """Demonstrate randomized benchmarking."""
    print("\n" + "=" * 70)
    print("DEMO 3: Randomized Benchmarking")
    print("=" * 70)

    # Setup RB experiment
    rb_exp = RBExperiment()

    # Define noise model
    error_rate = 0.01
    print(f"\nSimulating gates with depolarizing error rate: {error_rate}")

    def noise(gate):
        return depolarizing_noise(gate, error_rate=error_rate)

    # Run RB
    sequence_lengths = [1, 5, 10, 15, 20, 30, 50]
    num_samples = 30
    print(f"Running RB with sequence lengths: {sequence_lengths}")
    print(f"Samples per length: {num_samples}")

    result = rb_exp.run_rb_experiment(
        sequence_lengths, num_samples=num_samples, noise_model=noise
    )

    print(f"\n✅ RB Results:")
    print(f"  Depolarizing parameter p: {result.fit_parameters['p']:.6f}")
    print(f"  Average gate fidelity: {result.average_fidelity:.6f}")
    print(f"  Gate infidelity (error rate): {result.gate_infidelity:.2e}")
    print(f"  Standard error: {result.std_error:.2e}")

    # Visualize RB decay
    fig, ax = plt.subplots(figsize=(10, 6))
    visualize_rb_decay(result, ax=ax, show_fit=True)
    plt.tight_layout()
    plt.savefig("examples/rb_decay_demo.png", dpi=150)
    print(f"\n✅ RB decay plot saved to examples/rb_decay_demo.png")


def demo_interleaved_rb():
    """Demonstrate interleaved RB for specific gate."""
    print("\n" + "=" * 70)
    print("DEMO 4: Interleaved Randomized Benchmarking")
    print("=" * 70)

    # Setup interleaved RB
    interleaved_rb = InterleavedRB()
    target_gate = interleaved_rb.clifford_group.H  # Hadamard gate
    print(f"\nTarget gate: Hadamard")

    # Define noise (target gate has slightly higher error)
    def noise(gate):
        # Check if gate is approximately Hadamard
        overlap = np.abs((gate.dag() * target_gate).tr())
        if np.abs(overlap - 2) < 0.1:
            return depolarizing_noise(gate, error_rate=0.02)  # 2% error
        else:
            return depolarizing_noise(gate, error_rate=0.005)  # 0.5% error

    sequence_lengths = [1, 5, 10, 15, 20]
    num_samples = 30

    print(f"Running interleaved RB...")
    print(f"  Standard Clifford error: 0.5%")
    print(f"  Hadamard gate error: 2.0%")

    standard, interleaved, F_gate = interleaved_rb.run_interleaved_rb(
        target_gate, sequence_lengths, num_samples, noise_model=noise
    )

    print(f"\n✅ Interleaved RB Results:")
    print(f"  Standard RB fidelity: {standard.average_fidelity:.6f}")
    print(f"  Interleaved RB fidelity: {interleaved.average_fidelity:.6f}")
    print(f"  Hadamard gate fidelity: {F_gate:.6f}")
    print(f"  Hadamard gate error: {(1 - F_gate) * 100:.2f}%")


def demo_fisher_information():
    """Demonstrate Fisher information and worst-case analysis."""
    print("\n" + "=" * 70)
    print("DEMO 5: Fisher Information & Worst-Case Analysis")
    print("=" * 70)

    # Setup system
    H_drift = 0.5 * qt.sigmaz()
    H_control = [qt.sigmax()]
    total_time = 10e-6
    amplitudes = np.ones((1, 100)) * 1e6
    U_target = qt.gates.hadamard_transform()

    tester = RobustnessTester(
        H_drift,
        H_control,
        amplitudes,
        total_time,
        U_target=U_target,
    )

    # Compute Fisher information
    print("\nComputing Fisher information...")
    fisher_det = tester.compute_fisher_information("detuning")
    fisher_amp = tester.compute_fisher_information("amplitude")

    print(f"  Fisher information (detuning): {fisher_det:.4e}")
    print(f"  Fisher information (amplitude): {fisher_amp:.4e}")
    print(f"  → Higher value = better parameter estimation precision")

    # Worst-case parameter search
    print("\nFinding worst-case parameters...")
    param_ranges = {"detuning": (-0.2, 0.2), "amplitude_error": (-0.1, 0.1)}

    worst_case = tester.find_worst_case_parameters(param_ranges, n_samples=20)

    print(f"\n✅ Worst-case analysis:")
    print(f"  Worst-case fidelity: {worst_case['worst_case_fidelity']:.6f}")
    print(f"  Worst-case parameters:")
    for param, val in worst_case["worst_case_params"].items():
        print(f"    {param}: {val:.4f}")

    # Compute robustness landscape
    print("\nComputing robustness landscape (this may take a moment)...")
    landscape = tester.compute_robustness_landscape(param_ranges, n_points=30)

    print(f"\n✅ Robustness landscape statistics:")
    print(f"  Mean fidelity: {landscape['mean_fidelity']:.6f}")
    print(f"  Std fidelity: {landscape['std_fidelity']:.6f}")
    print(f"  Min fidelity: {landscape['min_fidelity']:.6f}")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("TASK 3 DEMONSTRATION: ENHANCED ROBUSTNESS & BENCHMARKING")
    print("=" * 70)
    print("\nThis script demonstrates:")
    print("  1. Filter function analysis")
    print("  2. Noise-tailored optimization")
    print("  3. Randomized benchmarking")
    print("  4. Interleaved RB")
    print("  5. Fisher information & worst-case analysis")

    try:
        demo_filter_functions()
        demo_noise_tailored_optimization()
        demo_randomized_benchmarking()
        demo_interleaved_rb()
        demo_fisher_information()

        print("\n" + "=" * 70)
        print("✅ ALL DEMONSTRATIONS COMPLETE")
        print("=" * 70)
        print("\nGenerated files:")
        print("  - examples/filter_function_demo.png")
        print("  - examples/rb_decay_demo.png")
        print("\nFor more details, see:")
        print("  - docs/TASK_3_SUMMARY.md")
        print("  - tests/unit/test_filter_functions.py")
        print("  - tests/unit/test_benchmarking.py")

    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
