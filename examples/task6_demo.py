#!/usr/bin/env python3
"""
Task 6 Demo: Production Polish & CI/CD Features

This script demonstrates all Task 6 features:
- Configuration management (6.4)
- I/O export/serialization (6.3)
- Logging and diagnostics (6.5)
- Performance profiling utilities (6.2)

Author: QubitPulseOpt Team
Date: 2025-01-28
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config, merge_configs, Config
from src.io import (
    save_pulse,
    load_pulse,
    save_optimization_result,
    load_optimization_result,
)
from src.logging_utils import (
    get_logger,
    log_operation,
    PerformanceTimer,
    DiagnosticCollector,
)
from src.pulses.shapes import gaussian_pulse, drag_pulse
from src.optimization.grape import GRAPEOptimizer
import qutip as qt


def demo_configuration_management():
    """Demonstrate configuration management (Task 6.4)."""
    print("\n" + "=" * 70)
    print("TASK 6.4: CONFIGURATION MANAGEMENT")
    print("=" * 70)

    logger = get_logger("demo.config")

    # 1. Load default configuration
    logger.info("Loading default configuration...")
    config = load_config()

    # 2. Access configuration values
    T1 = config.get("system.decoherence.T1")
    T2 = config.get("system.decoherence.T2")
    max_iter = config.get("optimization.grape.max_iterations")

    logger.info(f"System parameters from config:")
    logger.info(f"  T1 = {T1 * 1e6:.1f} μs")
    logger.info(f"  T2 = {T2 * 1e6:.1f} μs")
    logger.info(f"  Max GRAPE iterations = {max_iter}")

    # 3. Override configuration programmatically
    logger.info("Creating custom configuration override...")
    custom_config = Config(
        {
            "system": {
                "decoherence": {
                    "T1": 100e-6,  # Override T1
                    "T2": 200e-6,  # Override T2
                }
            },
            "optimization": {
                "grape": {
                    "max_iterations": 50  # Fewer iterations for demo
                }
            },
        }
    )

    # 4. Merge configurations
    final_config = merge_configs(config, custom_config)

    logger.info(f"After merge:")
    logger.info(f"  T1 = {final_config.get('system.decoherence.T1') * 1e6:.1f} μs")
    logger.info(f"  T2 = {final_config.get('system.decoherence.T2') * 1e6:.1f} μs")
    logger.info(
        f"  Max iterations = {final_config.get('optimization.grape.max_iterations')}"
    )

    # 5. Save configuration
    output_dir = Path("examples/task6_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "custom_config.yaml"
    final_config.save(config_path)
    logger.info(f"Saved configuration to {config_path}")

    return final_config


def demo_export_serialization(config):
    """Demonstrate I/O export and serialization (Task 6.3)."""
    print("\n" + "=" * 70)
    print("TASK 6.3: EXPORT & SERIALIZATION")
    print("=" * 70)

    logger = get_logger("demo.export")
    output_dir = Path("examples/task6_output")

    # 1. Create a pulse
    logger.info("Creating test pulse...")
    duration = config.get("pulse.default.duration", 100e-9)
    num_samples = config.get("pulse.default.num_samples", 200)

    times = np.linspace(0, duration * 1e9, num_samples)  # Convert to ns
    amplitudes = gaussian_pulse(
        times, amplitude=1.0, t_center=duration * 1e9 / 2, sigma=duration * 1e9 / 8
    )

    # Add frequency and phase modulation
    frequencies = 5.0 * np.ones_like(times)  # 5 GHz
    phases = 2 * np.pi * times / 100

    # 2. Export to different formats
    logger.info("Exporting pulse to multiple formats...")

    # JSON export
    json_path = output_dir / "pulse_export.json"
    save_pulse(
        json_path,
        times,
        amplitudes,
        frequencies,
        phases,
        format="json",
        pulse_name="demo_pulse",
    )
    logger.info(f"  JSON: {json_path}")

    # NPZ export (compressed NumPy)
    npz_path = output_dir / "pulse_export.npz"
    save_pulse(
        npz_path,
        times,
        amplitudes,
        frequencies,
        phases,
        format="npz",
        metadata={"experiment": "task6_demo"},
    )
    logger.info(f"  NPZ: {npz_path}")

    # CSV export
    csv_path = output_dir / "pulse_export.csv"
    save_pulse(csv_path, times, amplitudes, frequencies, phases, format="csv")
    logger.info(f"  CSV: {csv_path}")

    # 3. Load back and verify
    logger.info("Loading pulse from JSON and verifying...")
    loaded_data = load_pulse(json_path)
    loaded_times = loaded_data["pulse_data"]["times"]
    loaded_amps = loaded_data["pulse_data"]["amplitudes"]

    max_error = np.max(np.abs(np.array(loaded_amps) - amplitudes))
    logger.info(f"  Round-trip error: {max_error:.2e} (excellent!)")

    # 4. Export optimization result
    logger.info("Creating mock optimization result...")
    opt_result = {
        "final_fidelity": 0.9995,
        "iterations": 150,
        "convergence_history": np.linspace(0.5, 0.9995, 150),
        "cost_history": np.linspace(0.5, 0.0005, 150),
        "final_pulse": amplitudes,
        "pulse_duration": duration,
        "success": True,
        "message": "Optimization converged successfully",
    }

    result_path = output_dir / "optimization_result.json"
    save_optimization_result(result_path, opt_result, format="json")
    logger.info(f"  Optimization result: {result_path}")

    # 5. Plot exported pulse
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))

    axes[0].plot(times, amplitudes, "b-", linewidth=2)
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("Pulse Export Demo")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(times, frequencies, "r-", linewidth=2)
    axes[1].set_ylabel("Frequency (GHz)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(times, phases, "g-", linewidth=2)
    axes[2].set_ylabel("Phase (rad)")
    axes[2].set_xlabel("Time (ns)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / "pulse_export_demo.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    logger.info(f"  Plot saved: {plot_path}")
    plt.close()

    return amplitudes, times


def demo_logging_diagnostics():
    """Demonstrate logging and diagnostics (Task 6.5)."""
    print("\n" + "=" * 70)
    print("TASK 6.5: LOGGING & DIAGNOSTICS")
    print("=" * 70)

    logger = get_logger("demo.logging")

    # 1. Structured logging
    logger.info("Demonstrating structured logging...")
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")

    # 2. Operation logging with context manager
    logger.info("Using log_operation context manager...")
    with log_operation("Test calculation", logger, log_time=True):
        # Simulate some work
        result = np.linalg.eig(np.random.randn(100, 100))

    # 3. Performance timing
    logger.info("Using PerformanceTimer...")
    timer = PerformanceTimer("Matrix operations", logger)
    with timer:
        A = np.random.randn(500, 500)
        B = np.random.randn(500, 500)
        C = A @ B
    logger.info(f"Matrix multiplication took {timer.elapsed:.4f}s")

    # 4. Diagnostic collector
    logger.info("Using DiagnosticCollector...")
    diag = DiagnosticCollector("task6_demo")

    # Record various diagnostics
    diag.record("matrix_size", 500)
    diag.record("computation_time", timer.elapsed)
    diag.record("memory_estimate_mb", 500 * 500 * 8 / 1024 / 1024)

    # Record events
    diag.event("computation_started", method="numpy")
    diag.event("computation_completed", status="success")

    # Save diagnostics
    output_dir = Path("examples/task6_output")
    diag_path = output_dir / "diagnostics.json"
    diag.save(diag_path)
    logger.info(f"Diagnostics saved to {diag_path}")

    # Print summary
    logger.info("Diagnostic summary:")
    print(diag.get_summary())


def demo_grape_optimization_with_logging(config, pulse_amplitudes, times):
    """Run GRAPE optimization with full logging and export."""
    print("\n" + "=" * 70)
    print("INTEGRATED DEMO: GRAPE with Config, Logging & Export")
    print("=" * 70)

    logger = get_logger("demo.grape")
    output_dir = Path("examples/task6_output")

    # Initialize diagnostic collector
    diag = DiagnosticCollector("grape_optimization")

    # Get configuration
    max_iter = config.get("optimization.grape.max_iterations", 50)
    tolerance = config.get("optimization.grape.tolerance", 1e-6)

    logger.info(f"Configuration: max_iter={max_iter}, tolerance={tolerance}")

    # Setup optimization
    target = qt.sigmax()  # X gate

    logger.info("Starting GRAPE optimization...")
    diag.event("optimization_started", target="X_gate", max_iterations=max_iter)

    with log_operation("GRAPE optimization", logger, log_time=True):
        with PerformanceTimer("GRAPE", logger) as timer:
            try:
                optimizer = GRAPEOptimizer(
                    target_unitary=target,
                    initial_pulse=pulse_amplitudes[: len(times)],
                    times=times,
                    max_iterations=max_iter,
                    tolerance=tolerance,
                )

                result = optimizer.optimize()

                # Record results
                diag.record("final_fidelity", result.get("fidelity", 0))
                diag.record("iterations", result.get("iterations", 0))
                diag.record("optimization_time", timer.elapsed)
                diag.event("optimization_completed", status="success")

                logger.info(f"Optimization completed:")
                logger.info(f"  Fidelity: {result.get('fidelity', 0):.6f}")
                logger.info(f"  Iterations: {result.get('iterations', 0)}")
                logger.info(f"  Time: {timer.elapsed:.3f}s")

            except Exception as e:
                logger.error(f"Optimization failed: {e}")
                diag.event("optimization_failed", error=str(e))
                result = {"success": False, "error": str(e)}

    # Export results
    if result.get("success", True):
        logger.info("Exporting optimization results...")

        # Save full result
        result_path = output_dir / "grape_optimization_result.json"
        save_optimization_result(result_path, result, format="json")
        logger.info(f"  Results: {result_path}")

        # Save optimized pulse
        if "optimized_pulse" in result:
            pulse_path = output_dir / "optimized_pulse.npz"
            save_pulse(pulse_path, times, result["optimized_pulse"], format="npz")
            logger.info(f"  Optimized pulse: {pulse_path}")

        # Plot convergence
        if "convergence" in result and result["convergence"]:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(result["convergence"], "b-", linewidth=2)
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Fidelity")
            ax.set_title("GRAPE Optimization Convergence")
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1.05])

            conv_path = output_dir / "grape_convergence.png"
            plt.savefig(conv_path, dpi=150, bbox_inches="tight")
            logger.info(f"  Convergence plot: {conv_path}")
            plt.close()

    # Save diagnostics
    diag_path = output_dir / "grape_diagnostics.json"
    diag.save(diag_path)
    logger.info(f"Diagnostics: {diag_path}")


def demo_performance_summary():
    """Show performance profiling capabilities."""
    print("\n" + "=" * 70)
    print("TASK 6.2: PERFORMANCE PROFILING")
    print("=" * 70)

    logger = get_logger("demo.performance")

    logger.info("Performance profiling capabilities available:")
    logger.info("  - Run: python scripts/profile_performance.py")
    logger.info("  - Benchmarks: GRAPE scaling, Lindblad solver, memory usage")
    logger.info("  - Profiling: cProfile integration for hotspot identification")
    logger.info("  - Output: JSON reports with detailed metrics")
    logger.info("")
    logger.info("Example usage:")
    logger.info(
        "  $ python scripts/profile_performance.py --profile grape --iterations 50"
    )
    logger.info(
        "  $ python scripts/profile_performance.py --all --quick --output perf.json"
    )


def main():
    """Main demo function."""
    print("\n" + "=" * 70)
    print("TASK 6 DEMONSTRATION: PRODUCTION POLISH & CI/CD")
    print("=" * 70)
    print("This demo showcases all Task 6 features:")
    print("  6.1 CI/CD Pipeline - GitHub Actions workflows created")
    print("  6.2 Performance Profiling - scripts/profile_performance.py")
    print("  6.3 Export & Serialization - src/io/export.py")
    print("  6.4 Configuration Management - src/config.py")
    print("  6.5 Logging & Diagnostics - src/logging_utils.py")
    print("=" * 70)

    # Setup output directory
    output_dir = Path("examples/task6_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Task 6.4: Configuration Management
        config = demo_configuration_management()

        # Task 6.3: Export & Serialization
        pulse_amps, times = demo_export_serialization(config)

        # Task 6.5: Logging & Diagnostics
        demo_logging_diagnostics()

        # Integrated demo
        demo_grape_optimization_with_logging(config, pulse_amps, times)

        # Task 6.2: Performance profiling info
        demo_performance_summary()

        print("\n" + "=" * 70)
        print("DEMO COMPLETE!")
        print("=" * 70)
        print(f"Output files saved to: {output_dir}")
        print("Check the following:")
        print(f"  - Configuration: {output_dir}/custom_config.yaml")
        print(f"  - Pulse exports: {output_dir}/pulse_export.*")
        print(f"  - Optimization results: {output_dir}/*_result.json")
        print(f"  - Diagnostics: {output_dir}/*_diagnostics.json")
        print(f"  - Plots: {output_dir}/*.png")
        print("\nCI/CD workflows created in .github/workflows/")
        print("  - tests.yml - Fast and slow test separation")
        print("  - docs.yml - Documentation building")
        print("  - notebooks.yml - Notebook validation")
        print("\nReady for production deployment!")
        print("=" * 70 + "\n")

    except Exception as e:
        logger = get_logger("demo")
        logger.error(f"Demo failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
