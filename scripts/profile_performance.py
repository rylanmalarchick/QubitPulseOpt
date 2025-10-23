#!/usr/bin/env python3
"""
Performance profiling script for QubitPulseOpt.

This script benchmarks key optimization algorithms and identifies
performance bottlenecks. It measures:
- GRAPE optimization scaling with pulse length
- Lindblad solver performance
- Memory usage patterns
- Hotspot identification

Usage:
    python scripts/profile_performance.py --output reports/performance.json
    python scripts/profile_performance.py --profile grape --iterations 50
    python scripts/profile_performance.py --all --verbose

Author: QubitPulseOpt Team
Date: 2025-01-28
"""

import argparse
import time
import json
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
import tracemalloc
import cProfile
import pstats
from io import StringIO

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.optimization.grape import GRAPEOptimizer
from src.noise.lindblad import LindbladSolver
from src.pulses.shapes import gaussian_pulse
import qutip as qt


class PerformanceProfiler:
    """
    Performance profiling and benchmarking suite.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results: Dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "benchmarks": {},
            "system_info": self._get_system_info(),
        }

    def _get_system_info(self) -> Dict[str, str]:
        """Get system and library version information."""
        import platform

        info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "qutip_version": qt.__version__,
        }

        try:
            import scipy

            info["scipy_version"] = scipy.__version__
        except ImportError:
            pass

        return info

    def log(self, message: str) -> None:
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(f"[PROFILE] {message}")

    def benchmark_grape_scaling(
        self, pulse_lengths: List[int] = None, iterations: int = 20
    ) -> Dict[str, Any]:
        """
        Benchmark GRAPE optimization scaling with pulse length.

        Args:
            pulse_lengths: List of pulse lengths to test
            iterations: Number of GRAPE iterations per test

        Returns:
            Benchmark results dictionary
        """
        if pulse_lengths is None:
            pulse_lengths = [50, 100, 200, 400, 800]

        self.log("Starting GRAPE scaling benchmark...")

        results = {
            "pulse_lengths": pulse_lengths,
            "execution_times": [],
            "memory_usage": [],
            "iterations_per_second": [],
        }

        # Target: Pauli X gate
        target = qt.sigmax()

        for n_steps in pulse_lengths:
            self.log(f"  Testing n_steps = {n_steps}...")

            # Create initial pulse
            times = np.linspace(0, 100, n_steps)
            initial_amps = gaussian_pulse(times, amplitude=1.0, t_center=50, sigma=20)

            # Start memory tracking
            tracemalloc.start()
            start_time = time.time()

            try:
                # Run GRAPE optimization
                optimizer = GRAPEOptimizer(
                    target_unitary=target,
                    initial_pulse=initial_amps,
                    times=times,
                    max_iterations=iterations,
                )

                result = optimizer.optimize()

                # Measure performance
                elapsed = time.time() - start_time
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                results["execution_times"].append(elapsed)
                results["memory_usage"].append(peak / 1024 / 1024)  # MB
                results["iterations_per_second"].append(iterations / elapsed)

                self.log(
                    f"    Time: {elapsed:.3f}s, Memory: {peak / 1024 / 1024:.1f} MB, "
                    f"Fidelity: {result.get('fidelity', 0):.6f}"
                )

            except Exception as e:
                self.log(f"    ERROR: {e}")
                results["execution_times"].append(None)
                results["memory_usage"].append(None)
                results["iterations_per_second"].append(None)
                tracemalloc.stop()

        # Compute scaling statistics
        valid_times = [t for t in results["execution_times"] if t is not None]
        if len(valid_times) > 1:
            # Estimate complexity (approximate as O(n^alpha))
            log_n = np.log(pulse_lengths[: len(valid_times)])
            log_t = np.log(valid_times)
            alpha = np.polyfit(log_n, log_t, 1)[0]
            results["estimated_complexity"] = f"O(n^{alpha:.2f})"
        else:
            results["estimated_complexity"] = "insufficient_data"

        return results

    def benchmark_lindblad_solver(
        self, durations: List[float] = None, num_steps: int = 100
    ) -> Dict[str, Any]:
        """
        Benchmark Lindblad solver performance.

        Args:
            durations: List of evolution durations to test (in ns)
            num_steps: Number of time steps per evolution

        Returns:
            Benchmark results dictionary
        """
        if durations is None:
            durations = [50, 100, 200, 500, 1000]

        self.log("Starting Lindblad solver benchmark...")

        results = {
            "durations": durations,
            "execution_times": [],
            "steps_per_second": [],
        }

        # Initial state
        psi0 = qt.basis(2, 0)

        # Hamiltonian
        H0 = 2 * np.pi * 5e9 * qt.sigmaz() / 2
        H_control = qt.sigmax()

        # Collapse operators (decoherence)
        T1 = 50e-6  # 50 μs
        T2 = 100e-6  # 100 μs
        gamma1 = 1 / T1
        gamma2 = 1 / (2 * T2)
        c_ops = [np.sqrt(gamma1) * qt.sigmam(), np.sqrt(gamma2) * qt.sigmaz()]

        for duration in durations:
            self.log(f"  Testing duration = {duration} ns...")

            times = np.linspace(0, duration, num_steps)
            pulse = gaussian_pulse(
                times, amplitude=1.0, t_center=duration / 2, sigma=duration / 8
            )

            start_time = time.time()

            try:
                solver = LindbladSolver(H0, [H_control], c_ops)
                result = solver.evolve(psi0, times, [pulse])

                elapsed = time.time() - start_time
                steps_per_sec = num_steps / elapsed

                results["execution_times"].append(elapsed)
                results["steps_per_second"].append(steps_per_sec)

                self.log(
                    f"    Time: {elapsed:.3f}s, Steps/s: {steps_per_sec:.1f}, "
                    f"Final population: {qt.expect(qt.num(2), result.states[-1]):.6f}"
                )

            except Exception as e:
                self.log(f"    ERROR: {e}")
                results["execution_times"].append(None)
                results["steps_per_second"].append(None)

        return results

    def profile_hotspots(
        self, profile_target: str = "grape", iterations: int = 50
    ) -> Dict[str, Any]:
        """
        Profile code to identify computational hotspots.

        Args:
            profile_target: What to profile ('grape', 'lindblad')
            iterations: Number of iterations for profiling

        Returns:
            Profiling results with top functions
        """
        self.log(f"Profiling hotspots for {profile_target}...")

        profiler = cProfile.Profile()

        if profile_target == "grape":
            # Profile GRAPE optimization
            target = qt.sigmax()
            times = np.linspace(0, 100, 200)
            initial_amps = gaussian_pulse(times, amplitude=1.0, t_center=50, sigma=20)

            profiler.enable()
            optimizer = GRAPEOptimizer(
                target_unitary=target,
                initial_pulse=initial_amps,
                times=times,
                max_iterations=iterations,
            )
            result = optimizer.optimize()
            profiler.disable()

        elif profile_target == "lindblad":
            # Profile Lindblad evolution
            psi0 = qt.basis(2, 0)
            H0 = 2 * np.pi * 5e9 * qt.sigmaz() / 2
            H_control = qt.sigmax()
            c_ops = [np.sqrt(0.01) * qt.sigmam()]

            times = np.linspace(0, 100, 200)
            pulse = gaussian_pulse(times, amplitude=1.0, t_center=50, sigma=20)

            profiler.enable()
            solver = LindbladSolver(H0, [H_control], c_ops)
            result = solver.evolve(psi0, times, [pulse])
            profiler.disable()

        else:
            raise ValueError(f"Unknown profile target: {profile_target}")

        # Extract statistics
        stream = StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.strip_dirs()
        stats.sort_stats("cumulative")
        stats.print_stats(20)  # Top 20 functions

        # Parse top functions
        top_functions = []
        lines = stream.getvalue().split("\n")

        for line in lines[5:25]:  # Skip header, get top 20
            if line.strip() and not line.startswith("---"):
                parts = line.split()
                if len(parts) >= 6:
                    top_functions.append(
                        {
                            "ncalls": parts[0],
                            "tottime": parts[1],
                            "percall": parts[2],
                            "cumtime": parts[3],
                            "function": " ".join(parts[5:]),
                        }
                    )

        results = {
            "profile_target": profile_target,
            "iterations": iterations,
            "top_functions": top_functions[:10],
            "full_stats": stream.getvalue(),
        }

        return results

    def benchmark_memory_usage(self, operations: List[str] = None) -> Dict[str, Any]:
        """
        Benchmark memory usage of key operations.

        Args:
            operations: List of operations to benchmark

        Returns:
            Memory usage results
        """
        if operations is None:
            operations = ["grape_small", "grape_large", "lindblad_evolution"]

        self.log("Starting memory usage benchmark...")

        results = {"operations": {}}

        for op_name in operations:
            self.log(f"  Testing {op_name}...")

            tracemalloc.start()

            try:
                if op_name == "grape_small":
                    target = qt.sigmax()
                    times = np.linspace(0, 100, 100)
                    initial_amps = gaussian_pulse(
                        times, amplitude=1.0, t_center=50, sigma=20
                    )
                    optimizer = GRAPEOptimizer(
                        target_unitary=target,
                        initial_pulse=initial_amps,
                        times=times,
                        max_iterations=20,
                    )
                    result = optimizer.optimize()

                elif op_name == "grape_large":
                    target = qt.sigmax()
                    times = np.linspace(0, 100, 500)
                    initial_amps = gaussian_pulse(
                        times, amplitude=1.0, t_center=50, sigma=20
                    )
                    optimizer = GRAPEOptimizer(
                        target_unitary=target,
                        initial_pulse=initial_amps,
                        times=times,
                        max_iterations=20,
                    )
                    result = optimizer.optimize()

                elif op_name == "lindblad_evolution":
                    psi0 = qt.basis(2, 0)
                    H0 = 2 * np.pi * 5e9 * qt.sigmaz() / 2
                    H_control = qt.sigmax()
                    c_ops = [np.sqrt(0.01) * qt.sigmam()]
                    times = np.linspace(0, 100, 500)
                    pulse = gaussian_pulse(times, amplitude=1.0, t_center=50, sigma=20)
                    solver = LindbladSolver(H0, [H_control], c_ops)
                    result = solver.evolve(psi0, times, [pulse])

                current, peak = tracemalloc.get_traced_memory()
                results["operations"][op_name] = {
                    "current_mb": current / 1024 / 1024,
                    "peak_mb": peak / 1024 / 1024,
                }

                self.log(
                    f"    Current: {current / 1024 / 1024:.1f} MB, "
                    f"Peak: {peak / 1024 / 1024:.1f} MB"
                )

            except Exception as e:
                self.log(f"    ERROR: {e}")
                results["operations"][op_name] = {"error": str(e)}

            finally:
                tracemalloc.stop()

        return results

    def run_all_benchmarks(self, quick: bool = False) -> Dict[str, Any]:
        """
        Run all benchmarks.

        Args:
            quick: If True, use reduced test parameters for faster execution

        Returns:
            Complete benchmark results
        """
        self.log("Running all performance benchmarks...")

        if quick:
            pulse_lengths = [50, 100, 200]
            durations = [50, 100, 200]
            iterations = 10
        else:
            pulse_lengths = [50, 100, 200, 400, 800]
            durations = [50, 100, 200, 500, 1000]
            iterations = 20

        # Run benchmarks
        self.results["benchmarks"]["grape_scaling"] = self.benchmark_grape_scaling(
            pulse_lengths, iterations
        )

        self.results["benchmarks"]["lindblad_solver"] = self.benchmark_lindblad_solver(
            durations
        )

        self.results["benchmarks"]["memory_usage"] = self.benchmark_memory_usage()

        self.results["benchmarks"]["hotspots_grape"] = self.profile_hotspots(
            "grape", iterations
        )

        self.results["benchmarks"]["hotspots_lindblad"] = self.profile_hotspots(
            "lindblad", iterations
        )

        return self.results

    def save_results(self, filepath: str) -> None:
        """
        Save benchmark results to JSON file.

        Args:
            filepath: Output file path
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        self.log(f"Results saved to {filepath}")

    def print_summary(self) -> None:
        """Print a summary of benchmark results."""
        print("\n" + "=" * 70)
        print("PERFORMANCE BENCHMARK SUMMARY")
        print("=" * 70)

        if "grape_scaling" in self.results["benchmarks"]:
            grape = self.results["benchmarks"]["grape_scaling"]
            print("\nGRAPE Scaling:")
            print(f"  Complexity: {grape.get('estimated_complexity', 'N/A')}")
            if grape["execution_times"]:
                avg_time = np.mean(
                    [t for t in grape["execution_times"] if t is not None]
                )
                print(f"  Average time: {avg_time:.3f}s")

        if "lindblad_solver" in self.results["benchmarks"]:
            lindblad = self.results["benchmarks"]["lindblad_solver"]
            print("\nLindblad Solver:")
            if lindblad["steps_per_second"]:
                avg_steps = np.mean(
                    [s for s in lindblad["steps_per_second"] if s is not None]
                )
                print(f"  Average steps/second: {avg_steps:.1f}")

        if "memory_usage" in self.results["benchmarks"]:
            memory = self.results["benchmarks"]["memory_usage"]
            print("\nMemory Usage:")
            for op_name, usage in memory["operations"].items():
                if "peak_mb" in usage:
                    print(f"  {op_name}: {usage['peak_mb']:.1f} MB (peak)")

        print("\n" + "=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Performance profiling for QubitPulseOpt"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="performance_report.json",
        help="Output file path for results",
    )
    parser.add_argument(
        "--profile",
        "-p",
        type=str,
        choices=["grape", "lindblad", "memory", "all"],
        default="all",
        help="Which benchmark to run",
    )
    parser.add_argument(
        "--iterations",
        "-i",
        type=int,
        default=20,
        help="Number of iterations for optimization benchmarks",
    )
    parser.add_argument(
        "--quick",
        "-q",
        action="store_true",
        help="Run quick benchmarks with reduced parameters",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Create profiler
    profiler = PerformanceProfiler(verbose=args.verbose)

    # Run requested benchmarks
    if args.profile == "all":
        profiler.run_all_benchmarks(quick=args.quick)
    elif args.profile == "grape":
        profiler.results["benchmarks"]["grape_scaling"] = (
            profiler.benchmark_grape_scaling(iterations=args.iterations)
        )
        profiler.results["benchmarks"]["hotspots"] = profiler.profile_hotspots(
            "grape", args.iterations
        )
    elif args.profile == "lindblad":
        profiler.results["benchmarks"]["lindblad_solver"] = (
            profiler.benchmark_lindblad_solver()
        )
        profiler.results["benchmarks"]["hotspots"] = profiler.profile_hotspots(
            "lindblad", args.iterations
        )
    elif args.profile == "memory":
        profiler.results["benchmarks"]["memory_usage"] = (
            profiler.benchmark_memory_usage()
        )

    # Save results
    profiler.save_results(args.output)

    # Print summary
    profiler.print_summary()


if __name__ == "__main__":
    main()
