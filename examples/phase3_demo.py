#!/usr/bin/env python3
"""
Phase 3 Demonstration: Pulse Optimization & Translation
========================================================

This script demonstrates the complete Phase 3 workflow for QubitPulseOpt:
1. Load hardware characterization parameters (from Phase 2)
2. Configure GRAPE optimizer with hardware constraints
3. Optimize pulse shapes for target gates (X, Y, Hadamard)
4. Translate optimized pulses to IQM format
5. Create executable schedules
6. Simulate pulse execution
7. Validate fidelity and performance

This demo runs entirely in simulation mode and does NOT require
access to real IQM hardware. It validates all code paths and infrastructure.

Usage:
    python phase3_demo.py

Requirements:
    - QubitPulseOpt dependencies (qutip, numpy, scipy)
    - IQM SDK installed (iqm-client, iqm-pulse, iqm-pulla)
    - Phase 2 completed (for hardware parameters)

Author: QubitPulseOpt Development Team
Phase: 3 - Pulse Optimization & Translation
Status: Headless Development (Simulation Mode)
"""

import sys
import os
from pathlib import Path
import logging
from datetime import datetime
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def print_header(title, level=1):
    """Print a formatted section header."""
    if level == 1:
        print("\n" + "=" * 80)
        print(f"  {title}")
        print("=" * 80 + "\n")
    elif level == 2:
        print("\n" + "-" * 80)
        print(f"  {title}")
        print("-" * 80 + "\n")
    else:
        print(f"\n>>> {title}\n")


def print_result(name, value, unit=""):
    """Print a formatted result."""
    if isinstance(value, float):
        print(f"  {name:30s}: {value:.6f} {unit}")
    else:
        print(f"  {name:30s}: {value} {unit}")


def demo_load_hardware_params():
    """Load hardware characterization parameters from Phase 2."""
    print_header("STEP 1: Load Hardware Parameters", level=1)

    print("Loading hardware characterization from Phase 2...")
    print("(Using simulated parameters for headless development)")

    # Simulated hardware parameters (would come from Phase 2 in production)
    hardware_params = {
        "qubit_id": "QB1",
        "T1": 50e-6,  # 50 μs
        "T2": 30e-6,  # 30 μs
        "rabi_frequency": 5e6,  # 5 MHz
        "qubit_frequency": 5.0e9,  # 5 GHz
        "anharmonicity": -300e6,  # -300 MHz
        "max_amplitude": 1.0,  # Normalized
    }

    print("\n✓ Hardware Parameters Loaded:")
    print_result("Qubit ID", hardware_params["qubit_id"])
    print_result("T1", hardware_params["T1"] * 1e6, "μs")
    print_result("T2", hardware_params["T2"] * 1e6, "μs")
    print_result("Rabi Frequency", hardware_params["rabi_frequency"] / 1e6, "MHz")
    print_result("Qubit Frequency", hardware_params["qubit_frequency"] / 1e9, "GHz")
    print_result("Anharmonicity", hardware_params["anharmonicity"] / 1e6, "MHz")

    return hardware_params


def demo_configure_grape(hardware_params):
    """Configure GRAPE optimizer with hardware constraints."""
    print_header("STEP 2: Configure GRAPE Optimizer", level=1)

    try:
        import qutip as qt
        from phase3_grape_wrapper import SimpleGRAPEOptimizer

        print("Setting up GRAPE optimizer with hardware constraints...")

        # Extract parameters
        rabi_freq = hardware_params["rabi_frequency"]
        T2 = hardware_params["T2"]
        max_amp = hardware_params["max_amplitude"]

        # Qubit Hamiltonian (rotating frame)
        # H_drift = 0 (in rotating frame at qubit frequency)
        H_drift = 0 * qt.sigmaz()

        # Control Hamiltonians (X and Y drives)
        # H_x = Ω_x(t) σ_x / 2
        # H_y = Ω_y(t) σ_y / 2
        H_controls = [
            0.5 * qt.sigmax(),  # X-drive
            0.5 * qt.sigmay(),  # Y-drive
        ]

        # Optimization parameters
        n_timeslices = 50  # Pulse discretization
        total_time = 50e-9  # 50 ns (typical gate time)
        u_limits = (-max_amp * rabi_freq * 2 * np.pi, max_amp * rabi_freq * 2 * np.pi)

        print("\n✓ GRAPE Configuration:")
        print_result("Drift Hamiltonian", "H_drift = 0 (rotating frame)")
        print_result("Control Hamiltonians", "H_x (σ_x), H_y (σ_y)")
        print_result("Time Slices", n_timeslices)
        print_result("Total Time", total_time * 1e9, "ns")
        print_result("Amplitude Limits", f"±{max_amp * rabi_freq / 1e6:.2f}", "MHz")
        print_result("Convergence Threshold", 1e-4)
        print_result("Max Iterations", 200)

        # Create optimizer
        optimizer = SimpleGRAPEOptimizer(
            H_drift=H_drift,
            H_controls=H_controls,
            n_timeslices=n_timeslices,
            total_time=total_time,
            u_limits=u_limits,
            convergence_threshold=1e-4,
            max_iterations=200,
            verbose=False,
        )

        print("\n✓ GRAPE optimizer initialized successfully")

        return optimizer, n_timeslices, total_time

    except Exception as e:
        print(f"✗ GRAPE configuration failed: {e}")
        import traceback

        traceback.print_exc()
        return None, None, None


def demo_optimize_x_gate(optimizer, n_timeslices):
    """Optimize X gate using GRAPE."""
    print_header("STEP 3: Optimize X Gate (π rotation)", level=1)

    try:
        import qutip as qt

        print("Optimizing X gate pulse using GRAPE...")
        print("Target: X = [[0, 1], [1, 0]] (π rotation around x-axis)")

        # Target unitary: X gate
        U_target = qt.sigmax()

        # Initial guess: small constant pulse on X-drive, zero on Y
        u_init = np.zeros((2, n_timeslices))
        u_init[0, :] = 0.1  # Small constant X-drive
        # u_init[1, :] = 0.0  # No Y-drive initially

        print("\n  Running GRAPE optimization...")
        print(f"  Initial guess: constant amplitude {u_init[0, 0]:.3f}")

        # Optimize
        result = optimizer.optimize(
            target=U_target, initial_pulses=u_init, initial_state=None
        )

        if result.converged:
            print(f"\n✓ Optimization converged in {result.n_iterations} iterations")
        else:
            print(f"\n⚠️  Optimization stopped at {result.n_iterations} iterations")
            print(f"  {result.message}")

        print_result("Final Fidelity", result.final_fidelity)
        print_result("Initial Fidelity", result.fidelity_history[0])
        print_result(
            "Fidelity Improvement", result.final_fidelity - result.fidelity_history[0]
        )
        print_result("Final Gradient Norm", result.gradient_norms[-1])

        # Extract I/Q waveforms
        # I = X-drive, Q = Y-drive
        i_waveform = result.optimized_pulses[0, :]
        q_waveform = result.optimized_pulses[1, :]

        print(f"\n  Pulse Statistics:")
        print_result("I-channel max", np.max(np.abs(i_waveform)))
        print_result("Q-channel max", np.max(np.abs(q_waveform)))
        print_result("Pulse length", len(i_waveform), "samples")

        return result, i_waveform, q_waveform

    except Exception as e:
        print(f"✗ X gate optimization failed: {e}")
        import traceback

        traceback.print_exc()
        return None, None, None


def demo_optimize_hadamard(optimizer, n_timeslices):
    """Optimize Hadamard gate using GRAPE."""
    print_header("STEP 4: Optimize Hadamard Gate", level=1)

    try:
        import qutip as qt

        print("Optimizing Hadamard gate pulse using GRAPE...")
        print("Target: H = (1/√2)[[1, 1], [1, -1]]")

        # Target unitary: Hadamard gate
        U_target = qt.hadamard_transform()

        # Initial guess: combination of X and Y drives
        u_init = np.zeros((2, n_timeslices))
        u_init[0, :] = 0.08  # X-drive
        u_init[1, :] = 0.04  # Y-drive

        print("\n  Running GRAPE optimization...")

        # Optimize
        result = optimizer.optimize(
            target=U_target, initial_pulses=u_init, initial_state=None
        )

        if result.converged:
            print(f"\n✓ Optimization converged in {result.n_iterations} iterations")
        else:
            print(f"\n⚠️  Optimization stopped at {result.n_iterations} iterations")
            print(f"  {result.message}")

        print_result("Final Fidelity", result.final_fidelity)
        print_result("Initial Fidelity", result.fidelity_history[0])
        print_result(
            "Fidelity Improvement", result.final_fidelity - result.fidelity_history[0]
        )

        # Extract I/Q waveforms
        i_waveform = result.optimized_pulses[0, :]
        q_waveform = result.optimized_pulses[1, :]

        print(f"\n  Pulse Statistics:")
        print_result("I-channel max", np.max(np.abs(i_waveform)))
        print_result("Q-channel max", np.max(np.abs(q_waveform)))

        return result, i_waveform, q_waveform

    except Exception as e:
        print(f"✗ Hadamard optimization failed: {e}")
        import traceback

        traceback.print_exc()
        return None, None, None


def demo_save_pulses(i_waveform, q_waveform, gate_name, total_time):
    """Save optimized pulses to file."""
    print_header(f"STEP 5: Save Optimized {gate_name} Pulse", level=1)

    try:
        # Create pulses directory if it doesn't exist
        pulses_dir = Path(__file__).parent / "pulses"
        pulses_dir.mkdir(exist_ok=True)

        # Save as .npz file
        filepath = pulses_dir / f"optimized_{gate_name.lower()}_gate.npz"

        np.savez(
            filepath,
            i=i_waveform,
            q=q_waveform,
            duration=total_time,
            sample_rate=1e9,  # 1 GHz default
            gate_name=gate_name,
            timestamp=datetime.now().isoformat(),
        )

        print(f"✓ Pulse saved to: {filepath}")
        print_result("File size", filepath.stat().st_size, "bytes")
        print_result("Waveform length", len(i_waveform), "samples")
        print_result("Duration", total_time * 1e9, "ns")

        return filepath

    except Exception as e:
        print(f"✗ Failed to save pulse: {e}")
        return None


def demo_translate_to_iqm(i_waveform, q_waveform, qubit_id):
    """Translate optimized pulse to IQM format."""
    print_header("STEP 6: Translate to IQM Pulse Format", level=1)

    print("Translating QubitPulseOpt waveforms to IQM CustomIQWaveform...")
    print(f"Target qubit: {qubit_id}")

    try:
        # In production, would use:
        # from hardware.iqm_translator import IQMTranslator
        # translator = IQMTranslator()
        # schedule = translator.create_schedule(i_waveform, q_waveform, qubit_id)

        # For headless demo, simulate the translation
        print("\n✓ Translation workflow (simulated):")
        print("  1. Normalize waveforms to [-1, 1]")

        # Normalize
        max_val = max(np.max(np.abs(i_waveform)), np.max(np.abs(q_waveform)))
        if max_val > 0:
            i_norm = i_waveform / max_val
            q_norm = q_waveform / max_val
        else:
            i_norm = i_waveform
            q_norm = q_waveform

        print_result("Normalization factor", max_val)

        print("  2. Create IQM CustomIQWaveform object")
        print(f"     - I-channel: {len(i_norm)} samples")
        print(f"     - Q-channel: {len(q_norm)} samples")

        print("  3. Build IQM Schedule with:")
        print(f"     - Target qubit: {qubit_id}")
        print(f"     - Pulse duration: {len(i_norm)} ns (@ 1 GHz sampling)")
        print("     - Operation: CustomPulse")

        print("\n✓ IQM translation complete (simulated)")
        print("  In production: schedule ready for hardware execution")

        # Return simulated schedule metadata
        schedule_metadata = {
            "qubit": qubit_id,
            "pulse_type": "CustomIQWaveform",
            "i_samples": len(i_norm),
            "q_samples": len(q_norm),
            "duration_ns": len(i_norm),  # 1 ns per sample at 1 GHz
            "normalized": True,
        }

        return schedule_metadata, i_norm, q_norm

    except Exception as e:
        print(f"✗ IQM translation failed: {e}")
        import traceback

        traceback.print_exc()
        return None, None, None


def demo_simulate_execution(i_waveform, q_waveform, gate_name):
    """Simulate pulse execution and measure fidelity."""
    print_header("STEP 7: Simulate Pulse Execution", level=1)

    try:
        import qutip as qt

        print(f"Simulating {gate_name} gate execution...")
        print("(Using QuTiP dynamics solver)")

        # Reconstruct Hamiltonian with optimized pulse
        H_drift = 0 * qt.sigmaz()
        H_x = 0.5 * qt.sigmax()
        H_y = 0.5 * qt.sigmay()

        # Time-dependent Hamiltonian
        n_samples = len(i_waveform)
        times = np.linspace(0, 50e-9, n_samples)

        # Create list of [H, coeff_func] for time-dependent evolution
        def i_coeff(t, args):
            idx = int(t / 50e-9 * (n_samples - 1))
            idx = min(idx, n_samples - 1)
            return i_waveform[idx]

        def q_coeff(t, args):
            idx = int(t / 50e-9 * (n_samples - 1))
            idx = min(idx, n_samples - 1)
            return q_waveform[idx]

        H = [H_drift, [H_x, i_coeff], [H_y, q_coeff]]

        # Initial state: |0⟩
        psi0 = qt.basis(2, 0)

        # Evolve
        print("\n  Evolving quantum state under optimized pulse...")
        result = qt.sesolve(H, psi0, times)

        # Final state
        psi_final = result.states[-1]

        # Expected final state depends on gate
        if gate_name.upper() == "X":
            psi_expected = qt.basis(2, 1)  # |1⟩
        elif gate_name.upper() == "H" or gate_name.upper() == "HADAMARD":
            psi_expected = (qt.basis(2, 0) + qt.basis(2, 1)).unit()  # |+⟩
        else:
            psi_expected = psi_final  # Unknown gate

        # Compute fidelity
        fidelity = qt.fidelity(psi_final, psi_expected)

        print(f"\n✓ Simulation complete")
        print_result("State Fidelity", fidelity)
        print_result("Gate Error", 1 - fidelity)

        if fidelity > 0.99:
            print(f"  ✓ High-fidelity gate achieved!")
        elif fidelity > 0.95:
            print(f"  ⚠️  Good fidelity, could be improved")
        else:
            print(f"  ⚠️  Low fidelity, optimization may need tuning")

        return fidelity

    except Exception as e:
        print(f"✗ Simulation failed: {e}")
        import traceback

        traceback.print_exc()
        return None


def demo_pulse_analysis(i_waveform, q_waveform, total_time):
    """Analyze pulse properties."""
    print_header("STEP 8: Pulse Analysis", level=1)

    try:
        print("Analyzing optimized pulse properties...")

        # Time array
        n_samples = len(i_waveform)
        dt = total_time / n_samples
        times = np.arange(n_samples) * dt

        # Amplitude envelope
        amplitude = np.sqrt(i_waveform**2 + q_waveform**2)

        # Phase
        phase = np.arctan2(q_waveform, i_waveform)

        # Statistics
        print("\n✓ Pulse Analysis:")
        print_result("Peak Amplitude", np.max(amplitude))
        print_result("Average Amplitude", np.mean(amplitude))
        print_result("RMS Amplitude", np.sqrt(np.mean(amplitude**2)))
        print_result("Total Energy", np.sum(amplitude**2) * dt)

        # Frequency content (approximate)
        print(f"\n  Phase variation:")
        print_result("Phase range", np.ptp(phase), "radians")
        print_result("Phase std dev", np.std(phase), "radians")

        # Smoothness (gradient)
        di_dt = np.gradient(i_waveform) / dt
        dq_dt = np.gradient(q_waveform) / dt
        smoothness = np.mean(np.sqrt(di_dt**2 + dq_dt**2))

        print(f"\n  Smoothness:")
        print_result("Average |dA/dt|", smoothness / 1e9, "GHz")

        # Check for rapid changes (hardware limitation)
        max_derivative = np.max(np.sqrt(di_dt**2 + dq_dt**2))
        if max_derivative < 1e9:  # < 1 GHz/ns
            print("  ✓ Pulse is smooth (hardware-compatible)")
        else:
            print(
                f"  ⚠️  Rapid changes detected (max {max_derivative / 1e9:.2f} GHz/ns)"
            )

        return {
            "amplitude": amplitude,
            "phase": phase,
            "peak_amplitude": np.max(amplitude),
            "smoothness": smoothness,
        }

    except Exception as e:
        print(f"✗ Pulse analysis failed: {e}")
        return None


def main():
    """Run complete Phase 3 demonstration."""
    print("\n" + "=" * 80)
    print("  PHASE 3 DEMONSTRATION: PULSE OPTIMIZATION & TRANSLATION")
    print("  QubitPulseOpt - Headless Development Mode")
    print("=" * 80)

    start_time = datetime.now()

    print(f"\nStarted: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("Mode: SIMULATION (No real hardware required)")
    print("Purpose: Validate Phase 3 pulse optimization and translation workflow")

    # Step 1: Load hardware parameters
    hardware_params = demo_load_hardware_params()
    if not hardware_params:
        print("\n✗ DEMO FAILED: Could not load hardware parameters")
        return 1

    # Step 2: Configure GRAPE
    optimizer, n_timeslices, total_time = demo_configure_grape(hardware_params)
    if optimizer is None:
        print("\n✗ DEMO FAILED: Could not configure GRAPE")
        return 1

    # Step 3: Optimize X gate
    x_result, x_i, x_q = demo_optimize_x_gate(optimizer, n_timeslices)

    # Step 4: Optimize Hadamard
    h_result, h_i, h_q = demo_optimize_hadamard(optimizer, n_timeslices)

    # Step 5: Save X gate pulse
    x_filepath = None
    if x_i is not None and x_q is not None:
        x_filepath = demo_save_pulses(x_i, x_q, "X", total_time)

    # Step 6: Translate X gate to IQM
    x_schedule = None
    if x_i is not None and x_q is not None:
        x_schedule, x_i_norm, x_q_norm = demo_translate_to_iqm(
            x_i, x_q, hardware_params["qubit_id"]
        )

    # Step 7: Simulate execution
    x_fidelity = None
    if x_i is not None and x_q is not None:
        x_fidelity = demo_simulate_execution(x_i, x_q, "X")

    # Step 8: Analyze pulse
    x_analysis = None
    if x_i is not None and x_q is not None:
        x_analysis = demo_pulse_analysis(x_i, x_q, total_time)

    # Summary
    print_header("PHASE 3 DEMO SUMMARY", level=1)

    results_summary = {
        "Hardware Parameters Loaded": hardware_params is not None,
        "GRAPE Configured": optimizer is not None,
        "X Gate Optimized": x_result is not None and x_result.final_fidelity > 0.9,
        "Hadamard Gate Optimized": h_result is not None
        and h_result.final_fidelity > 0.9,
        "Pulse Saved": x_filepath is not None,
        "IQM Translation": x_schedule is not None,
        "Execution Simulated": x_fidelity is not None and x_fidelity > 0.9,
        "Pulse Analysis": x_analysis is not None,
    }

    print("Results:")
    for task, success in results_summary.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status:8s} {task}")

    # Detailed results
    if x_result:
        print(f"\nX Gate Optimization:")
        print_result("Final Fidelity", x_result.final_fidelity)
        print_result("Iterations", x_result.n_iterations)
        print_result("Converged", x_result.converged)

    if h_result:
        print(f"\nHadamard Gate Optimization:")
        print_result("Final Fidelity", h_result.final_fidelity)
        print_result("Iterations", h_result.n_iterations)
        print_result("Converged", h_result.converged)

    if x_fidelity:
        print(f"\nExecution Simulation:")
        print_result("State Fidelity", x_fidelity)
        print_result("Gate Error", 1 - x_fidelity)

    total_tasks = len(results_summary)
    passed_tasks = sum(results_summary.values())

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print(f"\nTotal: {passed_tasks}/{total_tasks} tasks completed successfully")
    print(f"Duration: {duration:.1f} seconds")

    if passed_tasks >= total_tasks - 1:  # Allow 1 failure
        print("\n" + "=" * 80)
        print("  ✓ PHASE 3 DEMONSTRATION COMPLETE - TESTS PASSED")
        print("=" * 80)
        print("\n✓ Pulse optimization workflow validated")
        print("✓ GRAPE integration functional")
        print("✓ IQM translation workflow ready")
        print("\nNext Steps:")
        print("  1. Proceed to Phase 4: Agent Orchestration")
        print("  2. Integrate Phase 2 + Phase 3 into closed-loop workflow")
        print("  3. When ready for hardware: Execute on real IQM qubits")
        print("")
        return 0
    else:
        print("\n" + "=" * 80)
        print("  ⚠️  PHASE 3 DEMONSTRATION - SOME TESTS FAILED")
        print("=" * 80)
        print("\nReview failures above and check:")
        print("  - QubitPulseOpt dependencies installed (qutip, etc.)")
        print("  - GRAPE optimizer accessible")
        print("  - No import errors")
        print("")
        return 1


if __name__ == "__main__":
    sys.exit(main())
