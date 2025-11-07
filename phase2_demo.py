#!/usr/bin/env python3
"""
Phase 2 Demonstration: Hardware Characterization
=================================================

This script demonstrates the complete Phase 2 workflow for QubitPulseOpt:
1. Hardware backend initialization (emulator mode for headless development)
2. T1 (energy relaxation) characterization
3. T2 (dephasing) characterization using both Hahn echo and Ramsey
4. Rabi oscillation measurements
5. Full qubit characterization suite
6. Standard Randomized Benchmarking (gate fidelity)
7. Interleaved Randomized Benchmarking (specific gate fidelity)
8. Configuration update based on characterization results

This demo runs entirely in emulator/simulator mode and does NOT require
access to real IQM hardware. It validates all code paths and infrastructure.

Usage:
    python phase2_demo.py

Requirements:
    - IQM SDK installed (iqm-client, iqm-pulse, iqm-pulla)
    - qiskit-experiments
    - All dependencies from requirements-hardware.txt

Author: QubitPulseOpt Development Team
Phase: 2 - Hardware Characterization
Status: Headless Development (Emulator Mode)
"""

import sys
import os
from pathlib import Path
import logging
from datetime import datetime

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


def print_result(name, value, unit="", stderr=None):
    """Print a formatted result."""
    if stderr is not None:
        print(f"  {name:30s}: {value:.6g} ± {stderr:.6g} {unit}")
    else:
        print(f"  {name:30s}: {value:.6g} {unit}")


def demo_backend_initialization():
    """Demonstrate backend initialization and connection."""
    print_header("STEP 1: Backend Initialization", level=1)

    try:
        from hardware.iqm_backend import IQMBackendManager
        from dotenv import load_dotenv

        # Load environment
        load_dotenv()

        print("Initializing IQM Backend Manager...")
        manager = IQMBackendManager()
        print(f"✓ Backend manager created: {manager}")

        # Get emulator backend (no hardware needed)
        print("\nConnecting to emulator backend (headless mode)...")
        backend = manager.get_backend(use_emulator=True)

        backend_name = (
            backend.name if hasattr(backend, "name") else type(backend).__name__
        )
        print(f"✓ Backend connected: {backend_name}")

        # Get backend info
        if hasattr(backend, "configuration"):
            config = backend.configuration()
            n_qubits = config.n_qubits
            print(f"  Number of qubits: {n_qubits}")
            print(
                f"  Simulator: {config.simulator if hasattr(config, 'simulator') else 'N/A'}"
            )

        print("\n✓ Backend initialization successful")
        return backend

    except Exception as e:
        print(f"✗ Backend initialization failed: {e}")
        import traceback

        traceback.print_exc()
        return None


def demo_t1_characterization(backend):
    """Demonstrate T1 characterization experiment."""
    print_header("STEP 2: T1 (Energy Relaxation) Characterization", level=1)

    try:
        from hardware.characterization import HardwareCharacterizer

        print("Creating HardwareCharacterizer...")
        char = HardwareCharacterizer(backend, default_shots=1024)
        print(f"✓ Characterizer initialized: {char}")

        print("\nRunning T1 experiment on qubit 0...")
        print("  (Measuring energy relaxation time)")

        t1_result = char.run_t1_experiment(qubit=0, shots=1024, use_emulator=True)

        if t1_result["success"]:
            print("\n✓ T1 Experiment Complete")
            print_result(
                "T1 Time", t1_result["value"] * 1e6, "μs", t1_result["stderr"] * 1e6
            )
            print(f"  Shots: {t1_result['shots']}")
            print(f"  Qubit: {t1_result['qubit']}")

            # Store for later
            return t1_result
        else:
            print(
                f"\n✗ T1 experiment failed: {t1_result.get('error', 'Unknown error')}"
            )
            return None

    except Exception as e:
        print(f"✗ T1 characterization failed: {e}")
        import traceback

        traceback.print_exc()
        return None


def demo_t2_characterization(backend):
    """Demonstrate T2 characterization experiments."""
    print_header("STEP 3: T2 (Dephasing) Characterization", level=1)

    try:
        from hardware.characterization import HardwareCharacterizer

        char = HardwareCharacterizer(backend, default_shots=1024)

        # Test both methods
        results = {}

        # T2 Hahn Echo
        print("Running T2 Hahn Echo experiment on qubit 0...")
        print("  (Refocuses low-frequency noise)")

        t2_hahn = char.run_t2_experiment(
            qubit=0, shots=1024, method="hahn", use_emulator=True
        )

        if t2_hahn["success"]:
            print("\n✓ T2 Hahn Echo Complete")
            print_result(
                "T2 (Hahn)", t2_hahn["value"] * 1e6, "μs", t2_hahn["stderr"] * 1e6
            )
            results["hahn"] = t2_hahn
        else:
            print(f"\n✗ T2 Hahn failed: {t2_hahn.get('error', 'Unknown')}")

        # T2 Ramsey
        print("\nRunning T2 Ramsey experiment on qubit 0...")
        print("  (Free evolution, measures T2*)")

        t2_ramsey = char.run_t2_experiment(
            qubit=0, shots=1024, method="ramsey", use_emulator=True
        )

        if t2_ramsey["success"]:
            print("\n✓ T2 Ramsey Complete")
            print_result(
                "T2* (Ramsey)",
                t2_ramsey["value"] * 1e6,
                "μs",
                t2_ramsey["stderr"] * 1e6,
            )
            results["ramsey"] = t2_ramsey
        else:
            print(f"\n✗ T2 Ramsey failed: {t2_ramsey.get('error', 'Unknown')}")

        print("\n✓ T2 characterization complete")
        return results

    except Exception as e:
        print(f"✗ T2 characterization failed: {e}")
        import traceback

        traceback.print_exc()
        return None


def demo_rabi_characterization(backend):
    """Demonstrate Rabi oscillation measurements."""
    print_header("STEP 4: Rabi Oscillation Characterization", level=1)

    try:
        from hardware.characterization import HardwareCharacterizer

        char = HardwareCharacterizer(backend, default_shots=1024)

        print("Running Rabi experiment on qubit 0...")
        print("  (Measuring pulse amplitude calibration)")

        rabi_result = char.run_rabi_experiment(qubit=0, shots=1024, use_emulator=True)

        if rabi_result["success"]:
            print("\n✓ Rabi Experiment Complete")
            print_result(
                "Rabi Frequency",
                rabi_result["rate"] / 1e6,
                "MHz",
                rabi_result["stderr"] / 1e6,
            )
            print_result("Rabi Period", 1.0 / rabi_result["rate"] * 1e9, "ns")
            print(f"  Shots: {rabi_result['shots']}")
            print(f"  Qubit: {rabi_result['qubit']}")

            return rabi_result
        else:
            print(f"\n✗ Rabi experiment failed: {rabi_result.get('error', 'Unknown')}")
            return None

    except Exception as e:
        print(f"✗ Rabi characterization failed: {e}")
        import traceback

        traceback.print_exc()
        return None


def demo_full_characterization(backend):
    """Demonstrate full qubit characterization suite."""
    print_header("STEP 5: Full Qubit Characterization Suite", level=1)

    try:
        from hardware.characterization import HardwareCharacterizer

        char = HardwareCharacterizer(backend, default_shots=1024)

        print("Running complete characterization suite on qubit 0...")
        print("  (T1, T2, and Rabi in one call)")

        full_results = char.characterize_qubit(
            qubit=0, shots=1024, experiments=["T1", "T2", "Rabi"], use_emulator=True
        )

        if full_results["success"]:
            print("\n✓ Full Characterization Complete")
            print("\nSummary:")
            summary = full_results["summary"]

            if "T1" in summary:
                print_result(
                    "T1", summary["T1"] * 1e6, "μs", summary.get("T1_stderr", 0) * 1e6
                )

            if "T2" in summary:
                print_result(
                    "T2", summary["T2"] * 1e6, "μs", summary.get("T2_stderr", 0) * 1e6
                )

            if "rabi_rate" in summary:
                print_result(
                    "Rabi Rate",
                    summary["rabi_rate"] / 1e6,
                    "MHz",
                    summary.get("rabi_rate_stderr", 0) / 1e6,
                )

            print(f"\n  Experiments run: {full_results['experiments_run']}")
            print(f"  Total shots: {full_results['shots']} per experiment")

            return full_results
        else:
            print(f"\n✗ Full characterization incomplete")
            return None

    except Exception as e:
        print(f"✗ Full characterization failed: {e}")
        import traceback

        traceback.print_exc()
        return None


def demo_randomized_benchmarking(backend):
    """Demonstrate Standard Randomized Benchmarking."""
    print_header("STEP 6: Standard Randomized Benchmarking", level=1)

    try:
        from hardware.characterization import HardwareCharacterizer

        char = HardwareCharacterizer(backend, default_shots=512)

        print("Running Standard RB on qubit 0...")
        print("  (Measuring average gate fidelity)")
        print("  Note: Using shorter sequences and fewer samples for demo")

        rb_result = char.run_randomized_benchmarking(
            qubits=0,
            lengths=[1, 10, 20, 50, 75, 100],  # Shorter for demo
            num_samples=5,  # Fewer samples for speed
            shots=512,
            seed=42,
            use_emulator=True,
        )

        if rb_result["success"]:
            print("\n✓ Randomized Benchmarking Complete")
            print_result(
                "Error per Clifford (EPC)",
                rb_result["epc"],
                "",
                rb_result["epc_stderr"],
            )
            print_result("Average Gate Fidelity", rb_result["fidelity"], "")

            if rb_result["alpha"] is not None:
                print_result(
                    "Depolarizing parameter α",
                    rb_result["alpha"],
                    "",
                    rb_result.get("alpha_stderr", 0),
                )

            print(f"\n  Sequence lengths: {rb_result['lengths']}")
            print(f"  Samples per length: {rb_result['num_samples']}")
            print(f"  Shots per circuit: {rb_result['shots']}")
            print(
                f"  Total circuits: {len(rb_result['lengths']) * rb_result['num_samples']}"
            )

            return rb_result
        else:
            print(f"\n✗ RB failed: {rb_result.get('error', 'Unknown')}")
            return None

    except Exception as e:
        print(f"✗ Randomized Benchmarking failed: {e}")
        import traceback

        traceback.print_exc()
        return None


def demo_interleaved_rb(backend):
    """Demonstrate Interleaved Randomized Benchmarking."""
    print_header("STEP 7: Interleaved Randomized Benchmarking", level=1)

    try:
        from hardware.characterization import HardwareCharacterizer
        from qiskit.circuit.library import XGate

        char = HardwareCharacterizer(backend, default_shots=512)

        print("Running Interleaved RB on qubit 0 for X gate...")
        print("  (Measuring specific gate fidelity)")
        print("  Target gate: X (π rotation)")

        # Create the gate to test
        target_gate = XGate()

        irb_result = char.run_interleaved_rb(
            qubits=0,
            interleaved_gate=target_gate,
            lengths=[1, 10, 20, 50],  # Shorter for demo
            num_samples=5,
            shots=512,
            seed=42,
            use_emulator=True,
        )

        if irb_result["success"]:
            print("\n✓ Interleaved RB Complete")
            print_result(
                "Interleaved EPC",
                irb_result["epc_interleaved"],
                "",
                irb_result["epc_interleaved_stderr"],
            )

            if irb_result["epc_standard"] is not None:
                print_result("Standard EPC", irb_result["epc_standard"], "")

            if irb_result["gate_error"] is not None:
                print_result(
                    "X Gate Error",
                    irb_result["gate_error"],
                    "",
                    irb_result.get("gate_error_stderr", 0),
                )

            if irb_result["gate_fidelity"] is not None:
                print_result("X Gate Fidelity", irb_result["gate_fidelity"], "")

            print(f"\n  Sequence lengths: {irb_result['lengths']}")
            print(f"  Samples per length: {irb_result['num_samples']}")

            return irb_result
        else:
            print(f"\n✗ Interleaved RB failed: {irb_result.get('error', 'Unknown')}")
            return None

    except Exception as e:
        print(f"✗ Interleaved RB failed: {e}")
        import traceback

        traceback.print_exc()
        return None


def demo_config_update(char_results):
    """Demonstrate configuration update based on characterization results."""
    print_header("STEP 8: Configuration Update", level=1)

    try:
        import sys
        from pathlib import Path

        # Add src to path if not already there
        src_path = str(Path(__file__).parent / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

        from config import Config

        print("Loading current configuration...")
        config = Config()

        print(f"✓ Config loaded from: {config.__class__.__name__}")

        if char_results and char_results.get("success"):
            summary = char_results.get("summary", {})

            print("\nUpdating qubit parameters from characterization:")

            # Update T1
            if "T1" in summary:
                t1_value = summary["T1"]
                print(f"  Setting T1 = {t1_value * 1e6:.2f} μs")
                # In a real implementation, would update config here
                # config.qubits[0].T1 = t1_value

            # Update T2
            if "T2" in summary:
                t2_value = summary["T2"]
                print(f"  Setting T2 = {t2_value * 1e6:.2f} μs")
                # config.qubits[0].T2 = t2_value

            # Update Rabi rate
            if "rabi_rate" in summary:
                rabi = summary["rabi_rate"]
                print(f"  Setting Rabi rate = {rabi / 1e6:.2f} MHz")
                # config.qubits[0].rabi_rate = rabi

            print("\n✓ Configuration update complete (simulated)")
            print("  Note: In production, config would be saved to YAML")

            return True
        else:
            print("\n⚠️  No characterization results to update config")
            return False

    except Exception as e:
        print(f"✗ Config update failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run complete Phase 2 demonstration."""
    print("\n" + "=" * 80)
    print("  PHASE 2 DEMONSTRATION: HARDWARE CHARACTERIZATION")
    print("  QubitPulseOpt - Headless Development Mode")
    print("=" * 80)

    start_time = datetime.now()

    print(f"\nStarted: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("Mode: EMULATOR (No real hardware required)")
    print("Purpose: Validate all Phase 2 infrastructure and code paths")

    # Step 1: Initialize backend
    backend = demo_backend_initialization()
    if not backend:
        print("\n✗ DEMO FAILED: Could not initialize backend")
        return 1

    # Step 2: T1 characterization
    t1_result = demo_t1_characterization(backend)

    # Step 3: T2 characterization
    t2_results = demo_t2_characterization(backend)

    # Step 4: Rabi characterization
    rabi_result = demo_rabi_characterization(backend)

    # Step 5: Full characterization suite
    full_char_results = demo_full_characterization(backend)

    # Step 6: Standard RB
    rb_result = demo_randomized_benchmarking(backend)

    # Step 7: Interleaved RB
    irb_result = demo_interleaved_rb(backend)

    # Step 8: Config update
    config_updated = demo_config_update(full_char_results)

    # Summary
    print_header("PHASE 2 DEMO SUMMARY", level=1)

    results_summary = {
        "Backend Initialization": backend is not None,
        "T1 Characterization": t1_result is not None
        and t1_result.get("success", False),
        "T2 Characterization": t2_results is not None and len(t2_results) > 0,
        "Rabi Characterization": rabi_result is not None
        and rabi_result.get("success", False),
        "Full Characterization": full_char_results is not None
        and full_char_results.get("success", False),
        "Standard RB": rb_result is not None and rb_result.get("success", False),
        "Interleaved RB": irb_result is not None and irb_result.get("success", False),
        "Config Update": config_updated,
    }

    print("Results:")
    for task, success in results_summary.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status:8s} {task}")

    total_tasks = len(results_summary)
    passed_tasks = sum(results_summary.values())

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print(f"\nTotal: {passed_tasks}/{total_tasks} tasks completed successfully")
    print(f"Duration: {duration:.1f} seconds")

    if passed_tasks == total_tasks:
        print("\n" + "=" * 80)
        print("  ✓ PHASE 2 DEMONSTRATION COMPLETE - ALL TESTS PASSED")
        print("=" * 80)
        print("\n✓ Infrastructure validated and ready for Phase 3")
        print("✓ All characterization tools working correctly")
        print("✓ Benchmarking pipeline functional")
        print("\nNext Steps:")
        print("  1. Proceed to Phase 3: Pulse Optimization & Translation")
        print(
            "  2. When ready for hardware: Book IQM time and run with use_emulator=False"
        )
        print("  3. Compare simulated vs. real characterization results")
        print("")
        return 0
    else:
        print("\n" + "=" * 80)
        print("  ⚠️  PHASE 2 DEMONSTRATION - SOME TESTS FAILED")
        print("=" * 80)
        print("\nReview failures above and check:")
        print("  - All dependencies installed (qiskit-experiments, etc.)")
        print("  - No import errors")
        print("  - Backend accessible")
        print("")
        return 1


if __name__ == "__main__":
    sys.exit(main())
