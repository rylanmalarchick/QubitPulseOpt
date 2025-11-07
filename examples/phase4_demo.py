#!/usr/bin/env python3
"""
Phase 4 Demo: QubitCalibrator AI Agent with ReAct Loop
=======================================================

This demo showcases the QubitCalibrator AI agent orchestrating autonomous
hardware-in-the-loop quantum control optimization using a ReAct (Reasoning + Acting)
loop architecture.

The agent autonomously:
1. Gets hardware topology
2. Characterizes qubit parameters (T1, T2, Rabi)
3. Updates simulation config with real data
4. Optimizes pulses using GRAPE
5. Translates and executes on hardware
6. Benchmarks fidelity using RB
7. Iterates until target fidelity achieved

This demo runs in emulator mode by default. Set use_emulator=False for real hardware.

Usage:
    python phase4_demo.py

Author: QubitPulseOpt Development Team
Phase: 4 - Agent Orchestration
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.agent import QubitCalibratorAgent, TraceLogger
from src.logging_utils import setup_logging

# Configure logging
setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)


def print_banner(text: str) -> None:
    """Print a formatted banner."""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80 + "\n")


def demo_basic_agent_workflow():
    """
    Demo 1: Basic agent workflow with single qubit calibration.
    """
    print_banner("DEMO 1: Basic Agent Workflow - Single Qubit Calibration")

    print("Initializing QubitCalibrator agent in emulator mode...")
    agent = QubitCalibratorAgent(
        use_emulator=True,
        max_iterations=15,
        session_id="demo1_basic_workflow",
    )

    print("Agent initialized successfully!")
    print(f"  - Backend: Emulator (AerSimulator)")
    print(f"  - Max iterations: 15")
    print(f"  - Session ID: demo1_basic_workflow")

    print("\n" + "-" * 80)
    print("Running calibration workflow for qubit QB1...")
    print("  Target gate: X")
    print("  Target fidelity: 0.99")
    print("-" * 80 + "\n")

    try:
        result = agent.run_calibration_workflow(
            target_qubit="QB1",
            target_gate="X",
            target_fidelity=0.99,
        )

        print("\n" + "-" * 80)
        print("WORKFLOW RESULTS")
        print("-" * 80)
        print(f"Status: {result['status']}")
        print(f"Goal Achieved: {result['goal_achieved']}")
        print(f"Total Iterations: {result['iterations']}")

        if "final_fidelity" in result:
            print(f"Final Fidelity: {result['final_fidelity']:.4f}")

        if "characterization" in result:
            char = result["characterization"]
            print(f"\nCharacterization Results:")
            if "t1" in char:
                print(f"  T1: {char['t1']:.2f} µs")
            if "t2" in char:
                print(f"  T2: {char['t2']:.2f} µs")
            if "frequency" in char:
                print(f"  Frequency: {char['frequency']:.2f} MHz")

        if "trace_path" in result:
            print(f"\nTrace saved to: {result['trace_path']}")

        print("\n✓ Demo 1 completed successfully!")
        return True

    except Exception as e:
        print(f"\n✗ Demo 1 failed: {e}")
        logger.error(f"Demo 1 error: {e}", exc_info=True)
        return False


def demo_trace_logging():
    """
    Demo 2: Structured trace logging and analysis.
    """
    print_banner("DEMO 2: Structured Trace Logging")

    print("Creating a simple trace logger...")
    trace_logger = TraceLogger(session_id="demo2_trace_example")

    print("Simulating a simple ReAct loop with logging...\n")

    # Simulate ReAct loop
    trace_logger.log_thought(
        "Need to understand hardware topology before proceeding.",
        context={"step": "initialization"},
    )

    trace_logger.log_action(
        "get_hardware_topology",
        {},
        context={"step": "topology_fetch"},
    )

    trace_logger.log_observation(
        {
            "status": "success",
            "num_qubits": 5,
            "topology": {"qubits": ["QB1", "QB2", "QB3", "QB4", "QB5"]},
        }
    )

    trace_logger.log_thought(
        "Topology retrieved. 5 qubits available. Starting characterization of QB1.",
        context={"step": "planning"},
    )

    trace_logger.log_action(
        "run_characterization_experiment",
        {"experiment_type": "T1", "qubits": ["QB1"]},
        context={"step": "characterization"},
    )

    trace_logger.log_observation(
        {
            "status": "success",
            "results": {"QB1": {"t1": 25.3, "t1_error": 0.5}},
        }
    )

    trace_logger.log_completion(
        {"final_status": "demo_complete"},
        status="completed",
    )

    # Save and display
    trace_path = trace_logger.save_trace()
    print(f"Trace saved to: {trace_path}")

    print("\nTrace Summary:")
    trace_logger.print_summary()

    # Get and display summary
    summary = trace_logger.get_summary()
    print("Key Metrics:")
    print(f"  - Duration: {summary['duration_seconds']:.2f} seconds")
    print(f"  - Total Steps: {summary['total_steps']}")
    print(f"  - Thoughts: {summary['total_thoughts']}")
    print(f"  - Actions: {summary['total_actions']}")
    print(f"  - Observations: {summary['total_observations']}")
    print(f"  - Success Rate: {summary['success_rate']:.1%}")

    print("\n✓ Demo 2 completed successfully!")
    return True


def demo_multi_qubit_calibration():
    """
    Demo 3: Multi-qubit sequential calibration.
    """
    print_banner("DEMO 3: Multi-Qubit Sequential Calibration")

    print("Initializing agent for multi-qubit calibration...")
    agent = QubitCalibratorAgent(
        use_emulator=True,
        max_iterations=10,
        session_id="demo3_multi_qubit",
    )

    qubits_to_calibrate = ["QB1", "QB2"]
    results = {}

    for qubit in qubits_to_calibrate:
        print(f"\n{'-' * 80}")
        print(f"Calibrating {qubit}...")
        print(f"{'-' * 80}\n")

        try:
            result = agent.calibrate_single_qubit(
                qubit=qubit,
                target_gate="X",
                target_fidelity=0.99,
            )

            results[qubit] = result
            print(f"\n✓ {qubit} calibration completed!")
            print(f"  Status: {result['status']}")
            print(f"  Iterations: {result['iterations']}")

            # Reset agent state for next qubit
            agent.reset()

        except Exception as e:
            print(f"\n✗ {qubit} calibration failed: {e}")
            logger.error(f"{qubit} calibration error: {e}", exc_info=True)
            results[qubit] = {"status": "error", "message": str(e)}

    print("\n" + "-" * 80)
    print("MULTI-QUBIT CALIBRATION SUMMARY")
    print("-" * 80)

    for qubit, result in results.items():
        status = result.get("status", "unknown")
        iterations = result.get("iterations", "N/A")
        print(f"{qubit}: {status} (iterations: {iterations})")

    print("\n✓ Demo 3 completed successfully!")
    return True


def demo_agent_state_management():
    """
    Demo 4: Agent state management and inspection.
    """
    print_banner("DEMO 4: Agent State Management")

    print("Creating agent and inspecting state...")
    agent = QubitCalibratorAgent(
        use_emulator=True,
        max_iterations=5,
        session_id="demo4_state_mgmt",
    )

    print("\nInitial state:")
    state = agent.get_state()
    print(f"  Topology: {state['topology']}")
    print(f"  Characterized qubits: {list(state['characterized_qubits'].keys())}")
    print(f"  Optimized pulses: {list(state['optimized_pulses'].keys())}")
    print(f"  Benchmarks: {list(state['benchmarks'].keys())}")
    print(f"  Goal achieved: {state['goal_achieved']}")

    print("\nRunning partial workflow (limited iterations)...")
    try:
        result = agent.run_calibration_workflow(
            target_qubit="QB1",
            target_gate="X",
            target_fidelity=0.99,
        )

        print("\nPost-workflow state:")
        state = agent.get_state()
        print(f"  Topology available: {state['topology'] is not None}")
        print(f"  Characterized qubits: {list(state['characterized_qubits'].keys())}")
        print(f"  Optimized pulses: {list(state['optimized_pulses'].keys())}")
        print(f"  Benchmarks: {list(state['benchmarks'].keys())}")
        print(f"  Goal achieved: {state['goal_achieved']}")

        print(f"\nWorkflow result:")
        print(f"  Status: {result['status']}")
        print(f"  Iterations used: {result['iterations']}/5")

        print("\nResetting agent state...")
        agent.reset()

        print("Post-reset state:")
        state = agent.get_state()
        print(f"  Topology: {state['topology']}")
        print(f"  Characterized qubits: {list(state['characterized_qubits'].keys())}")
        print(f"  Goal achieved: {state['goal_achieved']}")

        print("\n✓ Demo 4 completed successfully!")
        return True

    except Exception as e:
        print(f"\n✗ Demo 4 failed: {e}")
        logger.error(f"Demo 4 error: {e}", exc_info=True)
        return False


def run_all_demos():
    """Run all Phase 4 demos."""
    print_banner("PHASE 4: QubitCalibrator AI Agent - Complete Demo Suite")

    print("This demo showcases the autonomous QubitCalibrator AI agent using")
    print("a ReAct (Reasoning + Acting) loop for hardware-in-the-loop quantum")
    print("control optimization.")
    print("\nRunning in EMULATOR mode (AerSimulator).")
    print("Set use_emulator=False in code for real IQM hardware.\n")

    demos = [
        ("Basic Agent Workflow", demo_basic_agent_workflow),
        ("Structured Trace Logging", demo_trace_logging),
        ("Multi-Qubit Calibration", demo_multi_qubit_calibration),
        ("Agent State Management", demo_agent_state_management),
    ]

    results = []
    for name, demo_func in demos:
        try:
            success = demo_func()
            results.append((name, success))
        except Exception as e:
            logger.error(f"Demo '{name}' crashed: {e}", exc_info=True)
            results.append((name, False))
            print(f"\n✗ Demo '{name}' crashed with error: {e}")

    # Summary
    print_banner("DEMO SUITE SUMMARY")

    passed = sum(1 for _, success in results if success)
    total = len(results)

    print(f"Results: {passed}/{total} demos passed\n")

    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")

    print("\n" + "=" * 80)

    if passed == total:
        print("🎉 All Phase 4 demos completed successfully!")
    else:
        print(f"⚠️  {total - passed} demo(s) failed. Check logs for details.")

    print("=" * 80 + "\n")

    print("Next steps:")
    print("  1. Review agent traces in agent_logs/")
    print("  2. Run on real hardware: set use_emulator=False and provide IQM_TOKEN")
    print("  3. Explore the Jupyter notebook: notebooks/agents/03_phase4_agent.ipynb")
    print("  4. Customize agent reasoning in qubit_calibrator_agent.py")
    print("  5. Add new tools in src/agent/tools.py")

    return passed == total


if __name__ == "__main__":
    success = run_all_demos()
    sys.exit(0 if success else 1)
