#!/usr/bin/env python3
"""
Task 4 Demo: Visualization & Interactive Tools

This script demonstrates all the visualization capabilities implemented in Task 4:
1. OptimizationDashboard - Real-time optimization monitoring
2. ParameterSweepViewer - Parameter space exploration
3. PulseComparisonViewer - Pulse design comparison
4. BlochViewer3D - 3D Bloch sphere visualization
5. BlochAnimator - State evolution animations
6. PulseReport - Comprehensive pulse characterization
7. OptimizationReport - Optimization process reporting
8. Publication-quality figure generation
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
import qutip as qt

# Import visualization modules
from src.visualization.dashboard import (
    OptimizationDashboard,
    ParameterSweepViewer,
    PulseComparisonViewer,
    BlochViewer3D,
)
from src.visualization.bloch_animation import (
    BlochAnimator,
    AnimationStyle,
    create_bloch_animation,
)
from src.visualization.reports import (
    PulseReport,
    OptimizationReport,
    generate_latex_table,
    create_publication_figure,
)

# Create output directory
OUTPUT_DIR = Path("examples/task4_output")
OUTPUT_DIR.mkdir(exist_ok=True)
print(f"Output directory: {OUTPUT_DIR}")


def demo_optimization_dashboard():
    """Demonstrate real-time optimization dashboard."""
    print("\n" + "=" * 60)
    print("Demo 1: Optimization Dashboard")
    print("=" * 60)

    dashboard = OptimizationDashboard(n_controls=2, interactive=False)

    # Simulate optimization process
    print("Simulating optimization process...")
    n_iterations = 50
    n_timesteps = 100

    for i in range(n_iterations):
        # Simulate converging fidelity
        fidelity = 1.0 - 0.5 * np.exp(-i / 10)
        gradient_norm = 0.5 * np.exp(-i / 8)
        time_elapsed = 0.1 + 0.05 * np.random.rand()

        # Simulate control pulses
        controls = np.zeros((2, n_timesteps))
        controls[0] = np.sin(
            np.linspace(0, 2 * np.pi * (i + 1) / n_iterations, n_timesteps)
        )
        controls[1] = np.cos(
            np.linspace(0, 2 * np.pi * (i + 1) / n_iterations, n_timesteps)
        )

        # Update dashboard
        dashboard.update(
            iteration=i,
            fidelity=fidelity,
            gradient_norm=gradient_norm,
            time_elapsed=time_elapsed,
            controls=controls,
            robustness=0.9 + 0.05 * i / n_iterations,
        )

    # Save dashboard
    output_file = OUTPUT_DIR / "optimization_dashboard.png"
    dashboard.save(str(output_file), dpi=200)
    print(f"✓ Dashboard saved to {output_file}")

    # Export data
    data = dashboard.export_data()
    print(f"✓ Tracked {len(data['iterations'])} iterations")
    print(f"  Final fidelity: {data['fidelities'][-1]:.6f}")
    print(f"  Final gradient norm: {data['gradient_norms'][-1]:.6e}")

    dashboard.close()


def demo_parameter_sweep():
    """Demonstrate parameter sweep visualization."""
    print("\n" + "=" * 60)
    print("Demo 2: Parameter Sweep Viewer")
    print("=" * 60)

    viewer = ParameterSweepViewer()

    # Simulate parameter sweep (e.g., pulse amplitude vs duration)
    print("Simulating parameter sweep...")
    alphas = np.linspace(0, 2, 50)  # Pulse amplitude
    betas = np.linspace(0, 10, 50)  # Pulse duration
    A, B = np.meshgrid(alphas, betas)

    # Simulate fidelity landscape with optimal region
    fidelities = np.exp(-((A - 1.0) ** 2 + (B - 5.0) ** 2) / 2.0) * (
        1 - 0.1 * np.random.rand(*A.shape)
    )

    # Create heatmap
    fig, axes = viewer.plot_heatmap(
        alphas,
        betas,
        fidelities,
        x_label="Pulse Amplitude",
        y_label="Pulse Duration",
        z_label="Fidelity",
        title="Parameter Optimization Landscape",
        cmap="viridis",
    )

    output_file = OUTPUT_DIR / "parameter_sweep_heatmap.png"
    fig.savefig(output_file, dpi=200, bbox_inches="tight")
    print(f"✓ Heatmap saved to {output_file}")
    plt.close(fig)

    # Create 3D surface plot
    fig, ax = viewer.plot_3d_surface(
        alphas,
        betas,
        fidelities,
        x_label="Amplitude",
        y_label="Duration",
        z_label="Fidelity",
        title="Fidelity Surface",
        cmap="plasma",
    )

    output_file = OUTPUT_DIR / "parameter_sweep_3d.png"
    fig.savefig(output_file, dpi=200, bbox_inches="tight")
    print(f"✓ 3D surface saved to {output_file}")
    plt.close(fig)

    # Find optimal parameters
    max_idx = np.unravel_index(np.argmax(fidelities), fidelities.shape)
    opt_amp = alphas[max_idx[1]]
    opt_dur = betas[max_idx[0]]
    max_fid = fidelities[max_idx]
    print(f"✓ Optimal parameters: amplitude={opt_amp:.3f}, duration={opt_dur:.3f}")
    print(f"  Maximum fidelity: {max_fid:.6f}")


def demo_pulse_comparison():
    """Demonstrate pulse comparison viewer."""
    print("\n" + "=" * 60)
    print("Demo 3: Pulse Comparison Viewer")
    print("=" * 60)

    viewer = PulseComparisonViewer()

    # Create different pulse designs
    n_points = 200
    times = np.linspace(0, 10, n_points)

    # Gaussian pulse
    pulse_gaussian = 2.0 * np.exp(-((times - 5) ** 2) / 2.0)

    # DRAG-like pulse
    pulse_drag = 2.0 * np.exp(-((times - 5) ** 2) / 2.0) - 0.5 * (times - 5) * np.exp(
        -((times - 5) ** 2) / 2.0
    )

    # Square pulse with smoothed edges
    pulse_square = np.ones_like(times)
    pulse_square[times < 2] = 0.5 * (1 + np.tanh((times[times < 2] - 1) * 5))
    pulse_square[times > 8] = 0.5 * (1 + np.tanh((9 - times[times > 8]) * 5))

    # Composite pulse
    pulse_composite = (
        np.sin(2 * np.pi * times / 10) * np.exp(-((times - 5) ** 2) / 5.0) * 1.5
    )

    pulses = [pulse_gaussian, pulse_drag, pulse_square, pulse_composite]
    labels = ["Gaussian", "DRAG", "Square", "Composite"]
    metrics = {
        "fidelity": [0.985, 0.993, 0.978, 0.990],
        "duration": [10.0, 10.0, 10.0, 10.0],
        "energy": [8.5, 9.2, 12.0, 10.1],
        "robustness": [0.88, 0.92, 0.85, 0.91],
    }

    print("Comparing pulse designs...")
    fig = viewer.compare_pulses(pulses, labels, times=times, metrics=metrics)

    output_file = OUTPUT_DIR / "pulse_comparison.png"
    fig.savefig(output_file, dpi=200, bbox_inches="tight")
    print(f"✓ Pulse comparison saved to {output_file}")
    plt.close(fig)

    print("✓ Compared 4 pulse designs with performance metrics")


def demo_bloch_viewer():
    """Demonstrate 3D Bloch sphere viewer."""
    print("\n" + "=" * 60)
    print("Demo 4: 3D Bloch Sphere Viewer")
    print("=" * 60)

    viewer = BlochViewer3D()

    # Create interesting quantum states
    state_0 = qt.basis(2, 0)
    state_1 = qt.basis(2, 1)
    state_plus = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
    state_minus = (qt.basis(2, 0) - qt.basis(2, 1)).unit()
    state_plus_i = (qt.basis(2, 0) + 1j * qt.basis(2, 1)).unit()
    state_custom = (qt.basis(2, 0) + 0.5j * qt.basis(2, 1)).unit()

    states = [state_0, state_1, state_plus, state_minus, state_plus_i, state_custom]
    labels = ["|0⟩", "|1⟩", "|+⟩", "|-⟩", "|+i⟩", "custom"]

    print("Plotting quantum states on Bloch sphere...")
    fig, ax = viewer.plot_states(
        states, labels=labels, show_sphere=True, alpha_sphere=0.15
    )

    output_file = OUTPUT_DIR / "bloch_states.png"
    fig.savefig(output_file, dpi=200, bbox_inches="tight")
    print(f"✓ Bloch states saved to {output_file}")
    plt.close(fig)

    # Create trajectory (Rabi oscillation)
    print("Creating state evolution trajectory...")
    times = np.linspace(0, 2 * np.pi, 100)
    rabi_states = [qt.Qobj([[np.cos(t / 2)], [np.sin(t / 2) * 1j]]) for t in times]

    fig, ax = viewer.plot_trajectory(rabi_states, colormap="plasma")

    output_file = OUTPUT_DIR / "bloch_trajectory.png"
    fig.savefig(output_file, dpi=200, bbox_inches="tight")
    print(f"✓ Bloch trajectory saved to {output_file}")
    plt.close(fig)


def demo_bloch_animation():
    """Demonstrate Bloch sphere animations."""
    print("\n" + "=" * 60)
    print("Demo 5: Bloch Sphere Animations")
    print("=" * 60)

    # Create Rabi oscillation trajectory
    print("Creating Rabi oscillation animation...")
    times = np.linspace(0, 4 * np.pi, 60)
    rabi_states = [qt.Qobj([[np.cos(t / 2)], [np.sin(t / 2) * 1j]]) for t in times]

    style = AnimationStyle(
        sphere_alpha=0.12,
        trajectory_linewidth=2.5,
        point_size=180,
        colormap="viridis",
        background_color="white",
    )

    animator = BlochAnimator(rabi_states, labels=["Rabi Oscillation"], style=style)
    anim = animator.create_animation(interval=80, trail_length=15, show_trail=True)

    # Save as GIF
    output_file = OUTPUT_DIR / "rabi_oscillation.gif"
    try:
        animator.save(str(output_file), fps=15, dpi=80)
        print(f"✓ Animation saved to {output_file}")
    except Exception as e:
        print(f"⚠ Could not save animation: {e}")
        print("  (This may require pillow or ffmpeg)")

    animator.close()

    # Create multiple trajectory comparison
    print("Creating multiple trajectory comparison...")
    # Fast precession
    angles1 = np.linspace(0, 4 * np.pi, 50)
    traj1 = [
        qt.Qobj([[np.cos(a / 2)], [np.exp(1j * a) * np.sin(a / 2)]]) for a in angles1
    ]

    # Slow precession
    angles2 = np.linspace(0, 2 * np.pi, 50)
    traj2 = [
        qt.Qobj([[np.cos(a / 2)], [np.exp(1j * a) * np.sin(a / 2)]]) for a in angles2
    ]

    animator2 = BlochAnimator([traj1, traj2], labels=["Fast", "Slow"])
    anim2 = animator2.create_animation(interval=100, trail_length=10)

    output_file = OUTPUT_DIR / "multiple_trajectories.gif"
    try:
        animator2.save(str(output_file), fps=12, dpi=80)
        print(f"✓ Multi-trajectory animation saved to {output_file}")
    except Exception as e:
        print(f"⚠ Could not save animation: {e}")

    animator2.close()


def demo_pulse_report():
    """Demonstrate comprehensive pulse reporting."""
    print("\n" + "=" * 60)
    print("Demo 6: Pulse Characterization Report")
    print("=" * 60)

    # Create optimized pulse
    times = np.linspace(0, 10, 200)
    pulse_optimized = (
        2.0 * np.exp(-((times - 5) ** 2) / 2.0) * np.sin(4 * np.pi * times / 10)
    )

    print("Generating pulse characterization report...")
    report = PulseReport(
        pulse_optimized,
        times=times,
        fidelity=0.995,
        target_gate="Hadamard",
        optimization_method="GRAPE",
        label="Optimized Pulse",
    )

    # Add comparison pulses
    pulse_simple = 1.5 * np.sin(2 * np.pi * times / 10)
    pulse_gaussian = 2.0 * np.exp(-((times - 5) ** 2) / 2.0)

    report.add_comparison(pulse_simple, "Simple Sine", {"fidelity": 0.92})
    report.add_comparison(pulse_gaussian, "Gaussian", {"fidelity": 0.98})

    # Generate full report
    output_file = OUTPUT_DIR / "pulse_report.png"
    fig = report.generate_full_report(filename=str(output_file))
    print(f"✓ Full report saved to {output_file}")
    plt.close(fig)

    # Export metrics in different formats
    print("Exporting pulse metrics...")

    latex_file = OUTPUT_DIR / "pulse_metrics.tex"
    report.export_metrics_table(str(latex_file), format="latex")
    print(f"✓ LaTeX table: {latex_file}")

    csv_file = OUTPUT_DIR / "pulse_metrics.csv"
    report.export_metrics_table(str(csv_file), format="csv")
    print(f"✓ CSV table: {csv_file}")

    json_file = OUTPUT_DIR / "pulse_metrics.json"
    report.export_metrics_table(str(json_file), format="json")
    print(f"✓ JSON data: {json_file}")

    # Print characteristics
    chars = report.characteristics
    print(f"\nPulse Characteristics:")
    print(f"  Duration: {chars.duration:.3f}")
    print(f"  Peak Amplitude: {chars.peak_amplitude:.4f}")
    print(f"  RMS Amplitude: {chars.rms_amplitude:.4f}")
    print(f"  Energy: {chars.energy:.4f}")
    print(f"  Bandwidth: {chars.bandwidth:.4f}")
    print(f"  Fidelity: {chars.fidelity:.6f}")


def demo_optimization_report():
    """Demonstrate optimization process reporting."""
    print("\n" + "=" * 60)
    print("Demo 7: Optimization Process Report")
    print("=" * 60)

    report = OptimizationReport(method="GRAPE", target="X Gate")

    # Simulate optimization iterations
    print("Simulating optimization process...")
    n_iterations = 100

    for i in range(n_iterations):
        fidelity = 1.0 - 0.5 * np.exp(-i / 15)
        gradient_norm = 0.8 * np.exp(-i / 12)
        robustness = 0.8 + 0.15 * (1 - np.exp(-i / 20))

        report.add_iteration(
            iteration=i,
            fidelity=fidelity,
            gradient_norm=gradient_norm,
            robustness=robustness,
        )

    report.finalize()

    # Generate summary
    output_file = OUTPUT_DIR / "optimization_report.png"
    fig = report.generate_summary(filename=str(output_file))
    print(f"✓ Optimization report saved to {output_file}")
    plt.close(fig)

    print(f"✓ Tracked {len(report.iterations)} iterations")
    print(f"  Initial fidelity: {report.fidelities[0]:.6f}")
    print(f"  Final fidelity: {report.fidelities[-1]:.6f}")
    print(f"  Improvement: {report.fidelities[-1] - report.fidelities[0]:.6f}")


def demo_latex_tables():
    """Demonstrate LaTeX table generation."""
    print("\n" + "=" * 60)
    print("Demo 8: LaTeX Table Generation")
    print("=" * 60)

    # Comparison of optimization methods
    data = {
        "Method": ["GRAPE", "Krotov", "CRAB", "DRAG"],
        "Fidelity": [0.9950, 0.9935, 0.9920, 0.9910],
        "Duration": [10.5, 12.3, 8.7, 11.2],
        "Energy": [5.2, 4.8, 6.1, 5.5],
        "Robustness": [0.92, 0.90, 0.88, 0.89],
    }

    print("Generating comparison table...")
    output_file = OUTPUT_DIR / "method_comparison.tex"
    latex_str = generate_latex_table(
        data,
        str(output_file),
        caption="Comparison of Quantum Control Optimization Methods",
        label="tab:method_comparison",
    )

    print(f"✓ LaTeX table saved to {output_file}")
    print("\nGenerated LaTeX code:")
    print("-" * 60)
    print(latex_str)
    print("-" * 60)


def demo_publication_figures():
    """Demonstrate publication-quality figure creation."""
    print("\n" + "=" * 60)
    print("Demo 9: Publication-Quality Figures")
    print("=" * 60)

    # Create sample data
    x = np.linspace(0, 10, 200)
    data1 = np.exp(-x / 5) * np.sin(2 * np.pi * x)
    data2 = np.exp(-x / 7) * np.cos(2 * np.pi * x)
    data3 = np.exp(-x / 6) * np.sin(2 * np.pi * x + np.pi / 4)

    print("Creating publication-quality figure...")
    fig = create_publication_figure(
        [data1, data2, data3],
        labels=["Method A", "Method B", "Method C"],
        xlabel="Time (μs)",
        ylabel="Amplitude (a.u.)",
        title="Pulse Optimization Comparison",
        figsize=(10, 6),
    )

    # Save in multiple formats
    for ext in ["png", "pdf"]:
        output_file = OUTPUT_DIR / f"publication_figure.{ext}"
        fig.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"✓ Saved {ext.upper()}: {output_file}")

    plt.close(fig)


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 80)
    print(" " * 20 + "TASK 4 VISUALIZATION DEMO")
    print(" " * 15 + "Quantum Control Simulation Project")
    print("=" * 80)

    try:
        demo_optimization_dashboard()
        demo_parameter_sweep()
        demo_pulse_comparison()
        demo_bloch_viewer()
        demo_bloch_animation()
        demo_pulse_report()
        demo_optimization_report()
        demo_latex_tables()
        demo_publication_figures()

        print("\n" + "=" * 80)
        print(" " * 25 + "ALL DEMOS COMPLETED!")
        print("=" * 80)
        print(f"\nAll outputs saved to: {OUTPUT_DIR.absolute()}")
        print("\nKey Features Demonstrated:")
        print("  ✓ Real-time optimization monitoring")
        print("  ✓ Parameter space exploration")
        print("  ✓ Pulse design comparison")
        print("  ✓ 3D Bloch sphere visualization")
        print("  ✓ State evolution animations")
        print("  ✓ Comprehensive pulse characterization")
        print("  ✓ Optimization process reporting")
        print("  ✓ LaTeX table generation")
        print("  ✓ Publication-quality figures")
        print("\n" + "=" * 80)

    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
