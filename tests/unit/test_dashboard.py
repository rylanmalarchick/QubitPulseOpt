"""
Unit tests for visualization dashboard module.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
from src.visualization.dashboard import (
    OptimizationDashboard,
    ParameterSweepViewer,
    PulseComparisonViewer,
    BlochViewer3D,
    OptimizationMetrics,
)


class TestOptimizationMetrics:
    """Tests for OptimizationMetrics dataclass."""

    def test_creation(self):
        """Test creating OptimizationMetrics."""
        metrics = OptimizationMetrics(
            iteration=10,
            fidelity=0.99,
            infidelity=0.01,
            gradient_norm=0.001,
            time_elapsed=1.5,
        )
        assert metrics.iteration == 10
        assert metrics.fidelity == 0.99
        assert metrics.custom_metrics is None

    def test_with_custom_metrics(self):
        """Test with custom metrics."""
        custom = {"robustness": 0.95, "energy": 2.5}
        metrics = OptimizationMetrics(
            iteration=5,
            fidelity=0.98,
            infidelity=0.02,
            gradient_norm=0.01,
            time_elapsed=2.0,
            custom_metrics=custom,
        )
        assert metrics.custom_metrics["robustness"] == 0.95
        assert metrics.custom_metrics["energy"] == 2.5


class TestOptimizationDashboard:
    """Tests for OptimizationDashboard."""

    def test_initialization(self):
        """Test dashboard initialization."""
        dashboard = OptimizationDashboard(n_controls=2, interactive=False)
        assert dashboard.n_controls == 2
        assert dashboard.interactive is False
        assert len(dashboard.iterations) == 0
        assert dashboard.fig is not None

    def test_single_update(self):
        """Test single dashboard update."""
        dashboard = OptimizationDashboard(n_controls=1, interactive=False)
        controls = np.random.randn(1, 50)

        dashboard.update(
            iteration=0,
            fidelity=0.95,
            gradient_norm=0.1,
            time_elapsed=0.5,
            controls=controls,
        )

        assert len(dashboard.iterations) == 1
        assert dashboard.fidelities[0] == 0.95
        assert len(dashboard.controls_history) == 1

    def test_multiple_updates(self):
        """Test multiple updates."""
        dashboard = OptimizationDashboard(n_controls=2, interactive=False)

        for i in range(10):
            fidelity = 0.5 + 0.05 * i
            gradient = 0.1 / (i + 1)
            controls = np.random.randn(2, 30)

            dashboard.update(
                iteration=i,
                fidelity=fidelity,
                gradient_norm=gradient,
                controls=controls,
            )

        assert len(dashboard.iterations) == 10
        assert len(dashboard.fidelities) == 10
        assert len(dashboard.gradient_norms) == 10
        assert dashboard.fidelities[-1] > dashboard.fidelities[0]

    def test_infidelity_computation(self):
        """Test automatic infidelity computation."""
        dashboard = OptimizationDashboard(interactive=False)
        dashboard.update(iteration=0, fidelity=0.99)

        assert len(dashboard.infidelities) == 1
        assert dashboard.infidelities[0] == pytest.approx(0.01)

    def test_custom_metrics(self):
        """Test custom metrics tracking."""
        dashboard = OptimizationDashboard(interactive=False)

        dashboard.update(
            iteration=0, fidelity=0.95, robustness=0.9, energy=2.5, custom_val=42.0
        )

        assert "robustness" in dashboard.custom_metrics
        assert "energy" in dashboard.custom_metrics
        assert "custom_val" in dashboard.custom_metrics
        assert dashboard.custom_metrics["robustness"][0] == 0.9

    def test_export_data(self):
        """Test data export."""
        dashboard = OptimizationDashboard(interactive=False)

        for i in range(5):
            dashboard.update(iteration=i, fidelity=0.9 + i * 0.01)

        data = dashboard.export_data()

        assert "iterations" in data
        assert "fidelities" in data
        assert len(data["iterations"]) == 5
        assert isinstance(data["iterations"], np.ndarray)

    def test_save(self, tmp_path):
        """Test saving dashboard figure."""
        dashboard = OptimizationDashboard(interactive=False)
        dashboard.update(iteration=0, fidelity=0.95, gradient_norm=0.01)

        output_file = tmp_path / "dashboard.png"
        dashboard.save(str(output_file))

        assert output_file.exists()

    def test_close(self):
        """Test closing dashboard."""
        dashboard = OptimizationDashboard(interactive=False)
        dashboard.update(iteration=0, fidelity=0.95)
        dashboard.close()

        # After closing, figure should be closed
        assert not plt.fignum_exists(dashboard.fig.number)

    def test_2d_controls(self):
        """Test with 2D control array."""
        dashboard = OptimizationDashboard(n_controls=3, interactive=False)
        controls = np.random.randn(3, 40)

        dashboard.update(iteration=0, controls=controls)

        assert len(dashboard.controls_history) == 1
        assert dashboard.controls_history[0].shape == (3, 40)

    def test_1d_controls(self):
        """Test with 1D control array (single control)."""
        dashboard = OptimizationDashboard(n_controls=1, interactive=False)
        controls = np.random.randn(40)

        dashboard.update(iteration=0, controls=controls)

        # Should handle 1D correctly
        assert len(dashboard.controls_history) == 1


class TestParameterSweepViewer:
    """Tests for ParameterSweepViewer."""

    def test_initialization(self):
        """Test viewer initialization."""
        viewer = ParameterSweepViewer(figsize=(10, 6))
        assert viewer.figsize == (10, 6)

    def test_plot_heatmap(self):
        """Test heatmap plotting."""
        viewer = ParameterSweepViewer()

        x = np.linspace(0, 1, 20)
        y = np.linspace(0, 2, 20)
        X, Y = np.meshgrid(x, y)
        z = np.sin(X * np.pi) * np.cos(Y * np.pi)

        fig, axes = viewer.plot_heatmap(
            x, y, z, x_label="Alpha", y_label="Beta", z_label="Fidelity"
        )

        assert fig is not None
        assert len(axes) == 3
        plt.close(fig)

    def test_plot_heatmap_with_limits(self):
        """Test heatmap with custom color limits."""
        viewer = ParameterSweepViewer()

        x = np.linspace(0, 1, 15)
        y = np.linspace(0, 1, 15)
        X, Y = np.meshgrid(x, y)
        z = X + Y

        fig, axes = viewer.plot_heatmap(x, y, z, vmin=0, vmax=2)

        assert fig is not None
        plt.close(fig)

    def test_plot_3d_surface(self):
        """Test 3D surface plotting."""
        viewer = ParameterSweepViewer()

        x = np.linspace(-1, 1, 15)
        y = np.linspace(-1, 1, 15)
        X, Y = np.meshgrid(x, y)
        z = np.exp(-(X**2 + Y**2))

        fig, ax = viewer.plot_3d_surface(x, y, z, x_label="X", y_label="Y", z_label="Z")

        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_different_array_sizes(self):
        """Test with different array sizes."""
        viewer = ParameterSweepViewer()

        x = np.linspace(0, 1, 10)
        y = np.linspace(0, 1, 15)
        X, Y = np.meshgrid(x, y)
        z = X * Y

        fig, axes = viewer.plot_heatmap(x, y, z)

        assert fig is not None
        assert z.shape == (len(y), len(x))
        plt.close(fig)


class TestPulseComparisonViewer:
    """Tests for PulseComparisonViewer."""

    def test_initialization(self):
        """Test viewer initialization."""
        viewer = PulseComparisonViewer(figsize=(12, 8))
        assert viewer.figsize == (12, 8)

    def test_compare_two_pulses(self):
        """Test comparing two pulses."""
        viewer = PulseComparisonViewer()

        pulse1 = np.sin(np.linspace(0, 2 * np.pi, 100))
        pulse2 = np.cos(np.linspace(0, 2 * np.pi, 100))
        pulses = [pulse1, pulse2]
        labels = ["Sin", "Cos"]

        fig = viewer.compare_pulses(pulses, labels)

        assert fig is not None
        plt.close(fig)

    def test_compare_with_metrics(self):
        """Test comparison with performance metrics."""
        viewer = PulseComparisonViewer()

        pulse1 = np.random.randn(100)
        pulse2 = np.random.randn(100)
        pulses = [pulse1, pulse2]
        labels = ["GRAPE", "Krotov"]
        metrics = {
            "fidelity": [0.99, 0.98],
            "duration": [10.0, 12.0],
            "energy": [5.0, 4.5],
        }

        fig = viewer.compare_pulses(pulses, labels, metrics=metrics)

        assert fig is not None
        plt.close(fig)

    def test_compare_with_custom_times(self):
        """Test with custom time array."""
        viewer = PulseComparisonViewer()

        times = np.linspace(0, 10, 100)
        pulse1 = np.exp(-times / 5) * np.sin(times)
        pulse2 = np.exp(-times / 3) * np.cos(times)

        fig = viewer.compare_pulses(
            [pulse1, pulse2], ["Pulse 1", "Pulse 2"], times=times
        )

        assert fig is not None
        plt.close(fig)

    def test_compare_multiple_pulses(self):
        """Test comparing multiple pulses."""
        viewer = PulseComparisonViewer()

        pulses = [np.random.randn(80) for _ in range(5)]
        labels = [f"Pulse {i + 1}" for i in range(5)]

        fig = viewer.compare_pulses(pulses, labels)

        assert fig is not None
        plt.close(fig)

    def test_compare_different_lengths(self):
        """Test comparing pulses of different lengths."""
        viewer = PulseComparisonViewer()

        pulse1 = np.random.randn(100)
        pulse2 = np.random.randn(80)
        pulse3 = np.random.randn(120)

        fig = viewer.compare_pulses(
            [pulse1, pulse2, pulse3], ["Short", "Medium", "Long"]
        )

        assert fig is not None
        plt.close(fig)


class TestBlochViewer3D:
    """Tests for BlochViewer3D."""

    def test_initialization(self):
        """Test viewer initialization."""
        viewer = BlochViewer3D(figsize=(8, 8))
        assert viewer.figsize == (8, 8)

    def test_state_to_bloch_basis_states(self):
        """Test conversion of basis states to Bloch vectors."""
        viewer = BlochViewer3D()

        # |0> should map to [0, 0, 1]
        state_0 = qt.basis(2, 0)
        bloch_0 = viewer._state_to_bloch(state_0)
        assert bloch_0[2] == pytest.approx(1.0)
        assert np.abs(bloch_0[0]) < 1e-10
        assert np.abs(bloch_0[1]) < 1e-10

        # |1> should map to [0, 0, -1]
        state_1 = qt.basis(2, 1)
        bloch_1 = viewer._state_to_bloch(state_1)
        assert bloch_1[2] == pytest.approx(-1.0)

    def test_state_to_bloch_superposition(self):
        """Test conversion of superposition states."""
        viewer = BlochViewer3D()

        # |+> = (|0> + |1>)/sqrt(2) should map to [1, 0, 0]
        state_plus = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
        bloch_plus = viewer._state_to_bloch(state_plus)
        assert bloch_plus[0] == pytest.approx(1.0)
        assert np.abs(bloch_plus[1]) < 1e-10
        assert np.abs(bloch_plus[2]) < 1e-10

        # |+i> = (|0> + i|1>)/sqrt(2) should map to [0, 1, 0]
        state_plus_i = (qt.basis(2, 0) + 1j * qt.basis(2, 1)).unit()
        bloch_plus_i = viewer._state_to_bloch(state_plus_i)
        assert np.abs(bloch_plus_i[0]) < 1e-10
        assert bloch_plus_i[1] == pytest.approx(1.0)
        assert np.abs(bloch_plus_i[2]) < 1e-10

    def test_plot_single_state(self):
        """Test plotting a single state."""
        viewer = BlochViewer3D()
        state = qt.basis(2, 0)

        fig, ax = viewer.plot_states([state], labels=["|0>"])

        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_plot_multiple_states(self):
        """Test plotting multiple states."""
        viewer = BlochViewer3D()

        states = [
            qt.basis(2, 0),
            qt.basis(2, 1),
            (qt.basis(2, 0) + qt.basis(2, 1)).unit(),
        ]
        labels = ["|0>", "|1>", "|+>"]

        fig, ax = viewer.plot_states(states, labels=labels)

        assert fig is not None
        plt.close(fig)

    def test_plot_without_sphere(self):
        """Test plotting without Bloch sphere."""
        viewer = BlochViewer3D()
        state = (qt.basis(2, 0) + qt.basis(2, 1)).unit()

        fig, ax = viewer.plot_states([state], show_sphere=False)

        assert fig is not None
        plt.close(fig)

    def test_plot_trajectory(self):
        """Test plotting a trajectory."""
        viewer = BlochViewer3D()

        # Create a trajectory (rotation around Z)
        angles = np.linspace(0, 2 * np.pi, 50)
        states = []
        for angle in angles:
            state = qt.Qobj(
                [[np.cos(angle / 2)], [np.exp(1j * angle) * np.sin(angle / 2)]]
            )
            states.append(state)

        fig, ax = viewer.plot_trajectory(states)

        assert fig is not None
        plt.close(fig)

    def test_plot_trajectory_custom_colormap(self):
        """Test trajectory with custom colormap."""
        viewer = BlochViewer3D()

        # Simple trajectory
        states = [
            qt.basis(2, 0),
            (qt.basis(2, 0) + qt.basis(2, 1)).unit(),
            qt.basis(2, 1),
        ]

        fig, ax = viewer.plot_trajectory(states, colormap="plasma")

        assert fig is not None
        plt.close(fig)

    def test_draw_bloch_sphere(self):
        """Test Bloch sphere drawing."""
        viewer = BlochViewer3D()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        viewer._draw_bloch_sphere(ax, alpha=0.2)

        assert ax is not None
        plt.close(fig)

    def test_empty_states_list(self):
        """Test with empty states list."""
        viewer = BlochViewer3D()

        # Should not crash
        fig, ax = viewer.plot_states([])

        assert fig is not None
        plt.close(fig)


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_dashboard_with_pulse_comparison(self):
        """Test dashboard alongside pulse comparison."""
        dashboard = OptimizationDashboard(n_controls=1, interactive=False)
        viewer = PulseComparisonViewer()

        # Simulate optimization
        pulse_history = []
        for i in range(10):
            controls = np.sin(np.linspace(0, 2 * np.pi * (i + 1) / 10, 50))
            fidelity = 0.5 + 0.05 * i

            dashboard.update(iteration=i, fidelity=fidelity, controls=controls)
            pulse_history.append(controls.copy())

        # Compare initial and final pulses
        fig = viewer.compare_pulses(
            [pulse_history[0], pulse_history[-1]], ["Initial", "Final"]
        )

        assert fig is not None
        dashboard.close()
        plt.close(fig)

    def test_parameter_sweep_with_bloch(self):
        """Test parameter sweep alongside Bloch visualization."""
        sweep_viewer = ParameterSweepViewer()
        bloch_viewer = BlochViewer3D()

        # Parameter sweep
        alphas = np.linspace(0, np.pi, 10)
        betas = np.linspace(0, np.pi, 10)
        fidelities = np.outer(np.sin(betas), np.cos(alphas))

        fig1, _ = sweep_viewer.plot_heatmap(alphas, betas, fidelities)

        # Bloch states at optimal parameters
        opt_state = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
        fig2, _ = bloch_viewer.plot_states([opt_state], labels=["Optimal"])

        assert fig1 is not None
        assert fig2 is not None
        plt.close(fig1)
        plt.close(fig2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
