"""
Interactive dashboards for quantum control optimization and analysis.

This module provides real-time visualization tools for monitoring optimization
progress, exploring parameter spaces, comparing pulse designs, and visualizing
quantum states on the Bloch sphere.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
import qutip as qt
from typing import Optional, List, Dict, Tuple, Callable, Any
from dataclasses import dataclass
import time
import warnings


@dataclass
class OptimizationMetrics:
    """Container for optimization metrics at each iteration."""

    iteration: int
    fidelity: float
    infidelity: float
    gradient_norm: float
    time_elapsed: float
    custom_metrics: Optional[Dict[str, float]] = None


class OptimizationDashboard:
    """
    Real-time dashboard for monitoring optimization progress.

    Features:
    - Live fidelity/infidelity plots
    - Gradient norm tracking
    - Control amplitude evolution
    - Custom metric plots

    Examples
    --------
    >>> dashboard = OptimizationDashboard(n_controls=2)
    >>> for iteration in range(100):
    ...     # Run optimization step
    ...     dashboard.update(iteration, fidelity=0.99, gradient_norm=0.01, controls=u)
    >>> dashboard.save("optimization_progress.png")
    """

    def __init__(
        self,
        n_controls: int = 1,
        figsize: Tuple[int, int] = (14, 10),
        interactive: bool = True,
    ):
        """
        Initialize optimization dashboard.

        Parameters
        ----------
        n_controls : int
            Number of control fields to visualize
        figsize : tuple
            Figure size (width, height)
        interactive : bool
            If True, enable interactive mode with live updates
        """
        self.n_controls = n_controls
        self.figsize = figsize
        self.interactive = interactive

        # Data storage
        self.iterations = []
        self.fidelities = []
        self.infidelities = []
        self.gradient_norms = []
        self.times = []
        self.controls_history = []
        self.custom_metrics = {}

        # Setup figure
        self.fig = None
        self.axes = {}
        self._setup_figure()

        if interactive:
            plt.ion()
            plt.show()

    def _setup_figure(self):
        """Setup the dashboard figure and axes."""
        self.fig = plt.figure(figsize=self.figsize)
        gs = GridSpec(3, 2, figure=self.fig, hspace=0.3, wspace=0.3)

        # Fidelity plot
        self.axes["fidelity"] = self.fig.add_subplot(gs[0, 0])
        self.axes["fidelity"].set_xlabel("Iteration")
        self.axes["fidelity"].set_ylabel("Fidelity")
        self.axes["fidelity"].set_title("Optimization Fidelity")
        self.axes["fidelity"].grid(True, alpha=0.3)

        # Infidelity (log scale)
        self.axes["infidelity"] = self.fig.add_subplot(gs[0, 1])
        self.axes["infidelity"].set_xlabel("Iteration")
        self.axes["infidelity"].set_ylabel("Infidelity (log)")
        self.axes["infidelity"].set_title("Infidelity Progress")
        self.axes["infidelity"].set_yscale("log")
        self.axes["infidelity"].grid(True, alpha=0.3)

        # Gradient norm
        self.axes["gradient"] = self.fig.add_subplot(gs[1, 0])
        self.axes["gradient"].set_xlabel("Iteration")
        self.axes["gradient"].set_ylabel("Gradient Norm")
        self.axes["gradient"].set_title("Gradient Norm")
        self.axes["gradient"].set_yscale("log")
        self.axes["gradient"].grid(True, alpha=0.3)

        # Time per iteration
        self.axes["time"] = self.fig.add_subplot(gs[1, 1])
        self.axes["time"].set_xlabel("Iteration")
        self.axes["time"].set_ylabel("Time (s)")
        self.axes["time"].set_title("Computation Time")
        self.axes["time"].grid(True, alpha=0.3)

        # Control amplitudes
        self.axes["controls"] = self.fig.add_subplot(gs[2, :])
        self.axes["controls"].set_xlabel("Time Step")
        self.axes["controls"].set_ylabel("Control Amplitude")
        self.axes["controls"].set_title("Control Fields Evolution")
        self.axes["controls"].grid(True, alpha=0.3)

    def update(
        self,
        iteration: int,
        fidelity: Optional[float] = None,
        infidelity: Optional[float] = None,
        gradient_norm: Optional[float] = None,
        time_elapsed: Optional[float] = None,
        controls: Optional[np.ndarray] = None,
        **custom_metrics,
    ):
        """
        Update dashboard with new optimization data.

        Parameters
        ----------
        iteration : int
            Current iteration number
        fidelity : float, optional
            Current fidelity value
        infidelity : float, optional
            Current infidelity (1 - fidelity)
        gradient_norm : float, optional
            Norm of gradient vector
        time_elapsed : float, optional
            Time elapsed for this iteration
        controls : ndarray, optional
            Current control amplitudes (n_controls, n_timesteps)
        **custom_metrics
            Additional metrics to track
        """
        self.iterations.append(iteration)

        if fidelity is not None:
            self.fidelities.append(fidelity)
        if infidelity is not None:
            self.infidelities.append(infidelity)
        elif fidelity is not None:
            self.infidelities.append(1 - fidelity)

        if gradient_norm is not None:
            self.gradient_norms.append(gradient_norm)
        if time_elapsed is not None:
            self.times.append(time_elapsed)
        if controls is not None:
            self.controls_history.append(controls.copy())

        # Store custom metrics
        for key, value in custom_metrics.items():
            if key not in self.custom_metrics:
                self.custom_metrics[key] = []
            self.custom_metrics[key].append(value)

        # Update plots
        self._update_plots()

        if self.interactive:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def _update_plots(self):
        """Update all plots with current data."""
        # Fidelity
        if self.fidelities:
            self.axes["fidelity"].clear()
            self.axes["fidelity"].plot(
                self.iterations, self.fidelities, "b-", linewidth=2
            )
            self.axes["fidelity"].set_xlabel("Iteration")
            self.axes["fidelity"].set_ylabel("Fidelity")
            self.axes["fidelity"].set_title("Optimization Fidelity")
            self.axes["fidelity"].grid(True, alpha=0.3)
            self.axes["fidelity"].set_ylim([0, 1.05])

        # Infidelity
        if self.infidelities:
            self.axes["infidelity"].clear()
            valid_infid = [max(1e-15, inf) for inf in self.infidelities]
            self.axes["infidelity"].semilogy(
                self.iterations, valid_infid, "r-", linewidth=2
            )
            self.axes["infidelity"].set_xlabel("Iteration")
            self.axes["infidelity"].set_ylabel("Infidelity (log)")
            self.axes["infidelity"].set_title("Infidelity Progress")
            self.axes["infidelity"].grid(True, alpha=0.3)

        # Gradient norm
        if self.gradient_norms:
            self.axes["gradient"].clear()
            self.axes["gradient"].semilogy(
                self.iterations, self.gradient_norms, "g-", linewidth=2
            )
            self.axes["gradient"].set_xlabel("Iteration")
            self.axes["gradient"].set_ylabel("Gradient Norm")
            self.axes["gradient"].set_title("Gradient Norm")
            self.axes["gradient"].grid(True, alpha=0.3)

        # Time
        if self.times:
            self.axes["time"].clear()
            self.axes["time"].plot(
                self.iterations[: len(self.times)], self.times, "m-", linewidth=2
            )
            self.axes["time"].set_xlabel("Iteration")
            self.axes["time"].set_ylabel("Time (s)")
            self.axes["time"].set_title("Computation Time")
            self.axes["time"].grid(True, alpha=0.3)

        # Controls
        if self.controls_history:
            self.axes["controls"].clear()
            latest_controls = self.controls_history[-1]
            if latest_controls.ndim == 1:
                latest_controls = latest_controls.reshape(1, -1)

            for i in range(min(self.n_controls, latest_controls.shape[0])):
                self.axes["controls"].plot(latest_controls[i], label=f"Control {i + 1}")

            self.axes["controls"].set_xlabel("Time Step")
            self.axes["controls"].set_ylabel("Control Amplitude")
            self.axes["controls"].set_title("Control Fields Evolution")
            self.axes["controls"].legend()
            self.axes["controls"].grid(True, alpha=0.3)

    def save(self, filename: str, dpi: int = 300):
        """Save dashboard to file."""
        self.fig.savefig(filename, dpi=dpi, bbox_inches="tight")

    def export_data(self) -> Dict[str, Any]:
        """Export all tracked data as dictionary."""
        return {
            "iterations": np.array(self.iterations),
            "fidelities": np.array(self.fidelities),
            "infidelities": np.array(self.infidelities),
            "gradient_norms": np.array(self.gradient_norms),
            "times": np.array(self.times),
            "controls_history": self.controls_history,
            "custom_metrics": self.custom_metrics,
        }

    def close(self):
        """Close the dashboard."""
        if self.interactive:
            plt.ioff()
        plt.close(self.fig)


class ParameterSweepViewer:
    """
    Interactive viewer for parameter sweep results.

    Visualizes how system performance varies across parameter spaces
    with heatmaps, contour plots, and cross-sections.

    Examples
    --------
    >>> viewer = ParameterSweepViewer()
    >>> alphas = np.linspace(0, 1, 50)
    >>> betas = np.linspace(0, 2, 50)
    >>> results = sweep_parameters(alphas, betas)
    >>> viewer.plot_heatmap(alphas, betas, results, 'Alpha', 'Beta', 'Fidelity')
    """

    def __init__(self, figsize: Tuple[int, int] = (12, 4)):
        """
        Initialize parameter sweep viewer.

        Parameters
        ----------
        figsize : tuple
            Figure size (width, height)
        """
        self.figsize = figsize

    def _plot_heatmap_panel(
        self,
        ax: plt.Axes,
        x_values: np.ndarray,
        y_values: np.ndarray,
        z_values: np.ndarray,
        x_label: str,
        y_label: str,
        z_label: str,
        title: str,
        cmap: str,
        vmin: Optional[float],
        vmax: Optional[float],
    ) -> None:
        """
        Plot heatmap panel.

        Parameters
        ----------
        ax : Axes
            Matplotlib axes
        x_values, y_values : ndarray
            Parameter values
        z_values : ndarray
            Metric values (2D)
        x_label, y_label, z_label : str
            Axis labels
        title : str
            Plot title
        cmap : str
            Colormap
        vmin, vmax : float, optional
            Color limits
        """
        im = ax.imshow(
            z_values,
            extent=[x_values[0], x_values[-1], y_values[0], y_values[-1]],
            origin="lower",
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f"{title} - Heatmap")
        plt.colorbar(im, ax=ax, label=z_label)

    def _plot_contour_panel(
        self,
        ax: plt.Axes,
        x_values: np.ndarray,
        y_values: np.ndarray,
        z_values: np.ndarray,
        x_label: str,
        y_label: str,
        title: str,
        cmap: str,
    ) -> None:
        """
        Plot contour panel.

        Parameters
        ----------
        ax : Axes
            Matplotlib axes
        x_values, y_values : ndarray
            Parameter values
        z_values : ndarray
            Metric values (2D)
        x_label, y_label : str
            Axis labels
        title : str
            Plot title
        cmap : str
            Colormap
        """
        X, Y = np.meshgrid(x_values, y_values)
        cs = ax.contour(X, Y, z_values, levels=10, cmap=cmap)
        ax.clabel(cs, inline=True, fontsize=8)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f"{title} - Contours")

    def _plot_crosssection_panel(
        self,
        ax: plt.Axes,
        x_values: np.ndarray,
        y_values: np.ndarray,
        z_values: np.ndarray,
        x_label: str,
        y_label: str,
        z_label: str,
        title: str,
    ) -> None:
        """
        Plot cross-section panel.

        Parameters
        ----------
        ax : Axes
            Matplotlib axes
        x_values, y_values : ndarray
            Parameter values
        z_values : ndarray
            Metric values (2D)
        x_label, y_label, z_label : str
            Axis labels
        title : str
            Plot title
        """
        mid_x = len(x_values) // 2
        mid_y = len(y_values) // 2

        ax.plot(
            x_values, z_values[mid_y, :], "b-", label=f"{y_label}={y_values[mid_y]:.2f}"
        )
        ax.plot(
            y_values, z_values[:, mid_x], "r-", label=f"{x_label}={x_values[mid_x]:.2f}"
        )
        ax.set_xlabel("Parameter Value")
        ax.set_ylabel(z_label)
        ax.set_title(f"{title} - Cross-sections")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def plot_heatmap(
        self,
        x_values: np.ndarray,
        y_values: np.ndarray,
        z_values: np.ndarray,
        x_label: str = "Parameter 1",
        y_label: str = "Parameter 2",
        z_label: str = "Metric",
        title: str = "Parameter Sweep",
        cmap: str = "viridis",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> Tuple[plt.Figure, np.ndarray]:
        """
        Create heatmap and contour plots of parameter sweep.

        Parameters
        ----------
        x_values : ndarray
            First parameter values (1D)
        y_values : ndarray
            Second parameter values (1D)
        z_values : ndarray
            Metric values (2D: len(y_values) x len(x_values))
        x_label, y_label, z_label : str
            Axis labels
        title : str
            Plot title
        cmap : str
            Colormap name
        vmin, vmax : float, optional
            Color scale limits

        Returns
        -------
        fig : Figure
        axes : array of Axes
        """
        fig, axes = plt.subplots(1, 3, figsize=self.figsize)

        # Heatmap panel
        self._plot_heatmap_panel(
            axes[0],
            x_values,
            y_values,
            z_values,
            x_label,
            y_label,
            z_label,
            title,
            cmap,
            vmin,
            vmax,
        )

        # Contour panel
        self._plot_contour_panel(
            axes[1], x_values, y_values, z_values, x_label, y_label, title, cmap
        )

        # Cross-section panel
        self._plot_crosssection_panel(
            axes[2], x_values, y_values, z_values, x_label, y_label, z_label, title
        )

        plt.tight_layout()
        return fig, axes

    def plot_3d_surface(
        self,
        x_values: np.ndarray,
        y_values: np.ndarray,
        z_values: np.ndarray,
        x_label: str = "Parameter 1",
        y_label: str = "Parameter 2",
        z_label: str = "Metric",
        title: str = "Parameter Sweep",
        cmap: str = "viridis",
    ) -> Tuple[plt.Figure, Axes3D]:
        """
        Create 3D surface plot of parameter sweep.

        Parameters
        ----------
        x_values, y_values, z_values : ndarray
            Parameter and metric values
        x_label, y_label, z_label : str
            Axis labels
        title : str
            Plot title
        cmap : str
            Colormap name

        Returns
        -------
        fig : Figure
        ax : Axes3D
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        X, Y = np.meshgrid(x_values, y_values)
        surf = ax.plot_surface(X, Y, z_values, cmap=cmap, alpha=0.8)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_zlabel(z_label)
        ax.set_title(title)
        plt.colorbar(surf, ax=ax, shrink=0.5, label=z_label)

        return fig, ax


class PulseComparisonViewer:
    """
    Interactive viewer for comparing multiple pulse designs.

    Features:
    - Side-by-side pulse shape comparison
    - Spectrum analysis
    - Performance metrics
    - State trajectory comparison

    Examples
    --------
    >>> viewer = PulseComparisonViewer()
    >>> pulses = [pulse1, pulse2, pulse3]
    >>> labels = ['GRAPE', 'Krotov', 'DRAG']
    >>> viewer.compare_pulses(pulses, labels)
    """

    def __init__(self, figsize: Tuple[int, int] = (14, 10)):
        """Initialize pulse comparison viewer."""
        self.figsize = figsize

    def _setup_comparison_figure(
        self, metrics: Optional[Dict[str, List[float]]]
    ) -> Tuple:
        """Setup figure and axes for pulse comparison."""
        if metrics:
            fig = plt.figure(figsize=self.figsize)
            gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
            ax_pulses = fig.add_subplot(gs[0, :])
            ax_spectrum = fig.add_subplot(gs[1, :])
            ax_metrics = fig.add_subplot(gs[2, :])
        else:
            fig, (ax_pulses, ax_spectrum) = plt.subplots(2, 1, figsize=self.figsize)
            ax_metrics = None
        return fig, ax_pulses, ax_spectrum, ax_metrics

    def _plot_pulse_shapes(
        self,
        ax: plt.Axes,
        pulses: List[np.ndarray],
        labels: List[str],
        times: Optional[np.ndarray],
        colors,
    ) -> None:
        """Plot pulse shapes on given axes."""
        for i, (pulse, label) in enumerate(zip(pulses, labels)):
            times_plot = (
                times
                if times is not None and len(pulse) == len(times)
                else np.arange(len(pulse))
            )
            ax.plot(times_plot, pulse, label=label, color=colors[i], linewidth=2)

        ax.set_xlabel("Time")
        ax.set_ylabel("Amplitude")
        ax.set_title("Pulse Shapes Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_pulse_spectra(
        self,
        ax: plt.Axes,
        pulses: List[np.ndarray],
        labels: List[str],
        times: Optional[np.ndarray],
        colors,
    ) -> None:
        """Plot pulse spectra on given axes."""
        for i, (pulse, label) in enumerate(zip(pulses, labels)):
            spectrum = np.fft.fft(pulse)
            dt = (
                times[1] - times[0]
                if times is not None and len(pulse) == len(times) and len(times) > 1
                else 1.0
            )
            freqs = np.fft.fftfreq(len(pulse), dt)
            pos_mask = freqs >= 0
            ax.semilogy(
                freqs[pos_mask],
                np.abs(spectrum[pos_mask]),
                label=label,
                color=colors[i],
                linewidth=2,
            )

        ax.set_xlabel("Frequency")
        ax.set_ylabel("Magnitude (log)")
        ax.set_title("Pulse Spectra Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_comparison_metrics(
        self,
        ax: plt.Axes,
        metrics: Dict[str, List[float]],
        labels: List[str],
        colors,
    ) -> None:
        """Plot performance metrics bar chart."""
        n_pulses = len(labels)
        metric_names = list(metrics.keys())
        x_pos = np.arange(len(metric_names))
        width = 0.8 / n_pulses

        for i, label in enumerate(labels):
            values = [metrics[name][i] for name in metric_names]
            ax.bar(x_pos + i * width, values, width, label=label, color=colors[i])

        ax.set_xlabel("Metric")
        ax.set_ylabel("Value")
        ax.set_title("Performance Metrics")
        ax.set_xticks(x_pos + width * (n_pulses - 1) / 2)
        ax.set_xticklabels(metric_names, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    def compare_pulses(
        self,
        pulses: List[np.ndarray],
        labels: List[str],
        times: Optional[np.ndarray] = None,
        metrics: Optional[Dict[str, List[float]]] = None,
    ) -> plt.Figure:
        """Compare multiple pulse designs with shapes, spectra, and metrics."""
        n_pulses = len(pulses)
        colors = plt.cm.tab10(np.linspace(0, 1, n_pulses))

        # Setup figure
        fig, ax_pulses, ax_spectrum, ax_metrics = self._setup_comparison_figure(metrics)

        # Plot pulse shapes and spectra
        self._plot_pulse_shapes(ax_pulses, pulses, labels, times, colors)
        self._plot_pulse_spectra(ax_spectrum, pulses, labels, times, colors)

        # Plot metrics if provided
        if metrics and ax_metrics:
            self._plot_comparison_metrics(ax_metrics, metrics, labels, colors)

        # Suppress tight_layout warning
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*tight_layout.*", category=UserWarning
            )
            plt.tight_layout()
        return fig


class BlochViewer3D:
    """
    Interactive 3D Bloch sphere viewer for quantum state visualization.

    Features:
    - 3D Bloch sphere rendering
    - Multiple state vectors
    - Trajectory plotting
    - Interactive rotation

    Examples
    --------
    >>> viewer = BlochViewer3D()
    >>> states = [qt.basis(2, 0), (qt.basis(2, 0) + qt.basis(2, 1)).unit()]
    >>> viewer.plot_states(states, labels=['|0>', '|+>'])
    """

    def __init__(self, figsize: Tuple[int, int] = (8, 8)):
        """Initialize 3D Bloch sphere viewer."""
        self.figsize = figsize

    def _plot_state_vectors(
        self,
        ax: Axes3D,
        states: List[qt.Qobj],
        labels: Optional[List[str]],
    ) -> None:
        """
        Plot state vectors on Bloch sphere.

        Parameters
        ----------
        ax : Axes3D
            3D axes
        states : list of Qobj
            Quantum states to plot
        labels : list of str, optional
            Labels for each state
        """
        colors = plt.cm.tab10(np.linspace(0, 1, len(states)))

        for i, state in enumerate(states):
            bloch_vec = self._state_to_bloch(state)

            # Plot vector arrow
            ax.quiver(
                0,
                0,
                0,
                bloch_vec[0],
                bloch_vec[1],
                bloch_vec[2],
                color=colors[i],
                arrow_length_ratio=0.15,
                linewidth=2.5,
            )

            # Plot point at tip
            ax.scatter(
                bloch_vec[0],
                bloch_vec[1],
                bloch_vec[2],
                color=colors[i],
                s=100,
                label=labels[i] if labels else f"State {i + 1}",
            )

    def _configure_bloch_axes(self, ax: Axes3D, has_states: bool) -> None:
        """
        Configure 3D axes for Bloch sphere.

        Parameters
        ----------
        ax : Axes3D
            3D axes
        has_states : bool
            Whether states were plotted
        """
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Bloch Sphere")

        if has_states:
            ax.legend()

        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlim([-1.1, 1.1])
        ax.set_ylim([-1.1, 1.1])
        ax.set_zlim([-1.1, 1.1])

    def plot_states(
        self,
        states: List[qt.Qobj],
        labels: Optional[List[str]] = None,
        show_sphere: bool = True,
        alpha_sphere: float = 0.1,
    ) -> Tuple[plt.Figure, Axes3D]:
        """
        Plot quantum states on Bloch sphere.

        Parameters
        ----------
        states : list of Qobj
            Quantum states to plot
        labels : list of str, optional
            Labels for each state
        show_sphere : bool
            Whether to draw the Bloch sphere
        alpha_sphere : float
            Transparency of sphere surface

        Returns
        -------
        fig : Figure
        ax : Axes3D
        """
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection="3d")

        if show_sphere:
            self._draw_bloch_sphere(ax, alpha=alpha_sphere)

        # Plot state vectors
        self._plot_state_vectors(ax, states, labels)

        # Configure axes
        self._configure_bloch_axes(ax, len(states) > 0)

        return fig, ax

    def _plot_colored_trajectory(
        self,
        ax: Axes3D,
        bloch_vecs: np.ndarray,
        colormap: str,
    ) -> None:
        """
        Plot trajectory with color gradient.

        Parameters
        ----------
        ax : Axes3D
            3D axes
        bloch_vecs : np.ndarray
            Bloch vectors (N x 3)
        colormap : str
            Colormap name
        """
        n_points = len(bloch_vecs)
        cmap = plt.colormaps.get_cmap(colormap)
        colors = cmap(np.linspace(0, 1, n_points))

        for i in range(n_points - 1):
            ax.plot(
                bloch_vecs[i : i + 2, 0],
                bloch_vecs[i : i + 2, 1],
                bloch_vecs[i : i + 2, 2],
                color=colors[i],
                linewidth=2,
            )

    def _mark_trajectory_endpoints(
        self,
        ax: Axes3D,
        bloch_vecs: np.ndarray,
    ) -> None:
        """
        Mark start and end points of trajectory.

        Parameters
        ----------
        ax : Axes3D
            3D axes
        bloch_vecs : np.ndarray
            Bloch vectors (N x 3)
        """
        # Mark start
        ax.scatter(
            bloch_vecs[0, 0],
            bloch_vecs[0, 1],
            bloch_vecs[0, 2],
            color="green",
            s=200,
            marker="o",
            label="Start",
            edgecolors="black",
        )
        # Mark end
        ax.scatter(
            bloch_vecs[-1, 0],
            bloch_vecs[-1, 1],
            bloch_vecs[-1, 2],
            color="red",
            s=200,
            marker="s",
            label="End",
            edgecolors="black",
        )

    def _configure_trajectory_axes(self, ax: Axes3D) -> None:
        """
        Configure 3D axes for trajectory plot.

        Parameters
        ----------
        ax : Axes3D
            3D axes
        """
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("State Evolution on Bloch Sphere")
        ax.legend()

        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlim([-1.1, 1.1])
        ax.set_ylim([-1.1, 1.1])
        ax.set_zlim([-1.1, 1.1])

    def plot_trajectory(
        self,
        states: List[qt.Qobj],
        show_sphere: bool = True,
        alpha_sphere: float = 0.1,
        colormap: str = "viridis",
    ) -> Tuple[plt.Figure, Axes3D]:
        """
        Plot state evolution trajectory on Bloch sphere.

        Parameters
        ----------
        states : list of Qobj
            Sequence of quantum states
        show_sphere : bool
            Whether to draw the Bloch sphere
        alpha_sphere : float
            Transparency of sphere surface
        colormap : str
            Colormap for trajectory

        Returns
        -------
        fig : Figure
        ax : Axes3D
        """
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection="3d")

        if show_sphere:
            self._draw_bloch_sphere(ax, alpha=alpha_sphere)

        # Convert states to Bloch vectors
        bloch_vecs = np.array([self._state_to_bloch(state) for state in states])

        # Plot trajectory
        self._plot_colored_trajectory(ax, bloch_vecs, colormap)

        # Mark endpoints
        self._mark_trajectory_endpoints(ax, bloch_vecs)

        # Configure axes
        self._configure_trajectory_axes(ax)

        return fig, ax

    def _draw_bloch_sphere(self, ax: Axes3D, alpha: float = 0.1):
        """Draw the Bloch sphere surface and axes."""
        # Sphere surface
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))

        ax.plot_surface(x, y, z, color="gray", alpha=alpha, linewidth=0)

        # Equator and meridians
        theta = np.linspace(0, 2 * np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), 0, "k-", alpha=0.3, linewidth=0.5)
        ax.plot(np.cos(theta), 0, np.sin(theta), "k-", alpha=0.3, linewidth=0.5)
        ax.plot(0, np.cos(theta), np.sin(theta), "k-", alpha=0.3, linewidth=0.5)

        # Axes
        ax.plot([0, 1.3], [0, 0], [0, 0], "k-", linewidth=1)
        ax.plot([0, 0], [0, 1.3], [0, 0], "k-", linewidth=1)
        ax.plot([0, 0], [0, 0], [0, 1.3], "k-", linewidth=1)

        # Labels at axes ends
        ax.text(1.4, 0, 0, "X", fontsize=12)
        ax.text(0, 1.4, 0, "Y", fontsize=12)
        ax.text(0, 0, 1.4, "Z", fontsize=12)

    def _state_to_bloch(self, state: qt.Qobj) -> np.ndarray:
        """Convert quantum state to Bloch vector coordinates."""
        if state.type != "ket":
            state = state.dag()

        # Pauli matrices
        sx = qt.sigmax()
        sy = qt.sigmay()
        sz = qt.sigmaz()

        # Compute expectation values
        x = qt.expect(sx, state)
        y = qt.expect(sy, state)
        z = qt.expect(sz, state)

        return np.array([x, y, z])
