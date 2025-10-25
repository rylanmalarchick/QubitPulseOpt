"""
Analysis and reporting tools for quantum control optimization.

This module provides tools for generating comprehensive reports on pulse
characteristics, optimization results, and publication-quality figures with
LaTeX table export capabilities.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import qutip as qt
from typing import Optional, List, Dict, Tuple, Any, Union
from dataclasses import dataclass, asdict
import json
from datetime import datetime
import warnings


@dataclass
class PulseCharacteristics:
    """Container for pulse characterization metrics."""

    duration: float
    peak_amplitude: float
    rms_amplitude: float
    energy: float
    bandwidth: float
    smoothness: float  # Second derivative metric
    fidelity: float
    robustness_metric: Optional[float] = None
    custom_metrics: Optional[Dict[str, float]] = None


class PulseReport:
    """
    Comprehensive pulse characterization and reporting.

    Generates detailed reports including:
    - Time-domain analysis
    - Frequency-domain analysis
    - Performance metrics
    - Robustness analysis
    - Comparison with other pulses

    Examples
    --------
    >>> report = PulseReport(pulse, times, fidelity=0.995)
    >>> report.add_comparison(other_pulse, "DRAG")
    >>> report.generate_full_report("pulse_analysis.png")
    >>> report.export_metrics_table("metrics.tex")
    """

    def __init__(
        self,
        pulse: np.ndarray,
        times: Optional[np.ndarray] = None,
        fidelity: Optional[float] = None,
        target_gate: Optional[str] = None,
        optimization_method: Optional[str] = None,
        label: str = "Main Pulse",
    ):
        """
        Initialize pulse report.

        Parameters
        ----------
        pulse : ndarray
            Control pulse amplitudes
        times : ndarray, optional
            Time array (if None, uses indices)
        fidelity : float, optional
            Gate fidelity achieved by this pulse
        target_gate : str, optional
            Name of target gate (e.g., 'X', 'Hadamard')
        optimization_method : str, optional
            Optimization method used (e.g., 'GRAPE', 'Krotov')
        label : str
            Label for this pulse
        """
        self.pulse = pulse
        self.times = times if times is not None else np.arange(len(pulse))
        self.fidelity = fidelity
        self.target_gate = target_gate
        self.optimization_method = optimization_method
        self.label = label

        # Storage for comparisons
        self.comparison_pulses = []
        self.comparison_labels = []
        self.comparison_metrics = []

        # Compute characteristics
        self.characteristics = self._compute_characteristics()

    def _compute_characteristics(self) -> PulseCharacteristics:
        """Compute comprehensive pulse characteristics."""
        dt = self.times[1] - self.times[0] if len(self.times) > 1 else 1.0
        duration = self.times[-1] - self.times[0]

        # Amplitude metrics
        peak_amplitude = np.max(np.abs(self.pulse))
        rms_amplitude = np.sqrt(np.mean(self.pulse**2))
        energy = np.sum(self.pulse**2) * dt

        # Frequency analysis
        spectrum = np.fft.fft(self.pulse)
        freqs = np.fft.fftfreq(len(self.pulse), dt)
        power_spectrum = np.abs(spectrum) ** 2

        # Bandwidth (frequency containing 90% of power)
        cumsum = np.cumsum(power_spectrum[: len(power_spectrum) // 2])
        cumsum /= cumsum[-1]
        idx_90 = np.where(cumsum >= 0.9)[0][0]
        bandwidth = freqs[idx_90]

        # Smoothness (average second derivative)
        if len(self.pulse) > 2:
            second_deriv = np.diff(self.pulse, n=2)
            smoothness = np.mean(np.abs(second_deriv))
        else:
            smoothness = 0.0

        return PulseCharacteristics(
            duration=float(duration),
            peak_amplitude=float(peak_amplitude),
            rms_amplitude=float(rms_amplitude),
            energy=float(energy),
            bandwidth=float(bandwidth),
            smoothness=float(smoothness),
            fidelity=float(self.fidelity) if self.fidelity is not None else np.nan,
        )

    def add_comparison(
        self, pulse: np.ndarray, label: str, metrics: Optional[Dict[str, float]] = None
    ):
        """
        Add a pulse for comparison.

        Parameters
        ----------
        pulse : ndarray
            Comparison pulse amplitudes
        label : str
            Label for comparison pulse
        metrics : dict, optional
            Additional metrics for this pulse
        """
        self.comparison_pulses.append(pulse)
        self.comparison_labels.append(label)
        self.comparison_metrics.append(metrics or {})

    def generate_full_report(
        self, filename: Optional[str] = None, figsize: Tuple[int, int] = (14, 10)
    ) -> plt.Figure:
        """
        Generate comprehensive visual report.

        Parameters
        ----------
        filename : str, optional
            If provided, save figure to this file
        figsize : tuple
            Figure size (width, height)

        Returns
        -------
        fig : Figure
        """
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

        # Time domain
        ax_time = fig.add_subplot(gs[0, :])
        self._plot_time_domain(ax_time)

        # Frequency domain
        ax_freq = fig.add_subplot(gs[1, 0])
        self._plot_frequency_domain(ax_freq)

        # Metrics table
        ax_metrics = fig.add_subplot(gs[1, 1])
        self._plot_metrics_table(ax_metrics)

        # Comparison (if available)
        if self.comparison_pulses:
            ax_comparison = fig.add_subplot(gs[2, :])
            self._plot_comparison(ax_comparison)

        # Title
        title = f"Pulse Analysis Report: {self.label}"
        if self.target_gate:
            title += f" (Target: {self.target_gate})"
        if self.optimization_method:
            title += f" [{self.optimization_method}]"
        fig.suptitle(title, fontsize=16, fontweight="bold")

        # Suppress tight_layout warning for axes that might not be compatible
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*tight_layout.*", category=UserWarning
            )
            plt.tight_layout()

        if filename:
            fig.savefig(filename, dpi=300, bbox_inches="tight")

        return fig

    def _plot_time_domain(self, ax: plt.Axes):
        """Plot time-domain pulse shape."""
        ax.plot(self.times, self.pulse, "b-", linewidth=2, label=self.label)

        # Add comparisons
        for pulse, label in zip(self.comparison_pulses, self.comparison_labels):
            times_comp = (
                self.times
                if len(pulse) == len(self.times)
                else np.linspace(self.times[0], self.times[-1], len(pulse))
            )
            ax.plot(times_comp, pulse, "--", linewidth=1.5, alpha=0.7, label=label)

        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Amplitude", fontsize=12)
        ax.set_title("Pulse Shape", fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend()

    def _plot_frequency_domain(self, ax: plt.Axes):
        """Plot frequency-domain spectrum."""
        dt = self.times[1] - self.times[0] if len(self.times) > 1 else 1.0

        spectrum = np.fft.fft(self.pulse)
        freqs = np.fft.fftfreq(len(self.pulse), dt)
        power = np.abs(spectrum) ** 2

        # Plot positive frequencies only
        pos_mask = freqs >= 0
        ax.semilogy(freqs[pos_mask], power[pos_mask], "b-", linewidth=2)

        ax.set_xlabel("Frequency", fontsize=12)
        ax.set_ylabel("Power (log scale)", fontsize=12)
        ax.set_title("Power Spectrum", fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3)

    def _plot_metrics_table(self, ax: plt.Axes):
        """Plot metrics as a table."""
        ax.axis("off")

        # Prepare data
        metrics_data = [
            ["Metric", "Value"],
            ["Duration", f"{self.characteristics.duration:.3f}"],
            ["Peak Amplitude", f"{self.characteristics.peak_amplitude:.4f}"],
            ["RMS Amplitude", f"{self.characteristics.rms_amplitude:.4f}"],
            ["Energy", f"{self.characteristics.energy:.4f}"],
            ["Bandwidth", f"{self.characteristics.bandwidth:.4f}"],
            ["Smoothness", f"{self.characteristics.smoothness:.4e}"],
        ]

        if not np.isnan(self.characteristics.fidelity):
            metrics_data.append(["Fidelity", f"{self.characteristics.fidelity:.6f}"])

        # Create table
        table = ax.table(
            cellText=metrics_data,
            cellLoc="left",
            loc="center",
            colWidths=[0.5, 0.5],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Style header row
        for i in range(2):
            table[(0, i)].set_facecolor("#4CAF50")
            table[(0, i)].set_text_props(weight="bold", color="white")

        ax.set_title("Pulse Metrics", fontsize=13, fontweight="bold", pad=20)

    def _plot_comparison(self, ax: plt.Axes):
        """Plot pulse comparison."""
        n_pulses = 1 + len(self.comparison_pulses)
        all_pulses = [self.pulse] + self.comparison_pulses
        all_labels = [self.label] + self.comparison_labels

        # Normalize to same time base for comparison
        max_len = max(len(p) for p in all_pulses)

        for i, (pulse, label) in enumerate(zip(all_pulses, all_labels)):
            if len(pulse) != max_len:
                times_plot = np.linspace(0, 1, len(pulse))
            else:
                times_plot = np.linspace(0, 1, len(pulse))
            ax.plot(times_plot, pulse, linewidth=2, label=label, alpha=0.8)

        ax.set_xlabel("Normalized Time", fontsize=12)
        ax.set_ylabel("Amplitude", fontsize=12)
        ax.set_title("Pulse Comparison", fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend()

    def export_metrics_table(
        self, filename: str, format: str = "latex"
    ) -> Optional[str]:
        """
        Export metrics as a formatted table.

        Parameters
        ----------
        filename : str
            Output filename
        format : str
            Output format ('latex', 'csv', 'json')

        Returns
        -------
        table_str : str
            Table content (also saved to file)
        """
        if format == "latex":
            return self._export_latex_table(filename)
        elif format == "csv":
            return self._export_csv_table(filename)
        elif format == "json":
            return self._export_json_table(filename)
        else:
            raise ValueError(f"Unknown format: {format}")

    def _export_latex_table(self, filename: str) -> str:
        """Export metrics as LaTeX table."""
        lines = []
        lines.append("\\begin{table}[h]")
        lines.append("\\centering")
        lines.append("\\begin{tabular}{|l|r|}")
        lines.append("\\hline")
        lines.append("\\textbf{Metric} & \\textbf{Value} \\\\")
        lines.append("\\hline")

        chars = self.characteristics
        lines.append(f"Duration & {chars.duration:.3f} \\\\")
        lines.append(f"Peak Amplitude & {chars.peak_amplitude:.4f} \\\\")
        lines.append(f"RMS Amplitude & {chars.rms_amplitude:.4f} \\\\")
        lines.append(f"Energy & {chars.energy:.4f} \\\\")
        lines.append(f"Bandwidth & {chars.bandwidth:.4f} \\\\")
        lines.append(f"Smoothness & {chars.smoothness:.4e} \\\\")

        if not np.isnan(chars.fidelity):
            lines.append(f"Fidelity & {chars.fidelity:.6f} \\\\")

        lines.append("\\hline")
        lines.append("\\end{tabular}")

        caption = f"Pulse characteristics for {self.label}"
        if self.target_gate:
            caption += f" (Target: {self.target_gate})"
        lines.append(f"\\caption{{{caption}}}")
        lines.append(f"\\label{{tab:pulse_{self.label.replace(' ', '_').lower()}}}")
        lines.append("\\end{table}")

        table_str = "\n".join(lines)

        with open(filename, "w") as f:
            f.write(table_str)

        return table_str

    def _export_csv_table(self, filename: str) -> str:
        """Export metrics as CSV."""
        lines = ["Metric,Value"]
        chars = self.characteristics

        lines.append(f"Duration,{chars.duration}")
        lines.append(f"Peak Amplitude,{chars.peak_amplitude}")
        lines.append(f"RMS Amplitude,{chars.rms_amplitude}")
        lines.append(f"Energy,{chars.energy}")
        lines.append(f"Bandwidth,{chars.bandwidth}")
        lines.append(f"Smoothness,{chars.smoothness}")

        if not np.isnan(chars.fidelity):
            lines.append(f"Fidelity,{chars.fidelity}")

        csv_str = "\n".join(lines)

        with open(filename, "w") as f:
            f.write(csv_str)

        return csv_str

    def _export_json_table(self, filename: str) -> str:
        """Export metrics as JSON."""
        # Convert characteristics to dict and ensure all values are JSON-serializable
        chars_dict = asdict(self.characteristics)
        # Convert any numpy types to native Python types
        for key, value in chars_dict.items():
            if isinstance(value, (np.integer, np.floating)):
                chars_dict[key] = float(value)
            elif isinstance(value, np.ndarray):
                chars_dict[key] = value.tolist()

        data = {
            "pulse_label": self.label,
            "target_gate": self.target_gate,
            "optimization_method": self.optimization_method,
            "characteristics": chars_dict,
            "timestamp": datetime.now().isoformat(),
        }

        json_str = json.dumps(data, indent=2)

        with open(filename, "w") as f:
            f.write(json_str)

        return json_str


class OptimizationReport:
    """
    Comprehensive optimization process report.

    Tracks and visualizes the entire optimization process including
    convergence, parameter evolution, and final performance.

    Examples
    --------
    >>> report = OptimizationReport()
    >>> for iter in range(100):
    ...     report.add_iteration(iter, fidelity=f, gradient_norm=g)
    >>> report.generate_summary("optimization_summary.png")
    """

    def __init__(self, method: str = "Unknown", target: Optional[str] = None):
        """
        Initialize optimization report.

        Parameters
        ----------
        method : str
            Optimization method name
        target : str, optional
            Target gate or operation
        """
        self.method = method
        self.target = target

        # Data storage
        self.iterations = []
        self.fidelities = []
        self.infidelities = []
        self.gradient_norms = []
        self.custom_metrics = {}

        self.start_time = datetime.now()
        self.end_time = None

    def add_iteration(
        self,
        iteration: int,
        fidelity: Optional[float] = None,
        gradient_norm: Optional[float] = None,
        **custom_metrics,
    ):
        """Add data from an optimization iteration."""
        self.iterations.append(iteration)

        if fidelity is not None:
            self.fidelities.append(fidelity)
            self.infidelities.append(1 - fidelity)

        if gradient_norm is not None:
            self.gradient_norms.append(gradient_norm)

        for key, value in custom_metrics.items():
            if key not in self.custom_metrics:
                self.custom_metrics[key] = []
            self.custom_metrics[key].append(value)

    def finalize(self):
        """Mark optimization as complete."""
        self.end_time = datetime.now()

    def _plot_fidelity_convergence(self, ax: plt.Axes) -> None:
        """Plot fidelity convergence over iterations."""
        if self.fidelities:
            ax.plot(
                self.iterations[: len(self.fidelities)],
                self.fidelities,
                "b-",
                linewidth=2,
            )
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Fidelity")
            ax.set_title("Fidelity Convergence")
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1.05])

    def _plot_infidelity_progress(self, ax: plt.Axes) -> None:
        """Plot infidelity on log scale over iterations."""
        if self.infidelities:
            valid_infid = [max(1e-15, inf) for inf in self.infidelities]
            ax.semilogy(
                self.iterations[: len(valid_infid)], valid_infid, "r-", linewidth=2
            )
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Infidelity (log)")
            ax.set_title("Infidelity Progress")
            ax.grid(True, alpha=0.3)

    def _plot_gradient_convergence(self, ax: plt.Axes) -> None:
        """Plot gradient norm convergence on log scale."""
        if self.gradient_norms:
            ax.semilogy(
                self.iterations[: len(self.gradient_norms)],
                self.gradient_norms,
                "g-",
                linewidth=2,
            )
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Gradient Norm (log)")
            ax.set_title("Gradient Convergence")
            ax.grid(True, alpha=0.3)

    def _create_summary_table(self, ax: plt.Axes) -> None:
        """Create summary statistics table."""
        ax.axis("off")

        summary_data = [["Metric", "Value"], ["Method", self.method]]

        if self.target:
            summary_data.append(["Target", self.target])
        if self.fidelities:
            summary_data.append(["Final Fidelity", f"{self.fidelities[-1]:.6f}"])
            summary_data.append(["Initial Fidelity", f"{self.fidelities[0]:.6f}"])

        summary_data.append(["Iterations", str(len(self.iterations))])

        if self.end_time:
            duration = (self.end_time - self.start_time).total_seconds()
            summary_data.append(["Duration (s)", f"{duration:.2f}"])

        table = ax.table(
            cellText=summary_data, cellLoc="left", loc="center", colWidths=[0.5, 0.5]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Style header
        for i in range(2):
            table[(0, i)].set_facecolor("#2196F3")
            table[(0, i)].set_text_props(weight="bold", color="white")

        ax.set_title("Optimization Summary", fontsize=13, fontweight="bold", pad=20)

    def generate_summary(
        self, filename: Optional[str] = None, figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """Generate optimization summary report with plots and statistics."""
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

        # Plot all subplots
        self._plot_fidelity_convergence(fig.add_subplot(gs[0, 0]))
        self._plot_infidelity_progress(fig.add_subplot(gs[0, 1]))
        self._plot_gradient_convergence(fig.add_subplot(gs[1, 0]))
        self._create_summary_table(fig.add_subplot(gs[1, 1]))

        # Main title
        title = f"Optimization Report: {self.method}"
        if self.target:
            title += f" (Target: {self.target})"
        fig.suptitle(title, fontsize=16, fontweight="bold")

        if filename:
            fig.savefig(filename, dpi=300, bbox_inches="tight")

        return fig


def generate_latex_table(
    data: Dict[str, List[Any]],
    filename: str,
    caption: str = "Results",
    label: str = "tab:results",
) -> str:
    """
    Generate a LaTeX table from data dictionary.

    Parameters
    ----------
    data : dict
        Dictionary mapping column names to data lists
    filename : str
        Output filename
    caption : str
        Table caption
    label : str
        LaTeX label for referencing

    Returns
    -------
    latex_str : str
        Generated LaTeX code

    Examples
    --------
    >>> data = {'Method': ['GRAPE', 'Krotov'], 'Fidelity': [0.99, 0.98]}
    >>> generate_latex_table(data, 'results.tex', 'Comparison')
    """
    columns = list(data.keys())
    n_cols = len(columns)
    n_rows = len(data[columns[0]])

    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append(f"\\begin{{tabular}}{{|{'|'.join(['c'] * n_cols)}|}}")
    lines.append("\\hline")

    # Header
    header = " & ".join(f"\\textbf{{{col}}}" for col in columns)
    lines.append(f"{header} \\\\")
    lines.append("\\hline")

    # Data rows
    for i in range(n_rows):
        row = []
        for col in columns:
            value = data[col][i]
            if isinstance(value, float):
                row.append(f"{value:.4f}")
            else:
                row.append(str(value))
        lines.append(" & ".join(row) + " \\\\")

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table}")

    latex_str = "\n".join(lines)

    with open(filename, "w") as f:
        f.write(latex_str)

    return latex_str


def _setup_publication_style(
    style: str, figsize: Tuple[int, int]
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Set up figure with publication style.

    Parameters
    ----------
    style : str
        Matplotlib style to use
    figsize : tuple
        Figure size (width, height)

    Returns
    -------
    fig : Figure
        Matplotlib figure
    ax : Axes
        Matplotlib axes
    """
    if style != "default":
        plt.style.use(style)
    return plt.subplots(figsize=figsize)


def _plot_publication_data(
    ax: plt.Axes,
    data: List[np.ndarray],
    labels: Optional[List[str]],
    **plot_kwargs,
) -> None:
    """
    Plot data series on axes.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes
    data : list of ndarray
        Data series to plot
    labels : list of str, optional
        Legend labels
    **plot_kwargs
        Additional plot arguments
    """
    for i, d in enumerate(data):
        label = labels[i] if labels and i < len(labels) else None
        ax.plot(d, label=label, linewidth=2, **plot_kwargs)


def _format_publication_axes(
    ax: plt.Axes,
    xlabel: str,
    ylabel: str,
    title: str,
    has_labels: bool,
) -> None:
    """
    Format axes for publication quality.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes
    xlabel, ylabel : str
        Axis labels
    title : str
        Figure title
    has_labels : bool
        Whether legend labels are present
    """
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    if title:
        ax.set_title(title, fontsize=16, fontweight="bold")

    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.3, linestyle="--")

    if has_labels:
        ax.legend(fontsize=12, framealpha=0.9)

    # Suppress tight_layout warning for axes that might not be compatible
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*tight_layout.*", category=UserWarning
        )
        plt.tight_layout()


def _save_publication_figure(fig: plt.Figure, filename: str) -> None:
    """
    Save figure with publication-quality settings.

    Parameters
    ----------
    fig : Figure
        Matplotlib figure
    filename : str
        Output filename
    """
    # Use high DPI for publication
    dpi = 300 if filename.endswith((".png", ".jpg")) else None
    fig.savefig(filename, dpi=dpi, bbox_inches="tight")


def create_publication_figure(
    data: Union[np.ndarray, List[np.ndarray]],
    labels: Optional[List[str]] = None,
    xlabel: str = "X",
    ylabel: str = "Y",
    title: str = "",
    filename: Optional[str] = None,
    style: str = "default",
    figsize: Tuple[int, int] = (8, 6),
    **plot_kwargs,
) -> plt.Figure:
    """
    Create publication-quality figure with proper formatting.

    Parameters
    ----------
    data : ndarray or list of ndarray
        Data to plot
    labels : list of str, optional
        Legend labels
    xlabel, ylabel : str
        Axis labels
    title : str
        Figure title
    filename : str, optional
        Save to file if provided
    style : str
        Matplotlib style to use
    figsize : tuple
        Figure size
    **plot_kwargs
        Additional arguments passed to plot()

    Returns
    -------
    fig : Figure

    Examples
    --------
    >>> x = np.linspace(0, 10, 100)
    >>> y1 = np.sin(x)
    >>> y2 = np.cos(x)
    >>> fig = create_publication_figure([y1, y2], labels=['sin', 'cos'],
    ...                                   filename='trig.pdf')
    """
    # Setup figure with style
    fig, ax = _setup_publication_style(style, figsize)

    # Normalize data to list
    data_list = [data] if isinstance(data, np.ndarray) else data

    # Plot all data series
    _plot_publication_data(ax, data_list, labels, **plot_kwargs)

    # Format axes
    _format_publication_axes(ax, xlabel, ylabel, title, labels is not None)

    # Save if requested
    if filename:
        _save_publication_figure(fig, filename)

    return fig
