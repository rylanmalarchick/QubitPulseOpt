"""
Unit tests for visualization reports module.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
import json
from src.visualization.reports import (
    PulseReport,
    OptimizationReport,
    PulseCharacteristics,
    generate_latex_table,
    create_publication_figure,
)


class TestPulseCharacteristics:
    """Tests for PulseCharacteristics dataclass."""

    def test_creation(self):
        """Test creating PulseCharacteristics."""
        chars = PulseCharacteristics(
            duration=10.0,
            peak_amplitude=1.5,
            rms_amplitude=0.8,
            energy=5.0,
            bandwidth=2.0,
            smoothness=0.01,
            fidelity=0.99,
        )
        assert chars.duration == 10.0
        assert chars.fidelity == 0.99
        assert chars.robustness_metric is None

    def test_with_custom_metrics(self):
        """Test with custom metrics."""
        custom = {"gate_time": 5.0, "error_rate": 0.001}
        chars = PulseCharacteristics(
            duration=10.0,
            peak_amplitude=1.0,
            rms_amplitude=0.5,
            energy=3.0,
            bandwidth=1.5,
            smoothness=0.005,
            fidelity=0.98,
            robustness_metric=0.95,
            custom_metrics=custom,
        )
        assert chars.robustness_metric == 0.95
        assert chars.custom_metrics["gate_time"] == 5.0


class TestPulseReport:
    """Tests for PulseReport class."""

    def test_initialization(self):
        """Test basic initialization."""
        pulse = np.sin(np.linspace(0, 2 * np.pi, 100))
        report = PulseReport(pulse, fidelity=0.99, label="Test Pulse")

        assert report.label == "Test Pulse"
        assert report.fidelity == 0.99
        assert report.characteristics is not None

    def test_with_custom_times(self):
        """Test initialization with custom time array."""
        pulse = np.random.randn(50)
        times = np.linspace(0, 10, 50)
        report = PulseReport(pulse, times=times)

        assert len(report.times) == 50
        assert report.times[0] == 0
        assert report.times[-1] == 10

    def test_with_gate_info(self):
        """Test with target gate and optimization method."""
        pulse = np.random.randn(80)
        report = PulseReport(
            pulse,
            target_gate="Hadamard",
            optimization_method="GRAPE",
            fidelity=0.995,
        )

        assert report.target_gate == "Hadamard"
        assert report.optimization_method == "GRAPE"

    def test_compute_characteristics(self):
        """Test pulse characteristics computation."""
        pulse = np.sin(np.linspace(0, 2 * np.pi, 100))
        times = np.linspace(0, 10, 100)
        report = PulseReport(pulse, times=times, fidelity=0.98)

        chars = report.characteristics
        assert chars.duration == pytest.approx(10.0, rel=1e-6)
        assert chars.peak_amplitude == pytest.approx(1.0, abs=1e-3)
        assert chars.fidelity == 0.98
        assert chars.energy > 0
        assert chars.bandwidth > 0

    def test_add_comparison(self):
        """Test adding comparison pulses."""
        pulse = np.random.randn(100)
        report = PulseReport(pulse, label="Main")

        other_pulse = np.random.randn(100)
        metrics = {"fidelity": 0.97, "duration": 8.0}
        report.add_comparison(other_pulse, "Comparison", metrics=metrics)

        assert len(report.comparison_pulses) == 1
        assert len(report.comparison_labels) == 1
        assert report.comparison_labels[0] == "Comparison"
        assert report.comparison_metrics[0]["fidelity"] == 0.97

    def test_multiple_comparisons(self):
        """Test adding multiple comparison pulses."""
        pulse = np.random.randn(100)
        report = PulseReport(pulse)

        for i in range(3):
            other = np.random.randn(100)
            report.add_comparison(other, f"Pulse {i + 1}")

        assert len(report.comparison_pulses) == 3
        assert len(report.comparison_labels) == 3

    def test_generate_full_report(self):
        """Test generating full visual report."""
        pulse = np.sin(np.linspace(0, 4 * np.pi, 150))
        times = np.linspace(0, 10, 150)
        report = PulseReport(
            pulse,
            times=times,
            fidelity=0.99,
            target_gate="X",
            optimization_method="GRAPE",
        )

        fig = report.generate_full_report()

        assert fig is not None
        assert len(fig.axes) >= 2
        plt.close(fig)

    def test_generate_report_with_comparison(self):
        """Test report generation with comparison pulses."""
        pulse = np.sin(np.linspace(0, 2 * np.pi, 100))
        report = PulseReport(pulse, fidelity=0.99)

        comparison = np.cos(np.linspace(0, 2 * np.pi, 100))
        report.add_comparison(comparison, "Cosine Pulse")

        fig = report.generate_full_report()

        assert fig is not None
        plt.close(fig)

    def test_save_full_report(self, tmp_path):
        """Test saving full report to file."""
        pulse = np.random.randn(80)
        report = PulseReport(pulse)

        output_file = tmp_path / "report.png"
        fig = report.generate_full_report(filename=str(output_file))

        assert output_file.exists()
        plt.close(fig)

    def test_export_latex_table(self, tmp_path):
        """Test exporting metrics as LaTeX table."""
        pulse = np.sin(np.linspace(0, 2 * np.pi, 100))
        report = PulseReport(pulse, fidelity=0.995, label="Test")

        output_file = tmp_path / "metrics.tex"
        latex_str = report.export_metrics_table(str(output_file), format="latex")

        assert output_file.exists()
        assert "\\begin{table}" in latex_str
        assert "\\end{table}" in latex_str
        assert "Fidelity" in latex_str

    def test_export_csv_table(self, tmp_path):
        """Test exporting metrics as CSV."""
        pulse = np.random.randn(50)
        report = PulseReport(pulse, fidelity=0.98)

        output_file = tmp_path / "metrics.csv"
        csv_str = report.export_metrics_table(str(output_file), format="csv")

        assert output_file.exists()
        assert "Metric,Value" in csv_str
        assert "Duration" in csv_str
        assert "Fidelity" in csv_str

    def test_export_json_table(self, tmp_path):
        """Test exporting metrics as JSON."""
        pulse = np.random.randn(60)
        report = PulseReport(
            pulse,
            fidelity=0.99,
            target_gate="Hadamard",
            optimization_method="Krotov",
        )

        output_file = tmp_path / "metrics.json"
        json_str = report.export_metrics_table(str(output_file), format="json")

        assert output_file.exists()

        # Parse JSON
        data = json.loads(json_str)
        assert "characteristics" in data
        assert data["target_gate"] == "Hadamard"
        assert data["optimization_method"] == "Krotov"

    def test_export_unknown_format_raises(self, tmp_path):
        """Test that unknown format raises error."""
        pulse = np.random.randn(50)
        report = PulseReport(pulse)

        with pytest.raises(ValueError, match="Unknown format"):
            report.export_metrics_table(str(tmp_path / "out.txt"), format="unknown")

    def test_characteristics_without_fidelity(self):
        """Test characteristics when fidelity is not provided."""
        pulse = np.random.randn(100)
        report = PulseReport(pulse)

        assert np.isnan(report.characteristics.fidelity)

    def test_short_pulse(self):
        """Test with very short pulse."""
        pulse = np.array([1.0, 2.0])
        report = PulseReport(pulse)

        chars = report.characteristics
        assert chars.peak_amplitude == 2.0
        assert chars.smoothness == 0.0  # Too short for second derivative

    def test_constant_pulse(self):
        """Test with constant pulse."""
        pulse = np.ones(100)
        report = PulseReport(pulse)

        chars = report.characteristics
        assert chars.peak_amplitude == 1.0
        assert chars.rms_amplitude == pytest.approx(1.0)


class TestOptimizationReport:
    """Tests for OptimizationReport class."""

    def test_initialization(self):
        """Test basic initialization."""
        report = OptimizationReport(method="GRAPE", target="X Gate")

        assert report.method == "GRAPE"
        assert report.target == "X Gate"
        assert len(report.iterations) == 0
        assert report.end_time is None

    def test_add_iteration(self):
        """Test adding iteration data."""
        report = OptimizationReport(method="GRAPE")

        report.add_iteration(iteration=0, fidelity=0.5, gradient_norm=0.1)

        assert len(report.iterations) == 1
        assert report.fidelities[0] == 0.5
        assert report.infidelities[0] == 0.5
        assert report.gradient_norms[0] == 0.1

    def test_multiple_iterations(self):
        """Test adding multiple iterations."""
        report = OptimizationReport(method="Krotov")

        for i in range(10):
            fidelity = 0.5 + 0.05 * i
            gradient = 0.1 / (i + 1)
            report.add_iteration(iteration=i, fidelity=fidelity, gradient_norm=gradient)

        assert len(report.iterations) == 10
        assert report.fidelities[-1] > report.fidelities[0]

    def test_custom_metrics_tracking(self):
        """Test tracking custom metrics."""
        report = OptimizationReport()

        report.add_iteration(
            iteration=0, fidelity=0.9, robustness=0.85, control_energy=5.0
        )

        assert "robustness" in report.custom_metrics
        assert "control_energy" in report.custom_metrics
        assert report.custom_metrics["robustness"][0] == 0.85

    def test_finalize(self):
        """Test finalizing report."""
        report = OptimizationReport()
        report.add_iteration(iteration=0, fidelity=0.9)
        report.finalize()

        assert report.end_time is not None
        assert report.end_time > report.start_time

    def test_generate_summary(self):
        """Test generating summary report."""
        report = OptimizationReport(method="GRAPE", target="Hadamard")

        for i in range(20):
            report.add_iteration(iteration=i, fidelity=0.5 + i * 0.02)

        report.finalize()
        fig = report.generate_summary()

        assert fig is not None
        assert len(fig.axes) >= 2
        plt.close(fig)

    def test_generate_summary_with_gradient(self):
        """Test summary with gradient information."""
        report = OptimizationReport(method="GRAPE")

        for i in range(15):
            report.add_iteration(
                iteration=i, fidelity=0.7 + i * 0.01, gradient_norm=0.1 / (i + 1)
            )

        fig = report.generate_summary()

        assert fig is not None
        plt.close(fig)

    def test_save_summary(self, tmp_path):
        """Test saving summary to file."""
        report = OptimizationReport()

        for i in range(10):
            report.add_iteration(iteration=i, fidelity=0.8 + i * 0.01)

        output_file = tmp_path / "summary.png"
        fig = report.generate_summary(filename=str(output_file))

        assert output_file.exists()
        plt.close(fig)

    def test_empty_report(self):
        """Test report with no iterations."""
        report = OptimizationReport()
        fig = report.generate_summary()

        assert fig is not None
        plt.close(fig)


class TestGenerateLatexTable:
    """Tests for generate_latex_table function."""

    def test_basic_table(self, tmp_path):
        """Test basic table generation."""
        data = {
            "Method": ["GRAPE", "Krotov", "CRAB"],
            "Fidelity": [0.99, 0.98, 0.97],
            "Time": [10.0, 12.0, 8.0],
        }

        output_file = tmp_path / "table.tex"
        latex_str = generate_latex_table(
            data, str(output_file), caption="Results", label="tab:results"
        )

        assert output_file.exists()
        assert "\\begin{table}" in latex_str
        assert "\\end{table}" in latex_str
        assert "GRAPE" in latex_str
        assert "Krotov" in latex_str
        assert "\\caption{Results}" in latex_str
        assert "\\label{tab:results}" in latex_str

    def test_numeric_formatting(self, tmp_path):
        """Test numeric value formatting."""
        data = {
            "Param": ["alpha", "beta"],
            "Value": [0.123456, 0.987654],
        }

        output_file = tmp_path / "numeric.tex"
        latex_str = generate_latex_table(data, str(output_file))

        # Should format floats with 4 decimal places
        assert "0.1235" in latex_str
        assert "0.9877" in latex_str

    def test_mixed_types(self, tmp_path):
        """Test table with mixed data types."""
        data = {
            "Name": ["Test1", "Test2"],
            "Value": [42, 100],
            "Fidelity": [0.99, 0.98],
        }

        output_file = tmp_path / "mixed.tex"
        latex_str = generate_latex_table(data, str(output_file))

        assert "Test1" in latex_str
        assert "42" in latex_str
        assert "0.99" in latex_str

    def test_single_row(self, tmp_path):
        """Test table with single row."""
        data = {
            "Method": ["GRAPE"],
            "Fidelity": [0.995],
        }

        output_file = tmp_path / "single.tex"
        latex_str = generate_latex_table(data, str(output_file))

        assert "GRAPE" in latex_str
        assert "0.995" in latex_str


class TestCreatePublicationFigure:
    """Tests for create_publication_figure function."""

    def test_single_dataset(self):
        """Test with single dataset."""
        data = np.sin(np.linspace(0, 2 * np.pi, 100))

        fig = create_publication_figure(
            data, xlabel="Time", ylabel="Amplitude", title="Sine Wave"
        )

        assert fig is not None
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_multiple_datasets(self):
        """Test with multiple datasets."""
        x = np.linspace(0, 10, 100)
        data = [np.sin(x), np.cos(x), np.sin(2 * x)]
        labels = ["sin(x)", "cos(x)", "sin(2x)"]

        fig = create_publication_figure(data, labels=labels, xlabel="x", ylabel="y")

        assert fig is not None
        plt.close(fig)

    def test_save_to_file(self, tmp_path):
        """Test saving figure to file."""
        data = np.random.randn(100)
        output_file = tmp_path / "figure.png"

        fig = create_publication_figure(data, filename=str(output_file))

        assert output_file.exists()
        plt.close(fig)

    def test_custom_figsize(self):
        """Test with custom figure size."""
        data = np.random.randn(50)

        fig = create_publication_figure(data, figsize=(10, 6))

        assert fig.get_size_inches()[0] == pytest.approx(10.0)
        assert fig.get_size_inches()[1] == pytest.approx(6.0)
        plt.close(fig)

    def test_with_labels_and_legend(self):
        """Test with labels showing legend."""
        data = [np.random.randn(50), np.random.randn(50)]
        labels = ["Dataset 1", "Dataset 2"]

        fig = create_publication_figure(data, labels=labels)

        # Should have legend
        assert fig.axes[0].get_legend() is not None
        plt.close(fig)

    def test_without_labels_no_legend(self):
        """Test without labels (no legend)."""
        data = [np.random.randn(50), np.random.randn(50)]

        fig = create_publication_figure(data)

        # Should not have legend
        assert fig.axes[0].get_legend() is None
        plt.close(fig)

    def test_pdf_save(self, tmp_path):
        """Test saving as PDF."""
        data = np.random.randn(50)
        output_file = tmp_path / "figure.pdf"

        fig = create_publication_figure(data, filename=str(output_file))

        assert output_file.exists()
        plt.close(fig)


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_pulse_and_optimization_reports(self):
        """Test creating both pulse and optimization reports."""
        # Optimization report
        opt_report = OptimizationReport(method="GRAPE", target="X")
        pulse_evolution = []

        for i in range(20):
            fidelity = 0.7 + i * 0.015
            pulse = np.sin(np.linspace(0, 2 * np.pi * (i + 1) / 20, 100))
            pulse_evolution.append(pulse)
            opt_report.add_iteration(iteration=i, fidelity=fidelity)

        opt_report.finalize()

        # Pulse report for final pulse
        pulse_report = PulseReport(
            pulse_evolution[-1],
            fidelity=opt_report.fidelities[-1],
            target_gate="X",
            optimization_method="GRAPE",
        )

        # Add comparison with initial pulse
        pulse_report.add_comparison(pulse_evolution[0], "Initial Pulse")

        # Generate both reports
        opt_fig = opt_report.generate_summary()
        pulse_fig = pulse_report.generate_full_report()

        assert opt_fig is not None
        assert pulse_fig is not None
        plt.close(opt_fig)
        plt.close(pulse_fig)

    def test_full_workflow_with_exports(self, tmp_path):
        """Test full workflow with all export formats."""
        pulse = np.sin(np.linspace(0, 4 * np.pi, 150))
        times = np.linspace(0, 10, 150)

        report = PulseReport(
            pulse,
            times=times,
            fidelity=0.995,
            target_gate="Hadamard",
            optimization_method="GRAPE",
        )

        # Generate visual report
        fig_file = tmp_path / "pulse_report.png"
        report.generate_full_report(filename=str(fig_file))

        # Export in multiple formats
        tex_file = tmp_path / "metrics.tex"
        csv_file = tmp_path / "metrics.csv"
        json_file = tmp_path / "metrics.json"

        report.export_metrics_table(str(tex_file), format="latex")
        report.export_metrics_table(str(csv_file), format="csv")
        report.export_metrics_table(str(json_file), format="json")

        assert fig_file.exists()
        assert tex_file.exists()
        assert csv_file.exists()
        assert json_file.exists()

    def test_comparison_table_generation(self, tmp_path):
        """Test generating comparison table for multiple methods."""
        methods = ["GRAPE", "Krotov", "CRAB"]
        fidelities = [0.995, 0.993, 0.990]
        times = [10.5, 12.3, 8.7]
        energies = [5.2, 4.8, 6.1]

        data = {
            "Method": methods,
            "Fidelity": fidelities,
            "Duration": times,
            "Energy": energies,
        }

        output_file = tmp_path / "comparison.tex"
        latex_str = generate_latex_table(
            data,
            str(output_file),
            caption="Optimization Method Comparison",
            label="tab:comparison",
        )

        assert output_file.exists()
        assert all(method in latex_str for method in methods)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
