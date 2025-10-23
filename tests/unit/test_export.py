"""
Unit tests for I/O export and serialization module.

Tests cover:
- JSON export/import for pulses
- NPZ export/import for pulses
- CSV export/import for pulses
- Optimization result serialization
- Configuration export
- Round-trip save/load consistency
- Metadata preservation
- Error handling

Author: QubitPulseOpt Team
Date: 2025-01-28
"""

import pytest
import numpy as np
import json
import tempfile
from pathlib import Path

from src.io.export import (
    PulseExporter,
    PulseLoader,
    save_pulse,
    load_pulse,
    save_optimization_result,
    load_optimization_result,
)


@pytest.fixture
def simple_pulse():
    """Create a simple test pulse."""
    times = np.linspace(0, 100, 1000)
    amplitudes = np.exp(-((times - 50) ** 2) / 200)
    return times, amplitudes


@pytest.fixture
def complex_pulse():
    """Create a complex pulse with frequency and phase modulation."""
    times = np.linspace(0, 100, 1000)
    amplitudes = np.exp(-((times - 50) ** 2) / 200)
    frequencies = 5.0 * np.ones_like(times)
    phases = 2 * np.pi * times / 100
    return times, amplitudes, frequencies, phases


@pytest.fixture
def optimization_result():
    """Create a mock optimization result."""
    return {
        "final_fidelity": 0.9995,
        "iterations": 150,
        "convergence_history": np.linspace(0.5, 0.9995, 150),
        "cost_history": np.linspace(0.5, 0.0005, 150),
        "final_pulse": np.random.randn(1000),
        "pulse_duration": 100.0,
        "success": True,
        "message": "Optimization converged",
    }


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestPulseExporter:
    """Test PulseExporter class."""

    def test_exporter_initialization(self):
        """Test exporter initialization with metadata."""
        metadata = {"author": "test", "experiment": "test_exp"}
        exporter = PulseExporter(metadata=metadata)

        assert exporter.metadata["author"] == "test"
        assert exporter.metadata["experiment"] == "test_exp"
        assert exporter.metadata["exporter_version"] == PulseExporter.VERSION

    def test_export_pulse_json_simple(self, simple_pulse, temp_dir):
        """Test JSON export of a simple pulse."""
        times, amplitudes = simple_pulse
        exporter = PulseExporter()

        filepath = temp_dir / "test_pulse.json"
        result_path = exporter.export_pulse_json(
            times, amplitudes, filepath=filepath, pulse_name="test_pulse"
        )

        assert result_path.exists()
        assert result_path == filepath

        # Verify file contents
        with open(filepath, "r") as f:
            data = json.load(f)

        assert data["pulse_name"] == "test_pulse"
        assert data["schema_version"] == PulseExporter.VERSION
        assert "export_timestamp" in data
        assert len(data["pulse_data"]["times"]) == len(times)
        assert len(data["pulse_data"]["amplitudes"]) == len(amplitudes)

    def test_export_pulse_json_complex(self, complex_pulse, temp_dir):
        """Test JSON export with frequency and phase modulation."""
        times, amplitudes, frequencies, phases = complex_pulse
        exporter = PulseExporter()

        filepath = temp_dir / "complex_pulse.json"
        exporter.export_pulse_json(
            times,
            amplitudes,
            frequencies=frequencies,
            phases=phases,
            filepath=filepath,
        )

        with open(filepath, "r") as f:
            data = json.load(f)

        assert "frequencies" in data["pulse_data"]
        assert "phases" in data["pulse_data"]
        assert len(data["pulse_data"]["frequencies"]) == len(frequencies)
        assert len(data["pulse_data"]["phases"]) == len(phases)

    def test_export_pulse_json_statistics(self, simple_pulse, temp_dir):
        """Test that statistics are computed correctly."""
        times, amplitudes = simple_pulse
        exporter = PulseExporter()

        filepath = temp_dir / "stats_pulse.json"
        exporter.export_pulse_json(times, amplitudes, filepath=filepath)

        with open(filepath, "r") as f:
            data = json.load(f)

        stats = data["statistics"]
        assert "max_amplitude" in stats
        assert "mean_amplitude" in stats
        assert "rms_amplitude" in stats
        assert "peak_to_peak" in stats

        # Verify values
        assert stats["max_amplitude"] == pytest.approx(np.max(np.abs(amplitudes)))
        assert stats["mean_amplitude"] == pytest.approx(np.mean(np.abs(amplitudes)))

    def test_export_pulse_npz(self, simple_pulse, temp_dir):
        """Test NPZ export."""
        times, amplitudes = simple_pulse
        exporter = PulseExporter()

        filepath = temp_dir / "test_pulse.npz"
        result_path = exporter.export_pulse_npz(filepath, times, amplitudes)

        assert result_path.exists()

        # Load and verify
        data = np.load(filepath, allow_pickle=True)
        assert "times" in data
        assert "amplitudes" in data
        assert "_metadata" in data

        np.testing.assert_array_almost_equal(data["times"], times)
        np.testing.assert_array_almost_equal(data["amplitudes"], amplitudes)

    def test_export_pulse_npz_with_metadata(self, simple_pulse, temp_dir):
        """Test NPZ export with custom metadata."""
        times, amplitudes = simple_pulse
        exporter = PulseExporter()

        metadata = {"experiment": "test", "qubit_id": 1}
        filepath = temp_dir / "pulse_meta.npz"
        exporter.export_pulse_npz(filepath, times, amplitudes, metadata=metadata)

        data = np.load(filepath, allow_pickle=True)
        metadata_str = str(data["_metadata"][0])
        loaded_meta = json.loads(metadata_str)

        assert loaded_meta["experiment"] == "test"
        assert loaded_meta["qubit_id"] == 1

    def test_export_pulse_csv(self, simple_pulse, temp_dir):
        """Test CSV export."""
        times, amplitudes = simple_pulse
        exporter = PulseExporter()

        filepath = temp_dir / "test_pulse.csv"
        exporter.export_pulse_csv(filepath, times, amplitudes)

        assert filepath.exists()

        # Load and verify
        data = np.loadtxt(filepath, delimiter=",", skiprows=1)
        assert data.shape[1] == 2  # time and amplitude columns
        np.testing.assert_array_almost_equal(data[:, 0], times)
        np.testing.assert_array_almost_equal(data[:, 1], amplitudes)

    def test_export_pulse_csv_complex(self, complex_pulse, temp_dir):
        """Test CSV export with all columns."""
        times, amplitudes, frequencies, phases = complex_pulse
        exporter = PulseExporter()

        filepath = temp_dir / "complex_pulse.csv"
        exporter.export_pulse_csv(
            filepath, times, amplitudes, frequencies=frequencies, phases=phases
        )

        data = np.loadtxt(filepath, delimiter=",", skiprows=1)
        assert data.shape[1] == 4  # All columns present

    def test_export_optimization_result_json(self, optimization_result, temp_dir):
        """Test optimization result export to JSON."""
        exporter = PulseExporter()
        filepath = temp_dir / "opt_result.json"

        exporter.export_optimization_result(
            filepath, optimization_result, format="json"
        )

        assert filepath.exists()

        with open(filepath, "r") as f:
            data = json.load(f)

        assert data["final_fidelity"] == pytest.approx(0.9995)
        assert data["iterations"] == 150
        assert data["success"] is True
        assert "_metadata" in data

    def test_export_optimization_result_npz(self, optimization_result, temp_dir):
        """Test optimization result export to NPZ."""
        exporter = PulseExporter()
        filepath = temp_dir / "opt_result.npz"

        exporter.export_optimization_result(filepath, optimization_result, format="npz")

        assert filepath.exists()

        data = np.load(filepath, allow_pickle=True)
        assert "convergence_history" in data
        assert "final_pulse" in data

    def test_export_experiment_config(self, temp_dir):
        """Test experiment configuration export."""
        exporter = PulseExporter()
        config = {
            "system_params": {"T1": 10e-6, "T2": 20e-6, "frequency": 5e9},
            "optimization_params": {"max_iter": 200, "tolerance": 1e-5},
            "pulse_params": {"duration": 100e-9, "shape": "gaussian"},
        }

        filepath = temp_dir / "config.json"
        exporter.export_experiment_config(filepath, config)

        assert filepath.exists()

        with open(filepath, "r") as f:
            data = json.load(f)

        assert "configuration" in data
        assert data["configuration"]["system_params"]["T1"] == 10e-6

    def test_export_qiskit_pulse(self, simple_pulse, temp_dir):
        """Test Qiskit-compatible pulse export."""
        times, amplitudes = simple_pulse
        exporter = PulseExporter()

        filepath = temp_dir / "qiskit_pulse.json"
        exporter.export_qiskit_pulse(filepath, times, amplitudes, dt=1.0)

        assert filepath.exists()

        with open(filepath, "r") as f:
            data = json.load(f)

        assert data["format"] == "qiskit_pulse_compatible"
        assert "real_amplitudes" in data
        assert "imag_amplitudes" in data
        assert data["dt"] == 1.0

    def test_export_qiskit_pulse_complex(self, temp_dir):
        """Test Qiskit export with complex amplitudes."""
        times = np.linspace(0, 100, 1000)
        amplitudes = np.exp(1j * 2 * np.pi * times / 100)
        exporter = PulseExporter()

        filepath = temp_dir / "qiskit_complex.json"
        exporter.export_qiskit_pulse(filepath, times, amplitudes)

        with open(filepath, "r") as f:
            data = json.load(f)

        assert len(data["real_amplitudes"]) == len(amplitudes)
        assert len(data["imag_amplitudes"]) == len(amplitudes)


class TestPulseLoader:
    """Test PulseLoader class."""

    def test_load_pulse_json(self, simple_pulse, temp_dir):
        """Test loading pulse from JSON."""
        times, amplitudes = simple_pulse
        exporter = PulseExporter()
        filepath = temp_dir / "test_pulse.json"
        exporter.export_pulse_json(times, amplitudes, filepath=filepath)

        # Load
        loader = PulseLoader()
        data = loader.load_pulse_json(filepath)

        assert "pulse_data" in data
        loaded_times = data["pulse_data"]["times"]
        loaded_amps = data["pulse_data"]["amplitudes"]

        np.testing.assert_array_almost_equal(loaded_times, times)
        np.testing.assert_array_almost_equal(loaded_amps, amplitudes)

    def test_load_pulse_npz(self, simple_pulse, temp_dir):
        """Test loading pulse from NPZ."""
        times, amplitudes = simple_pulse
        exporter = PulseExporter()
        filepath = temp_dir / "test_pulse.npz"
        exporter.export_pulse_npz(filepath, times, amplitudes)

        # Load
        loader = PulseLoader()
        data = loader.load_pulse_npz(filepath)

        np.testing.assert_array_almost_equal(data["times"], times)
        np.testing.assert_array_almost_equal(data["amplitudes"], amplitudes)
        assert "metadata" in data

    def test_load_pulse_csv(self, simple_pulse, temp_dir):
        """Test loading pulse from CSV."""
        times, amplitudes = simple_pulse
        exporter = PulseExporter()
        filepath = temp_dir / "test_pulse.csv"
        exporter.export_pulse_csv(filepath, times, amplitudes)

        # Load
        loader = PulseLoader()
        data = loader.load_pulse_csv(filepath, has_header=True)

        assert "time" in data
        assert "amplitude" in data
        np.testing.assert_array_almost_equal(data["time"], times)
        np.testing.assert_array_almost_equal(data["amplitude"], amplitudes)

    def test_load_optimization_result_json(self, optimization_result, temp_dir):
        """Test loading optimization result from JSON."""
        exporter = PulseExporter()
        filepath = temp_dir / "opt_result.json"
        exporter.export_optimization_result(
            filepath, optimization_result, format="json"
        )

        # Load
        loader = PulseLoader()
        data = loader.load_optimization_result(filepath, format="json")

        assert data["final_fidelity"] == pytest.approx(0.9995)
        assert data["iterations"] == 150
        assert isinstance(data["convergence_history"], np.ndarray)

    def test_load_optimization_result_npz(self, optimization_result, temp_dir):
        """Test loading optimization result from NPZ."""
        exporter = PulseExporter()
        filepath = temp_dir / "opt_result.npz"
        exporter.export_optimization_result(filepath, optimization_result, format="npz")

        # Load
        loader = PulseLoader()
        data = loader.load_optimization_result(filepath, format="npz")

        assert "convergence_history" in data
        assert "final_pulse" in data

    def test_load_experiment_config(self, temp_dir):
        """Test loading experiment configuration."""
        exporter = PulseExporter()
        config = {"system_params": {"T1": 10e-6, "T2": 20e-6}}
        filepath = temp_dir / "config.json"
        exporter.export_experiment_config(filepath, config)

        # Load
        loader = PulseLoader()
        loaded_config = loader.load_experiment_config(filepath)

        assert loaded_config["system_params"]["T1"] == 10e-6


class TestRoundTripConsistency:
    """Test round-trip save/load consistency."""

    def test_json_roundtrip(self, simple_pulse, temp_dir):
        """Test JSON export-import roundtrip."""
        times, amplitudes = simple_pulse
        filepath = temp_dir / "roundtrip.json"

        # Export
        save_pulse(filepath, times, amplitudes, format="json")

        # Import
        data = load_pulse(filepath, format="json")

        loaded_times = data["pulse_data"]["times"]
        loaded_amps = data["pulse_data"]["amplitudes"]

        np.testing.assert_array_almost_equal(loaded_times, times)
        np.testing.assert_array_almost_equal(loaded_amps, amplitudes)

    def test_npz_roundtrip(self, simple_pulse, temp_dir):
        """Test NPZ export-import roundtrip."""
        times, amplitudes = simple_pulse
        filepath = temp_dir / "roundtrip.npz"

        # Export
        save_pulse(filepath, times, amplitudes, format="npz")

        # Import
        data = load_pulse(filepath, format="npz")

        np.testing.assert_array_almost_equal(data["times"], times)
        np.testing.assert_array_almost_equal(data["amplitudes"], amplitudes)

    def test_csv_roundtrip(self, simple_pulse, temp_dir):
        """Test CSV export-import roundtrip."""
        times, amplitudes = simple_pulse
        filepath = temp_dir / "roundtrip.csv"

        # Export
        save_pulse(filepath, times, amplitudes, format="csv")

        # Import
        data = load_pulse(filepath, format="csv")

        np.testing.assert_array_almost_equal(data["time"], times)
        np.testing.assert_array_almost_equal(data["amplitude"], amplitudes)

    def test_complex_pulse_roundtrip(self, complex_pulse, temp_dir):
        """Test roundtrip with complex pulse data."""
        times, amplitudes, frequencies, phases = complex_pulse
        filepath = temp_dir / "complex_roundtrip.json"

        # Export
        save_pulse(
            filepath,
            times,
            amplitudes,
            frequencies=frequencies,
            phases=phases,
            format="json",
        )

        # Import
        data = load_pulse(filepath)
        pulse_data = data["pulse_data"]

        np.testing.assert_array_almost_equal(pulse_data["times"], times)
        np.testing.assert_array_almost_equal(pulse_data["amplitudes"], amplitudes)
        np.testing.assert_array_almost_equal(pulse_data["frequencies"], frequencies)
        np.testing.assert_array_almost_equal(pulse_data["phases"], phases)

    def test_optimization_result_roundtrip(self, optimization_result, temp_dir):
        """Test optimization result roundtrip."""
        filepath = temp_dir / "opt_roundtrip.json"

        # Save
        save_optimization_result(filepath, optimization_result, format="json")

        # Load
        loaded = load_optimization_result(filepath)

        assert loaded["final_fidelity"] == pytest.approx(
            optimization_result["final_fidelity"]
        )
        assert loaded["iterations"] == optimization_result["iterations"]
        np.testing.assert_array_almost_equal(
            loaded["convergence_history"], optimization_result["convergence_history"]
        )


class TestMetadataPreservation:
    """Test that metadata is preserved through save/load cycles."""

    def test_metadata_in_json(self, simple_pulse, temp_dir):
        """Test metadata preservation in JSON export."""
        times, amplitudes = simple_pulse
        metadata = {"experiment": "test_001", "qubit": "Q1", "date": "2025-01-28"}

        exporter = PulseExporter(metadata=metadata)
        filepath = temp_dir / "meta_test.json"
        exporter.export_pulse_json(times, amplitudes, filepath=filepath)

        # Load and check
        with open(filepath, "r") as f:
            data = json.load(f)

        assert data["metadata"]["experiment"] == "test_001"
        assert data["metadata"]["qubit"] == "Q1"

    def test_additional_metadata(self, simple_pulse, temp_dir):
        """Test additional metadata parameter."""
        times, amplitudes = simple_pulse
        exporter = PulseExporter()

        additional = {"temperature": 10e-3, "power": -20}
        filepath = temp_dir / "additional_meta.json"
        exporter.export_pulse_json(
            times, amplitudes, filepath=filepath, additional_metadata=additional
        )

        with open(filepath, "r") as f:
            data = json.load(f)

        assert data["metadata"]["temperature"] == 10e-3
        assert data["metadata"]["power"] == -20


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_format(self, simple_pulse, temp_dir):
        """Test invalid format raises error."""
        times, amplitudes = simple_pulse
        filepath = temp_dir / "test.xyz"

        with pytest.raises(ValueError, match="Unsupported format"):
            save_pulse(filepath, times, amplitudes, format="xyz")

    def test_load_nonexistent_file(self, temp_dir):
        """Test loading nonexistent file raises error."""
        filepath = temp_dir / "nonexistent.json"

        with pytest.raises(FileNotFoundError):
            load_pulse(filepath)

    def test_empty_arrays(self, temp_dir):
        """Test handling of empty arrays."""
        times = np.array([])
        amplitudes = np.array([])
        filepath = temp_dir / "empty.json"

        # Should not raise error
        save_pulse(filepath, times, amplitudes, format="json")

        # Load it back
        data = load_pulse(filepath)
        assert len(data["pulse_data"]["times"]) == 0

    def test_mismatched_array_lengths(self, temp_dir):
        """Test handling of mismatched array lengths."""
        times = np.linspace(0, 100, 1000)
        amplitudes = np.random.randn(500)  # Different length

        filepath = temp_dir / "mismatched.json"

        # Should still save (arrays are independent in JSON)
        save_pulse(filepath, times, amplitudes, format="json")

    def test_auto_format_detection(self, simple_pulse, temp_dir):
        """Test automatic format detection from file extension."""
        times, amplitudes = simple_pulse

        # Save with explicit format
        filepath_json = temp_dir / "auto.json"
        save_pulse(filepath_json, times, amplitudes, format="json")

        # Load with auto-detection
        data = load_pulse(filepath_json)
        assert data is not None


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_save_pulse_convenience(self, simple_pulse, temp_dir):
        """Test save_pulse convenience function."""
        times, amplitudes = simple_pulse
        filepath = temp_dir / "convenience.json"

        result = save_pulse(filepath, times, amplitudes, format="json")
        assert result.exists()

    def test_load_pulse_convenience(self, simple_pulse, temp_dir):
        """Test load_pulse convenience function."""
        times, amplitudes = simple_pulse
        filepath = temp_dir / "convenience.npz"

        save_pulse(filepath, times, amplitudes, format="npz")
        data = load_pulse(filepath)

        assert "times" in data
        assert "amplitudes" in data

    def test_optimization_convenience_functions(self, optimization_result, temp_dir):
        """Test optimization result convenience functions."""
        filepath = temp_dir / "opt_convenience.json"

        # Save
        save_optimization_result(filepath, optimization_result)

        # Load
        loaded = load_optimization_result(filepath)

        assert loaded["final_fidelity"] == pytest.approx(
            optimization_result["final_fidelity"]
        )


class TestNumPyDtypeHandling:
    """Test handling of various NumPy dtypes."""

    def test_complex_dtype_export(self, temp_dir):
        """Test export of complex-valued arrays."""
        times = np.linspace(0, 100, 100)
        amplitudes = np.exp(1j * 2 * np.pi * times / 100)

        filepath = temp_dir / "complex.json"
        save_pulse(filepath, times, amplitudes, format="json")

        # Verify JSON contains real/imag pairs
        with open(filepath, "r") as f:
            data = json.load(f)

        # Complex arrays should be saved as [real, imag] pairs
        amp_data = data["pulse_data"]["amplitudes"]
        assert isinstance(amp_data[0], list)
        assert len(amp_data[0]) == 2  # [real, imag]

    def test_integer_dtype_export(self, temp_dir):
        """Test export of integer arrays."""
        times = np.arange(0, 100, dtype=np.int32)
        amplitudes = np.arange(0, 100, dtype=np.int64)

        filepath = temp_dir / "integer.json"
        save_pulse(filepath, times, amplitudes, format="json")

        data = load_pulse(filepath)
        # Should be converted to float for compatibility
        assert isinstance(data["pulse_data"]["times"], np.ndarray)

    def test_float32_export(self, temp_dir):
        """Test export of float32 arrays."""
        times = np.linspace(0, 100, 100, dtype=np.float32)
        amplitudes = np.random.randn(100).astype(np.float32)

        filepath = temp_dir / "float32.npz"
        save_pulse(filepath, times, amplitudes, format="npz")

        data = load_pulse(filepath)
        np.testing.assert_array_almost_equal(data["times"], times, decimal=5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
