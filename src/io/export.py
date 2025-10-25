"""
Export and serialization utilities for quantum control pulses and results.

This module provides functions to save and load pulse sequences, optimization
results, and experimental configurations in various formats suitable for
hardware backends and data analysis.

Supported formats:
- JSON: Human-readable, metadata-rich format
- NPZ: NumPy compressed format for large arrays
- CSV: Simple tabular format for time-series data
- Qiskit Pulse: Compatibility layer (basic)

Author: QubitPulseOpt Team
Date: 2025-01-28
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
import warnings

# Power of 10 compliance: Import loop bounds
from ..constants import MAX_DICT_ITEMS


class PulseExporter:
    """
    Export pulse sequences and optimization results to various formats.

    Attributes:
        version (str): Schema version for exported files
        metadata (dict): Additional metadata to include in exports
    """

    VERSION = "1.0.0"

    def __init__(self, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize the exporter with optional metadata.

        Args:
            metadata: Additional metadata to include in all exports
        """
        self.metadata = metadata or {}
        self.metadata["exporter_version"] = self.VERSION
        self.metadata["export_timestamp"] = None  # Set at export time

    def _build_pulse_data_dict(
        self,
        times: np.ndarray,
        amplitudes: np.ndarray,
        frequencies: Optional[np.ndarray],
        phases: Optional[np.ndarray],
    ) -> dict:
        """
        Build pulse data dictionary.

        Args:
            times: Time points array
            amplitudes: Pulse amplitude envelope
            frequencies: Optional frequency modulation
            phases: Optional phase modulation

        Returns:
            Pulse data dictionary
        """
        pulse_data = {
            "times": self._to_list(times),
            "amplitudes": self._to_list(amplitudes),
            "num_samples": len(times),
            "duration": float(times[-1] - times[0]) if len(times) > 1 else 0.0,
            "sample_rate": float(len(times) / (times[-1] - times[0]))
            if len(times) > 1 and times[-1] != times[0]
            else 0.0,
        }

        # Add optional fields
        if frequencies is not None:
            pulse_data["frequencies"] = self._to_list(frequencies)
        if phases is not None:
            pulse_data["phases"] = self._to_list(phases)

        return pulse_data

    def _compute_pulse_statistics(self, amplitudes: np.ndarray) -> dict:
        """
        Compute pulse statistics.

        Args:
            amplitudes: Pulse amplitude envelope

        Returns:
            Statistics dictionary
        """
        if len(amplitudes) > 0:
            # For complex arrays, use absolute values for statistics
            amp_abs = np.abs(amplitudes)
            return {
                "max_amplitude": float(np.max(amp_abs)),
                "mean_amplitude": float(np.mean(amp_abs)),
                "rms_amplitude": float(np.sqrt(np.mean(amp_abs**2))),
                "peak_to_peak": float(np.ptp(amp_abs)),
            }
        else:
            return {
                "max_amplitude": 0.0,
                "mean_amplitude": 0.0,
                "rms_amplitude": 0.0,
                "peak_to_peak": 0.0,
            }

    def export_pulse_json(
        self,
        times: np.ndarray,
        amplitudes: np.ndarray,
        frequencies: Optional[np.ndarray] = None,
        phases: Optional[np.ndarray] = None,
        filepath: Union[str, Path] = "pulse.json",
        pulse_name: str = "pulse",
        additional_metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Export a pulse sequence to JSON format.

        Args:
            times: Time points array (ns or μs)
            amplitudes: Pulse amplitude envelope
            frequencies: Optional frequency modulation
            phases: Optional phase modulation
            filepath: Output file path
            pulse_name: Name/identifier for the pulse
            additional_metadata: Additional metadata for this specific pulse

        Returns:
            Path to the exported file
        """
        filepath = Path(filepath)

        # Build pulse data
        pulse_data = self._build_pulse_data_dict(times, amplitudes, frequencies, phases)

        # Prepare complete data structure
        data = {
            "schema_version": self.VERSION,
            "export_timestamp": datetime.utcnow().isoformat() + "Z",
            "pulse_name": pulse_name,
            "metadata": {**self.metadata, **(additional_metadata or {})},
            "pulse_data": pulse_data,
            "statistics": self._compute_pulse_statistics(amplitudes),
        }

        # Write to file
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        return filepath

    def export_pulse_npz(
        self,
        filepath: Union[str, Path],
        times: np.ndarray,
        amplitudes: np.ndarray,
        frequencies: Optional[np.ndarray] = None,
        phases: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **additional_arrays,
    ) -> Path:
        """
        Export pulse data to compressed NumPy format (.npz).

        Args:
            filepath: Output file path
            times: Time points array
            amplitudes: Pulse amplitude envelope
            frequencies: Optional frequency modulation
            phases: Optional phase modulation
            metadata: Metadata dictionary (will be JSON-serialized)
            **additional_arrays: Additional arrays to save

        Returns:
            Path to the exported file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Prepare arrays to save
        arrays = {
            "times": times,
            "amplitudes": amplitudes,
        }

        if frequencies is not None:
            arrays["frequencies"] = frequencies
        if phases is not None:
            arrays["phases"] = phases

        # Add any additional arrays
        arrays.update(additional_arrays)

        # Save metadata as JSON string
        full_metadata = {
            **self.metadata,
            **(metadata or {}),
            "export_timestamp": datetime.utcnow().isoformat() + "Z",
            "schema_version": self.VERSION,
        }
        arrays["_metadata"] = np.array([json.dumps(full_metadata)])

        # Save compressed
        np.savez_compressed(filepath, **arrays)

        return filepath

    def export_pulse_csv(
        self,
        filepath: Union[str, Path],
        times: np.ndarray,
        amplitudes: np.ndarray,
        frequencies: Optional[np.ndarray] = None,
        phases: Optional[np.ndarray] = None,
        delimiter: str = ",",
        header: bool = True,
    ) -> Path:
        """
        Export pulse data to CSV format for simple analysis.

        Args:
            filepath: Output file path
            times: Time points array
            amplitudes: Pulse amplitude envelope
            frequencies: Optional frequency modulation
            phases: Optional phase modulation
            delimiter: CSV delimiter
            header: Whether to include header row

        Returns:
            Path to the exported file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Build column data
        columns = [times, amplitudes]
        column_names = ["time", "amplitude"]

        if frequencies is not None:
            columns.append(frequencies)
            column_names.append("frequency")
        if phases is not None:
            columns.append(phases)
            column_names.append("phase")

        # Stack into array
        data = np.column_stack(columns)

        # Write with header
        if header:
            header_str = delimiter.join(column_names)
            np.savetxt(
                filepath, data, delimiter=delimiter, header=header_str, comments=""
            )
        else:
            np.savetxt(filepath, data, delimiter=delimiter)

        return filepath

    def export_optimization_result(
        self, filepath: Union[str, Path], result: Dict[str, Any], format: str = "json"
    ) -> Path:
        """
        Export optimization results including convergence history.

        Args:
            filepath: Output file path
            result: Optimization result dictionary containing:
                - final_pulse: optimized pulse
                - fidelity: final fidelity
                - iterations: number of iterations
                - convergence_history: fidelity vs iteration
                - cost_history: cost function history
                - etc.
            format: Export format ('json' or 'npz')

        Returns:
            Path to the exported file
        """
        filepath = Path(filepath)

        if format.lower() == "json":
            return self._export_optimization_json(filepath, result)
        elif format.lower() == "npz":
            return self._export_optimization_npz(filepath, result)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _convert_value_for_json(self, value: Any) -> Any:
        """
        Convert a single value for JSON serialization.

        Rule 4: Helper method to reduce nesting depth.
        Rule 5: Add type assertion.

        Args:
            value: Value to convert

        Returns:
            JSON-serializable value
        """
        if isinstance(value, np.ndarray):
            return self._to_list(value)
        elif isinstance(value, (np.integer, np.floating)):
            return float(value)
        elif isinstance(value, dict):
            return self._convert_for_json(value)
        else:
            return value

    def _export_optimization_json(self, filepath: Path, result: Dict[str, Any]) -> Path:
        """Export optimization result to JSON."""
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Convert arrays to lists for JSON serialization
        json_result = {}
        # Rule 2: Add explicit bound with assertion
        assert len(result) <= MAX_DICT_ITEMS, (
            f"Result dict has {len(result)} items, exceeds {MAX_DICT_ITEMS}"
        )

        for i, (key, value) in enumerate(result.items()):
            assert i < MAX_DICT_ITEMS, f"Exceeded {MAX_DICT_ITEMS} items"

            # Rule 1: Flatten nesting with helper function
            json_result[key] = self._convert_value_for_json(value)

        # Add metadata
        json_result["_metadata"] = {
            **self.metadata,
            "export_timestamp": datetime.utcnow().isoformat() + "Z",
            "schema_version": self.VERSION,
            "result_type": "optimization",
        }

        with open(filepath, "w") as f:
            json.dump(json_result, f, indent=2)

        return filepath

    def _export_optimization_npz(self, filepath: Path, result: Dict[str, Any]) -> Path:
        """Export optimization result to NPZ."""
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Separate arrays from scalars/metadata
        arrays = {}
        metadata = {
            **self.metadata,
            "export_timestamp": datetime.utcnow().isoformat() + "Z",
            "schema_version": self.VERSION,
        }

        # Rule 2: Add explicit bound with assertion
        assert len(result) <= MAX_DICT_ITEMS, (
            f"Result dict has {len(result)} items, exceeds {MAX_DICT_ITEMS}"
        )

        for i, (key, value) in enumerate(result.items()):
            assert i < MAX_DICT_ITEMS, f"Exceeded {MAX_DICT_ITEMS} items"
            if isinstance(value, np.ndarray):
                arrays[key] = value
            else:
                metadata[key] = value

        # Save metadata as JSON string
        arrays["_metadata"] = np.array([json.dumps(metadata, default=str)])

        np.savez_compressed(filepath, **arrays)

        return filepath

    def export_experiment_config(
        self, filepath: Union[str, Path], config: Dict[str, Any]
    ) -> Path:
        """
        Export experimental configuration to JSON.

        Args:
            filepath: Output file path
            config: Configuration dictionary containing:
                - system_params: qubit parameters (frequencies, T1, T2, etc.)
                - optimization_params: optimizer settings
                - pulse_params: pulse shape parameters
                - etc.

        Returns:
            Path to the exported file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        export_data = {
            "schema_version": self.VERSION,
            "export_timestamp": datetime.utcnow().isoformat() + "Z",
            "config_type": "experiment",
            "metadata": self.metadata,
            "configuration": self._convert_for_json(config),
        }

        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2)

        return filepath

    def export_qiskit_pulse(
        self,
        filepath: Union[str, Path],
        times: np.ndarray,
        amplitudes: np.ndarray,
        dt: float = 1.0,
        pulse_name: str = "custom_pulse",
    ) -> Path:
        """
        Export pulse in a Qiskit-compatible format (basic implementation).

        Note: This is a simplified format. For full Qiskit integration,
        use Qiskit Pulse library directly with this data.

        Args:
            filepath: Output file path
            times: Time points
            amplitudes: Pulse amplitudes (complex or real)
            dt: Time step in ns
            pulse_name: Name for the pulse

        Returns:
            Path to the exported file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Ensure amplitudes are complex
        if not np.iscomplexobj(amplitudes):
            amplitudes_complex = amplitudes.astype(complex)
        else:
            amplitudes_complex = amplitudes

        data = {
            "schema_version": self.VERSION,
            "export_timestamp": datetime.utcnow().isoformat() + "Z",
            "format": "qiskit_pulse_compatible",
            "pulse_name": pulse_name,
            "dt": float(dt),
            "samples": len(amplitudes),
            "real_amplitudes": self._to_list(np.real(amplitudes_complex)),
            "imag_amplitudes": self._to_list(np.imag(amplitudes_complex)),
            "metadata": self.metadata,
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        return filepath

    # Helper methods

    @staticmethod
    def _to_list(arr: np.ndarray) -> List:
        """Convert numpy array to Python list, handling special dtypes."""
        if np.iscomplexobj(arr):
            # Convert complex to list of [real, imag] pairs
            return [[float(x.real), float(x.imag)] for x in arr]
        else:
            return arr.astype(float).tolist()

    @staticmethod
    def _convert_for_json(obj: Any) -> Any:
        """Recursively convert objects to JSON-serializable format."""
        if isinstance(obj, np.ndarray):
            return PulseExporter._to_list(obj)
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: PulseExporter._convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [PulseExporter._convert_for_json(item) for item in obj]
        else:
            return obj


class PulseLoader:
    """
    Load pulse sequences and optimization results from exported files.
    """

    @staticmethod
    def load_pulse_json(filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Load pulse data from JSON file.

        Args:
            filepath: Path to JSON file

        Returns:
            Dictionary containing pulse data and metadata
        """
        filepath = Path(filepath)

        with open(filepath, "r") as f:
            data = json.load(f)

        # Convert lists back to numpy arrays
        pulse_data = data.get("pulse_data", {})
        pulse_data["times"] = np.array(pulse_data.get("times", []))
        pulse_data["amplitudes"] = np.array(pulse_data.get("amplitudes", []))

        if "frequencies" in pulse_data:
            pulse_data["frequencies"] = np.array(pulse_data["frequencies"])
        if "phases" in pulse_data:
            pulse_data["phases"] = np.array(pulse_data["phases"])

        data["pulse_data"] = pulse_data

        return data

    @staticmethod
    def load_pulse_npz(filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Load pulse data from NPZ file.

        Args:
            filepath: Path to NPZ file

        Returns:
            Dictionary containing pulse data and metadata
        """
        filepath = Path(filepath)

        data = np.load(filepath, allow_pickle=True)

        # Extract metadata if present
        result = {}
        metadata = None

        if "_metadata" in data:
            metadata_str = str(data["_metadata"][0])
            metadata = json.loads(metadata_str)
            result["metadata"] = metadata

        # Load arrays
        for key in data.files:
            if key != "_metadata":
                result[key] = data[key]

        return result

    @staticmethod
    def load_pulse_csv(
        filepath: Union[str, Path], delimiter: str = ",", has_header: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Load pulse data from CSV file.

        Args:
            filepath: Path to CSV file
            delimiter: CSV delimiter
            has_header: Whether file has header row

        Returns:
            Dictionary with column data
        """
        filepath = Path(filepath)

        # Read data
        if has_header:
            data = np.genfromtxt(filepath, delimiter=delimiter, names=True)
            # Extract column names and data
            # Rule 2: Add explicit bound with assertion
            assert len(data.dtype.names) <= MAX_DICT_ITEMS, (
                f"CSV has {len(data.dtype.names)} columns, exceeds {MAX_DICT_ITEMS}"
            )
            result = {}
            for i, name in enumerate(data.dtype.names):
                assert i < MAX_DICT_ITEMS, f"Exceeded {MAX_DICT_ITEMS} columns"
                result[name] = data[name]
        else:
            data = np.loadtxt(filepath, delimiter=delimiter)
            # Default column names
            if data.ndim == 1:
                result = {"column_0": data}
            else:
                n_cols = data.shape[1]
                assert n_cols <= MAX_DICT_ITEMS, (
                    f"CSV has {n_cols} columns, exceeds {MAX_DICT_ITEMS}"
                )
                result = {}
                for i in range(n_cols):
                    assert i < MAX_DICT_ITEMS, f"Exceeded {MAX_DICT_ITEMS} columns"
                    result[f"column_{i}"] = data[:, i]

        return result

    @staticmethod
    def load_optimization_result(
        filepath: Union[str, Path], format: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load optimization results from file.

        Args:
            filepath: Path to result file
            format: File format ('json' or 'npz'), auto-detected if None

        Returns:
            Dictionary containing optimization results
        """
        filepath = Path(filepath)

        if format is None:
            format = filepath.suffix.lstrip(".")

        if format == "json":
            with open(filepath, "r") as f:
                data = json.load(f)

            # Convert lists to arrays where appropriate
            for key, value in data.items():
                if isinstance(value, list) and key not in ["_metadata", "metadata"]:
                    try:
                        data[key] = np.array(value)
                    except (ValueError, TypeError):
                        pass  # Keep as list if conversion fails

            return data

        elif format == "npz":
            return PulseLoader.load_pulse_npz(filepath)

        else:
            raise ValueError(f"Unsupported format: {format}")

    @staticmethod
    def load_experiment_config(filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Load experimental configuration from JSON file.

        Args:
            filepath: Path to configuration file

        Returns:
            Configuration dictionary
        """
        filepath = Path(filepath)

        with open(filepath, "r") as f:
            data = json.load(f)

        return data.get("configuration", data)


# Convenience functions


def save_pulse(
    filepath: Union[str, Path],
    times: np.ndarray,
    amplitudes: np.ndarray,
    frequencies: Optional[np.ndarray] = None,
    phases: Optional[np.ndarray] = None,
    format: str = "json",
    **kwargs,
) -> Path:
    """
    Save pulse data to file (convenience function).

    Args:
        filepath: Output file path
        times: Time points
        amplitudes: Pulse amplitudes
        frequencies: Optional frequency modulation
        phases: Optional phase modulation
        format: Export format ('json', 'npz', or 'csv')
        **kwargs: Additional arguments passed to exporter

    Returns:
        Path to saved file
    """
    exporter = PulseExporter()

    if format == "json":
        return exporter.export_pulse_json(
            times, amplitudes, frequencies, phases, filepath, **kwargs
        )
    elif format == "npz":
        return exporter.export_pulse_npz(
            filepath, times, amplitudes, frequencies, phases, **kwargs
        )
    elif format == "csv":
        return exporter.export_pulse_csv(
            filepath, times, amplitudes, frequencies, phases, **kwargs
        )
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_pulse(
    filepath: Union[str, Path], format: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load pulse data from file (convenience function).

    Args:
        filepath: Path to pulse file
        format: File format, auto-detected if None

    Returns:
        Dictionary with pulse data
    """
    filepath = Path(filepath)

    if format is None:
        format = filepath.suffix.lstrip(".")

    loader = PulseLoader()

    if format == "json":
        return loader.load_pulse_json(filepath)
    elif format == "npz":
        return loader.load_pulse_npz(filepath)
    elif format == "csv":
        return loader.load_pulse_csv(filepath)
    else:
        raise ValueError(f"Unsupported format: {format}")


def save_optimization_result(
    filepath: Union[str, Path], result: Dict[str, Any], format: str = "json"
) -> Path:
    """
    Save optimization results (convenience function).

    Args:
        filepath: Output file path
        result: Optimization result dictionary
        format: Export format ('json' or 'npz')

    Returns:
        Path to saved file
    """
    exporter = PulseExporter()
    return exporter.export_optimization_result(filepath, result, format)


def load_optimization_result(
    filepath: Union[str, Path], format: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load optimization results (convenience function).

    Args:
        filepath: Path to result file
        format: File format, auto-detected if None

    Returns:
        Optimization result dictionary
    """
    loader = PulseLoader()
    return loader.load_optimization_result(filepath, format)
