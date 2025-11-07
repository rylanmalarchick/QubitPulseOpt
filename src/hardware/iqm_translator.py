"""
IQM Translator
==============

This module provides the core translation layer between QubitPulseOpt simulations
and IQM Resonance quantum hardware. It converts optimized pulse waveforms (I/Q arrays)
into hardware-executable schedules using the IQM Pulse API.

Classes
-------
IQMTranslator : Main translation engine for QubitPulseOpt → IQM SDK conversion

API Translation
---------------
QubitPulseOpt Asset              →  IQM SDK Target
-------------------              →  --------------
numpy.ndarray (I/Q waveforms)    →  iqm.pulse.CustomIQWaveforms
pulse duration & sample_rate     →  iqm.pulse.ScheduleBuilder timing
target qubit ID                  →  iqm.pulse.Schedule qubit mapping

Example
-------
>>> import numpy as np
>>> from src.hardware.iqm_translator import IQMTranslator
>>>
>>> # Load optimized pulse from QubitPulseOpt
>>> pulse_data = np.load('pulses/optimized_x_gate.npz')
>>> i_waveform = pulse_data['i']
>>> q_waveform = pulse_data['q']
>>>
>>> # Translate to IQM schedule
>>> translator = IQMTranslator()
>>> schedule = translator.create_schedule(
...     i_waveform=i_waveform,
...     q_waveform=q_waveform,
...     target_qubit='QB1',
...     sample_rate=1e9  # 1 GHz
... )
>>>
>>> # Execute on hardware
>>> backend = get_iqm_backend()
>>> result = translator.execute_schedule(schedule, backend, shots=1024)

References
----------
- IQM Pulse API: https://iqm-finland.github.io/iqm-pulse/
- Scope of Work: newscopeofwork.md (Section 1.2, Table 1)
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import json

logger = logging.getLogger(__name__)


class IQMTranslator:
    """
    Translates QubitPulseOpt pulse waveforms to IQM hardware schedules.

    This class bridges the gap between simulation (QubitPulseOpt) and
    hardware execution (IQM Resonance). It handles:
    - I/Q waveform conversion to IQM pulse objects
    - Schedule construction with proper timing
    - Pulse execution on IQM backends
    - Result parsing and fidelity extraction

    Attributes
    ----------
    default_sample_rate : float
        Default AWG sampling rate (Hz), typically 1 GHz for IQM hardware
    waveform_tolerance : float
        Numerical tolerance for waveform amplitude validation

    Example
    -------
    >>> translator = IQMTranslator()
    >>> schedule = translator.create_schedule(i_array, q_array, 'QB1')
    >>> result = translator.execute_schedule(schedule, backend)
    """

    def __init__(self, default_sample_rate: float = 1e9):
        """
        Initialize the IQM translator.

        Parameters
        ----------
        default_sample_rate : float, default=1e9
            Default sampling rate for AWG in Hz (1 GHz is standard for IQM)
        """
        self.default_sample_rate = default_sample_rate
        self.waveform_tolerance = 1e-6

        logger.info(
            f"IQMTranslator initialized (sample_rate={default_sample_rate:.2e} Hz)"
        )

    def load_pulse_from_file(self, filepath: str) -> Dict[str, np.ndarray]:
        """
        Load a QubitPulseOpt pulse from .npz or .json file.

        Parameters
        ----------
        filepath : str
            Path to pulse file (.npz or .json format)

        Returns
        -------
        pulse_data : dict
            Dictionary with keys:
            - 'i': np.ndarray - I channel waveform
            - 'q': np.ndarray - Q channel waveform
            - 'duration': float - Pulse duration (if available)
            - 'sample_rate': float - Sample rate (if available)

        Raises
        ------
        FileNotFoundError
            If file does not exist
        ValueError
            If file format is invalid or required data missing
        """
        logger.info(f"Loading pulse from file: {filepath}")

        if filepath.endswith(".npz"):
            # Load NumPy compressed archive
            try:
                data = np.load(filepath)
                pulse_data = {
                    "i": data["i"] if "i" in data else data.get("I", None),
                    "q": data["q"] if "q" in data else data.get("Q", None),
                }

                # Optional metadata
                if "duration" in data:
                    pulse_data["duration"] = float(data["duration"])
                if "sample_rate" in data:
                    pulse_data["sample_rate"] = float(data["sample_rate"])

                # Validate required fields
                if pulse_data["i"] is None or pulse_data["q"] is None:
                    raise ValueError("NPZ file must contain 'i' and 'q' arrays")

                logger.info(f"Loaded pulse: I/Q shape={pulse_data['i'].shape}")
                return pulse_data

            except Exception as e:
                raise ValueError(f"Failed to load .npz file: {e}") from e

        elif filepath.endswith(".json"):
            # Load JSON format
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)

                pulse_data = {
                    "i": np.array(data["i"]),
                    "q": np.array(data["q"]),
                }

                if "duration" in data:
                    pulse_data["duration"] = float(data["duration"])
                if "sample_rate" in data:
                    pulse_data["sample_rate"] = float(data["sample_rate"])

                logger.info(
                    f"Loaded pulse from JSON: I/Q length={len(pulse_data['i'])}"
                )
                return pulse_data

            except Exception as e:
                raise ValueError(f"Failed to load .json file: {e}") from e

        else:
            raise ValueError(f"Unsupported file format: {filepath}")

    def validate_waveforms(
        self, i_waveform: np.ndarray, q_waveform: np.ndarray
    ) -> Tuple[bool, str]:
        """
        Validate I/Q waveforms for hardware compatibility.

        Parameters
        ----------
        i_waveform : np.ndarray
            In-phase component
        q_waveform : np.ndarray
            Quadrature component

        Returns
        -------
        is_valid : bool
            True if waveforms are valid
        message : str
            Validation message (error description if invalid)

        Notes
        -----
        Checks performed:
        - Both arrays must be 1D numpy arrays
        - Both must have same length
        - Amplitudes should be normalized (typically |I|, |Q| ≤ 1)
        - No NaN or Inf values
        """
        # Check types
        if not isinstance(i_waveform, np.ndarray) or not isinstance(
            q_waveform, np.ndarray
        ):
            return False, "Waveforms must be numpy arrays"

        # Check dimensions
        if i_waveform.ndim != 1 or q_waveform.ndim != 1:
            return (
                False,
                f"Waveforms must be 1D (got {i_waveform.ndim}D and {q_waveform.ndim}D)",
            )

        # Check lengths match
        if len(i_waveform) != len(q_waveform):
            return False, f"I/Q length mismatch: {len(i_waveform)} vs {len(q_waveform)}"

        # Check for NaN/Inf
        if np.any(~np.isfinite(i_waveform)) or np.any(~np.isfinite(q_waveform)):
            return False, "Waveforms contain NaN or Inf values"

        # Check amplitude normalization (warning, not error)
        max_i = np.max(np.abs(i_waveform))
        max_q = np.max(np.abs(q_waveform))
        if (
            max_i > 1.0 + self.waveform_tolerance
            or max_q > 1.0 + self.waveform_tolerance
        ):
            logger.warning(
                f"Waveforms may not be normalized: max|I|={max_i:.3f}, max|Q|={max_q:.3f}"
            )

        # Check if waveforms are too short
        if len(i_waveform) < 2:
            return False, f"Waveforms too short: {len(i_waveform)} samples"

        return True, "Validation passed"

    def create_schedule(
        self,
        i_waveform: np.ndarray,
        q_waveform: np.ndarray,
        target_qubit: str,
        sample_rate: Optional[float] = None,
        gate_name: str = "custom_gate",
    ) -> Any:
        """
        Create an IQM pulse schedule from QubitPulseOpt I/Q waveforms.

        This is the core translation function. It converts numpy arrays
        from QubitPulseOpt into an iqm.pulse.Schedule object that can
        be executed on IQM hardware.

        Parameters
        ----------
        i_waveform : np.ndarray
            In-phase waveform component (1D array)
        q_waveform : np.ndarray
            Quadrature waveform component (1D array)
        target_qubit : str
            Target qubit identifier (e.g., 'QB1', 'QB2')
        sample_rate : float, optional
            AWG sampling rate in Hz. If None, uses default_sample_rate
        gate_name : str, default='custom_gate'
            Name for the custom gate operation

        Returns
        -------
        schedule : iqm.pulse.Schedule
            IQM pulse schedule ready for execution

        Raises
        ------
        ValueError
            If waveform validation fails
        ImportError
            If iqm-pulse library is not installed

        Example
        -------
        >>> i = np.array([0.0, 0.5, 1.0, 0.5, 0.0])
        >>> q = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        >>> schedule = translator.create_schedule(i, q, 'QB1')
        """
        # Validate waveforms
        is_valid, message = self.validate_waveforms(i_waveform, q_waveform)
        if not is_valid:
            raise ValueError(f"Waveform validation failed: {message}")

        logger.info(
            f"Creating schedule: qubit={target_qubit}, "
            f"waveform_length={len(i_waveform)}, gate={gate_name}"
        )

        # Use default sample rate if not provided
        if sample_rate is None:
            sample_rate = self.default_sample_rate

        # Calculate pulse duration
        duration = len(i_waveform) / sample_rate
        logger.debug(f"Pulse duration: {duration * 1e9:.2f} ns")

        try:
            # Import IQM pulse libraries
            from iqm.pulse import Schedule, ScheduleBuilder
            from iqm.pulse.gate_implementation import CustomIQWaveforms

            # Create custom IQ waveforms object
            # This is the direct translation from QubitPulseOpt arrays to IQM API
            custom_waveforms = CustomIQWaveforms(
                wave_i=i_waveform.tolist(),  # Convert numpy to list
                wave_q=q_waveform.tolist(),
                duration=duration,
                name=gate_name,
            )

            # Build the schedule
            builder = ScheduleBuilder()

            # Add the custom gate to the target qubit
            # Note: Actual API may vary - this is based on documentation patterns
            builder.add_gate(
                gate=custom_waveforms,
                qubits=[target_qubit],
                time=0.0,  # Start at t=0
            )

            # Compile to schedule
            schedule = builder.build()

            logger.info(f"Schedule created successfully for {target_qubit}")
            return schedule

        except ImportError as e:
            raise ImportError(
                "iqm-pulse not installed. Install with: pip install iqm-pulse"
            ) from e
        except Exception as e:
            logger.error(f"Failed to create IQM schedule: {e}")
            raise ValueError(f"Schedule creation failed: {e}") from e

    def execute_schedule(
        self, schedule: Any, backend: Any, shots: int = 1024, memory: bool = False
    ) -> Dict[str, Any]:
        """
        Execute an IQM pulse schedule on hardware.

        Parameters
        ----------
        schedule : iqm.pulse.Schedule
            Pulse schedule to execute
        backend : IQMPullaBackend
            IQM hardware backend
        shots : int, default=1024
            Number of measurement shots
        memory : bool, default=False
            If True, return individual shot results

        Returns
        -------
        result : dict
            Execution result with keys:
            - 'counts': Dict[str, int] - Measurement outcome counts
            - 'shots': int - Number of shots executed
            - 'success': bool - Execution success flag
            - 'job_id': str - IQM job identifier
            - 'raw_result': Any - Raw backend result object

        Example
        -------
        >>> backend = manager.get_backend()
        >>> result = translator.execute_schedule(schedule, backend, shots=512)
        >>> print(f"Counts: {result['counts']}")
        """
        logger.info(f"Executing schedule on {backend.name} with {shots} shots")

        try:
            # Execute the schedule
            # Note: Actual execution API depends on iqm-pulla version
            job = backend.run(schedule, shots=shots, memory=memory)

            # Wait for completion
            result_obj = job.result()

            # Parse results
            result = {
                "counts": result_obj.get_counts()
                if hasattr(result_obj, "get_counts")
                else {},
                "shots": shots,
                "success": result_obj.success
                if hasattr(result_obj, "success")
                else True,
                "job_id": job.job_id() if hasattr(job, "job_id") else "unknown",
                "raw_result": result_obj,
            }

            logger.info(f"Execution complete. Job ID: {result['job_id']}")
            return result

        except Exception as e:
            logger.error(f"Schedule execution failed: {e}")
            return {
                "counts": {},
                "shots": shots,
                "success": False,
                "job_id": "failed",
                "error": str(e),
                "raw_result": None,
            }

    def translate_and_execute(
        self,
        pulse_filepath: str,
        target_qubit: str,
        backend: Any,
        shots: int = 1024,
        gate_name: str = "qpo_custom_gate",
    ) -> Dict[str, Any]:
        """
        Complete workflow: load pulse, translate, and execute on hardware.

        This is a convenience method that combines all translation steps.

        Parameters
        ----------
        pulse_filepath : str
            Path to QubitPulseOpt pulse file (.npz or .json)
        target_qubit : str
            Target qubit (e.g., 'QB1')
        backend : IQMPullaBackend
            IQM hardware backend
        shots : int, default=1024
            Number of measurement shots
        gate_name : str, default='qpo_custom_gate'
            Name for the custom gate

        Returns
        -------
        result : dict
            Complete execution result including:
            - 'pulse_data': Loaded pulse information
            - 'schedule': Created IQM schedule
            - 'execution': Execution result from hardware
            - 'success': Overall success flag

        Example
        -------
        >>> result = translator.translate_and_execute(
        ...     'pulses/x_gate_optimized.npz',
        ...     'QB1',
        ...     backend,
        ...     shots=1024
        ... )
        >>> if result['success']:
        ...     print(f"Counts: {result['execution']['counts']}")
        """
        logger.info(f"Full translation workflow: {pulse_filepath} → {target_qubit}")

        try:
            # Step 1: Load pulse
            pulse_data = self.load_pulse_from_file(pulse_filepath)

            # Step 2: Create schedule
            sample_rate = pulse_data.get("sample_rate", self.default_sample_rate)
            schedule = self.create_schedule(
                i_waveform=pulse_data["i"],
                q_waveform=pulse_data["q"],
                target_qubit=target_qubit,
                sample_rate=sample_rate,
                gate_name=gate_name,
            )

            # Step 3: Execute
            execution_result = self.execute_schedule(schedule, backend, shots)

            # Combine all results
            result = {
                "pulse_data": {
                    "filepath": pulse_filepath,
                    "i_shape": pulse_data["i"].shape,
                    "q_shape": pulse_data["q"].shape,
                    "sample_rate": sample_rate,
                },
                "schedule": schedule,
                "execution": execution_result,
                "success": execution_result["success"],
            }

            logger.info(f"Translation workflow completed successfully")
            return result

        except Exception as e:
            logger.error(f"Translation workflow failed: {e}")
            return {
                "pulse_data": {},
                "schedule": None,
                "execution": {"success": False, "error": str(e)},
                "success": False,
                "error": str(e),
            }

    def __repr__(self) -> str:
        """String representation of the translator."""
        return f"IQMTranslator(sample_rate={self.default_sample_rate:.2e} Hz)"
