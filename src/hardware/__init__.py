"""
Hardware Integration Module
============================

This module provides the integration layer between QubitPulseOpt simulations
and real quantum hardware, specifically IQM Resonance quantum processing units.

The primary function is to translate optimized pulse waveforms from QubitPulseOpt
into hardware-executable schedules using the IQM SDK (iqm-pulse, iqm-pulla).

Modules
-------
iqm_translator : Translates QubitPulseOpt I/Q arrays to IQM pulse schedules
iqm_backend : Wrapper for IQM hardware backend access and authentication
characterization : Hardware characterization utilities using qiskit-experiments

Key Classes
-----------
IQMTranslator : Main translation engine (QubitPulseOpt → IQM SDK)
IQMBackendManager : Manages authentication and backend access
HardwareCharacterizer : Runs T1, T2, Rabi experiments on hardware

API Translation Map
-------------------
QubitPulseOpt Asset              →  IQM SDK Target
-------------------              →  --------------
numpy.ndarray (I/Q waveforms)    →  iqm.pulse.CustomIQWaveforms
pulse duration & sample_rate     →  iqm.pulse.ScheduleBuilder timing
GRAPEResult.final_fidelity       →  Observation log (for comparison)
config/default_config.yaml       →  Hardware-calibrated config

Example
-------
>>> from src.hardware import IQMTranslator, IQMBackendManager
>>> # Load optimized pulse from QubitPulseOpt
>>> pulse_data = np.load('pulses/optimized_x_gate.npz')
>>> i_waveform = pulse_data['i']
>>> q_waveform = pulse_data['q']
>>>
>>> # Translate and execute on hardware
>>> backend_mgr = IQMBackendManager()
>>> backend = backend_mgr.get_backend()
>>> translator = IQMTranslator()
>>> schedule = translator.create_schedule(i_waveform, q_waveform, qubit='QB1')
>>> result = translator.execute_schedule(schedule, backend)

Security Note
-------------
This module uses the IQM_TOKEN environment variable for authentication.
The token should be stored in a .env file (which MUST be in .gitignore).
Never hardcode credentials or log the token value.

References
----------
- IQM Pulse API: https://iqm-finland.github.io/iqm-pulse/
- IQM Pulla: https://iqm-finland.github.io/iqm-pulla/
- Scope of Work: newscopeofwork.md (Part 1, Section 1.2)
"""

from .iqm_translator import IQMTranslator
from .iqm_backend import IQMBackendManager
from .characterization import HardwareCharacterizer

__all__ = [
    "IQMTranslator",
    "IQMBackendManager",
    "HardwareCharacterizer",
]

__version__ = "1.0.0"
__author__ = "QubitPulseOpt Development Team"
