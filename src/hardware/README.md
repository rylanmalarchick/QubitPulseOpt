# Hardware Integration Module

This module provides the integration layer between QubitPulseOpt simulations and real quantum hardware on the IQM Resonance platform.

## Overview

The hardware integration enables a **closed-loop "sim-to-real" workflow**:

1. **Characterize** live hardware parameters (T1, T2, frequency)
2. **Update** QubitPulseOpt simulation with real-world data
3. **Optimize** custom pulses using hardware-aware configuration
4. **Execute** optimized pulses on real quantum hardware
5. **Benchmark** actual fidelity using Randomized Benchmarking
6. **Analyze** sim-to-real gap and iterate

## Module Structure

```
src/hardware/
├── __init__.py                  # Module exports
├── README.md                    # This file
├── iqm_backend.py              # IQM Resonance backend management
├── iqm_translator.py           # QubitPulseOpt → IQM pulse translation
└── characterization.py         # Hardware characterization (T1, T2, Rabi)
```

## Core Components

### 1. IQM Backend Manager (`iqm_backend.py`)

Manages authentication and connection to IQM Resonance quantum hardware.

**Key Features:**
- Secure token-based authentication via environment variables
- Backend selection (real hardware vs. emulator)
- Hardware topology queries
- Connection verification

**Example:**
```python
from src.hardware.iqm_backend import IQMBackendManager

# Initialize (reads IQM_TOKEN from environment)
manager = IQMBackendManager()

# Get hardware backend
backend = manager.get_backend(use_emulator=False)

# Get topology information
topology = manager.get_hardware_topology()
print(f"Available qubits: {topology['qubits']}")
```

### 2. IQM Translator (`iqm_translator.py`)

Translates optimized pulse waveforms from QubitPulseOpt into hardware-executable schedules.

**Translation Map:**

| QubitPulseOpt Asset | IQM SDK Target |
|---------------------|----------------|
| `numpy.ndarray` (I/Q waveforms) | `iqm.pulse.CustomIQWaveforms` |
| Pulse duration & sample_rate | `iqm.pulse.ScheduleBuilder` timing |
| Target qubit ID | `iqm.pulse.Schedule` qubit mapping |

**Example:**
```python
from src.hardware.iqm_translator import IQMTranslator
import numpy as np

# Load optimized pulse from QubitPulseOpt
pulse_data = np.load('pulses/optimized_x_gate.npz')

# Translate and execute
translator = IQMTranslator()
result = translator.translate_and_execute(
    pulse_filepath='pulses/optimized_x_gate.npz',
    target_qubit='QB1',
    backend=backend,
    shots=1024
)
print(f"Execution success: {result['success']}")
```

### 3. Hardware Characterizer (`characterization.py`)

Runs standard quantum hardware characterization experiments using `qiskit-experiments`.

**Supported Experiments:**
- **T1**: Energy relaxation time measurement
- **T2Hahn**: Dephasing time (Hahn echo)
- **T2Ramsey**: Dephasing time (Ramsey interferometry)
- **Rabi**: Rabi oscillation frequency

**Example:**
```python
from src.hardware.characterization import HardwareCharacterizer

# Initialize
characterizer = HardwareCharacterizer(backend)

# Measure T1
t1_result = characterizer.run_t1_experiment(qubit='QB1', shots=1024)
print(f"T1 = {t1_result['value']*1e6:.2f} μs")

# Measure T2
t2_result = characterizer.run_t2_experiment(qubit='QB1', method='hahn')
print(f"T2 = {t2_result['value']*1e6:.2f} μs")

# Full characterization suite
results = characterizer.characterize_qubit(qubit='QB1')
print(f"Summary: T1={results['summary']['T1']:.2e} s")
```

## Installation

### Required Dependencies

Install hardware integration dependencies:

```bash
pip install -r requirements-hardware.txt
```

Key packages:
- `iqm-client>=17.0` - IQM authentication and API access
- `iqm-pulse>=8.0` - Pulse definition library
- `iqm-pulla[qiskit]>=6.0` - Pulse execution with Qiskit integration
- `qiskit>=1.0.0` - Qiskit framework
- `qiskit-experiments>=0.6.0` - Characterization experiments
- `python-dotenv>=1.0.0` - Environment variable management

### Authentication Setup

1. Create a `.env` file in the project root:
```bash
IQM_TOKEN=your_iqm_api_token_here
```

2. **IMPORTANT**: Ensure `.env` is in `.gitignore`:
```bash
echo ".env" >> .gitignore
```

3. The token will be automatically loaded by the `IQMBackendManager`.

**Security Notes:**
- NEVER commit the `.env` file to version control
- NEVER hardcode the IQM_TOKEN in source code
- NEVER log or print the token value

## Usage Workflow

### Complete Closed-Loop Example

```python
from src.hardware import IQMBackendManager, IQMTranslator, HardwareCharacterizer
from src.config import Config
from src.optimization.grape import GRAPEOptimizer
from src.hamiltonian.single_qubit import SingleQubitSystem

# 1. Initialize hardware connection
backend_mgr = IQMBackendManager()
backend = backend_mgr.get_backend(use_emulator=False)  # Use real hardware

# 2. Characterize hardware (Hardware-to-Sim)
characterizer = HardwareCharacterizer(backend)
results = characterizer.characterize_qubit(qubit='QB1', shots=2048)

t1_measured = results['summary']['T1']
t2_measured = results['summary']['T2']
print(f"Measured: T1={t1_measured*1e6:.2f} μs, T2={t2_measured*1e6:.2f} μs")

# 3. Update simulation config
config = Config()
config.load_from_yaml('config/default_config.yaml')
config.set('system.decoherence.T1', t1_measured)
config.set('system.decoherence.T2', t2_measured)
config.save_to_yaml('config/hardware_calibrated.yaml')

# 4. Optimize pulse with hardware-aware config
system = SingleQubitSystem(config)
target_unitary = system.sigma_x()  # X-gate
optimizer = GRAPEOptimizer(system, target_unitary, config)
result = optimizer.optimize()

print(f"Simulated fidelity: {result.final_fidelity:.5f}")

# 5. Translate and execute on hardware
translator = IQMTranslator()
exec_result = translator.translate_and_execute(
    pulse_filepath='pulses/optimized_x_gate.npz',
    target_qubit='QB1',
    backend=backend,
    shots=1024
)

print(f"Hardware execution: {exec_result['success']}")

# 6. Benchmark with Randomized Benchmarking (RB)
# (Full RB implementation pending - see Phase 4 in scope of work)
```

## Testing

### Emulator Testing (No Credits)

Always test with the emulator first to avoid consuming hardware credits:

```python
# Use emulator backend
backend = backend_mgr.get_backend(use_emulator=True)

# All operations work the same, but don't consume credits
characterizer = HardwareCharacterizer(backend)
t1_result = characterizer.run_t1_experiment(
    qubit=0, 
    shots=1024, 
    use_emulator=True
)
```

### Verify Connection

```python
backend_mgr = IQMBackendManager()
if backend_mgr.verify_connection():
    print("✓ Connection to IQM Resonance verified")
else:
    print("✗ Connection failed")
```

## Cost Management

Hardware execution consumes IQM credits. Best practices:

1. **Always test on emulator first**: Use `use_emulator=True`
2. **Minimize shots**: Start with low shot counts (256-512)
3. **Use dry runs**: Verify workflows before real execution
4. **Monitor credit usage**: Track costs in agent logs

**Estimated Costs** (approximate):
- T1 experiment (1024 shots): ~30 credits
- T2 experiment (1024 shots): ~45 credits
- Pulse execution (1024 shots): ~30 credits
- RB experiment (5 lengths, 10 samples, 512 shots): ~250 credits

## Error Handling

The module includes comprehensive error handling:

```python
try:
    backend = backend_mgr.get_backend()
    result = characterizer.run_t1_experiment(qubit='QB1')
    
    if result['success']:
        t1_value = result['value']
    else:
        print(f"Experiment failed: {result.get('error')}")
        
except Exception as e:
    print(f"Hardware error: {e}")
    # Fall back to emulator or default values
```

## API Reference

### IQMBackendManager

**Constructor:**
```python
IQMBackendManager(dotenv_path: Optional[str] = None)
```

**Methods:**
- `get_backend(backend_name: Optional[str] = None, use_emulator: bool = False)` → Backend
- `get_hardware_topology()` → Dict[str, Any]
- `get_available_backends()` → List[str]
- `verify_connection()` → bool

### IQMTranslator

**Constructor:**
```python
IQMTranslator(default_sample_rate: float = 1e9)
```

**Methods:**
- `create_schedule(i_waveform, q_waveform, target_qubit, sample_rate, gate_name)` → Schedule
- `execute_schedule(schedule, backend, shots, memory)` → Dict[str, Any]
- `translate_and_execute(pulse_filepath, target_qubit, backend, shots)` → Dict[str, Any]
- `load_pulse_from_file(filepath)` → Dict[str, np.ndarray]
- `validate_waveforms(i_waveform, q_waveform)` → Tuple[bool, str]

### HardwareCharacterizer

**Constructor:**
```python
HardwareCharacterizer(backend, default_shots: int = 1024, analysis_timeout: float = 300.0)
```

**Methods:**
- `run_t1_experiment(qubit, shots, delays, use_emulator)` → Dict[str, Any]
- `run_t2_experiment(qubit, shots, method, delays, use_emulator)` → Dict[str, Any]
- `run_rabi_experiment(qubit, shots, amplitudes, use_emulator)` → Dict[str, Any]
- `characterize_qubit(qubit, shots, experiments, use_emulator)` → Dict[str, Any]

## Troubleshooting

### "IQM_TOKEN not found"
- Ensure `.env` file exists in project root
- Verify the file contains `IQM_TOKEN=your_token_here`
- Check that `python-dotenv` is installed

### "iqm-pulla not installed"
```bash
pip install iqm-pulla[qiskit]
```

### "qiskit-experiments not installed"
```bash
pip install qiskit-experiments
```

### Connection timeout
- Verify internet connection
- Check IQM Resonance service status
- Verify token is valid and not expired

### Low fidelity on hardware
- Re-run characterization with higher shot count
- Check for parameter drift (re-characterize)
- Verify pulse translation is correct
- Consider crosstalk and noise sources

## Phase-by-Phase Development

This module is developed incrementally following the scope of work:

- [x] **Phase 1**: Hardware handshake and connection verification
- [ ] **Phase 2**: Open-loop characterization (T1, T2, Rabi)
- [ ] **Phase 3**: Pulse translation and execution
- [ ] **Phase 4**: Closed-loop optimization
- [ ] **Phase 5**: DRAG pulse validation

See `notebooks/hardware/` for phase-specific implementation notebooks.

## References

- **IQM Pulse API**: https://iqm-finland.github.io/iqm-pulse/
- **IQM Pulla**: https://iqm-finland.github.io/iqm-pulla/
- **IQM Client**: https://iqm-finland.github.io/iqm-client/
- **Qiskit Experiments**: https://qiskit.org/ecosystem/experiments/
- **Scope of Work**: `newscopeofwork.md`

## Contributing

When adding new hardware features:

1. Add appropriate error handling and logging
2. Include emulator testing support
3. Document credit costs
4. Add security checks for token handling
5. Update this README

## License

This module is part of QubitPulseOpt and follows the same license (MIT).

---

**Author**: QubitPulseOpt Development Team  
**Version**: 1.0.0  
**Last Updated**: 2025