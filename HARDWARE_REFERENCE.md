# QubitPulseOpt Hardware Integration - Quick Reference Card

## 🚀 Quick Setup (5 Minutes)

```bash
# 1. Install dependencies
./setup_hardware.sh

# 2. Configure token
echo "IQM_TOKEN=your_token_here" > .env

# 3. Verify installation
python test_hardware_setup.py

# 4. Test Phase 1
cd notebooks/hardware
jupyter notebook 01_phase1_hardware_handshake.ipynb
```

---

## 📦 Essential Imports

```python
# Hardware integration
from src.hardware import (
    IQMBackendManager,
    IQMTranslator,
    HardwareCharacterizer
)

# Agent tools
from src.agent.tools import AgentTools

# QubitPulseOpt core
from src.config import Config
from src.optimization.grape import GRAPEOptimizer
from src.hamiltonian.single_qubit import SingleQubitSystem
```

---

## 🔌 Backend Connection

```python
# Initialize backend manager (reads IQM_TOKEN from .env)
backend_mgr = IQMBackendManager()

# Get emulator backend (FREE - no credits)
backend = backend_mgr.get_backend(use_emulator=True)

# Get real hardware backend (PAID - uses credits)
backend = backend_mgr.get_backend(use_emulator=False)

# Get topology
topology = backend_mgr.get_hardware_topology()
print(f"Qubits: {topology['qubits']}")
print(f"Backend: {topology['backend_name']}")
```

---

## 📊 Hardware Characterization

```python
# Initialize characterizer
characterizer = HardwareCharacterizer(backend)

# Measure T1 (energy relaxation)
t1_result = characterizer.run_t1_experiment(
    qubit='QB1',
    shots=1024,
    use_emulator=True  # Change to False for real hardware
)
print(f"T1 = {t1_result['value']*1e6:.2f} μs")

# Measure T2 (dephasing - Hahn echo)
t2_result = characterizer.run_t2_experiment(
    qubit='QB1',
    shots=1024,
    method='hahn',
    use_emulator=True
)
print(f"T2 = {t2_result['value']*1e6:.2f} μs")

# Measure Rabi frequency
rabi_result = characterizer.run_rabi_experiment(
    qubit='QB1',
    shots=1024,
    use_emulator=True
)
print(f"Rabi rate = {rabi_result['rate']/1e6:.2f} MHz")

# Full characterization suite
results = characterizer.characterize_qubit(
    qubit='QB1',
    shots=2048,
    use_emulator=True
)
print(results['summary'])
```

---

## 🔄 Closed-Loop Workflow

```python
# 1. Connect to hardware
backend_mgr = IQMBackendManager()
backend = backend_mgr.get_backend(use_emulator=True)

# 2. Characterize qubit (Hardware-to-Sim)
characterizer = HardwareCharacterizer(backend)
results = characterizer.characterize_qubit(qubit='QB1', shots=1024)
t1 = results['summary']['T1']
t2 = results['summary']['T2']

# 3. Update QubitPulseOpt config
config = Config()
config.load_from_yaml('config/default_config.yaml')
config.set('system.decoherence.T1', t1)
config.set('system.decoherence.T2', t2)
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
    pulse_filepath='pulses/x_gate_optimized.npz',
    target_qubit='QB1',
    backend=backend,
    shots=1024
)
print(f"Hardware execution: {exec_result['success']}")
```

---

## 🛠️ Agent Tools Usage

```python
# Initialize agent tools
tools = AgentTools(backend_manager=backend_mgr)

# Get hardware topology
topology = tools.get_hardware_topology()

# Run characterization
t1_result = tools.run_characterization_experiment(
    experiment_name='T1',
    qubits=['QB1'],
    experiment_options={'shots': 1024},
    run_on_emulator=True
)

# Update simulation config
tools.update_simulation_config(
    params={
        'system.decoherence.T1': 39.7e-6,
        'system.decoherence.T2': 25.2e-6,
    },
    new_config_name='config/hardware_run_001.yaml'
)

# Run optimization
tools.run_qpo_optimization(
    config_file='config/hardware_run_001.yaml',
    target_gate='X',
    output_artifact='pulses/x_gate_001.npz'
)

# Execute pulse on hardware
tools.translate_and_execute_pulse(
    pulse_artifact='pulses/x_gate_001.npz',
    qubits=['QB1'],
    run_on_emulator=False
)
```

---

## 🔄 Pulse Translation

```python
# Initialize translator
translator = IQMTranslator()

# Method 1: From file
result = translator.translate_and_execute(
    pulse_filepath='pulses/my_pulse.npz',
    target_qubit='QB1',
    backend=backend,
    shots=1024
)

# Method 2: From arrays
import numpy as np
i_waveform = np.array([...])  # I channel
q_waveform = np.array([...])  # Q channel

schedule = translator.create_schedule(
    i_waveform=i_waveform,
    q_waveform=q_waveform,
    target_qubit='QB1',
    sample_rate=1e9,  # 1 GHz
    gate_name='custom_x'
)

result = translator.execute_schedule(schedule, backend, shots=1024)
print(f"Counts: {result['counts']}")
```

---

## 💰 Cost Management

### Always Test on Emulator First!

```python
# ✅ FREE: No credits consumed
backend = backend_mgr.get_backend(use_emulator=True)
characterizer = HardwareCharacterizer(backend)
t1 = characterizer.run_t1_experiment(qubit=0, use_emulator=True)

# ⚠️ PAID: Consumes IQM credits
backend = backend_mgr.get_backend(use_emulator=False)
t1 = characterizer.run_t1_experiment(qubit=0, use_emulator=False)
```

### Minimize Shots

```python
# Start with minimal shots for testing
result = characterizer.run_t1_experiment(qubit=0, shots=256)

# Increase only if you need better accuracy
result = characterizer.run_t1_experiment(qubit=0, shots=2048)
```

### Estimated Costs (Real Hardware)

| Operation | Shots | Credits |
|-----------|-------|---------|
| T1 measurement | 1024 | ~30 |
| T2 measurement | 1024 | ~45 |
| Pulse execution | 1024 | ~30 |
| RB experiment | 512 | ~250 |
| **Full workflow** | - | **~400-500** |

---

## 🐛 Common Issues & Fixes

### "IQM_TOKEN not found"
```bash
# Create .env file
echo "IQM_TOKEN=your_actual_token" > .env

# Verify it's not in git
grep ".env" .gitignore
```

### "ImportError: No module named iqm"
```bash
pip install -r requirements-hardware.txt
```

### "Connection timeout"
```python
# Test with emulator first
backend = backend_mgr.get_backend(use_emulator=True)
```

### Verify installation
```bash
python test_hardware_setup.py
```

---

## 📁 File Locations

```
QubitPulseOpt/
├── .env                    # IQM_TOKEN (NEVER commit!)
├── setup_hardware.sh       # Installation script
├── test_hardware_setup.py  # Verification script
│
├── src/
│   ├── hardware/          # Hardware integration modules
│   │   ├── iqm_backend.py
│   │   ├── iqm_translator.py
│   │   └── characterization.py
│   │
│   └── agent/             # Agent tools
│       └── tools.py
│
├── notebooks/hardware/    # Testing notebooks
│   └── 01_phase1_hardware_handshake.ipynb
│
├── config/               # Configuration files
├── pulses/              # Pulse artifacts (.npz)
└── agent_logs/          # Agent execution logs
```

---

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| `HARDWARE_QUICKSTART.md` | 5-minute getting started |
| `src/hardware/README.md` | Complete API reference |
| `HARDWARE_INTEGRATION_STATUS.md` | Implementation status |
| `DELIVERY_SUMMARY.md` | What was built |
| `newscopeofwork.md` | Original scope of work |

---

## 🔐 Security Checklist

- [ ] `.env` file created with IQM_TOKEN
- [ ] `.env` is in `.gitignore`
- [ ] Never commit `.env` to git
- [ ] Never share IQM_TOKEN publicly
- [ ] Never log or print token value
- [ ] Use emulator for development
- [ ] Minimize shot counts

---

## ✅ Testing Workflow

### Phase 1: Connection
```bash
cd notebooks/hardware
jupyter notebook 01_phase1_hardware_handshake.ipynb
# Execute all cells
```

### Phase 2: Characterization
```python
characterizer = HardwareCharacterizer(backend)
results = characterizer.characterize_qubit(qubit='QB1')
# Verify T1, T2, Rabi measurements
```

### Phase 3: Pulse Execution
```python
translator = IQMTranslator()
result = translator.translate_and_execute(
    pulse_filepath='pulses/test_pulse.npz',
    target_qubit='QB1',
    backend=backend
)
# Verify execution succeeds
```

---

## 🎯 Target Metrics

| Metric | Target | Phase |
|--------|--------|-------|
| Authentication success | 100% | 1 |
| Connection latency | < 5s | 1 |
| T1 accuracy | ±10% | 2 |
| T2 accuracy | ±10% | 2 |
| Hardware fidelity | > 99.9% | 4 |
| Sim-to-real gap | < 0.1% | 4 |

---

## 🔗 External Links

- IQM Portal: https://resonance.iqm.fi/
- IQM Pulse Docs: https://iqm-finland.github.io/iqm-pulse/
- IQM Pulla Docs: https://iqm-finland.github.io/iqm-pulla/
- Qiskit Experiments: https://qiskit.org/ecosystem/experiments/

---

## 💡 Pro Tips

1. **Always test on emulator first** before using real hardware
2. **Start with low shot counts** (256-512) and increase if needed
3. **Check credit balance** regularly on IQM portal
4. **Run verification script** after any configuration changes
5. **Keep notebooks** for reproducible workflows
6. **Log everything** - agent_logs/ directory is your friend

---

**Quick Command Reference**

```bash
# Setup
./setup_hardware.sh

# Verify
python test_hardware_setup.py

# Test
jupyter notebook notebooks/hardware/01_phase1_hardware_handshake.ipynb

# Check logs
ls -lh agent_logs/
```

---

**Status**: 🟢 Phase 1 Ready | 🟡 Phase 2-3 Infrastructure Ready | 🔴 Phase 4-5 Planned

**For detailed information, see the full documentation!**

---

*QubitPulseOpt Hardware Integration v1.0.0*