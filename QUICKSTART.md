# QubitPulseOpt - Quick Start Guide

Get up and running with quantum optimal control in 5 minutes.

---

## Prerequisites

- Python 3.8 or higher
- Git
- 2 GB free disk space

---

## Installation

```bash
# Clone repository
git clone https://github.com/rylanmalarchick/QubitPulseOpt.git
cd QubitPulseOpt

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements-dev.txt

# Verify installation
pytest tests/ -v --maxfail=3
```

---

## Your First Optimization

### Option 1: Run Demo Script

```bash
# Phase 1: Hamiltonian simulation
python examples/phase1_demo.py

# Phase 2: GRAPE pulse optimization (recommended starting point)
python examples/phase2_demo.py

# Phase 3: DRAG pulse implementation
python examples/phase3_demo.py

# Phase 4: Benchmarking
python examples/phase4_demo.py
```

### Option 2: Interactive Python

```python
from src.optimization import GRAPEOptimizer
from src.hamiltonian import DriftHamiltonian, ControlHamiltonian

# Define qubit (5 GHz superconducting transmon)
drift = DriftHamiltonian(omega_0=5.0)
control = ControlHamiltonian()

# Set realistic decoherence
T1 = 50e-6  # 50 microseconds
T2 = 70e-6  # 70 microseconds

# Initialize GRAPE optimizer for X-gate
optimizer = GRAPEOptimizer(
    drift_hamiltonian=drift,
    control_hamiltonian=control,
    target_gate='X',
    duration=20e-9,  # 20 nanoseconds
    num_steps=100,
    T1=T1,
    T2=T2
)

# Optimize!
result = optimizer.optimize(max_iterations=200)

print(f"✓ Fidelity: {result['fidelity']*100:.2f}%")
print(f"✓ Converged in {result['iterations']} iterations")

# Visualize
optimizer.plot_convergence()
optimizer.plot_pulse()
```

**Expected output**: Fidelity > 99.5% in ~150 iterations

---

## Understanding the Results

After running `phase2_demo.py`, you'll see:

1. **Pulse Shape Plot**: Complex, non-intuitive waveform discovered by GRAPE
   - Compare to simple Gaussian baseline
   - Notice pre-compensating features and error-canceling undershoots

2. **Convergence Plot**: Fidelity vs. iteration
   - Rapid initial improvement (first 50 iterations)
   - Fine-tuning phase (iterations 50-150)
   - Final convergence to >99.9%

3. **Bloch Sphere Trajectory**: Quantum state evolution
   - Starting point: |0⟩ (north pole)
   - Ending point: |1⟩ (south pole)
   - Path shows perfect X-gate rotation

---

## Hardware Integration (Advanced)

### Prerequisites

```bash
# Install hardware dependencies
pip install -r requirements-hardware.txt

# Configure IQM credentials (DO NOT COMMIT THIS FILE)
echo "IQM_TOKEN=your_token_here" > .env
```

### Test Hardware Connection

```python
from src.hardware import IQMBackendManager

# Initialize (starts with emulator by default)
manager = IQMBackendManager()
backend = manager.get_backend(use_emulator=True)

print(f"✓ Connected to: {backend}")
```

### Run Hardware Validation

```bash
# Dry-run (emulator, no credits consumed)
python scripts/hardware_validation_async.py --dry-run

# Real hardware (requires IQM credits)
# python scripts/hardware_validation_async.py --submit-only
```

---

## Project Structure

```
QubitPulseOpt/
├── src/
│   ├── hamiltonian/       # Quantum system modeling
│   ├── optimization/      # GRAPE algorithm
│   ├── pulses/           # Pulse generators (DRAG, Gaussian)
│   └── hardware/         # IQM hardware integration
├── tests/                # 570+ unit/integration tests
├── docs/                 # Documentation & figures
├── phase1_demo.py        # Hamiltonian demo
├── phase2_demo.py        # GRAPE optimization demo ⭐
├── phase3_demo.py        # DRAG pulse demo
└── phase4_demo.py        # Benchmarking demo
```

---

## Common Tasks

### Generate Figures

```bash
python scripts/generate_figures.py
# Output: docs/figures/*.png
```

### Run Tests

```bash
# Fast unit tests
pytest tests/ -v -m "not slow"

# Full test suite with coverage
pytest tests/ --cov=src --cov-report=html
```

### View Documentation

```bash
# Open technical supplement
open docs/TECHNICAL_SUPPLEMENT.md

# View generated figures
open docs/figures/
```

---

## Troubleshooting

### Import Errors

```bash
# Ensure virtual environment is activated
which python  # Should show venv/bin/python

# Reinstall dependencies
pip install -r requirements-dev.txt --force-reinstall
```

### Low Fidelity Results

- Check coherence times (T1, T2) are realistic (10-100 µs)
- Increase number of time steps (num_steps=200)
- Extend optimization iterations (max_iterations=300)
- Verify target gate specification

### Hardware Connection Issues

```bash
# Test token is configured
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('✓' if os.getenv('IQM_TOKEN') else '✗ Missing')"

# Use emulator for development
backend = manager.get_backend(use_emulator=True)
```

---

## Next Steps

1. **Explore Demos**: Run all four phase demos to see different features
2. **Read Documentation**: Check `docs/TECHNICAL_SUPPLEMENT.md` for theory
3. **View Figures**: Examine `docs/figures/` for visualization examples
4. **Modify Parameters**: Experiment with different T1/T2, gate times, target gates
5. **Hardware Integration**: Set up IQM connection (see `HARDWARE_REFERENCE.md`)

---

## Key Resources

- **Main Documentation**: `README.md`
- **Technical Details**: `docs/TECHNICAL_SUPPLEMENT.md`
- **Hardware Guide**: `HARDWARE_REFERENCE.md`
- **API Reference**: Docstrings in `src/` modules
- **Test Examples**: `tests/` directory

---

## Support

**Found a bug?** Open an issue on GitHub  
**Have a question?** Check `docs/` or reach out via email

---

**You're ready to optimize quantum gates! 🚀**

Start with: `python examples/phase2_demo.py`
