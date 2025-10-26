# QubitPulseOpt: Quantum Optimal Control for High-Fidelity Gates

<div align="center">

![QubitPulseOpt](https://img.shields.io/badge/Quantum-Control-blueviolet?style=for-the-badge)
[![Tests](https://github.com/rylanmalarchick/QubitPulseOpt/workflows/Tests/badge.svg)](https://github.com/rylanmalarchick/QubitPulseOpt/actions/workflows/tests.yml)
[![Compliance](https://github.com/rylanmalarchick/QubitPulseOpt/workflows/Compliance/badge.svg)](https://github.com/rylanmalarchick/QubitPulseOpt/actions/workflows/compliance.yml)
[![codecov](https://codecov.io/gh/rylanmalarchick/QubitPulseOpt/branch/main/graph/badge.svg)](https://codecov.io/gh/rylanmalarchick/QubitPulseOpt)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Power of 10](https://img.shields.io/badge/Power%20of%2010-97.5%25-success)](docs/POWER_OF_10_COMPLIANCE.md)

**Professional-grade quantum pulse optimization framework implementing GRAPE/Krotov methods for high-fidelity single-qubit gates with comprehensive noise modeling and benchmarking.**

[Features](#-features) • [Quick Start](#-quick-start) • [Documentation](#-documentation) • [Results](#-key-results) • [Contributing](#-contributing)

</div>

---

## Why This Project?

Quantum computers promise exponential speedups, but their power hinges on **gate fidelity**—the ability to execute quantum operations with minimal error. In real hardware, qubits suffer from:

- **Decoherence**: T1/T2 decay times as short as 10-100 μs
- **Control noise**: Amplitude and frequency fluctuations in driving fields  
- **Leakage**: Unwanted transitions to non-computational states

**QubitPulseOpt** addresses these challenges using optimal control theory to design electromagnetic pulses that:

Achieve **99.9%+ fidelity** for single-qubit gates  
Remain robust under realistic noise (T1=10μs, T2=20μs)  
Complete in **20-50 ns** (compatible with gate-based quantum algorithms)  
Meet **NASA JPL Power-of-10** coding standards for safety-critical systems

### Real-World Impact

- **NISQ Devices**: Optimized gates reduce error rates → longer coherent computations
- **Quantum Error Correction**: High-fidelity gates lower QEC overhead  
- **Algorithm Development**: Reliable primitives enable focus on higher-level protocols

---

## Key Results

| Metric | Value | Benchmark |
|--------|-------|-----------|
| **X-Gate Fidelity** | 99.94% | Industry target: >99.9% |
| **Gate Duration** | 20 ns | Typical: 20-50 ns |
| **T1 Tolerance** | 10 μs | Superconducting qubits |
| **T2 Tolerance** | 20 μs | State-of-the-art |
| **Test Coverage** | 95.8% | 573+ tests passing |
| **Power-of-10 Compliance** | 97.5% | NASA/JPL standards |
| **Optimization Speed** | <50 iterations | GRAPE convergence |

---

## Features

### Core Capabilities

- **Pulse Design**: Gaussian, DRAG, BB1, CORPSE, SK1 composite pulses
- **Optimization Algorithms**: GRAPE (gradient ascent), Krotov method, CRAB
- **Noise Modeling**: Lindblad master equation with T1/T2 decoherence
- **Gate Library**: X, Y, Z, H, T gates + arbitrary single-qubit rotations
- **Benchmarking**: Randomized benchmarking, process tomography, filter functions
- **Visualization**: Interactive dashboards, Bloch sphere animations, publication-quality plots

### Advanced Features

- **Robustness Analysis**: Filter function formalism for noise susceptibility
- **Parameter Sweeps**: Automated fidelity vs. T1/T2/amplitude/frequency
- **Composite Pulses**: Error-suppressing sequences (BB1, CORPSE, SK1)
- **Hardware Export**: Pulse sequences in JSON/NPZ/AWG-compatible formats
- **Performance Profiling**: Optimized with NumPy/SciPy, parallel execution support

### Software Engineering

- **573+ Tests**: Comprehensive pytest suite with 95.8% coverage
- **CI/CD Pipeline**: Automated testing on Python 3.9-3.12, linting, compliance checks
- **Pre-commit Hooks**: Black, isort, flake8, Power-of-10 enforcement
- **Documentation**: 8 Jupyter notebooks, LaTeX science document, API docs
- **Safety-Critical Standards**: NASA JPL Power-of-10 compliant (97.5%)

---

## Quick Start

### 5-Minute Setup

```bash
# Clone repository
git clone https://github.com/rylanmalarchick/QubitPulseOpt.git
cd QubitPulseOpt

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install qutip numpy scipy matplotlib jupyter pytest

# Verify installation
pytest tests/unit -v -m "not slow" --maxfail=3

# Launch interactive notebooks
jupyter notebook
```

**First notebook to try:** `notebooks/08_end_to_end_workflow.ipynb`

### Example: Optimize an X-Gate

```python
from src.optimization.grape_optimizer import GRAPEOptimizer
from src.hamiltonian.single_qubit import SingleQubitHamiltonian
import qutip as qt

# Define system
omega_d = 5.0 * 2 * np.pi  # 5 GHz qubit
H_drift = -0.5 * omega_d * qt.sigmaz()
H_ctrl = [qt.sigmax()]  # Control Hamiltonian

# Target: X-gate (π rotation around X-axis)
U_target = qt.sigmax()

# Initialize GRAPE optimizer
optimizer = GRAPEOptimizer(
    H_drift=H_drift,
    H_ctrl=H_ctrl,
    U_target=U_target,
    T=20.0,  # 20 ns gate time
    n_steps=100
)

# Optimize
result = optimizer.optimize(max_iter=100, ftol=1e-6)

print(f"Final fidelity: {result.fidelity:.6f}")
print(f"Converged in {result.n_iter} iterations")

# Visualize
optimizer.plot_convergence()
optimizer.plot_pulse()
```

**Expected output:** F > 0.999 in ~50 iterations

---

## Repository Structure

```
QubitPulseOpt/
├── src/                          # Core library (production code)
│   ├── hamiltonian/              # System definitions (H₀, H_ctrl)
│   ├── pulses/                   # Waveform generators (Gaussian, DRAG, composite)
│   ├── optimization/             # GRAPE, Krotov, CRAB optimizers
│   ├── noise/                    # Lindblad master equation, decoherence
│   ├── benchmarking/             # Randomized benchmarking, process tomography
│   ├── gates/                    # Gate library (X, Y, Z, H, T, rotations)
│   ├── analysis/                 # Filter functions, sensitivity analysis
│   └── visualization/            # Dashboards, animations, reports
├── tests/                        # 573+ tests (pytest)
│   ├── unit/                     # Fast unit tests (< 1 min)
│   └── integration/              # Integration tests
├── notebooks/                    # 8 Jupyter tutorials
│   ├── 01_drift_dynamics.ipynb
│   ├── 02_rabi_oscillations.ipynb
│   ├── 03_decoherence_and_lindblad.ipynb
│   ├── 04_advanced_pulse_shaping.ipynb
│   ├── 05_gate_optimization.ipynb
│   ├── 06_robustness_analysis.ipynb
│   ├── 07_visualization_gallery.ipynb
│   └── 08_end_to_end_workflow.ipynb
├── docs/                         # Comprehensive documentation
│   ├── science/                  # LaTeX theory document
│   ├── DEVELOPER_GUIDE_PRECOMMIT.md
│   ├── REMAINING_TASKS_CHECKLIST.md
│   └── POWER_OF_10_COMPLIANCE.md
├── examples/                     # Standalone demo scripts
├── scripts/                      # Utilities (compliance checker, setup)
├── .github/workflows/            # CI/CD pipelines
└── config/                       # Configuration files
```

---

## Documentation

### For Users

- **[Quick Start Guide](docs/QUICKSTART.md)** - 10-minute walkthrough
- **[Jupyter Notebooks](notebooks/)** - 8 interactive tutorials covering all features
- **[API Documentation](https://rylanmalarchick.github.io/QubitPulseOpt/)** - Auto-generated from docstrings
- **[Science Document](docs/quantum_control_theory.pdf)** - LaTeX theory & derivations

### For Developers

- **[Contributing Guide](CONTRIBUTING.md)** - How to contribute (includes Power-of-10 rules)
- **[Developer Guide](docs/DEVELOPER_GUIDE_PRECOMMIT.md)** - Pre-commit hooks, testing, CI/CD
- **[Testing Guide](tests/README_TESTING.md)** - Test infrastructure and practices
- **[Code Quality](docs/POWER_OF_10_COMPLIANCE.md)** - Power-of-10 compliance report

### Theory & Background

- **[Krotov Algorithm](docs/science/krotov_theory.md)** - Mathematical foundations
- **[Filter Functions](docs/science/filter_functions_theory.md)** - Noise susceptibility analysis
- **[Randomized Benchmarking](docs/science/randomized_benchmarking_theory.md)** - Gate fidelity characterization
- **[Composite Pulses](docs/science/composite_pulses_theory.md)** - Error suppression techniques

---

## Technical Approach

### Physics Foundation

QubitPulseOpt models a driven qubit using the time-dependent Hamiltonian:

```
H(t) = H₀ + Ω(t)·H_ctrl
     = -½ω_d σ_z + Ω(t)σ_x
```

where:
- **H₀**: Drift Hamiltonian (free evolution, ω_d = qubit frequency)
- **Ω(t)**: Time-dependent control amplitude (the pulse we optimize)
- **H_ctrl**: Control Hamiltonian (coupling operator)

### Optimization Framework

**GRAPE (Gradient Ascent Pulse Engineering):**

Maximizes gate fidelity F = |⟨ψ_target|U(T)|ψ_init⟩|² by:

1. Discretizing time: t ∈ [0, T] → {t₀, t₁, ..., t_N}
2. Parameterizing pulse: Ω(t) → {Ω₁, Ω₂, ..., Ω_N}
3. Computing gradient: ∂F/∂Ω_k using chain rule
4. Updating: Ω_k^(new) = Ω_k^(old) + α·∂F/∂Ω_k

Convergence: Typically 50-100 iterations to F > 0.999

### Noise Modeling

**Lindblad Master Equation:**

```
dρ/dt = -i[H(t), ρ] + γ₁·L[σ₋](ρ) + γ₂·L[σ_z](ρ)
```

- **γ₁ = 1/T₁**: Amplitude damping (energy relaxation)
- **γ₂ = 1/T₂**: Dephasing (phase randomization)
- **L[c](ρ)**: Lindblad superoperator

Realistic parameters:
- T₁ = 10-100 μs (superconducting qubits)
- T₂ = 20-200 μs (often T₂ ≈ 2T₁)

---

## Demonstrations

### Bloch Sphere Evolution

*[Placeholder: Add bloch_evolution.gif after running demo materials generator]*

### Optimization Convergence

*[Placeholder: Add optimization_convergence.gif after running demo materials generator]*

### Robustness Analysis

*[Placeholder: Add parameter_sweep.png after running demo materials generator]*

**To generate demo materials:**

```bash
python scripts/generate_demo_materials.py
```

---

## Testing & Quality Assurance

### Test Suite

```bash
# Fast tests (< 1 minute)
pytest tests/unit -v -m "not slow"

# Full suite including slow optimization tests
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=src --cov-report=html

# Parallel execution (faster)
pytest tests/unit -v -m "not slow" -n auto
```

### Code Quality Checks

```bash
# Format code
black src/ tests/
isort src/ tests/

# Lint
flake8 src/ tests/

# Type checking (optional)
mypy src/

# Power-of-10 compliance
python scripts/compliance/power_of_10_checker.py src --verbose
```

### CI/CD Pipeline

Every push and PR triggers:

Tests on Python 3.9, 3.10, 3.11, 3.12  
Code coverage tracking (Codecov)  
Linting (Black, isort, flake8)  
Power-of-10 compliance check (≥97% threshold)  
Security scanning (Bandit, Safety)  
Documentation build (Sphinx → GitHub Pages)

---

## Development Setup

### Pre-commit Hooks

QubitPulseOpt uses pre-commit hooks to ensure code quality:

```bash
# Install pre-commit
pip install pre-commit

# Install git hooks
pre-commit install

# (Optional) Run on all files
pre-commit run --all-files
```

**Hooks include:**
- Black code formatting
- isort import sorting
- flake8 linting
- Trailing whitespace removal
- Power-of-10 compliance checks

See [`docs/DEVELOPER_GUIDE_PRECOMMIT.md`](docs/DEVELOPER_GUIDE_PRECOMMIT.md) for details.

### Power-of-10 Coding Standards

This project follows NASA JPL's **Power-of-10 Rules** for safety-critical code:

1. Restrict to simple control flow (no goto)
2. All loops have fixed upper bounds
3. No dynamic memory allocation after init
4. Functions limited to 60 lines
5. Minimum 2 assertions per function
6. Data declared at smallest scope
7. Return values checked
8. Limited preprocessor use
9. Pointer use restricted
10. All warnings enabled

**Current compliance: 97.5%** ([Full report](docs/POWER_OF_10_COMPLIANCE.md))

---

## Background & Motivation

### From Robotics to Quantum Control

This project applies control theory principles from real-time robotics to quantum systems:

| **Robotics (AirHound Drone)** | **Quantum Control (QubitPulseOpt)** |
|--------------------------------|--------------------------------------|
| Stabilize yaw under IMU noise | Steer qubit under decoherence |
| Loop closure: ~10 ms | Gate time: ~20 ns (10⁶× faster!) |
| PID controller | Optimal control (GRAPE/Krotov) |
| Sensor fusion (Kalman filter) | Filter function analysis |
| Real-time constraints | Coherence time constraints |

### Key Parallels

- **Noisy Signals**: IMU drift ↔ T1/T2 decay
- **Latency Optimization**: ROS2 pipelines ↔ Gate duration minimization
- **Robustness**: Wind gusts ↔ Control amplitude noise
- **System Identification**: Motor calibration ↔ Hamiltonian tomography

**Unique Challenge**: Quantum systems are fundamentally probabilistic and irreversible—you can't "re-run" a failed measurement. Control must be right the first time.

### Related Projects

- **AirHound**: Autonomous drone navigation with yaw stabilization
- **NASA NCCS**: High-altitude imagery processing with deep learning noise reduction
- **QubitPulseOpt**: Quantum optimal control (this project)

See [`docs/PORTFOLIO_CONNECTIONS.md`](docs/PORTFOLIO_CONNECTIONS.md) for detailed narrative.

---

## Contributing

Contributions are welcome! Please see [`CONTRIBUTING.md`](CONTRIBUTING.md) for guidelines.

### Quick Contribution Workflow

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Install pre-commit hooks: `pre-commit install`
4. Make your changes
5. Run tests: `pytest tests/ -v`
6. Commit: `git commit -m 'Add amazing feature'`
7. Push: `git push origin feature/amazing-feature`
8. Open a Pull Request

**All PRs must:**
- Pass all CI checks (tests, linting, compliance)
- Maintain ≥95% test coverage
- Follow Power-of-10 standards
- Include tests for new features

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

### Scientific Foundations

- **QuTiP**: The Quantum Toolbox in Python ([qutip.org](https://qutip.org))
- **GRAPE Algorithm**: Khaneja et al., J. Magn. Reson. (2005)
- **Krotov Method**: Tannor & Somlói, J. Chem. Phys. (1992)
- **Filter Functions**: Green et al., PRL (2013)
- **Composite Pulses**: Levitt, Prog. NMR Spectrosc. (1986)

### Software & Tools

- **NumPy/SciPy**: Numerical computing foundation
- **Matplotlib**: Visualization and plotting
- **pytest**: Testing framework
- **Black/isort/flake8**: Code quality tools
- **GitHub Actions**: CI/CD infrastructure

### Inspiration & Background

- **NASA JPL Power-of-10**: Safety-critical coding standards
- **MIT OpenCourseWare**: 8.370 Quantum Information Science
- **Nielsen & Chuang**: "Quantum Computation and Quantum Information"

---

## Contact & Links

- **GitHub**: [rylanmalarchick/QubitPulseOpt](https://github.com/rylanmalarchick/QubitPulseOpt)
- **Documentation**: [rylanmalarchick.github.io/QubitPulseOpt](https://rylanmalarchick.github.io/QubitPulseOpt)
- **Issues**: [GitHub Issues](https://github.com/rylanmalarchick/QubitPulseOpt/issues)
- **Discussions**: [GitHub Discussions](https://github.com/rylanmalarchick/QubitPulseOpt/discussions)

---

## Project Status

**Version**: 1.0.0 (Production Ready)  
**Status**: Active Development  
**Last Updated**: 2024

### Milestones

- Phase 1: Core simulation infrastructure
- Phase 2: GRAPE optimization
- Phase 3: Advanced features (benchmarking, robustness)
- Phase 4: Documentation & notebooks
- Phase 5: CI/CD & production polish
- 🔄 Phase 6 (Current): Portfolio integration & advanced features

See [`docs/REMAINING_TASKS_CHECKLIST.md`](docs/REMAINING_TASKS_CHECKLIST.md) for detailed roadmap.

---

<div align="center">

**⭐ Star this repo if you find it useful! ⭐**

**Made with ❤️ for the quantum computing community**

</div>