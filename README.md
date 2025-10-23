# QubitPulseOpt: Optimal Pulse Engineering for Single-Qubit Gates

[![Tests](https://github.com/YOUR_USERNAME/quantumControls/workflows/Tests/badge.svg)](https://github.com/YOUR_USERNAME/quantumControls/actions/workflows/tests.yml)
[![Documentation](https://github.com/YOUR_USERNAME/quantumControls/workflows/Documentation/badge.svg)](https://github.com/YOUR_USERNAME/quantumControls/actions/workflows/docs.yml)
[![Notebooks](https://github.com/YOUR_USERNAME/quantumControls/workflows/Notebooks/badge.svg)](https://github.com/YOUR_USERNAME/quantumControls/actions/workflows/notebooks.yml)
[![codecov](https://codecov.io/gh/YOUR_USERNAME/quantumControls/branch/main/graph/badge.svg)](https://codecov.io/gh/YOUR_USERNAME/quantumControls)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Project Context
**From Perception to Coherence:** This project applies control theory principles familiar from real-time robotics (e.g., AirHound's drone yaw stabilization under sensor noise) to quantum systems. Instead of steering a quadrotor with noisy IMU data, we shape electromagnetic pulses to drive qubit state transitions while battling decoherence—a temporal "noise budget" analogous to loop closure times in ROS2 pipelines.

**Academic Foundation:** Builds on prior work in noisy signal processing (NASA deep learning for high altitude imagery) by treating T1/T2 decay as systematic corruption of quantum information channels.

---

## Objectives (SOW-Aligned)
1. **Simulate** drift Hamiltonian evolution (free precession on Bloch sphere)
2. **Optimize** control pulses for high-fidelity X/Y/H gates (F > 0.999)
3. **Characterize** robustness under realistic noise models (T1=10μs, T2=20μs)
4. **Demonstrate** GRAPE/CRAB algorithms with visual diagnostics

**Success Criteria:** See `docs/Scope of Work*.md` Section B (KPIs).

---

## Quick Start
### 1. Environment Setup

**Option A: Python venv (Recommended - Fast Setup)**
```bash
# Create virtual environment
python3 -m venv venv

# Activate environment
source venv/bin/activate  # On Linux/Mac
# OR: venv\Scripts\activate  # On Windows

# Install dependencies
pip install qutip numpy scipy matplotlib jupyter pytest pytest-cov black flake8 ipykernel

# Register Jupyter kernel
python -m ipykernel install --user --name=qubitpulseopt --display-name="Python (QubitPulseOpt)"

# Verify installation
python -c "import qutip; print(qutip.about())"

# Or use the helper script
source scripts/activate_env.sh
```

**Option B: Conda (Alternative - Full Isolation)**
```bash
# Create conda environment (CPU-only QuTiP)
conda env create -f environment.yml
conda activate qubitpulseopt

# Verify installation
python -c "import qutip; print(qutip.about())"
```

### 2. Run Tests
```bash
# Run fast unit tests (< 1 minute)
pytest tests/unit -v -m "not slow" --cov=src

# Run all tests including slow optimization tests
pytest tests/ -v --cov=src

# Run with parallel execution
pytest tests/unit -v -m "not slow" -n auto
```

### 3. Explore Notebooks
```bash
# Launch Jupyter
jupyter notebook

# Open any notebook from notebooks/ directory
# Recommended starting points:
# - 01_basic_pulse_design.ipynb - Introduction to pulse design
# - 02_grape_optimization.ipynb - GRAPE optimization tutorial
# - 08_end_to_end_workflow.ipynb - Complete workflow example
```

---

## Repository Structure
```
QubitPulseOpt/
├── src/                  # Core simulation modules
│   ├── hamiltonian/      # System definitions (H₀ + Hc)
│   ├── pulses/           # Waveform generators (Gaussian, DRAG, composite)
│   ├── optimization/     # GRAPE/CRAB/filtering implementations
│   ├── noise/            # Decoherence models (Lindblad)
│   ├── benchmarking/     # Randomized benchmarking, filter functions
│   ├── visualization/    # Dashboards, animations, reports
│   ├── io/               # Export/import utilities
│   └── config.py         # Configuration management
├── notebooks/            # Interactive demos (8 complete tutorials)
├── tests/                # Pytest suite (573+ tests)
│   ├── unit/             # Fast unit tests
│   └── integration/      # Integration tests (if present)
├── docs/                 # Comprehensive documentation
├── examples/             # Standalone example scripts
├── config/               # Configuration files
├── data/                 # Simulation outputs
└── .github/workflows/    # CI/CD pipelines
```

---

## Science Documentation

A comprehensive **LaTeX science document** covering all theoretical foundations, mathematical derivations, and implementation details is maintained in `docs/science/quantum_control_theory.tex`.

**Pre-compiled PDF:** [`docs/quantum_control_theory.pdf`](docs/quantum_control_theory.pdf)

### Current Coverage
- **Phase 1.1:** Computational infrastructure, reproducibility, environment design philosophy
- **Phase 1.2:** Drift Hamiltonian derivation, analytical propagator, Bloch sphere dynamics, validation methodology

### Building the PDF (Optional)
If you modify the LaTeX source:
```bash
cd docs/science/
pdflatex quantum_control_theory.tex
pdflatex quantum_control_theory.tex  # Run twice for references
cp quantum_control_theory.pdf ../
```

**Requirements:** Standard LaTeX distribution (TeX Live, MiKTeX) with packages: `amsmath`, `physics`, `hyperref`, `cleveref`, `listings`, `tikz`.

---

## Features

### Core Capabilities
- **Pulse Design:** Gaussian, DRAG, composite pulses, adiabatic techniques
- **Optimization:** GRAPE, CRAB algorithms with customizable cost functions
- **Noise Modeling:** T1/T2 decoherence, filter function analysis
- **Benchmarking:** Randomized benchmarking, sensitivity analysis
- **Visualization:** Interactive dashboards, Bloch sphere animations, publication-quality reports
- **Export:** JSON, NPZ, hardware-compatible formats

### Advanced Features
- **Robustness Analysis:** Filter functions, uncertainty quantification
- **Gate Library:** Optimized single-qubit gates (X, Y, Z, H, T, arbitrary rotations)
- **Composite Pulses:** BB1, CORPSE, SK1 for error suppression
- **Performance:** Parallel execution, profiled hotspots, optimized solvers

## Documentation

- **Notebooks:** 8 comprehensive tutorials covering all features
- **API Documentation:** Detailed docstrings with examples
- **Science Document:** LaTeX theory document with derivations
- **Examples:** Standalone scripts demonstrating key workflows
- **Status Tracking:** `docs/PHASE_3_STATUS.md` for project progress

## Testing & Quality

- **573+ Tests:** Comprehensive unit and integration test coverage
- **CI/CD:** Automated testing on Python 3.9, 3.10, 3.11
- **Code Quality:** Linting with flake8, formatting with black
- **Nightly Builds:** Slow optimization tests run on schedule
- **Notebook Validation:** All notebooks tested for execution

## Milestones (Complete)
- ✅ **Phase 1:** Drift dynamics + unitary evolution validated
- ✅ **Phase 2:** GRAPE optimizer converges for X-gate (F>0.999)
- ✅ **Phase 3:** Advanced features - robustness, benchmarking, visualization
- ✅ **Phase 4:** Complete documentation, notebooks, and examples
- 🔄 **Phase 5 (Current):** Production polish, CI/CD, performance optimization

---

## References
- SOW Document: `docs/Scope of Work*.md`
- QuTiP Docs: [https://qutip.org/docs/latest/](https://qutip.org/docs/latest/)
- GRAPE Tutorial: [arXiv:quant-ph/0504128](https://arxiv.org/abs/quant-ph/0504128)
