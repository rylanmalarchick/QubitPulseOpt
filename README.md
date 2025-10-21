# QubitPulseOpt: Optimal Pulse Engineering for Single-Qubit Gates

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
```bash
# Create conda environment (CPU-only QuTiP)
conda env create -f environment.yml
conda activate qubitpulseopt

# Verify installation
python -c "import qutip; print(qutip.about())"
```

### 2. Run Baseline Demo (Week 1 Milestone)
```bash
# Interactive exploration
jupyter notebook notebooks/01_drift_dynamics.ipynb

# Automated validation
pytest tests/ -v --cov=src
```

---

## Repository Structure
```
QubitPulseOpt/
├── src/                  # Core simulation modules
│   ├── hamiltonian/      # System definitions (H₀ + Hc)
│   ├── pulses/           # Waveform generators (Gaussian, DRAG, etc.)
│   ├── optimization/     # GRAPE/CRAB implementations
│   └── noise/            # Decoherence models (Lindblad)
├── notebooks/            # Interactive demos (ReAct loop outputs)
├── tests/                # Pytest suite (physics validation)
├── docs/                 # SOW + design notes
├── data/                 # Simulation outputs (CSV/plots)
└── agent_logs/           # AI decision traceability (JSON)
```

---

## Milestones (4-Week Plan)
- **Week 1:** Drift dynamics + unitary evolution validated
- **Week 2:** GRAPE optimizer converges for X-gate (F>0.99)
- **Week 3:** Noise robustness + ML hyperparameter search
- **Week 4:** Final report + Git tag `v1.0-shippable`

---

## References
- SOW Document: `docs/Scope of Work*.md`
- QuTiP Docs: [https://qutip.org/docs/latest/](https://qutip.org/docs/latest/)
- GRAPE Tutorial: [arXiv:quant-ph/0504128](https://arxiv.org/abs/quant-ph/0504128)
