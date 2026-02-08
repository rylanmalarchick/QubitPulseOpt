# QubitPulseOpt — Agent Context for PRA Submission

## Mission

You are working on **QubitPulseOpt**, a Python framework for quantum optimal control pulse optimization. The goal is to complete experiments and submit to **Physical Review A** (free, high-prestige physics journal for quantum control).

**Target submission: ~Feb 24, 2026**

Existing preprint: arXiv:2511.12799. IEEE TQE draft: `paper/ieee_tqe/main_full_extended.tex` (504 lines). PRA draft at `paper/pra/main.tex` (~460 lines, compiled PDF at `paper/pra/main.pdf`, 8 pages).

## Current State (Updated Feb 7, 2026)

### All Tasks 1-4 COMPLETE

| Task | Status | Details |
|------|--------|---------|
| Task 1: Clean up invalid results | DONE | Invalid files deleted |
| Task 2: Re-run error_budget + robustness | DONE | Valid results with beta=0.398 |
| Task 3: Commit all source fixes | DONE | Commit `cab4917` (not pushed) |
| Task 4: Rewrite PRA paper | DONE | All data, narrative, discussion updated |

### DRAG Beta Formula: FIXED everywhere (source + paper)

The correct formula from Motzoi et al. PRL 103, 110501 (2009):

```python
anharmonicity_radns = 2.0 * np.pi * anharmonicity_mhz * 1e-3  # MHz -> rad/ns
optimal_beta = -1.0 / (2.0 * anharmonicity_radns)
```

For alpha/2pi = -200 MHz: beta = 0.398 (amplitude-independent, O(1)).

Paper Eq. 8: `\beta = -\frac{1}{2\alpha}` — CORRECT.

### GRAPE Gradient Fix: DONE

`src/optimization/grape.py` — gradient ordering corrected from `U_before * dU * U_after` to `U_after * dU * U_before`. Fidelity clamping added. H-gate converges at 100% in 160 iterations.

### Experiment Results: ALL VALID

| Directory | File | Key Results |
|-----------|------|-------------|
| `results/drag_vs_grape/` | `benchmark_20260207_202538.json` | 2-level, 40ns: DRAG F=0.993, GRAPE F=1.0 |
| `results/multilevel_comparison/` | `3level_xgate_20260207_202907.json` | 3-level sweep 10-100ns: DRAG F=0.967-0.999999 |
| `results/error_budget/` | `error_budget_20260207_203830.json` | 2-level error budget |
| `results/error_budget/` | `error_budget_combined_20260207_203843.json` | Combined 2+3-level error budget |
| `results/robustness/` | `robustness_20260207_203855.json` | Detuning + amplitude sweeps |

### Paper: REWRITTEN with correct data

**Key numbers in paper (all verified against result files):**

| Metric | Gaussian | DRAG | GRAPE |
|--------|----------|------|-------|
| 3-level F (20ns, coherent) | 0.972 | 0.9995 | 1.000 |
| 3-level infidelity (full decoherence) | 2.9e-2 | 8.4e-4 | 7.2e-4 |
| GRAPE advantage over Gaussian | 39x | — | — |
| GRAPE advantage over DRAG | — | 1.2x | — |
| Detuning min fidelity | 0.937 | 0.990 | 0.931 |
| Amplitude min fidelity | 0.965 | 0.990 | 0.994 |

**New narrative:** DRAG works well with correct beta. GRAPE's advantage over DRAG is small (1.2x) at 20ns. DRAG is more robust to detuning than GRAPE. GRAPE's real advantage is at short gate times and guaranteed zero coherent error.

### Git Status

**Committed (not pushed):** `cab4917` — source fixes, experiments, results, paper directory
**Uncommitted changes:** Paper rewrite (`paper/pra/main.tex`), CLAUDE.md update, `.bak` file to clean up

### What Remains

- [ ] Commit paper rewrite
- [ ] Push to origin
- [ ] Task 5 (STRETCH): 2-qubit CZ gate extension
- [ ] Expand bibliography to 30+ references (currently ~25)
- [ ] Write PRA cover letter (`paper/pra/cover_letter.tex`)

## Codebase Structure

```
src/
├── pulses/
│   ├── shapes.py          — Gaussian, Blackman, cosine, DRAG, custom pulse shapes
│   ├── drag.py            — Full DRAG implementation (820 lines)
│   ├── adiabatic.py       — STIRAP, Landau-Zener sweeps
│   └── composite.py       — BB1, CORPSE, SK1 composite sequences
├── optimization/
│   ├── grape.py           — GRAPE optimizer (1256 lines)
│   ├── krotov.py          — Krotov optimizer (1068 lines)
│   ├── gates.py           — UniversalGates (922 lines)
│   ├── robustness.py      — RobustnessTester (1321 lines)
│   ├── filter_functions.py — Noise spectral density (740 lines)
│   ├── benchmarking.py    — Randomized benchmarking (761 lines)
│   └── compilation.py     — GateCompiler (899 lines)
├── hamiltonian/
│   ├── drift.py, control.py, evolution.py, lindblad.py, stochastic.py
├── hardware/
│   ├── iqm_backend.py, iqm_translator.py, characterization.py
├── analysis/, visualization/, io/
├── config.py, constants.py

scripts/
├── benchmark_drag_vs_grape.py
├── experiment_3level_comparison.py
├── experiment_criticality.py
├── experiment_error_budget.py
├── experiment_robustness.py

paper/
├── arxiv_v1/          — Original arXiv preprint
├── ieee_tqe/          — IEEE TQE submission
└── pra/               — PRA submission (UPDATED)
    ├── main.tex       — 460 lines, all data corrected
    ├── main.pdf       — 8 pages, compiled
    ├── references.bib — 25 references
    └── cover_letter.tex
```

## Key Physics Constraints

- Fidelity in [0, 1]; unitarity: U†U = I
- T2 <= 2*T1 (fundamental decoherence constraint)
- Transmon anharmonicity: alpha/2pi in [-350, -150] MHz
- DRAG beta = -1/(2*alpha) where alpha is in rad/ns. For alpha/2pi = -200 MHz, beta = 0.398
- Single-qubit gate times: 20-100ns; two-qubit: 100-500ns

## Environment

- Python 3.12+ (use `.venv/bin/python`, NOT bare `python3`)
- QuTiP 5.2.1, NumPy, SciPy
- Run tests: `.venv/bin/python -m pytest tests/ -v --tb=short`
- LaTeX: `/usr/local/texlive/2024/bin/x86_64-linux/pdflatex`
- Style: snake_case functions, PascalCase classes, type hints, docstrings with physics refs
- Git: conventional commits (feat, fix, test, docs)
