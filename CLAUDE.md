# QubitPulseOpt — Agent Context for PRA Submission

## Mission

You are working on **QubitPulseOpt**, a Python framework for quantum optimal control pulse optimization. The goal is to complete experiments and submit to **Physical Review A** (free, high-prestige physics journal for quantum control).

**Target submission: ~Feb 24, 2026**

Existing preprint: arXiv:2511.12799. IEEE TQE draft: `paper/ieee_tqe/main_full_extended.tex` (504 lines). PRA draft exists at `paper/pra/main.tex` (33.9KB, compiled PDF at `paper/pra/main.pdf`).

## Current State (Updated Feb 7, 2026)

### DRAG Beta Formula: FIXED in all locations

The critical DRAG beta bug has been **resolved** in source and all scripts. Every location now uses the correct formula from Motzoi et al. PRL 103, 110501 (2009):

```python
anharmonicity_radns = 2.0 * np.pi * anharmonicity_mhz * 1e-3  # MHz -> rad/ns
optimal_beta = -1.0 / (2.0 * anharmonicity_radns)
```

For alpha/2pi = -200 MHz: beta = 0.398 (amplitude-independent, O(1)).

**Verified locations (all correct):**

| File | Line(s) | Status |
|------|---------|--------|
| `src/pulses/drag.py` | L245 (optimize_beta), L614 (_compute_drag_pulse_params), L694 (leakage_error_estimate) | CORRECT + sanity warning if |beta|>5 |
| `scripts/benchmark_drag_vs_grape.py` | L129 | CORRECT |
| `scripts/experiment_3level_comparison.py` | L245 | CORRECT |
| `scripts/experiment_error_budget.py` | L101, L649 | CORRECT |
| `scripts/experiment_robustness.py` | L106, L185 | CORRECT |

### GRAPE Gradient Fix: DONE

`src/optimization/grape.py` — gradient ordering corrected from `U_before * dU * U_after` to `U_after * dU * U_before`. Fidelity clamping added. H-gate converges at 100% in 160 iterations.

### Other Source Changes (uncommitted)

- `src/optimization/gates.py` — Added unitarity validation to `_rotation_gate()` and `rotation_from_euler_angles()`
- `src/hamiltonian/evolution.py` — Added `_validate_unitary()` after matrix exponentiation in `propagator()`
- `tests/unit/test_drag.py` — Modified (test updates for beta fix)

### Experiment Results: MIXED — some valid, some need re-run

| Directory | Valid File | Invalid Files | Action Needed |
|-----------|-----------|---------------|---------------|
| `results/drag_vs_grape/` | `benchmark_20260207_202538.json` (beta=0.398, DRAG fid=0.993) | 5 older files with wrong beta (0.05, 398.9, 2.507) | Delete invalid, keep valid |
| `results/multilevel_comparison/` | `3level_xgate_20260207_202907.json` (beta=0.398 constant, DRAG avg fid=0.996) | `3level_xgate_20260207_195218.json` (beta varies 0.63-6.27) | Delete invalid, keep valid |
| `results/error_budget/` | **NONE** | All 4 files show DRAG fid=0.747 (2-level) / 0.872 (3-level) — bad beta | **Re-run entirely** |
| `results/robustness/` | **NONE** | Single file shows DRAG fid 0.67-0.82 — bad beta | **Re-run entirely** |

### Paper Directory

```
paper/
├── arxiv_v1/          — Original arXiv preprint (Dec 23)
├── figures/           — Empty
├── ieee_tqe/          — IEEE TQE submission (Feb 3): main.tex, cover_letter.tex, 9 PNG figures
└── pra/               — PRA submission (Feb 7): main.tex (33.9KB), main.pdf, references.bib, cover_letter.tex
```

The `preprint/` directory files were deleted (show as `D` in git status) and reorganized into `paper/`.

## Git Status (26 dirty items — NOTHING committed since Feb 6)

**Last commit:** `0830d76` (Feb 6) — "docs: remove emoji from README preprint link"

**Modified tracked files:**
- `src/hamiltonian/evolution.py` — unitarity validation
- `src/optimization/gates.py` — unitarity validation
- `src/optimization/grape.py` — gradient fix + fidelity clamping
- `src/pulses/drag.py` — beta formula fix
- `src/qubitpulseopt.egg-info/PKG-INFO, SOURCES.txt, top_level.txt` — egg-info updates
- `tests/pytest.log` — test log
- `tests/unit/test_drag.py` — test updates

**Deleted tracked files:**
- `preprint/preprint.pdf`, `preprint.tex`, `preprint_backup_pre_corrections.tex`, `preprint_original_backup.tex`

**Untracked (new):**
- `CLAUDE.md`, `opencode.json`, `agent_docs/`
- `paper/` (entire PRA + IEEE TQE + arxiv_v1)
- `results/` (all experiment outputs)
- `scripts/benchmark_drag_vs_grape.py`, `experiment_*.py` (5 experiment scripts)
- `src/analysis/`, `src/hamiltonian/stochastic.py`, `src/rl_env.py`, `src/train_agent.py`

## Task List (Priority Order)

### Task 1: Clean up results — delete invalid files [5 min]

Remove results generated with wrong beta. Keep ONLY:
- `results/drag_vs_grape/benchmark_20260207_202538.json`
- `results/multilevel_comparison/3level_xgate_20260207_202907.json`

Delete everything else in those two directories.

### Task 2: Re-run error_budget and robustness experiments [~30 min compute]

These have NO valid results. Run:
```bash
.venv/bin/python scripts/experiment_error_budget.py
.venv/bin/python scripts/experiment_robustness.py
```

After running, verify:
- Error budget: DRAG coherent fidelity should be ~0.993+ in 2-level (not 0.747)
- Robustness: DRAG fidelities should be >0.95 at nominal parameters

### Task 3: Commit all work [10 min]

Suggested commit structure (can be one or split):
```bash
git add src/pulses/drag.py src/optimization/grape.py src/optimization/gates.py src/hamiltonian/evolution.py
git add tests/unit/test_drag.py
git add scripts/ results/ paper/ src/analysis/ src/hamiltonian/stochastic.py
git add CLAUDE.md
git rm preprint/preprint.pdf preprint/preprint.tex preprint/preprint_backup_pre_corrections.tex preprint/preprint_original_backup.tex
git commit -m "fix: correct DRAG beta formula, fix GRAPE gradient, add PRA experiments

- DRAG beta: -alpha/(2*Omega) -> -1/(2*alpha) per Motzoi PRL 103, 110501
- GRAPE gradient ordering: U_before*dU*U_after -> U_after*dU*U_before
- Added unitarity validation in gates.py and evolution.py
- 4 PRA experiment scripts with results (2 of 4 complete with valid data)
- PRA paper draft at paper/pra/main.tex
- Reorganized paper directory (arxiv_v1, ieee_tqe, pra)"
```

Do NOT commit: `opencode.json`, `agent_docs/`, `src/rl_env.py`, `src/train_agent.py` (RL code is unrelated).

### Task 4: Paper finalization for PRA [writing task]

The PRA draft exists at `paper/pra/main.tex`. Needs:
1. Update results sections with corrected DRAG data
2. Add 3-level comparison results (valid data exists)
3. Add error budget and robustness results (after Task 2)
4. Expand bibliography from 9 to 30+ references
5. Write significance statement for PRA cover letter
6. PACS: 03.67.Lx (quantum control), 85.25.Cp (Josephson devices)

### Task 5: 2-Qubit CZ Gate [STRETCH — only if tasks 1-4 complete]

Extend to CZ = diag(1,1,1,-1) in 2-qubit system.

## Codebase Structure

```
src/
├── pulses/
│   ├── shapes.py          — Gaussian, Blackman, cosine, DRAG, custom pulse shapes
│   ├── drag.py            — Full DRAG implementation (820 lines): DRAGPulse, optimize_beta(), leakage analysis
│   ├── adiabatic.py       — STIRAP, Landau-Zener sweeps
│   └── composite.py       — BB1, CORPSE, SK1 composite sequences
├── optimization/
│   ├── grape.py           — GRAPE optimizer (1256 lines): GRAPEOptimizer, GRAPEResult
│   ├── krotov.py          — Krotov optimizer (1068 lines): KrotovOptimizer, KrotovResult
│   ├── gates.py           — UniversalGates: H, S, T, Pauli optimization (922 lines)
│   ├── robustness.py      — RobustnessTester: detuning/amplitude sweeps, Fisher info (1321 lines)
│   ├── filter_functions.py — Noise spectral density, filter function analysis (740 lines)
│   ├── benchmarking.py    — Randomized benchmarking: Clifford group, RB, IRB (761 lines)
│   └── compilation.py     — GateCompiler: circuit compilation (899 lines)
├── hamiltonian/
│   ├── drift.py           — DriftHamiltonian: H₀ = (ω₀/2)σz
│   ├── control.py         — ControlHamiltonian: time-dependent driving
│   ├── evolution.py       — TimeEvolution: analytical + numerical (+ unitarity validation)
│   ├── lindblad.py        — DecoherenceParams + Lindblad master equation solver
│   └── stochastic.py      — SME for weak measurement trajectories (NEW, untracked)
├── analysis/              — Analysis utilities (NEW, untracked)
├── hardware/
│   ├── iqm_backend.py     — IQM REST API backend (933 lines)
│   ├── iqm_translator.py  — Pulse → IQM format translation (511 lines)
│   └── characterization.py — T1/T2/Rabi/RB characterization (960 lines)
├── visualization/, io/
├── config.py, constants.py
└── rl_env.py, train_agent.py — RL (not relevant to PRA submission)

scripts/
├── benchmark_drag_vs_grape.py          — Main benchmark (~415 lines)
├── experiment_3level_comparison.py     — 3-level transmon comparison (~493 lines)
├── experiment_criticality.py           — Criticality analysis (~135 lines)
├── experiment_error_budget.py          — Error budget analysis (~1079 lines)
├── experiment_robustness.py            — Robustness sweeps (~476 lines)
└── (11 other scripts: verification, preprint, profiling, IQM)

results/                    — Experiment outputs (see validity table above)
paper/                      — Paper drafts (arxiv_v1, ieee_tqe, pra)
config/default_config.yaml  — Default params
```

## Test Status

```
53 failed, 770 passed, 3 skipped, 2 xpassed in 38.29s
```

53 failures are pre-existing (Krotov edge cases, pulse shape tests shifted by drag changes, statistical tests). Not blocking for PRA submission.

## Key Physics Constraints

- Fidelity in [0, 1]; unitarity: U†U = I
- T2 <= 2*T1 (fundamental decoherence constraint)
- Transmon anharmonicity: alpha/2pi in [-350, -150] MHz (typically ~-200 to -300 MHz)
- Single-qubit gate times: 20-100ns; two-qubit: 100-500ns
- **DRAG beta = -1/(2*alpha) where alpha is in rad/ns. For alpha/2pi = -200 MHz, beta ≈ 0.398. This is O(1) and independent of gate time/amplitude.**

## Environment

- Python 3.12+ (use `.venv/bin/python`, NOT bare `python3` — system python is 3.14)
- QuTiP 5.2.1, NumPy, SciPy
- Run tests: `.venv/bin/python -m pytest tests/ -v --tb=short`
- Style: snake_case functions, PascalCase classes, type hints, docstrings with physics references (DOI/arXiv)
- Git: conventional commits (feat, fix, test, docs)
