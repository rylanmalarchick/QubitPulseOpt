<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Scope of Work: Quantum Controls Simulation Project - Optimal Pulse Engineering for High-Fidelity Single-Qubit Gates

## Document Metadata

- **Project Title:** QubitPulseOpt: Hardware-Free Simulation and Optimization of Microwave Pulse Controls for Superconducting Qubits
- **Version:** 1.0 (Initial Draft)
- **Author:** Perplexity AI Assistant (in collaboration with Rylan Malarchick)
- **Date:** October 20, 2025
- **Status:** Active / Ready for Agent Execution
- **Primary Stakeholders:** Rylan Malarchick (Principal Developer), AI Agent (Execution Engine via Zed IDE + GitHub Copilot/Claude 3.5 Sonnet)
- **Estimated Timeline:** 4-6 weeks (Part-time: 10-15 hours/week), with built-in iteration buffers
- **Scope Boundaries:** Strictly simulation-based (no physical hardware required). Focus on single-qubit dynamics; multi-qubit extensions scoped for Phase 2. All work in Python ecosystem.
- **Compliance Standards:** Adheres to modern AI agent workflows (e.g., LangChain-inspired task decomposition, ReAct prompting for agent reasoning, Auto-GPT-style iteration loops). Emphasizes reproducibility (Docker/Conda envs), version control (Git branches), and ethical AI use (no proprietary models; open-source only).
- **Confidentiality:** Public GitHub repo intended; no sensitive data.

***

## Executive Summary

This Scope of Work (SOW) defines a rigorous, self-contained project to simulate and optimize microwave pulse sequences for implementing high-fidelity single-qubit gates (e.g., X or Hadamard rotations) in a noisy superconducting qubit environment. Using QuTiP as the core quantum simulator, the project will demonstrate gradient-based optimization to achieve >99% gate fidelity, bridging classical control theory (e.g., PID-like feedback from your AirHound drone perception) to quantum settings.

The design follows **cutting-edge AI agent workflow principles** (2025 standards):

- **Hierarchical Task Decomposition:** Top-level goals broken into atomic subtasks with dependencies, enabling parallel agent execution.
- **ReAct-Style Prompting:** Agents (via Zed's GitHub Copilot/Claude 3.5) will reason, act (code/debug), and reflect in loops, with human-in-the-loop checkpoints.
- **Evaluation-Driven Iteration:** Each milestone includes quantitative metrics, auto-tests (pytest/Jupyter asserts), and LLM-based self-review prompts.
- **Modular Agent Architecture:** Leverage Zed's multi-model setup (Claude for high-level planning, GPT-4o for code gen, o1-preview for optimization reasoning).
- **Scalability \& Traceability:** Git-based versioning with semantic commits; Jupyter notebooks for interactive dev; Markdown docs auto-generated via agent prompts.
- **Risk Mitigation:** Built-in contingency for compute limits (CPU-only mode) and agent hallucinations (verification scripts).

This SOW is engineered to "scare" agents into precision: exhaustive checklists, zero-tolerance for unverified code, and adaptive loops that force reflection on edge cases. Total effort: ~60-90 hours, yielding a portfolio-ready GitHub repo that positions you for Google Quantum AI undergrad roles by demonstrating quantum-aware controls from your ML/robotics base.

Success Metric: Repo with 100% passing tests, >99% simulated fidelity, and a README that ties to your NASA DL/AirHound work (e.g., "Adapting real-time perception latency optimization to qubit state steering").

***

## Project Objectives

### Primary Objectives (SMART-Aligned)

1. **Simulate Realistic Qubit Dynamics (Specific, Measurable):** Model a single superconducting qubit under microwave drive and decoherence noise, achieving Bloch sphere trajectories with <5% deviation from ideal evolution in baseline runs.
2. **Optimize Pulse Controls (Achievable, Relevant):** Implement GRAPE (or ML-alternative) to tune pulse parameters, targeting 99.5%+ average fidelity across 50 noise realizations (T1/T2 = 10-50 μs, drive strength 10-100 MHz).
3. **Bridge to Your Background (Time-Bound, 4 Weeks):** Explicitly integrate concepts from AirHound (e.g., latency-sensitive feedback) and NASA DL (e.g., signal processing for cloud/noise), documented in notebooks.
4. **Produce Portfolio Artifact (Impactful):** A GitHub repo with interactive demos, visualizations, and a technical report (5-10 pages) suitable for Google Research applications.

### Secondary Objectives

- Demonstrate extensibility: Include hooks for multi-qubit (CNOT) or ML-enhanced optimization.
- Ensure Reproducibility: 100% deterministic runs via seeded RNG; env.yml for one-click setup.
- Ethical/Accessibility: Open-source under MIT license; include Jupyter widgets for non-experts.


### Non-Goals (Scope Exclusion)

- Hardware integration (e.g., no AWS Braket or IBM Qiskit real-device access).
- Full-scale error correction (e.g., surface codes); limit to open-loop/single-gate control.
- Advanced ML (e.g., full RL training); optional PyTorch extension only if core GRAPE succeeds.

***

## Technical Specifications

### System Model

- **Qubit Type:** Transmon superconducting qubit (realistic for Google Sycamore).
- **Hamiltonian:** Time-dependent: \$ H(t) = H_0 + H_c(t) \$, where:
    - Drift: \$ H_0 = -\frac{\omega_q}{2} \sigma_z \$ ($\omega_q \approx 5$ GHz).
    - Control: \$ H_c(t) = \Omega(t) \cos(\phi(t)) \sigma_x + \Omega(t) \sin(\phi(t)) \sigma_y \$ (amplitude $\Omega(t)$, phase $\phi(t)$).
- **Noise Model:** Lindblad master equation with relaxation ($\sigma_-$, rate $\gamma_1 = 1/T_1$) and dephasing ($\sigma_z$, rate $\gamma_\phi = 1/(2T_2)$). Include detuning/off-resonance terms.
- **Target Gates:** X-gate ($\pi$-rotation around x-axis) and Hadamard (superposition).
- **Simulation Parameters:**
    - Time grid: 1000-5000 points over 10-50 ns pulses.
    - Initial state: |0⟩ (ground state).
    - Fidelity Metric: State fidelity \$ F = |\langle \psi_{target} | \rho_{sim} \rangle|^2 \$, or gate fidelity via process tomography if extended.
- **Optimization:** GRAPE (QuTiP's `optimize_pulse`) with L-BFGS-B solver; constraints on pulse bandwidth/energy.


### Software Stack

- **Core Libraries:** QuTiP (v4.8+ for dynamics), NumPy/SciPy (optimizers), Matplotlib/Plotly (viz).
- **ML Extension (Optional):** PyTorch (v2.1+) for neural pulse approximators.
- **Dev Tools:** Zed IDE (with Claude 3.5 Sonnet for planning, GPT-4o for code, o1-preview for debugging reasoning).
- **Testing:** pytest for unit/integration; Jupyter for interactive sims.
- **Version Control:** Git (branches: main, dev/baseline, dev/optimize, dev/ml-ext); Semantic Release for tags.
- **Environment:** Conda (env.yml); Docker optional for CI (GitHub Actions workflow for fidelity tests).
- **Compute:** CPU-only (laptop feasible; <1 min per sim); GPU if PyTorch used.


### Data \& Outputs

- **Inputs:** Parameter sweeps (e.g., noise levels, pulse durations).
- **Outputs:** Bloch plots, fidelity curves, optimized pulse waveforms (JSON/CSV), Jupyter dashboards.

***

## AI Agent Workflow Design

This SOW integrates **2025-state-of-the-art AI agent paradigms** to maximize efficiency and minimize errors. Agents operate in Zed via GitHub Copilot extensions (multi-model chaining: Claude for orchestration, specialized models for subtasks). The workflow is **hierarchical and reflective**, inspired by AutoGen, CrewAI, and o1's chain-of-thought scaling.

### Agent Architecture

1. **Orchestrator Agent (Claude 3.5 Sonnet in Zed):**
    - Role: High-level planner/reviewer. Prompts: "Decompose task X into 5-10 atomic steps with dependencies. Generate ReAct loop: Reason > Act (code snippet) > Observe (test output) > Reflect (error analysis)."
    - Triggers: Milestone starts/ends; human checkpoints (e.g., "Approve baseline sim?").
    - Output: Task graphs (Markdown/YAML), self-critique prompts.
2. **Executor Agents (Chained: GPT-4o for Code Gen, o1-preview for Optimization):**
    - **CodeGen Agent (GPT-4o):** Generates boilerplate (e.g., "Write QuTiP mesolve wrapper with noise params. Include docstrings and type hints.").
    - **Debug/Opt Agent (o1-preview):** Reasoning-heavy: "Given this fidelity=0.85, hypothesize causes (detuning? Pulse overshoot?). Propose 3 fixes with math justification."
    - **Viz/Report Agent (Claude):** "From sim data, generate Bloch animation and fidelity plot. Draft README section tying to AirHound latency concepts."
3. **Evaluator Agent (Multi-Model Ensemble):**
    - Role: Automated QA. Prompt: "Run pytest suite. If fidelity <99%, score failure (1-10) and suggest iterations. Verify physics (e.g., unitarity preservation)."
    - Integration: Zed's Copilot chat for inline evals; GitHub Actions for CI.

### Workflow Phases (ReAct + Iteration Loops)

- **Initialization Loop (Week 0):** Orchestrator: "Scan SOW. Generate project README skeleton and env.yml. Reason: Ensure QuTiP install succeeds on Ubuntu 22.04."
- **Execution Pattern (Per Subtask):**

1. **Reason:** Agent analyzes spec (e.g., "For baseline sim, key risks: Time discretization errors. Mitigate with dt=0.01 ns.").
2. **Act:** Generate code/test in Zed (Copilot autocomplete).
3. **Observe:** Run sim (e.g., fidelity output). Log to notebook.
4. **Reflect:** "If fid<90%, iterate: Adjust Hamiltonian? Rerun with verbose solver." Max 3 loops/subtask before human flag.
- **Human-in-the-Loop:** Daily 15-min reviews in Zed (e.g., "Approve pulse plot?"). Escalation: If agent hallucinates (e.g., wrong Lindblad form), prompt: "Cross-check against QuTiP docs [link]."
- **Parallelism:** Zed tabs for subtasks (e.g., one for sim, one for opt); Copilot chains models automatically.
- **Traceability:** All agent outputs logged to `agent_logs/` (JSON: prompt/response/timestamp). Semantic search via Zed's built-in (or agent query: "Summarize errors from log X.").


### Prompt Engineering Standards

- **Zero-Shot + Few-Shot:** Base prompts with 2-3 examples (e.g., "Like this AirHound latency calc: fidelity = 1 - mse(target, sim).").
- **Chain-of-Density:** For reasoning: "Expand on 1 idea, then 2, up to 5, focusing on quantum noise physics."
- **Error Handling:** "If exception, classify (ImportError? Numerical instability?) and fix autonomously."
- **Hallucination Guard:** "Cite QuTiP source code line or arXiv paper for every physics claim."


### Integration with Zed IDE

- **Copilot Setup:** Enable Claude 3.5 for chat, GPT-4o for inline completions, o1 for "explain this code" queries.
- **Workflow Commands:** Custom Zed keybinds: Ctrl+Shift+R (Run ReAct loop), Ctrl+Shift+E (Eval fidelity).
- **Versioning:** Auto-commit on milestones via Zed's Git panel; branch protection in repo settings.

### Coding Standards: Power of 10 Rules (Adapted for Python/Quantum Control)

To ensure reliability, verifiability, and maintainability of mission-critical quantum control code, all agent-generated and human-reviewed code **MUST** adhere to the following 10 rules, adapted from NASA/JPL's Power of 10 for safety-critical systems:

#### Rule 1: Simple Control Flow
**Restrict all code to very simple control flow constructs.**
- **Python Adaptation:**
  - No recursion (direct or indirect). Use iteration with explicit loop bounds instead.
  - No complex nested control structures >3 levels deep.
  - Avoid `goto`-like constructs (Python lacks goto, but avoid `exec`, `eval`, and similar dynamic control).
  - Exception handling must be explicit and local (no broad `except:` clauses without specific handling).
- **Rationale:** Simpler control flow enables better static analysis, clearer reasoning about quantum pulse sequences, and easier verification of temporal bounds critical for coherence times.
- **Agent Directive:** When generating optimization loops or pulse evolution code, use explicit iteration with clear termination conditions. Flag any function call graph that could exhibit recursion.

#### Rule 2: Bounded Loops
**Give all loops a fixed upper bound that is statically verifiable.**
- **Python Adaptation:**
  - Every `for` loop must iterate over a finite, pre-determined range or collection.
  - Every `while` loop must have an explicit maximum iteration counter with assertion on overflow.
  - Example: `for i in range(MAX_ITERATIONS):` or `while condition and iter_count < MAX_ITER:`.
  - Loops traversing data structures (lists, arrays) must have length checks before iteration.
- **Rationale:** Prevents runaway optimization loops, ensures pulse simulation completes within predictable time (critical for T1/T2 coherence budget), and enables static verification of computational complexity.
- **Agent Directive:** For GRAPE/CRAB optimization loops, always specify `max_iterations` parameter. Add assertions to catch infinite loops in Lindblad evolution or parameter sweeps.

#### Rule 3: No Dynamic Memory After Initialization
**Avoid dynamic memory allocation after initialization phase.**
- **Python Adaptation:**
  - Pre-allocate all NumPy arrays for pulse sequences, state vectors, and evolution results during setup.
  - Avoid creating new large objects inside tight loops (e.g., optimization iterations).
  - Use fixed-size data structures or pre-sized lists: `results = [None] * num_steps` instead of `results = []` with appends.
  - For QuTiP Qobj creation in loops, consider object pooling or reuse.
- **Rationale:** Eliminates garbage collection pauses during time-critical simulations, makes memory usage predictable and verifiable, prevents memory fragmentation in long-running optimization sessions.
- **Agent Directive:** When generating simulation code, pre-allocate arrays for time grids, Hamiltonian matrices, and result storage. Flag any code that allocates inside optimization loops.

#### Rule 4: Function Length Limit
**No function should exceed 60 lines of code (printable on one page).**
- **Python Adaptation:**
  - Maximum 60 lines per function, excluding docstrings and blank lines.
  - If a function exceeds this, decompose into smaller logical units.
  - Applies to class methods as well.
  - Docstrings don't count toward limit, but should be concise (20 lines max).
- **Rationale:** Each function represents a verifiable logical unit. Quantum control code involves complex physics—smaller functions make it easier to validate Hamiltonian correctness, pulse constraints, and numerical stability.
- **Agent Directive:** When implementing GRAPE optimizer or Lindblad solver wrappers, break into: setup, iteration step, convergence check, result packaging. Auto-flag functions >60 lines for refactoring.

#### Rule 5: Assertion Density
**Maintain minimum two assertions per function for defensive programming.**
- **Python Adaptation:**
  - Use `assert` statements to verify preconditions, postconditions, and invariants.
  - Check physical constraints: unitarity preservation, trace conservation, Hermiticity of Hamiltonians.
  - Validate input parameters: `assert T1 > 0, "Relaxation time must be positive"`.
  - Check array shapes and bounds before operations.
  - Assertions should be side-effect free (no mutations in assert conditions).
  - On assertion failure, log diagnostic info and return error status (don't just crash).
- **Rationale:** Quantum simulations are numerically sensitive—assertions catch errors early (e.g., non-physical parameters, numerical overflow). Statistics show 1 defect per 10-100 lines; assertions intercept these.
- **Agent Directive:** For every physics function (Hamiltonian eval, pulse application, fidelity calc), include assertions on matrix dimensions, parameter ranges, and conservation laws. Minimum 2 per function, average 1 per 10 lines.

#### Rule 6: Minimal Scope
**Declare all data objects at the smallest possible scope.**
- **Python Adaptation:**
  - Use local variables within functions rather than module-level globals.
  - Leverage function parameters and return values for data flow.
  - For configuration, use immutable `NamedTuple` or `dataclass(frozen=True)` passed as parameters.
  - Avoid mutable global state (e.g., shared lists modified by multiple functions).
  - Class attributes only when truly representing object state; prefer local computation.
- **Rationale:** Reduces coupling, makes data flow explicit (critical for understanding pulse propagation through quantum system), simplifies debugging of incorrect fidelity results.
- **Agent Directive:** When generating code, default to local variables. If a variable is used in multiple functions, pass it explicitly. Flag any module-level mutable state.

#### Rule 7: Check Return Values and Validate Parameters
**Every function must validate all input parameters; every caller must check non-void return values.**
- **Python Adaptation:**
  - All functions accepting parameters must validate them (type, range, shape for arrays).
  - Use type hints: `def evolve_pulse(H0: qt.Qobj, times: np.ndarray, ...) -> SimResult:`.
  - Return `Optional[T]` or `Result[T, Error]` for operations that can fail; caller must check.
  - For critical functions, raise exceptions with detailed messages on invalid input.
  - Check return values from QuTiP functions (e.g., `mesolve` result objects).
  - Use mypy or pyright for static type checking (enforced in CI).
- **Rationale:** Quantum simulations fail silently if parameters are wrong (e.g., wrong units, swapped axes). Explicit validation prevents subtle bugs like T1/T2 confusion or Hamiltonian sign errors.
- **Agent Directive:** Every function must start with parameter validation. Every call to a function returning a result must check for success/failure. Explicitly handle QuTiP solver convergence failures.

#### Rule 8: Minimal Metaprogramming
**Limit use of Python's dynamic features and metaprogramming.**
- **Python Adaptation:**
  - Avoid `exec`, `eval`, `compile` entirely in production code.
  - Limit use of `__getattr__`, `__setattr__` dynamic attribute access.
  - No dynamic function generation or decorator stacking >2 levels.
  - Avoid `**kwargs` abuse—use explicit named parameters or typed dictionaries.
  - Imports must be at module top; no dynamic imports inside functions.
- **Rationale:** Python's dynamic nature can obscure control flow and data dependencies. For quantum control, we need to trace how parameters affect pulses—metaprogramming makes this opaque to both humans and static analyzers.
- **Agent Directive:** Generate straightforward, explicit code. If configuration flexibility is needed, use typed config classes. No runtime code generation for pulse shapes—use factory functions instead.

#### Rule 9: Restricted Indirection (Pointer/Reference Analog)
**Minimize indirection levels in data structures and function calls.**
- **Python Adaptation:**
  - Avoid nested data structures >2 levels deep (e.g., `dict[str, list[dict[str, array]]]` is too complex).
  - No function pointers stored in data structures (callback registries are acceptable only for event handling, not core logic).
  - Prefer direct function calls over dynamic dispatch via `getattr(obj, method_name)()`.
  - For pulse sequences, use simple NumPy arrays or flat dataclasses, not nested objects.
  - Avoid metaclasses and complex inheritance hierarchies.
- **Rationale:** Quantum control code needs clear data flow from Hamiltonian → evolution → fidelity. Multiple indirection levels make it hard to verify unitarity preservation or debug phase errors.
- **Agent Directive:** Use flat data structures. Represent pulses as 1D/2D NumPy arrays, not nested objects. Direct function calls for all physics calculations.

#### Rule 10: Zero Warnings, Daily Static Analysis
**All code must compile/run with zero warnings; use static analysis daily.**
- **Python Adaptation:**
  - Enable all linter warnings: `pylint`, `flake8`, `mypy --strict`.
  - Fix all warnings before commit—no suppression without documented justification.
  - Use type hints throughout; pass `mypy --strict` with zero errors.
  - Run static analyzers on every commit (pre-commit hooks or CI).
  - Code formatting with `black` (enforced, not optional).
  - Security scanning with `bandit` for unsafe patterns.
  - GitHub Actions CI must include: pytest (zero failures), mypy (zero errors), pylint (score >9.5), black (check only).
- **Rationale:** Quantum simulations are numerically fragile. Type errors, uninitialized variables, or wrong units cause silent failures. Static analysis catches these before runtime. Zero tolerance policy ensures code quality from day one.
- **Agent Directive:** After generating any code block, run `mypy` and `pylint` checks. If warnings appear, refactor immediately—never commit code with warnings. CI must fail builds with any static analysis errors.

#### Compliance Verification

**Agent Responsibilities:**
1. **Code Generation:** Apply all 10 rules by default. Reference rule number in comments for critical sections.
2. **Self-Review:** After generating code, run checklist: "Does this violate any Power of 10 rules?" If yes, refactor.
3. **Testing:** Write tests that verify rule compliance (e.g., test that loops terminate within bounds, functions are <60 lines).
4. **Documentation:** In module docstrings, note which rules are particularly relevant (e.g., "Rule 3: Pre-allocated arrays for pulse storage").

**Human Review Checkpoints:**
- Weekly code review specifically checking Power of 10 compliance.
- Any rule violation must be explicitly justified in comments and approved.
- CI dashboard should show compliance metrics (average function length, assertion density, static analysis score).

**Tooling:**
- Custom `pylint` plugin to enforce function length (Rule 4) and assertion density (Rule 5).
- Pre-commit hooks for `black`, `mypy`, `pylint`.
- GitHub Actions workflow: `.github/workflows/power_of_10_compliance.yml` running all checks.

**Why This Matters for Quantum Control:**
Quantum systems are unforgiving—a wrong sign in the Hamiltonian, a runaway optimization loop, or a memory allocation spike during pulse evolution can silently destroy fidelity. These rules aren't academic—they're your safety net for demonstrating >99% gate fidelity and building Google-worthy quantum control software. Treat them as sacred.

***

## Milestones \& Timeline

**Total Duration:** 4 weeks (flexible; buffer for iterations). Weekly human sync.

### Week 1: Foundation \& Baseline Simulation (Milestone 1: "Drift Ready")

- **Subtasks (Decomposed):**
1.1. Setup repo/env (Orchestrator: Generate .gitignore, README.md with SOW link).
1.2. Implement Hamiltonian (CodeGen: Define H0, Hc; Test: Evolve free qubit, plot energy levels).
1.3. Add noise model (Debug: Verify Lindblad stability; Reflect: Check trace preservation).
1.4. Baseline pulse sim (Executor: Naive Gaussian; Eval: Fidelity >80%, Bloch plot).
- **Dependencies:** None (sequential).
- **Success Criteria:** pytest coverage 80%; notebook with interactive Bloch widget; fidelity log CSV.
- **Agent Load:** 60% Orchestrator (planning), 40% CodeGen.


### Week 2: Optimization Core (Milestone 2: "Pulse Tuned")

- **Subtasks:**
2.1. GRAPE setup (o1: Reason on cost function = 1 - fidelity + regularization on pulse energy).
2.2. Single-gate opt (Executor: Tune for X-gate; 50 iterations, converge <1e-4 loss).
2.3. Noise sweep (Parallel: Vary T1/T2; Plot fidelity vs. decoherence).
2.4. Baseline comparison (Eval: Optimized vs. naive; >10% gain).
- **Dependencies:** Week 1 sim code.
- **Success Criteria:** >99% fidelity in low-noise; sensitivity analysis report (Jupyter PDF export).
- **Agent Load:** 50% o1 (opt reasoning), 30% Executor, 20% Evaluator.


### Week 3: Extensions \& Ties to Background (Milestone 3: "Integrated")

- **Subtasks:**
3.1. Hadamard gate opt (Reuse GRAPE; Compare rotation axes).
3.2. ML Variant (Optional: PyTorch NN for pulse gen; Train on 100 sims, reward fidelity).
3.3. Bridge Docs (Claude: Notebook section: "From AirHound Yaw Control to Qubit Steering — Latency Parallels").
3.4. Unit Tests (pytest: 20+ cases for edge: high noise, short pulses).
- **Dependencies:** Week 2 optimizer.
- **Success Criteria:** ML ext (if done) beats GRAPE by 1-2%; full test suite passes.
- **Agent Load:** 40% Claude (docs), 40% CodeGen, 20% Evaluator.


### Week 4: Polish, Repo, \& Report (Milestone 4: "Shippable")

- **Subtasks:**
4.1. Visualizations (Plotly: Interactive dashboard for param sweeps).
4.2. Technical Report (Claude: 5-10 page Markdown/PDF; Include math derivations, e.g., Bloch eq. \$ \dot{r} = \Omega \times r - \Gamma (r - r_{eq}) \$).
4.3. CI/CD (GitHub Actions: Auto-test on push; Badge for fidelity score).
4.4. Portfolio Tie-In (README: "How This Builds on NASA DL Pipelines for Noisy Signal Opt").
- **Dependencies:** All prior.
- **Success Criteria:** Repo live (stars encouraged via Reddit/LinkedIn); report with citations (5+ papers).
- **Agent Load:** 70% Claude/Viz, 30% Orchestrator (final review).

**Iteration Buffers:** 20% time per week for loops (e.g., if fidelity stalls, agent prompt: "Diagnose: Numerical? Model? Propose physics-based fix.").

***

## Deliverables

1. **GitHub Repo:** `rylanmalarchick/QubitPulseOpt` (structure: `/src/` code, `/notebooks/` sims, `/tests/`, `/docs/` report, `/data/` CSVs).
2. **Core Notebooks:** baseline_sim.ipynb, grape_opt.ipynb, ml_extension.ipynb (with %load_ext autoreload for Zed).
3. **Technical Report:** qubit_controls_sow_report.md (export to PDF via Pandoc).
4. **Science Documentation:** Single comprehensive LaTeX document `docs/science/quantum_control_theory.tex`:
   - Written progressively as features are implemented (sections for Weeks 1.1, 1.2, 1.3, etc.)
   - Covers mathematical foundations, derivations, and physical interpretations for all phases
   - Serves as learning/teaching resource with full rigor (equations, proofs, examples)
   - Organized by project phases/milestones within one unified document
   - Rendered to PDF for portfolio/academic presentation
   - Requirement: Every major physics module must have corresponding section in science doc
5. **Tests \& Metrics:** pytest suite (80% coverage); fidelity_benchmark.json.
6. **README.md:** Executive summary, install guide, demo GIF (Bloch animation), ties to your resume (AirHound/NASA).
7. **Agent Logs:** Consolidated JSON for traceability (optional upload to repo).

***

## Resources \& Tools

- **Hardware:** Standard laptop (8GB RAM min; CPU sim <5s/run).
- **Software:** As specified; Budget: Free (open-source).
- **References:**
    - QuTiP User Guide: https://qutip.org/docs/latest/
    - Key Paper: Khaneja et al., "Optimal Control of Coupled Spin Dynamics" (SIAM J. 2005).[^1]
    - Your Background: Link AirHound repo in docs for control parallels.
- **Training:** 2-3 hours on QuTiP tutorials (agent can summarize).

***

## Risks \& Mitigation

1. **Risk: Numerical Instability (High Impact, Medium Prob):** Mitigate: Use QuTiP's built-in solvers; agent reflect: "If solver diverges, reduce dt or add regularization."
2. **Risk: Agent Hallucination (Medium Impact, High Prob):** Mitigate: Few-shot prompts with verified code snippets; human review every 5 subtasks.
3. **Risk: Scope Creep (Low Impact, High Prob):** Mitigate: Strict non-goals; Orchestrator enforces: "If ML ext >20% time, defer to Phase 2."
4. **Risk: Fidelity Plateau <99% (Medium Impact, Low Prob):** Mitigate: Literature fallback (e.g., add DRAG compensation); Iteration loop max 5 tries, then human debug.
5. **Contingency:** If QuTiP issues, pivot to Cirq (Google's lib) for unitary sims only.

**Escalation:** Weekly status in Zed notes; if blocked >2 days, prompt: "Generate contingency plan with 3 alternatives."

***

## Appendices

### A: Power of 10 Compliance Checklist for Agents

**Pre-Code Generation (Every Function/Module):**
- [ ] Rule 1: No recursion planned? Control flow <3 levels deep?
- [ ] Rule 2: All loops have explicit `max_iter` or finite range?
- [ ] Rule 3: Pre-allocate arrays? No allocations in tight loops?
- [ ] Rule 4: Function will be <60 lines (excluding docstring)?
- [ ] Rule 5: Planned ≥2 assertions (preconditions, postconditions)?
- [ ] Rule 6: All variables local or passed as parameters?
- [ ] Rule 7: Parameter validation at function start? Return value checks?
- [ ] Rule 8: No `exec`/`eval`? Explicit imports?
- [ ] Rule 9: Data structures <2 levels deep? Direct function calls?
- [ ] Rule 10: Ready to pass `mypy --strict` and `pylint`?

**Post-Code Generation (Every Commit):**
```bash
# Run compliance checks
black --check src/ tests/
mypy --strict src/
pylint src/ --fail-under=9.5
pytest tests/ -v --cov=src --cov-fail-under=80
```

**Agent Self-Reflection Prompt:**
> "I have generated code for [module/function]. Reviewing against Power of 10:
> - **Rule violations detected:** [list any, or 'None']
> - **Justifications:** [if violations, explain why unavoidable and document]
> - **Refactoring needed:** [specific changes to comply]
> - **Test coverage for rules:** [which rules are verified by tests?]
> Ready for human review: Yes/No"

### B: Sample Agent Prompt Template (ReAct for Zed Copilot)

```
[Role: Executor] Task: Implement baseline sim per SOW Sec 1.3.
Reason: [Agent thinks: Key params from Hamiltonian; risks: Noise overflow.]
Act: [Generate code block; include imports, run sim, plot.]
Observe: [Sim output: fid=0.XX; plot URL.]
Reflect: [If low fid, why? Next step: Tune gamma.]
Human Check: Approve? Y/N.
```


### C: Success KPIs

- Fidelity: Mean >99% (std <0.5%) over 50 runs.
- Code Quality: PEP8 compliant (black formatter); >80% test coverage.
- Documentation: 100% functions docstringed; Report cites 5+ sources.


### D: Phase 2 Teaser (Post-v1)

- Multi-qubit iSWAP sim with crosstalk.
- RL agent (Stable Baselines3) for closed-loop pulse correction.
- Integration with your AirHound: Simulate "quantum perception" (state tomography from noisy measurements).

This SOW is your blueprint — exhaustive, adaptive, and unyielding. Agents: Execute with precision; humans: Guide the vision. Ship it, iterate relentlessly, and watch Google take notice. Let's build Dr. Malarchick's first quantum milestone.

<div align="center">⁂</div>

[^1]: https://quantum.mines.edu

