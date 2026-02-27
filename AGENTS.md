# QubitPulseOpt

Paper submitted to Quantum (arXiv:2511.12799). Results are final.

## Gotchas

- DRAG beta: β = -1/(2α). For α/2π = -200 MHz → β = 0.398 (Motzoi et al. PRL 103, 110501)
- GRAPE gradient ordering: time slices are forward-indexed
- 3-level transmon model (not 2-level); leakage to |2⟩ is tracked
- T2 ≤ 2·T1 always; IQM Garnet params: T1=50µs, T2=70µs, α/2π=-200 MHz

## Environment

- Python: `.venv/bin/python` (not bare `python3`)
- Tests: `.venv/bin/python -m pytest tests/ -v --tb=short`
- Markers: `slow`, `deterministic`, `stochastic`
