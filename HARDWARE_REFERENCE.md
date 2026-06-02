# Hardware Reference: IQM Garnet

This document records the IQM Garnet device parameters QubitPulseOpt targets and
how the framework connects to IQM hardware. Device figures are taken from the
public IQM whitepaper (arXiv:2408.12433) rather than internal notes; the live
IQM calibration API was not accessible during this work.

> **Status:** All results in this project are from simulation using
> hardware-representative parameters. No circuits were executed on a physical
> QPU. The code below is connectivity/translation infrastructure, validated
> against the IQM REST API surface but not against live quantum execution.

## Device: IQM Garnet (20-qubit)

IQM Garnet is a 20 computational-qubit superconducting processor (plus 30 tunable
couplers) on a square-lattice topology with nearest-neighbor connectivity. Qubits
and couplers are flux-tunable transmons; readout is frequency-multiplexed dispersive.

### Verified figures (IQM whitepaper, arXiv:2408.12433)

| Quantity | Value | Notes |
|----------|-------|-------|
| Relaxation time T₁ | ~40 µs (order of) | "improve T₁ from the order of 40 µs to above 100 µs" (whitepaper §5) |
| Median single-qubit gate error | 9 × 10⁻⁴ (F ≈ 99.91%) | randomized benchmarking, distance-2 groups |
| Median CZ (two-qubit) error | 5 × 10⁻³ (F ≈ 99.5%) | interleaved RB |
| Median readout error | 3 × 10⁻² | simultaneous QPU readout |
| ZZ coupling (tunable) | up to 50 MHz, off at idle | tunable-coupler design |
| Quantum Volume | 2⁵ = 32 | volumetric benchmark |
| CLOPS | 2600 | circuit layer operations per second |
| Qubit frequencies | ~5–6 GHz | dispersive readout band (whitepaper Fig. 3) |

The whitepaper does **not** publish a median T₂ or a per-qubit anharmonicity.

### Parameters used in this project

Simulations use hardware-representative parameters consistent with the figures above:

| Parameter | Value | Source |
|-----------|-------|--------|
| T₁ | 37 µs | representative; consistent with whitepaper "order of 40 µs" |
| T₂ | 9.6 µs | representative transmon dephasing time (not in whitepaper) |
| Anharmonicity α/2π | −200 MHz | representative transmon value (not in whitepaper) |
| Qubit frequency ω_q/2π | 5.0 GHz | representative |
| Baseline gate time | 20 ns | matches whitepaper CZ duration range (20–40 ns) |

These are the values in `verified_results/hardware_parameters.json` and the paper
(`paper/quantum/`). The error-budget math is self-consistent with them
(ε_T₁ ≈ T/2T₁ ≈ 2.7×10⁻⁴, ε_φ ≈ T/T₂ − T/2T₁ ≈ 1.8×10⁻³ at T = 20 ns).

## Connecting to IQM hardware

IQM Garnet is reachable through IQM Resonance and AWS Braket. The framework's
integration code lives in:

- `src/hardware/iqm_backend.py` — backend manager (auth via `IQM_TOKEN`, backend
  selection, job submission/polling over the IQM REST API).
- `src/hardware/iqm_translator.py` — converts optimized I/Q pulse arrays into IQM
  pulse-schedule waveforms.
- `scripts/query_iqm_calibration.py` — queries system topology and (when the API
  is reachable) live calibration data.

```python
from scripts.query_iqm_calibration import query_iqm_system

system_info = query_iqm_system()   # requires IQM_TOKEN and network access
print(system_info["name"], system_info["qubits"])
```

Optional hardware dependencies are listed in `requirements-hardware.txt`.

## Reference

M. Algaba et al., *Technology and Performance Benchmarks of IQM's 20-Qubit
Quantum Computer*, arXiv:2408.12433 (2024).
