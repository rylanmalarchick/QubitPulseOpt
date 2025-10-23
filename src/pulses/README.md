# Pulses Module

Advanced pulse shaping techniques for quantum control.

## Overview

This module provides state-of-the-art pulse shaping methods for high-fidelity quantum gates, including:

- **DRAG Pulses**: Derivative Removal by Adiabatic Gate for leakage suppression
- **Composite Pulses**: Error-robust pulse sequences (BB1, CORPSE, SK1, Knill)
- **Adiabatic Techniques**: Landau-Zener sweeps and STIRAP for robust population transfer

## Modules

### `drag.py` - DRAG Pulses

DRAG (Derivative Removal by Adiabatic Gate) pulses suppress leakage errors in multi-level systems by adding a derivative correction to the quadrature component.

**Key Features:**
- Gaussian envelope with derivative correction: Q(t) = β * dI(t)/dt
- Optimal β parameter calculation: β_opt = -α/(2Ω)
- 3-level system leakage analysis
- QuTiP integration for time-dependent evolution
- Gate pulse generation (X, Y, X/2, Y/2, Hadamard)

**Example:**
```python
from src.pulses.drag import create_drag_x_gate

# Create X gate with DRAG correction
pulse = create_drag_x_gate(
    amplitude=1.0,
    sigma=0.1,
    beta=0.5,
    anharmonicity=-0.3
)

# Evaluate envelope at time t
I, Q = pulse.envelope(t=0.5)

# Get QuTiP coefficient functions
H_coeff_x, H_coeff_y = pulse.qutip_coefficients()
```

**Reference:**
- Motzoi et al., Phys. Rev. Lett. 103, 110501 (2009)

---

### `composite.py` - Composite Pulses

Composite pulse sequences achieve error robustness by combining multiple pulses with specific phases and angles to cancel systematic errors to first order.

**Key Sequences:**
- **BB1**: Broadband detuning error cancellation
- **CORPSE**: Detuning + amplitude error correction
- **SK1**: Solovay-Kitaev decomposition for arbitrary rotations
- **Knill**: Optimized 5-pulse error-correcting sequence

**Key Features:**
- Error robustness analysis (detuning, amplitude, phase)
- Robustness radius calculation
- Sequence comparison and benchmarking
- Gate fidelity with systematic errors
- Pulse envelope generation

**Example:**
```python
from src.pulses.composite import create_bb1_x, create_corpse

# BB1 sequence for X gate
bb1 = create_bb1_x(amplitude=1.0, pulse_duration=0.1)

# CORPSE sequence
corpse = create_corpse(theta=np.pi, amplitude=1.0, pulse_duration=0.1)

# Analyze error robustness
errors, fidelities = bb1.analyze_detuning_robustness(
    detuning_range=np.linspace(-0.2, 0.2, 50)
)

# Compute robustness radius
radius = bb1.robustness_radius(
    detuning_range=(-0.2, 0.2),
    amplitude_range=(0.9, 1.1),
    threshold=0.99
)
```

**References:**
- Wimperis, J. Magn. Reson. A 109, 221 (1994) [CORPSE]
- Wimperis, Phys. Rev. A 49, 3266 (1994) [BB1]

---

### `adiabatic.py` - Adiabatic Techniques

Adiabatic passage methods achieve robust population transfer by slowly varying the Hamiltonian, keeping the system in an instantaneous eigenstate.

**Key Techniques:**
- **Landau-Zener Sweeps**: Two-level avoided crossing sweeps
- **STIRAP**: Stimulated Raman Adiabatic Passage for three-level systems
- **Adiabaticity Checker**: General analysis tool for time-dependent Hamiltonians

**Key Features:**
- Multiple sweep profiles (linear, tanh, gaussian)
- Dark state tracking in STIRAP
- Counter-intuitive pulse ordering
- Transfer efficiency calculation
- Adiabaticity parameter γ(t) = E_gap²/|dH/dt|
- Sweep time optimization

**Example:**
```python
from src.pulses.adiabatic import create_landau_zener_sweep, create_stirap_pulse

# Landau-Zener sweep
lz = create_landau_zener_sweep(
    delta_range=(-10.0, 10.0),
    coupling=2.0,
    sweep_time=5.0,
    sweep_type='linear'
)

# Check adiabaticity
metrics = lz.check_adiabaticity(threshold=10.0)
print(f"Transition probability: {metrics.transition_probability:.4f}")

# STIRAP pulse
stirap = create_stirap_pulse(
    omega_pump=10.0,
    omega_stokes=10.0,
    pulse_duration=20.0,
    delay=-2.0,  # Counter-intuitive ordering
    pulse_shape='gaussian'
)

# Calculate transfer efficiency
efficiency = stirap.transfer_efficiency(n_points=100)
print(f"Transfer efficiency: {efficiency:.2%}")
```

**References:**
- Landau, Phys. Z. Sowjetunion 2, 46 (1932)
- Zener, Proc. R. Soc. Lond. A 137, 696 (1932)
- Bergmann et al., Rev. Mod. Phys. 70, 1003 (1998) [STIRAP]

---

## Installation & Dependencies

The pulses module requires:
- NumPy
- SciPy
- QuTiP (>= 4.7)

Already installed if you have the main project environment.

---

## Testing

Run unit tests for the pulses module:

```bash
# All pulse tests
pytest tests/unit/test_drag.py tests/unit/test_composite.py tests/unit/test_adiabatic.py -v

# Individual modules
pytest tests/unit/test_drag.py -v
pytest tests/unit/test_composite.py -v
pytest tests/unit/test_adiabatic.py -v
```

**Test Coverage:**
- `test_drag.py`: 32 tests
- `test_composite.py`: 44 tests
- `test_adiabatic.py`: 38 tests
- **Total**: 114 tests

---

## Usage Guidelines

### Choosing a Pulse Technique

**Use DRAG when:**
- You have a multi-level system (transmon, etc.)
- Leakage to non-computational states is a problem
- You need fast gates with minimal pulse duration
- Anharmonicity is known

**Use Composite Pulses when:**
- You have systematic errors (detuning, amplitude drift)
- You need robustness without calibration
- Gate time is less critical than fidelity
- Errors are within the first-order correction range (~10%)

**Use Adiabatic Techniques when:**
- You need maximum robustness to parameter variations
- Long gate times are acceptable
- Population transfer (not coherent superposition) is the goal
- You have coupling to lossy intermediate states (STIRAP)

### Performance Considerations

- **DRAG**: Fast (single pulse), moderate fidelity (99%+)
- **Composite**: Medium speed (5× single pulse), high fidelity (99.9%+)
- **Adiabatic**: Slow (10-100× single pulse), very high robustness

---

## Mathematical Background

### DRAG Pulse Theory

For a three-level system with levels |0⟩, |1⟩, |2⟩ and anharmonicity α:

```
H = ω₀|0⟩⟨0| + ω₁|1⟩⟨1| + ω₂|2⟩⟨2| + Ω(t)(|0⟩⟨1| + |1⟩⟨0|)
```

where ω₂ - ω₁ = ω₁ - ω₀ + α (α < 0 for transmons).

The DRAG pulse adds a derivative term to suppress leakage:
```
Ω(t) = Ω_I(t) + i Ω_Q(t)
Ω_Q(t) = β dΩ_I(t)/dt
```

Optimal β = -α/(2Ω₀) minimizes leakage to second order.

### Composite Pulse Theory

Composite pulses cancel errors using operator identities. For BB1:
```
U_BB1 = U_X(φ) U_Y(π) U_X(2π-2φ) U_Y(π) U_X(φ)
```

where φ = arccos(-1/4). This cancels detuning errors δ to first order:
```
F ≈ 1 - O(δ²)  (vs. F ≈ 1 - O(δ) for single pulse)
```

### Adiabatic Theorem

For time-dependent Hamiltonian H(t), adiabatic evolution requires:
```
γ(t) = (E_n - E_m)² / |⟨m|dH/dt|n⟩| ≫ 1  for all t
```

Landau-Zener transition probability for linear sweep:
```
P_LZ = exp(-π Ω² / (2 |dΔ/dt|))
```

---

## Integration with Other Modules

The pulse module integrates with:

- **`optimization/`**: Use pulses as initial guesses for GRAPE/Krotov
- **`dynamics/`**: Pulses work with Lindblad dynamics (decoherence)
- **`robustness/`**: Analyze pulse robustness to noise and errors
- **`visualization/`**: Plot pulse envelopes and trajectories

---

## Examples & Notebooks

See demo notebooks for detailed examples:
- `notebooks/04_advanced_pulse_shaping.ipynb`: DRAG, composite, and adiabatic demonstrations
- `notebooks/05_gate_optimization.ipynb`: Integration with GRAPE/Krotov
- `notebooks/06_robustness_analysis.ipynb`: Error analysis and benchmarking

---

## Contributing

To add a new pulse type:

1. Create new class in appropriate file or new file in `src/pulses/`
2. Implement key methods: `envelope()`, `qutip_coefficients()`, etc.
3. Add comprehensive unit tests in `tests/unit/`
4. Update this README with usage examples
5. Add to documentation and demo notebooks

---

## License

MIT License - See LICENSE file in project root

---

## Citation

If you use this code in research, please cite:

```bibtex
@software{quantum_controls_2025,
  title = {Quantum Controls: Advanced Pulse Shaping and Optimization},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/quantumControls}
}
```

---

**Last Updated:** 2025-01-27  
**Module Version:** 1.0.0  
**Status:** Production Ready ✅