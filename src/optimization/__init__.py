"""
Optimal Control Module
======================

This module implements gradient-based optimal control methods for quantum systems,
including GRAPE (Gradient Ascent Pulse Engineering) and Krotov's method.

Optimal control aims to find pulse shapes Ω(t) that maximize gate fidelity
while satisfying physical constraints (amplitude limits, bandwidth, etc.).

Methods Implemented:
-------------------
1. GRAPE: Piecewise-constant pulses optimized via gradient ascent
2. Krotov: Monotonically convergent method with smooth pulse updates

Physics Background:
------------------
The goal is to find control fields that implement a target unitary U_target
with maximum fidelity:

    F = |⟨ψ_target|U_control|ψ_init⟩|²

Or for unitary gates:

    F = (1/N) |Tr(U_target† U_control)|

The optimization iteratively improves the pulse shape by computing gradients
of the fidelity with respect to control parameters.

References:
----------
- Khaneja et al., J. Magn. Reson. 172, 296 (2005) - GRAPE
- Floether et al., New J. Phys. 14, 073023 (2012) - GRAPE review
- Reich et al., J. Chem. Phys. 136, 104103 (2012) - Krotov
- QuTiP documentation: https://qutip.org/docs/latest/guide/guide-control.html

Author: Orchestrator Agent
Date: 2025-01-27
SOW Reference: Phase 2 - Optimal Control Theory
"""

from .grape import GRAPEOptimizer, GRAPEResult
from .krotov import KrotovOptimizer, KrotovResult
from .robustness import RobustnessTester, RobustnessResult, compare_pulse_robustness
from .gates import (
    UniversalGates,
    GateResult,
    euler_angles_from_unitary,
    rotation_from_euler_angles,
)
from .compilation import GateCompiler, CompiledCircuit, EulerDecomposition
from .filter_functions import (
    FilterFunctionCalculator,
    FilterFunctionResult,
    NoisePSD,
    NoiseInfidelityCalculator,
    NoiseTailoredOptimizer,
    visualize_filter_function,
    compute_filter_function_sum_rule,
    analyze_pulse_noise_sensitivity,
)
from .benchmarking import (
    CliffordGroup,
    RBSequenceGenerator,
    RBExperiment,
    InterleavedRB,
    RBResult,
    depolarizing_noise,
    amplitude_damping_noise,
    visualize_rb_decay,
)

__all__ = [
    "GRAPEOptimizer",
    "GRAPEResult",
    "KrotovOptimizer",
    "KrotovResult",
    "RobustnessTester",
    "RobustnessResult",
    "compare_pulse_robustness",
    "UniversalGates",
    "GateResult",
    "euler_angles_from_unitary",
    "rotation_from_euler_angles",
    "GateCompiler",
    "CompiledCircuit",
    "EulerDecomposition",
    "FilterFunctionCalculator",
    "FilterFunctionResult",
    "NoisePSD",
    "NoiseInfidelityCalculator",
    "NoiseTailoredOptimizer",
    "visualize_filter_function",
    "compute_filter_function_sum_rule",
    "analyze_pulse_noise_sensitivity",
    "CliffordGroup",
    "RBSequenceGenerator",
    "RBExperiment",
    "InterleavedRB",
    "RBResult",
    "depolarizing_noise",
    "amplitude_damping_noise",
    "visualize_rb_decay",
]
