"""
Power of 10 Compliance Constants
=================================

This module defines explicit bounds for all loops and iterations in the quantum
control project, in compliance with NASA/JPL Power of 10 Rule 2:

    "Give all loops a fixed upper bound. It must be trivially possible for a
     checking tool to prove statically that the loop cannot exceed a preset
     upper bound on the number of iterations."

All constants are conservative upper bounds that should never be exceeded in
normal operation. If exceeded, an assertion will fire to prevent unbounded
execution.

Usage:
    from src.constants import MAX_ITERATIONS, MAX_PARAMS

    for i in range(min(actual_iterations, MAX_ITERATIONS)):
        assert i < MAX_ITERATIONS, f"Exceeded {MAX_ITERATIONS} iterations"
        # ... loop body ...

Author: Agent
Date: 2025-01-29
SOW Reference: Task 7 - Power of 10 Compliance
"""

# =============================================================================
# Optimization Loop Bounds
# =============================================================================

# Maximum number of optimization iterations
# Used in: GRAPE, Krotov, gradient descent, BFGS, etc.
MAX_ITERATIONS = 10000

# Maximum number of function evaluations in scipy.optimize
MAX_FUNCTION_EVALS = 50000

# Maximum number of gradient evaluations
MAX_GRADIENT_EVALS = 10000

# Maximum number of line search steps
MAX_LINE_SEARCH_STEPS = 100

# Maximum backtracking steps in optimization
MAX_BACKTRACKING_STEPS = 50


# =============================================================================
# System Size Bounds
# =============================================================================

# Maximum number of qubits in the system
# 20 qubits → 2^20 = 1M dimensional Hilbert space (memory limit)
MAX_QUBITS = 20

# Maximum Hilbert space dimension
# For composite systems: dim = prod(subsystem_dims)
MAX_HILBERT_DIM = 1048576  # 2^20

# Maximum number of energy levels per subsystem
MAX_LEVELS_PER_SUBSYSTEM = 10


# =============================================================================
# Control Parameter Bounds
# =============================================================================

# Maximum number of control parameters in optimization
# (e.g., pulse amplitudes, phases, frequencies)
MAX_PARAMS = 10000

# Maximum number of control Hamiltonians
MAX_CONTROL_HAMILTONIANS = 100

# Maximum number of time slices in discretized pulse
MAX_TIMESLICES = 100000

# Maximum number of pulse segments (piecewise definition)
MAX_PULSE_SEGMENTS = 10000


# =============================================================================
# Monte Carlo & Sampling Bounds
# =============================================================================

# Maximum number of Monte Carlo samples for robustness testing
MAX_MONTE_CARLO_SAMPLES = 100000

# Maximum number of random trajectories in stochastic simulation
MAX_TRAJECTORIES = 10000

# Maximum number of bootstrap resamples
MAX_BOOTSTRAP_SAMPLES = 10000

# Maximum number of parameter sweep points
MAX_SWEEP_POINTS = 10000


# =============================================================================
# Data Structure Bounds
# =============================================================================

# Maximum number of items in parameter dictionaries
MAX_DICT_ITEMS = 10000

# Maximum depth of nested configuration structures
MAX_CONFIG_DEPTH = 10

# Maximum number of environment variables to process
MAX_ENV_VARS = 1000

# Maximum number of results to store in history
MAX_HISTORY_LENGTH = 100000

# Maximum number of benchmark runs
MAX_BENCHMARK_RUNS = 10000


# =============================================================================
# Visualization Bounds
# =============================================================================

# Maximum number of plot points per trace
MAX_PLOT_POINTS = 100000

# Maximum number of animation frames
MAX_ANIMATION_FRAMES = 10000

# Maximum number of subplots in a figure
MAX_SUBPLOTS = 100

# Maximum number of data series per plot
MAX_PLOT_SERIES = 100


# =============================================================================
# File I/O Bounds
# =============================================================================

# Maximum number of files to process in batch
MAX_FILES_BATCH = 10000

# Maximum number of lines to read from a file
MAX_FILE_LINES = 1000000

# Maximum number of export formats to process
MAX_EXPORT_FORMATS = 100


# =============================================================================
# Numerical Solver Bounds
# =============================================================================

# Maximum number of timesteps in ODE/Lindblad evolution
MAX_SOLVER_TIMESTEPS = 1000000

# Maximum number of solver substeps (adaptive step size)
MAX_SOLVER_SUBSTEPS = 100000

# Maximum number of Krylov subspace iterations
MAX_KRYLOV_ITERATIONS = 1000


# =============================================================================
# Matrix Operation Bounds
# =============================================================================

# Maximum number of matrix exponentiations
MAX_MATRIX_EXPM_CALLS = 100000

# Maximum number of eigenvalue computations
MAX_EIGENVALUE_CALLS = 10000

# Maximum number of SVD decompositions
MAX_SVD_CALLS = 10000


# =============================================================================
# Testing & Validation Bounds
# =============================================================================

# Maximum number of test cases
MAX_TEST_CASES = 10000

# Maximum number of validation checks
MAX_VALIDATION_CHECKS = 10000

# Maximum number of assertions per function (soft limit)
RECOMMENDED_ASSERTIONS_PER_FUNCTION = 2


# =============================================================================
# Physics Constants (for assertion bounds)
# =============================================================================

# Fidelity bounds (should be [0, 1] but allow small numerical error)
MIN_FIDELITY = -0.01
MAX_FIDELITY = 1.01

# Probability bounds
MIN_PROBABILITY = -1e-10
MAX_PROBABILITY = 1.0 + 1e-10

# Energy bounds (in units of ħω, typical for quantum systems)
MIN_ENERGY = -1000.0  # GHz
MAX_ENERGY = 1000.0  # GHz

# Pulse amplitude bounds (normalized units)
MIN_PULSE_AMPLITUDE = -100.0
MAX_PULSE_AMPLITUDE = 100.0

# Time bounds (in microseconds, typical for superconducting qubits)
MIN_TIME = 0.0
MAX_TIME = 1000.0  # 1 ms

# Decoherence time bounds (T1, T2 in microseconds)
MIN_DECOHERENCE_TIME = 0.1  # 100 ns
MAX_DECOHERENCE_TIME = 10000.0  # 10 ms


# =============================================================================
# Assertion Helpers
# =============================================================================


def assert_iteration_bound(iteration: int, max_iter: int = MAX_ITERATIONS) -> None:
    """
    Assert that iteration count is within bounds.

    Args:
        iteration: Current iteration number
        max_iter: Maximum allowed iterations

    Raises:
        AssertionError: If iteration exceeds bound
    """
    assert iteration < max_iter, (
        f"Iteration {iteration} exceeds maximum {max_iter}. "
        f"Possible infinite loop or unbounded iteration."
    )


def assert_parameter_count(n_params: int, max_params: int = MAX_PARAMS) -> None:
    """
    Assert that parameter count is within bounds.

    Args:
        n_params: Number of parameters
        max_params: Maximum allowed parameters

    Raises:
        AssertionError: If parameter count exceeds bound
    """
    assert n_params <= max_params, (
        f"Parameter count {n_params} exceeds maximum {max_params}. "
        f"Consider reducing problem size or increasing MAX_PARAMS."
    )


def assert_system_size(dim: int, max_dim: int = MAX_HILBERT_DIM) -> None:
    """
    Assert that Hilbert space dimension is within bounds.

    Args:
        dim: Hilbert space dimension
        max_dim: Maximum allowed dimension

    Raises:
        AssertionError: If dimension exceeds bound
    """
    assert dim <= max_dim, (
        f"Hilbert space dimension {dim} exceeds maximum {max_dim}. "
        f"Memory requirements may be too large."
    )


def assert_fidelity_valid(fidelity: float) -> None:
    """
    Assert that fidelity is within valid bounds.

    Args:
        fidelity: Fidelity value to check

    Raises:
        AssertionError: If fidelity is outside [0, 1] with small tolerance
    """
    assert MIN_FIDELITY <= fidelity <= MAX_FIDELITY, (
        f"Fidelity {fidelity} outside valid range [{MIN_FIDELITY}, {MAX_FIDELITY}]. "
        f"Check numerical stability and computation."
    )


# =============================================================================
# Module Metadata
# =============================================================================

__all__ = [
    # Optimization
    "MAX_ITERATIONS",
    "MAX_FUNCTION_EVALS",
    "MAX_GRADIENT_EVALS",
    "MAX_LINE_SEARCH_STEPS",
    "MAX_BACKTRACKING_STEPS",
    # System
    "MAX_QUBITS",
    "MAX_HILBERT_DIM",
    "MAX_LEVELS_PER_SUBSYSTEM",
    # Control
    "MAX_PARAMS",
    "MAX_CONTROL_HAMILTONIANS",
    "MAX_TIMESLICES",
    "MAX_PULSE_SEGMENTS",
    # Sampling
    "MAX_MONTE_CARLO_SAMPLES",
    "MAX_TRAJECTORIES",
    "MAX_BOOTSTRAP_SAMPLES",
    "MAX_SWEEP_POINTS",
    # Data structures
    "MAX_DICT_ITEMS",
    "MAX_CONFIG_DEPTH",
    "MAX_ENV_VARS",
    "MAX_HISTORY_LENGTH",
    "MAX_BENCHMARK_RUNS",
    # Visualization
    "MAX_PLOT_POINTS",
    "MAX_ANIMATION_FRAMES",
    "MAX_SUBPLOTS",
    "MAX_PLOT_SERIES",
    # File I/O
    "MAX_FILES_BATCH",
    "MAX_FILE_LINES",
    "MAX_EXPORT_FORMATS",
    # Solvers
    "MAX_SOLVER_TIMESTEPS",
    "MAX_SOLVER_SUBSTEPS",
    "MAX_KRYLOV_ITERATIONS",
    # Matrix ops
    "MAX_MATRIX_EXPM_CALLS",
    "MAX_EIGENVALUE_CALLS",
    "MAX_SVD_CALLS",
    # Testing
    "MAX_TEST_CASES",
    "MAX_VALIDATION_CHECKS",
    "RECOMMENDED_ASSERTIONS_PER_FUNCTION",
    # Physics bounds
    "MIN_FIDELITY",
    "MAX_FIDELITY",
    "MIN_PROBABILITY",
    "MAX_PROBABILITY",
    "MIN_ENERGY",
    "MAX_ENERGY",
    "MIN_PULSE_AMPLITUDE",
    "MAX_PULSE_AMPLITUDE",
    "MIN_TIME",
    "MAX_TIME",
    "MIN_DECOHERENCE_TIME",
    "MAX_DECOHERENCE_TIME",
    # Helpers
    "assert_iteration_bound",
    "assert_parameter_count",
    "assert_system_size",
    "assert_fidelity_valid",
]
