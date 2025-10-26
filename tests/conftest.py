"""
Pytest configuration and shared fixtures for QubitPulseOpt test suite.

This module provides:
1. Seed fixtures for deterministic and stochastic tests
2. Common test utilities
3. Pytest hooks for test infrastructure

Author: Orchestrator Agent
Date: 2025-01-27
Task: 1.5 - Stochastic Test Infrastructure
"""

import pytest
import numpy as np

# Conditional imports to avoid errors when dependencies not available
try:
    import qutip as qt

    QUTIP_AVAILABLE = True
except ImportError:
    QUTIP_AVAILABLE = False
    qt = None


# ============================================================================
# Seed Management for Deterministic vs Stochastic Tests
# ============================================================================

# Default seed for deterministic tests
DETERMINISTIC_SEED = 42

# Seeds for statistical ensemble tests
STATISTICAL_SEEDS = list(range(100, 200))  # 100 different seeds


@pytest.fixture
def deterministic_seed():
    """
    Fixture providing a fixed seed for deterministic tests.

    Use this fixture to ensure reproducible results in unit tests.

    Example:
        def test_optimization(deterministic_seed):
            np.random.seed(deterministic_seed)
            # ... test code with random initialization ...
    """
    return DETERMINISTIC_SEED


@pytest.fixture
def statistical_seeds():
    """
    Fixture providing multiple seeds for statistical ensemble tests.

    Use this for tests that verify stochastic behavior over multiple runs.

    Example:
        def test_optimization_success_rate(statistical_seeds):
            successes = 0
            for seed in statistical_seeds[:10]:  # Use first 10 seeds
                np.random.seed(seed)
                result = optimize(...)
                if result.fidelity > threshold:
                    successes += 1
            assert successes >= 8  # 80% success rate
    """
    return STATISTICAL_SEEDS


@pytest.fixture(autouse=True)
def reset_random_state():
    """
    Automatically reset random state before each test.

    This ensures test isolation by resetting both NumPy and QuTiP
    random number generators.
    """
    # Save original state
    np_state = np.random.get_state()

    # Reset to a known state for isolation
    np.random.seed(DETERMINISTIC_SEED)

    # Run the test
    yield

    # Restore original state (optional - for cleanup)
    # np.random.set_state(np_state)


@pytest.fixture
def stochastic_seed(request):
    """
    Fixture providing different seeds for each stochastic test run.

    This uses pytest's internal randomization to provide variety
    while still being reproducible with --randomly-seed option.

    Example:
        @pytest.mark.stochastic
        def test_random_behavior(stochastic_seed):
            np.random.seed(stochastic_seed)
            # ... test stochastic optimization ...
    """
    # Use pytest-randomly plugin seed if available, otherwise use test name hash
    if hasattr(request.config, "getoption"):
        random_seed = request.config.getoption("--randomly-seed", default=None)
        if random_seed is not None:
            return int(random_seed)

    # Fallback: hash of test name for consistent but varied seeds
    test_name = request.node.nodeid
    seed = hash(test_name) % (2**31)  # Keep seed in valid range
    return seed


# ============================================================================
# Quantum System Fixtures
# ============================================================================


@pytest.fixture
def single_qubit_drift():
    """Standard single-qubit drift Hamiltonian (zero for universal control)."""
    if not QUTIP_AVAILABLE:
        pytest.skip("QuTiP not available")
    return 2 * np.pi * 0.0 * qt.sigmaz() / 2


@pytest.fixture
def single_qubit_controls():
    """Standard single-qubit control Hamiltonians (X and Y)."""
    if not QUTIP_AVAILABLE:
        pytest.skip("QuTiP not available")
    return [qt.sigmax(), qt.sigmay()]


@pytest.fixture
def qubit_ground_state():
    """Single qubit ground state |0⟩."""
    if not QUTIP_AVAILABLE:
        pytest.skip("QuTiP not available")
    return qt.basis(2, 0)


@pytest.fixture
def qubit_excited_state():
    """Single qubit excited state |1⟩."""
    if not QUTIP_AVAILABLE:
        pytest.skip("QuTiP not available")
    return qt.basis(2, 1)


# ============================================================================
# Test Utilities
# ============================================================================


def assert_fidelity_above(result, threshold, msg=None):
    """
    Helper to assert fidelity is above threshold with informative message.

    Args:
        result: GateResult or similar object with final_fidelity attribute
        threshold: Minimum acceptable fidelity
        msg: Optional custom message
    """
    if msg is None:
        msg = f"Fidelity {result.final_fidelity:.6f} below threshold {threshold:.6f}"
    assert result.final_fidelity > threshold, msg


def assert_unitary(operator, atol=1e-10):
    """
    Assert that an operator is unitary (U† U = I).

    Args:
        operator: QuTiP Qobj to check
        atol: Absolute tolerance for identity check
    """
    if not QUTIP_AVAILABLE:
        pytest.skip("QuTiP not available")

    dim = operator.shape[0]
    product = operator.dag() * operator
    identity = qt.qeye(dim)

    np.testing.assert_allclose(
        product.full(),
        identity.full(),
        atol=atol,
        err_msg=f"Operator is not unitary: U† U = {product}",
    )


# ============================================================================
# Pytest Hooks
# ============================================================================


def pytest_configure(config):
    """
    Configure pytest with custom markers and settings.

    This is called once at the start of the test session.
    """
    # Ensure markers are registered (though they should be in pytest.ini)
    config.addinivalue_line(
        "markers",
        "stochastic: marks tests with inherent randomness (may need multiple runs)",
    )
    config.addinivalue_line(
        "markers",
        "deterministic: marks tests that should always produce same result with fixed seed",
    )
    config.addinivalue_line(
        "markers",
        "flaky: marks tests known to be flaky (will auto-retry with pytest-rerunfailures)",
    )


def pytest_collection_modifyitems(config, items):
    """
    Modify test collection to add markers automatically.

    This hook runs after test collection and can add markers based on
    test names or other heuristics.
    """
    for item in items:
        # Auto-mark optimization tests as potentially stochastic
        if "optimization" in item.nodeid.lower() and "stochastic" not in item.keywords:
            # Only add if not explicitly marked as deterministic
            if "deterministic" not in item.keywords:
                item.add_marker(pytest.mark.optimization)

        # Auto-mark slow tests based on name patterns
        if any(
            pattern in item.nodeid.lower()
            for pattern in ["long", "slow", "integration"]
        ):
            if "slow" not in item.keywords:
                item.add_marker(pytest.mark.slow)


# ============================================================================
# Session-level reporting
# ============================================================================


@pytest.fixture(scope="session", autouse=True)
def print_test_environment(request):
    """Print test environment information at session start."""
    print("\n" + "=" * 70)
    print("QubitPulseOpt Test Suite - Stochastic Test Infrastructure")
    print("=" * 70)
    print(f"Deterministic seed: {DETERMINISTIC_SEED}")
    print(f"NumPy version: {np.__version__}")
    if QUTIP_AVAILABLE:
        print(f"QuTiP version: {qt.__version__}")
    else:
        print("QuTiP: Not available")

    # Check for pytest-rerunfailures
    if hasattr(request.config, "pluginmanager"):
        if request.config.pluginmanager.has_plugin("rerunfailures"):
            print("✓ pytest-rerunfailures plugin available")
        else:
            print("✗ pytest-rerunfailures plugin not found")

    print("=" * 70 + "\n")
