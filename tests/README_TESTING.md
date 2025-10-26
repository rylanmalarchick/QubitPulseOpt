# QubitPulseOpt Testing Infrastructure

## Overview

This document describes the testing infrastructure for QubitPulseOpt, with special emphasis on handling **stochastic optimization tests** that have inherent randomness.

## Test Categories

### 1. Deterministic Tests (`@pytest.mark.deterministic`)

Tests that should always produce the same result when run with a fixed seed.

**Characteristics:**
- Use `np.random.seed(deterministic_seed)` fixture
- Should pass 100% of the time
- Fast execution (< 5 seconds each)
- Suitable for rapid feedback during development

**Example:**
```python
@pytest.mark.deterministic
def test_s_gate_optimization(gate_optimizer, deterministic_seed):
    """Test S gate optimization with fixed seed."""
    np.random.seed(deterministic_seed)
    result = gate_optimizer.optimize_phase_gate(
        phase=np.pi / 2,
        gate_time=15.0,
        n_timeslices=20,
        max_iterations=30,
    )
    assert result.final_fidelity > 0.65
```

### 2. Stochastic Tests (`@pytest.mark.stochastic`)

Tests for algorithms with inherent randomness that may occasionally fail.

**Characteristics:**
- Test behavior over multiple random initializations
- May use `@pytest.mark.flaky(reruns=2)` for auto-retry
- Use fixed seed for reproducibility when possible
- Threshold assertions account for stochastic nature

**Example:**
```python
@pytest.mark.stochastic
@pytest.mark.flaky(reruns=2, reruns_delay=1)
def test_hadamard_high_fidelity(gate_optimizer, deterministic_seed):
    """Test Hadamard achieves high fidelity."""
    np.random.seed(deterministic_seed)
    result = gate_optimizer.optimize_hadamard(
        gate_time=30.0,
        n_timeslices=30,
        max_iterations=50,
        n_starts=5,
    )
    assert result.final_fidelity > 0.75
```

### 3. Statistical Tests (`@pytest.mark.statistical`)

Tests that run optimization multiple times and verify success rates.

**Characteristics:**
- Run 10-100 trials with different seeds
- Test statistical properties (mean, variance, success rate)
- Slower execution (30s - 5min each)
- Provide robust evidence of algorithmic behavior

**Example:**
```python
@pytest.mark.statistical
@pytest.mark.slow
def test_hadamard_success_rate(gate_optimizer, statistical_seeds):
    """Test Hadamard optimization succeeds in 70%+ of runs."""
    num_trials = 20
    successes = 0
    
    for seed in statistical_seeds[:num_trials]:
        np.random.seed(seed)
        result = gate_optimizer.optimize_hadamard(...)
        if result.final_fidelity > 0.75:
            successes += 1
    
    success_rate = successes / num_trials
    assert success_rate >= 0.70
```

### 4. Slow Tests (`@pytest.mark.slow`)

Tests that take > 10 seconds to run.

**Characteristics:**
- Long-running optimizations
- High iteration counts or many timeslices
- Run in nightly CI, not on every commit

## Running Tests

### Fast Development Cycle (Deterministic Tests Only)
```bash
# Run only deterministic, fast tests (~5 minutes)
pytest -m "deterministic and not slow"

# Or exclude stochastic tests
pytest -m "not stochastic and not slow"
```

### Standard Test Suite (Unit Tests)
```bash
# Run all unit tests, exclude statistical tests (~15-20 minutes)
pytest -m "not statistical and not slow"

# Or run all tests except statistical
pytest tests/unit/ -m "not statistical"
```

### Full Test Suite (All Tests)
```bash
# Run everything including statistical tests (~30-40 minutes)
pytest

# Run with verbose output
pytest -v

# Run with captured print statements
pytest -s
```

### Specific Test Categories
```bash
# Run only stochastic tests
pytest -m "stochastic"

# Run only statistical tests
pytest -m "statistical"

# Run only optimization tests
pytest -m "optimization"

# Run integration tests
pytest -m "integration"
```

### Parallel Execution
```bash
# Run tests in parallel (requires pytest-xdist)
pytest -n auto

# Run with 4 workers
pytest -n 4
```

## Test Fixtures

### Seed Fixtures

- **`deterministic_seed`**: Fixed seed (42) for reproducible tests
- **`statistical_seeds`**: List of 100 seeds for ensemble testing
- **`stochastic_seed`**: Hash-based seed unique to each test
- **`reset_random_state`**: Auto-reset RNG before each test (autouse)

### Quantum System Fixtures

- **`single_qubit_drift`**: Standard drift Hamiltonian
- **`single_qubit_controls`**: X and Y control Hamiltonians
- **`qubit_ground_state`**: |0⟩ state
- **`qubit_excited_state`**: |1⟩ state

### Usage Example
```python
def test_optimization(gate_optimizer, deterministic_seed):
    """Test uses deterministic seed from fixture."""
    np.random.seed(deterministic_seed)  # seed = 42
    result = gate_optimizer.optimize_hadamard(...)
    assert result.final_fidelity > threshold
```

## Flaky Test Handling

### Automatic Retries with pytest-rerunfailures

Tests marked with `@pytest.mark.flaky(reruns=N)` will automatically retry on failure.

**Configuration:**
```python
# In pytest.ini
--reruns=0        # Default: no auto-retry
--reruns-delay=1  # 1 second delay between retries

# Per-test override
@pytest.mark.flaky(reruns=3, reruns_delay=2)
def test_unstable_optimization(...):
    ...
```

**When to Use:**
- Known stochastic tests that fail < 20% of the time
- Tests sensitive to random initialization
- RB experiments with occasional fit failures

**When NOT to Use:**
- Tests that fail deterministically
- Tests with bugs (fix the bug instead!)
- Tests that fail > 50% of the time (use statistical tests instead)

### Current Flaky Tests

1. **`test_sdg_gate_optimization`** - Phase gate optimization, `reruns=2`
2. **`test_hadamard_high_fidelity`** - Hadamard gate, `reruns=2`
3. **`test_run_rb_experiment_with_noise`** - RB fit degeneracy, `reruns=3`

## Best Practices

### ✅ DO

1. **Use fixed seeds for deterministic tests**
   ```python
   np.random.seed(deterministic_seed)
   ```

2. **Use realistic thresholds for stochastic tests**
   ```python
   # Good: achievable threshold
   assert result.final_fidelity > 0.75
   
   # Bad: overly optimistic threshold
   assert result.final_fidelity > 0.99
   ```

3. **Mark tests appropriately**
   ```python
   @pytest.mark.deterministic
   @pytest.mark.stochastic
   @pytest.mark.statistical
   @pytest.mark.slow
   ```

4. **Write statistical tests for success rates**
   ```python
   # Test that algorithm succeeds 80% of the time
   success_rate = sum(1 for _ in range(20) if optimize() > threshold) / 20
   assert success_rate >= 0.80
   ```

5. **Use `@pytest.mark.flaky` sparingly and document why**
   ```python
   @pytest.mark.flaky(reruns=2, reruns_delay=1)
   def test_challenging_optimization(...):
       """
       Test occasionally fails due to unlucky random initialization.
       Flaky marker allows retry to reduce false negatives.
       """
   ```

### ❌ DON'T

1. **Don't run stochastic tests without seeds**
   ```python
   # Bad: non-reproducible
   def test_optimization(...):
       # No seed set!
       result = optimizer.optimize(...)
   ```

2. **Don't use `@pytest.mark.flaky` to hide bugs**
   ```python
   # Bad: test has a real bug
   @pytest.mark.flaky(reruns=10)  # Red flag!
   def test_broken_function(...):
       assert broken_function() > 0  # Fix the function instead!
   ```

3. **Don't make tests unnecessarily slow**
   ```python
   # Bad: huge budgets for unit test
   result = optimize(max_iterations=10000, n_timeslices=500)
   
   # Good: realistic budgets for unit test
   result = optimize(max_iterations=30, n_timeslices=20)
   ```

4. **Don't use overly tight thresholds**
   ```python
   # Bad: 99.9% threshold for stochastic unit test
   assert result.final_fidelity > 0.999
   
   # Good: realistic threshold
   assert result.final_fidelity > 0.70
   ```

## CI/CD Integration

### Fast CI (Every Commit)
Run deterministic tests only for quick feedback (< 5 min):
```yaml
# .github/workflows/fast-tests.yml
- name: Fast Tests
  run: pytest -m "deterministic and not slow" --maxfail=5
```

### Standard CI (Pull Requests)
Run all unit tests including stochastic (< 20 min):
```yaml
# .github/workflows/pr-tests.yml
- name: Unit Tests
  run: pytest -m "not statistical" --reruns=2
```

### Nightly CI (Scheduled)
Run full test suite including statistical tests (< 40 min):
```yaml
# .github/workflows/nightly-tests.yml
- name: Full Test Suite
  run: pytest -v --reruns=3
```

## Debugging Flaky Tests

### 1. Run Test Multiple Times
```bash
# Run test 10 times to check for flakiness
pytest tests/unit/test_gates.py::test_sdg_gate -v --count=10

# Run until failure (requires pytest-repeat)
pytest tests/unit/test_gates.py::test_sdg_gate -v --count=100 -x
```

### 2. Check with Different Seeds
```python
# In test file, temporarily add:
@pytest.mark.parametrize("seed", [42, 100, 101, 102, 103])
def test_optimization(gate_optimizer, seed):
    np.random.seed(seed)
    # ... test code ...
```

### 3. Add Debug Output
```python
def test_optimization(gate_optimizer, deterministic_seed):
    np.random.seed(deterministic_seed)
    result = gate_optimizer.optimize_hadamard(...)
    
    # Debug output
    print(f"Final fidelity: {result.final_fidelity:.6f}")
    print(f"Iterations: {result.metadata.get('iterations', 'N/A')}")
    print(f"Pulse max: {np.max(np.abs(result.optimized_pulses)):.4f}")
    
    assert result.final_fidelity > threshold
```

### 4. Convert to Statistical Test
If test is too flaky, convert to statistical ensemble test:
```python
# Before: flaky test
@pytest.mark.flaky(reruns=5)  # Fails 30% of the time!
def test_optimization(...):
    result = optimize(...)
    assert result.fidelity > 0.75

# After: statistical test
@pytest.mark.statistical
def test_optimization_success_rate(statistical_seeds):
    successes = sum(
        1 for seed in statistical_seeds[:20]
        if (np.random.seed(seed), optimize(...).fidelity > 0.75)[1]
    )
    assert successes >= 14  # 70% success rate
```

## Test Statistics

### Current Test Suite Performance

| Category | Count | Time | Pass Rate |
|----------|-------|------|-----------|
| Deterministic | ~580 | ~5 min | 100% |
| Stochastic | ~30 | ~10 min | 95-99% |
| Statistical | ~10 | ~15 min | 100% |
| **Total** | **~620** | **~28 min** | **~99%** |

### Known Flaky Tests (< 5% failure rate)
1. `test_sdg_gate_optimization` - 2-5% failure rate
2. `test_hadamard_high_fidelity` - 1-3% failure rate
3. `test_run_rb_experiment_with_noise` - 3-8% failure rate

All flaky tests have `@pytest.mark.flaky(reruns=2-3)` and should retry automatically.

## References

- [pytest documentation](https://docs.pytest.org/)
- [pytest-rerunfailures](https://github.com/pytest-dev/pytest-rerunfailures)
- [pytest markers](https://docs.pytest.org/en/stable/mark.html)
- Task 1.5: Stochastic Test Infrastructure (REMAINING_TASKS_CHECKLIST.md)

## Maintenance

This testing infrastructure was implemented as part of **Task 1.5: Stochastic Test Infrastructure**.

**Last Updated:** 2025-01-27  
**Author:** Orchestrator Agent  
**Status:** ✅ Complete

For questions or improvements, see `docs/REMAINING_TASKS_CHECKLIST.md` or contact the development team.