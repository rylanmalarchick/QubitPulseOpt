# Contributing to QubitPulseOpt

Thank you for your interest in contributing to QubitPulseOpt! This document provides guidelines and instructions for contributing to the project.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Code Standards](#code-standards)
- [Testing Requirements](#testing-requirements)
- [Submitting Changes](#submitting-changes)
- [Review Process](#review-process)

---

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please:
- Be respectful and considerate in all interactions
- Focus on constructive feedback
- Accept differing viewpoints gracefully
- Prioritize the project's best interests

---

## Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/QubitPulseOpt.git
cd QubitPulseOpt

# Add upstream remote
git remote add upstream https://github.com/ORIGINAL_OWNER/QubitPulseOpt.git
```

### 2. Set Up Development Environment

**Option A: Automated Setup (Recommended)**
```bash
# Run the automated setup script
./scripts/setup_dev_env.sh
```

**Option B: Manual Setup**
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# OR: venv\Scripts\activate  # Windows

# Install core dependencies
pip install qutip numpy scipy matplotlib jupyter

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Verify setup
pre-commit run --all-files
pytest tests/unit -v -m "not slow"
```

### 3. Create a Feature Branch

```bash
# Update your main branch
git checkout main
git pull upstream main

# Create a feature branch
git checkout -b feature/your-feature-name
```

---

## Development Workflow

### Making Changes

1. **Write Code**: Make your changes following our [Code Standards](#code-standards)
2. **Write Tests**: Add tests for new functionality (see [Testing Requirements](#testing-requirements))
3. **Run Tests Locally**: `pytest tests/ -v`
4. **Check Compliance**: `python scripts/compliance/power_of_10_checker.py src --verbose`

### Pre-commit Hooks

Pre-commit hooks run automatically on `git commit` and check:
- Code formatting (Black)
- Import sorting (isort)
- Linting (Flake8)
- Power of 10 compliance
- Syntax validation
- Trailing whitespace, end-of-file issues

**Manual execution:**
```bash
# Run all hooks on staged files
pre-commit run

# Run all hooks on all files
pre-commit run --all-files

# Run specific hook
pre-commit run black --all-files
```

**Bypassing hooks** (use sparingly):
```bash
# Skip all hooks
git commit --no-verify -m "WIP: work in progress"

# Skip specific hooks
SKIP=flake8,power-of-10-compliance git commit -m "Message"
```

See [`docs/DEVELOPER_GUIDE_PRECOMMIT.md`](docs/DEVELOPER_GUIDE_PRECOMMIT.md) for detailed information.

---

## Code Standards

### Power of 10 Compliance

We follow adapted NASA/JPL Power of 10 rules for safety-critical code:

1. **Simple Control Flow**: No recursion, nesting depth <3 levels
2. **Bounded Loops**: All loops must have explicit upper bounds
3. **No Dynamic Allocation After Init**: Pre-allocate arrays where possible
4. **Function Length**: ≤60 lines per function
5. **Assertion Density**: ≥2 assertions per function
6. **Minimal Scope**: Use local variables, explicit data flow
7. **Check Return Values**: Validate all inputs/outputs
8. **Minimal Metaprogramming**: Avoid `exec`, `eval`
9. **Restricted Indirection**: Flat data structures
10. **Zero Warnings**: Code must pass all static analysis

**Check compliance:**
```bash
python scripts/compliance/power_of_10_checker.py src --verbose
```

### Code Style

- **Formatter**: Black (88 character line length)
- **Import Sorting**: isort (Black-compatible profile)
- **Linter**: Flake8 with plugins (docstrings, bugbear)
- **Docstring Convention**: NumPy style

**Example:**
```python
def optimize_pulse(
    hamiltonian: Hamiltonian,
    target_gate: np.ndarray,
    max_iterations: int = 100,
) -> OptimizationResult:
    """
    Optimize control pulse for target gate operation.

    Parameters
    ----------
    hamiltonian : Hamiltonian
        System Hamiltonian defining dynamics
    target_gate : np.ndarray
        Target unitary gate (2x2 complex matrix)
    max_iterations : int, optional
        Maximum optimization iterations (default: 100)

    Returns
    -------
    OptimizationResult
        Optimization results including final fidelity and pulse

    Raises
    ------
    ValueError
        If hamiltonian is not properly initialized
    AssertionError
        If target_gate is not unitary

    Examples
    --------
    >>> ham = Hamiltonian(omega=1.0)
    >>> target = pauli_x()
    >>> result = optimize_pulse(ham, target, max_iterations=50)
    >>> print(f"Fidelity: {result.fidelity:.4f}")
    """
    # Assertions (Power of 10 Rule 5)
    assert hamiltonian is not None, "Hamiltonian cannot be None"
    assert target_gate.shape == (2, 2), "Target must be 2x2 matrix"
    assert max_iterations > 0, "Iterations must be positive"
    
    # Implementation with bounded loop (Power of 10 Rule 2)
    for iteration in range(max_iterations):
        # ... implementation ...
        pass
    
    return result
```

### File Organization

```
QubitPulseOpt/
├── src/                    # Source code (main package)
│   ├── hamiltonian/        # Hamiltonian definitions
│   ├── pulses/             # Pulse generators
│   ├── optimization/       # Optimization algorithms
│   ├── noise/              # Noise models
│   ├── benchmarking/       # Benchmarking tools
│   └── visualization/      # Visualization utilities
├── tests/                  # Test suite
│   ├── unit/               # Unit tests
│   └── integration/        # Integration tests
├── notebooks/              # Jupyter notebooks
├── docs/                   # Documentation
├── examples/               # Example scripts
└── scripts/                # Utility scripts
```

---

## Testing Requirements

### Test Categories

We use pytest markers to categorize tests:
- `@pytest.mark.unit`: Fast, isolated unit tests
- `@pytest.mark.integration`: Multi-component integration tests
- `@pytest.mark.slow`: Tests requiring >1 second
- `@pytest.mark.deterministic`: Reproducible tests with fixed seeds
- `@pytest.mark.stochastic`: Tests with random elements
- `@pytest.mark.statistical`: Tests validating statistical properties
- `@pytest.mark.flaky(reruns=3)`: Tests that may fail intermittently

### Writing Tests

**Required for new features:**
```python
import pytest
import numpy as np
from src.optimization.gates import optimize_gate


class TestGateOptimization:
    """Test suite for gate optimization."""

    @pytest.mark.unit
    @pytest.mark.deterministic
    def test_x_gate_optimization_basic(self, deterministic_seed):
        """Test basic X-gate optimization converges."""
        # Setup
        target = np.array([[0, 1], [1, 0]], dtype=complex)
        
        # Execute
        result = optimize_gate(
            target, 
            n_timeslices=10, 
            max_iterations=50,
            random_seed=42
        )
        
        # Assertions
        assert result.fidelity > 0.99, "Fidelity should exceed 0.99"
        assert result.converged, "Optimization should converge"
        assert len(result.pulse) == 10, "Pulse length should match timeslices"

    @pytest.mark.stochastic
    @pytest.mark.flaky(reruns=2)
    def test_x_gate_optimization_stochastic(self):
        """Test X-gate optimization with random initialization."""
        # For stochastic tests, use ensemble statistics
        fidelities = []
        n_trials = 10
        
        for _ in range(n_trials):
            result = optimize_gate(target, n_timeslices=10)
            fidelities.append(result.fidelity)
        
        # Statistical assertions
        mean_fidelity = np.mean(fidelities)
        assert mean_fidelity > 0.95, "Mean fidelity should be high"
        assert np.std(fidelities) < 0.1, "Results should be consistent"
```

### Running Tests

```bash
# Fast unit tests only
pytest tests/unit -v -m "not slow"

# All tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=src --cov-report=html

# Parallel execution
pytest tests/unit -v -n auto

# Specific markers
pytest tests/ -v -m "deterministic"
pytest tests/ -v -m "not (slow or flaky)"
```

See [`tests/README_TESTING.md`](tests/README_TESTING.md) for comprehensive testing guide.

---

## Submitting Changes

### Commit Messages

Use clear, descriptive commit messages:

```
# Good
Add GRAPE optimizer with line search
Fix fidelity calculation for non-unitary operators
Update benchmarking to handle edge cases

# Bad
Fix bug
Update code
WIP
```

**Format:**
```
<type>: <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Formatting, missing semicolons, etc.
- `refactor`: Code restructuring without behavior change
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Example:**
```
feat: Add Krotov optimization algorithm

Implement Krotov's method for quantum optimal control with:
- Convergence criteria based on gradient norm
- Configurable step size and regularization
- Support for bounded controls

Closes #123
```

### Pull Request Process

1. **Ensure Quality**:
   ```bash
   # Run all checks
   pre-commit run --all-files
   pytest tests/ -v --cov=src
   python scripts/compliance/power_of_10_checker.py src
   ```

2. **Update Documentation**:
   - Add/update docstrings
   - Update relevant markdown docs
   - Add examples if appropriate

3. **Push Changes**:
   ```bash
   git push origin feature/your-feature-name
   ```

4. **Create Pull Request**:
   - Go to GitHub and create a PR
   - Use the PR template
   - Link related issues
   - Provide clear description of changes

5. **Address Review Feedback**:
   - Respond to comments
   - Make requested changes
   - Push updates to the same branch

### PR Checklist

- [ ] Code follows Power of 10 standards
- [ ] All tests pass locally
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] Pre-commit hooks pass
- [ ] No merge conflicts with main
- [ ] Commit messages are clear and descriptive
- [ ] PR description explains the change

---

## Review Process

### What Reviewers Look For

1. **Correctness**: Does the code work as intended?
2. **Test Coverage**: Are there adequate tests?
3. **Code Quality**: Does it follow our standards?
4. **Documentation**: Is it well-documented?
5. **Design**: Is the approach sound?
6. **Performance**: Any unnecessary overhead?

### Response Time

- Initial review: Within 3-5 business days
- Follow-up reviews: Within 1-2 business days
- Urgent fixes: Within 1 business day

### Approval Criteria

PRs require:
- ✅ All CI checks passing
- ✅ At least one approving review
- ✅ No unresolved conversations
- ✅ Up-to-date with main branch

---

## Common Tasks

### Adding a New Optimization Algorithm

1. Create module in `src/optimization/`
2. Implement with base class interface
3. Add unit tests in `tests/unit/test_optimization/`
4. Add example notebook
5. Update documentation

### Adding a New Pulse Type

1. Create pulse generator in `src/pulses/`
2. Add validation and bounds checking
3. Add unit tests
4. Add visualization example
5. Document mathematical form

### Fixing a Bug

1. Write a failing test that reproduces the bug
2. Fix the bug
3. Verify test now passes
4. Add regression test if needed
5. Document fix in commit message

---

## Getting Help

- **Documentation**: [`docs/`](docs/)
- **Testing Guide**: [`tests/README_TESTING.md`](tests/README_TESTING.md)
- **Pre-commit Guide**: [`docs/DEVELOPER_GUIDE_PRECOMMIT.md`](docs/DEVELOPER_GUIDE_PRECOMMIT.md)
- **Issues**: Open a GitHub issue for questions or bugs
- **Discussions**: Use GitHub Discussions for general questions

---

## Recognition

Contributors will be:
- Listed in `CONTRIBUTORS.md`
- Acknowledged in release notes
- Attributed in commit history

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to QubitPulseOpt!** 🎉
