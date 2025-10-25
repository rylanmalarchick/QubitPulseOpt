# Task 7 - Phase 4: Pre-existing Test Failures

**Date**: 2025-01-27
**Status**: NOT STARTED
**Priority**: Medium
**Test Failures**: 21 tests (3.3% of total suite)

## Overview

This document tracks pre-existing test failures discovered during Task 7 Phase 2 assertion work. These failures existed before the Power of 10 compliance work began and are NOT caused by Task 7 changes. They represent separate algorithmic and logic bugs that require dedicated debugging.

**Note**: These failures were confirmed to exist in commit `a9e7a61` (before Task 7 work began).

## Current Test Status

- **Total tests**: 635
- **Passing**: 609 (96.0%)
- **Failing**: 21 (3.3%)
- **Skipped**: 3 (0.5%)
- **xfailed**: 2 (0.3%)

## Failure Categories

### 1. GRAPE Optimizer Failures (9 tests) - HIGH PRIORITY

**Issue**: GRAPE optimizer produces very low fidelities (0.33-0.46) instead of expected >0.95 for standard gates.

**Root Cause**: Algorithmic issues in gradient-based optimization
- Learning rate either overshoots or makes minimal progress
- Line search not functioning properly
- Possible gradient computation or normalization issues
- Step-size control inadequate

**Affected Tests**:
```
tests/unit/test_gates.py::TestHadamardGateOptimization::test_hadamard_high_fidelity
tests/unit/test_gates.py::TestHadamardGateOptimization::test_hadamard_metadata
tests/unit/test_gates.py::TestPhaseGateOptimization::test_s_gate_optimization
tests/unit/test_gates.py::TestPhaseGateOptimization::test_t_gate_optimization
tests/unit/test_gates.py::TestPhaseGateOptimization::test_z_gate_optimization
tests/unit/test_gates.py::TestPhaseGateOptimization::test_sdg_gate_optimization
tests/unit/test_gates.py::TestPauliGateOptimization::test_z_gate_optimization
tests/unit/test_gates.py::TestArbitraryRotations::test_rotation_about_z_axis
tests/unit/test_gates.py::TestArbitraryRotations::test_rotation_about_arbitrary_axis
```

**Reproduction**:
```bash
cd QubitPulseOpt
. venv/bin/activate
python -m pytest tests/unit/test_gates.py::TestHadamardGateOptimization::test_hadamard_high_fidelity -xvs
```

**Expected Behavior**: Hadamard gate optimization should achieve fidelity >0.95

**Actual Behavior**: Achieves fidelity ~0.33-0.46

**Investigation Findings**:
- Default learning rate (0.1) often overshoots and degrades fidelity
- Small learning rates (0.001-0.01) produce minimal improvement
- Gradient computation appears correct (verified manually)
- Line-search and stability features not working as expected

**Recommended Fix Strategy**:
1. Debug line-search implementation in `src/optimization/grape.py`
2. Review gradient normalization and sign conventions
3. Implement adaptive learning rate (e.g., Adam, RMSprop)
4. Add gradient clipping and momentum properly
5. Test with numerical gradient checks (finite differences)
6. Add detailed logging to track fidelity evolution

**Files to Investigate**:
- `src/optimization/grape.py` (lines 400-700 - optimization loop)
- `tests/unit/test_gates.py` (lines 200-400 - gate optimization tests)

**Estimated Effort**: 6-8 hours
- 2-3 hours: Reproduce and isolate issue
- 2-3 hours: Debug line search and gradient handling
- 2 hours: Test fixes and validate across all gate types

---

### 2. Randomized Benchmarking Failures (8 tests) - MEDIUM PRIORITY

**Issue**: Clifford group closure test and RB experiment execution failures.

**Affected Tests**:
```
tests/unit/test_benchmarking.py::TestCliffordGroup::test_clifford_closure
tests/unit/test_benchmarking.py::TestRBExperiment::test_run_rb_experiment_ideal
tests/unit/test_benchmarking.py::TestRBExperiment::test_rb_result_attributes
tests/unit/test_benchmarking.py::TestInterleavedRB::test_run_interleaved_rb
tests/unit/test_benchmarking.py::TestInterleavedRB::test_interleaved_rb_with_noise
tests/unit/test_benchmarking.py::TestVisualization::test_visualize_rb_decay_no_fit
tests/unit/test_benchmarking.py::TestIntegration::test_rb_with_multiple_noise_levels
tests/unit/test_benchmarking.py::TestEdgeCases::test_rb_with_short_sequences_only
```

**Specific Issue (test_clifford_closure)**:
```
AssertionError: Clifford group not closed under composition
```

The test multiplies two Clifford gates and expects the result to be in the Clifford group, but the product is not found. This suggests:
- Numerical precision issues in gate comparison
- Incomplete Clifford group generation
- Global phase handling issues

**Reproduction**:
```bash
cd QubitPulseOpt
. venv/bin/activate
python -m pytest tests/unit/test_benchmarking.py::TestCliffordGroup::test_clifford_closure -xvs
```

**Recommended Fix Strategy**:
1. Review Clifford group generation in `src/optimization/benchmarking.py`
2. Check gate comparison tolerance (allow for numerical errors)
3. Verify global phase handling (Cliffords are defined up to global phase)
4. Add debug output to show which product is not matching
5. Compare with QuTiP's built-in Clifford utilities

**Files to Investigate**:
- `src/optimization/benchmarking.py` (Clifford group class)
- `tests/unit/test_benchmarking.py` (RB tests)

**Estimated Effort**: 4-6 hours
- 2 hours: Understand Clifford group implementation
- 2 hours: Debug closure test and gate matching
- 1-2 hours: Fix RB experiment execution issues

---

### 3. Euler Decomposition Failures (4 tests) - MEDIUM PRIORITY

**Issue**: Euler angle decomposition returns 0.0 fidelity instead of >0.999 for standard gates.

**Affected Tests**:
```
tests/unit/test_compilation.py::TestEulerDecomposition::test_decompose_hadamard
tests/unit/test_compilation.py::TestEulerDecomposition::test_decompose_s_gate
tests/unit/test_compilation.py::TestEulerDecomposition::test_decompose_t_gate
tests/unit/test_compilation.py::TestEulerDecomposition::test_decompose_arbitrary_unitary
```

**Specific Issue (test_decompose_hadamard)**:
```
assert decomp.fidelity > 0.999
E   assert 0.0 > 0.999
E    +  where 0.0 = EulerDecomposition(φ₁=-180.00°, θ=90.00°, φ₂=0.00°, F=0.00000000).fidelity
```

The decomposition produces correct-looking angles but reports 0.0 fidelity, suggesting:
- Fidelity calculation bug in Euler decomposition
- Unitary reconstruction from angles may be incorrect
- Missing or incorrect global phase handling

**Reproduction**:
```bash
cd QubitPulseOpt
. venv/bin/activate
python -m pytest tests/unit/test_compilation.py::TestEulerDecomposition::test_decompose_hadamard -xvs
```

**Recommended Fix Strategy**:
1. Review `decompose_to_euler()` function in `src/optimization/compilation.py`
2. Check unitary reconstruction from Euler angles
3. Verify fidelity calculation between original and reconstructed unitary
4. Compare with standard Euler decomposition references (e.g., Nielsen & Chuang)
5. Add unit test for angle→unitary→angle round-trip

**Files to Investigate**:
- `src/optimization/compilation.py` (lines 500-600 - Euler decomposition)
- `tests/unit/test_compilation.py` (TestEulerDecomposition class)

**Estimated Effort**: 3-4 hours
- 1 hour: Review Euler decomposition math
- 1-2 hours: Debug fidelity calculation
- 1 hour: Test and validate fix

---

## Overall Resolution Plan

### Phase 4.1: GRAPE Optimizer (HIGH PRIORITY)
**Goal**: Achieve >95% fidelity for all standard gate optimizations
**Effort**: 6-8 hours
**Success Criteria**: All 9 GRAPE gate tests passing

### Phase 4.2: Euler Decomposition (MEDIUM PRIORITY)
**Goal**: Correct fidelity calculation in Euler decomposition
**Effort**: 3-4 hours
**Success Criteria**: All 4 Euler decomposition tests passing

### Phase 4.3: Randomized Benchmarking (MEDIUM PRIORITY)
**Goal**: Fix Clifford group closure and RB execution
**Effort**: 4-6 hours
**Success Criteria**: All 8 RB tests passing

### Total Estimated Effort: 13-18 hours (2-3 days)

---

## Testing Strategy

After each fix:
1. Run specific test: `pytest tests/unit/test_X.py::TestY::test_z -xvs`
2. Run category tests: `pytest tests/unit/test_gates.py -xvs` (for GRAPE)
3. Run full suite: `pytest tests/ -q --tb=no`
4. Verify no regressions in previously passing tests

---

## Success Metrics

**Target**: 100% test pass rate (635/635 tests passing)

**Current**: 96.0% (609/635 passing)
**After Phase 4**: 100% (635/635 passing)

---

## Dependencies

- Python 3.12+
- QuTiP 5.2.1
- NumPy, SciPy
- pytest

---

## References

### GRAPE Optimization
- Khaneja et al., J. Magn. Reson. 172, 296 (2005) - Original GRAPE paper
- Machnes et al., Phys. Rev. Lett. 120, 150401 (2018) - GRAPE improvements

### Euler Decomposition
- Nielsen & Chuang, "Quantum Computation and Quantum Information", Chapter 4
- Shende et al., IEEE Trans. CAD 25, 1000 (2006) - Universal gate decomposition

### Randomized Benchmarking
- Magesan et al., Phys. Rev. Lett. 106, 180504 (2011) - Scalable RB
- Gambetta et al., Phys. Rev. A 85, 042311 (2012) - Interleaved RB

---

## Notes

- All failures confirmed pre-existing (present in commit `a9e7a61`)
- No failures introduced by Task 7 assertion work
- These are algorithmic/logic bugs, not compliance issues
- Should be tracked as separate issues from Power of 10 compliance
- Can be addressed in parallel with Phase 2.2 (function decomposition)

---

## Conclusion

Phase 4 represents the final cleanup of pre-existing test failures discovered during Task 7. While these are not Power of 10 compliance issues, resolving them will:

1. Bring test suite to 100% passing
2. Fix critical GRAPE optimizer performance
3. Improve code quality and reliability
4. Validate benchmarking and compilation modules

**Recommendation**: Address in priority order (GRAPE → Euler → RB) to maximize impact.