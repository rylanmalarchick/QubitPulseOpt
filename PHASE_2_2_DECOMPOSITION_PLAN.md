# Phase 2.2: Function Decomposition Plan

**Date**: 2025-01-27
**Status**: IN PROGRESS
**Goal**: Decompose 48 functions > 60 lines into smaller units (≤60 lines each)

## Overview

Power of 10 Rule 4 requires functions to be ≤60 lines to improve:
- **Readability**: Easier to understand small, focused functions
- **Testability**: Each function can be unit tested independently
- **Maintainability**: Changes localized to specific helpers
- **Debuggability**: Smaller call stacks, clearer error traces

## Strategy

### Priority-Based Approach
Tackle largest functions first for maximum impact:
1. **Priority 1 (Lines 150+)**: 5 functions - Immediate decomposition
2. **Priority 2 (Lines 100-149)**: 10 functions - High priority
3. **Priority 3 (Lines 70-99)**: 14 functions - Medium priority
4. **Priority 4 (Lines 61-69)**: 19 functions - Low priority

### Decomposition Pattern
```python
# BEFORE: Monolithic function (100+ lines)
def large_function(params):
    # Setup (10 lines)
    # Validation (15 lines)
    # Core logic block 1 (30 lines)
    # Core logic block 2 (25 lines)
    # Post-processing (20 lines)
    return result

# AFTER: Decomposed into helpers
def large_function(params):
    validated_params = _validate_params(params)
    intermediate = _compute_core_logic_1(validated_params)
    result = _compute_core_logic_2(intermediate)
    return _post_process_result(result)

def _validate_params(params):
    # 15 lines
    return validated

def _compute_core_logic_1(params):
    # 30 lines
    return intermediate

def _compute_core_logic_2(data):
    # 25 lines
    return result

def _post_process_result(result):
    # 20 lines
    return final
```

## Function Inventory (48 Functions)

### Priority 1: CRITICAL (150+ lines) - 5 functions

1. **`src/optimization/grape.py::optimize_unitary`** - 230 lines
   - Largest function in codebase
   - Main GRAPE optimization loop
   - Extract: initialization, iteration loop, convergence check, result assembly
   
2. **`src/optimization/krotov.py::optimize_unitary`** - 132 lines
   - Krotov optimization loop
   - Extract: propagator computation, gradient computation, control update
   
3. **`src/optimization/grape.py::GRAPEOptimizer.__init__`** - 136 lines
   - Already mostly validation (converted to ValueError in Phase 2.1)
   - Extract: control validation, time validation, optimization params validation
   
4. **`src/optimization/krotov.py::KrotovOptimizer.__init__`** - 123 lines
   - Similar to GRAPE init
   - Extract: validation helper functions
   
5. **`src/pulses/drag.py::compare_with_gaussian`** - 119 lines
   - DRAG vs Gaussian comparison
   - Extract: simulation, analysis, plotting

### Priority 2: HIGH (100-149 lines) - 10 functions

6. **`src/optimization/gates.py::_optimize_gate`** - 116 lines
7. **`src/pulses/drag.py::scan_beta_parameter`** - 112 lines
8. **`src/visualization/dashboard.py::compare_pulses`** - 112 lines
9. **`src/visualization/reports.py::generate_summary`** - 107 lines
10. **`src/io/export.py::HardwareConfig.__init__`** - 104 lines
11. **`src/pulses/shapes.py::gaussian_pulse`** - 103 lines
12. **`src/optimization/gates.py::optimize_rotation`** - 102 lines
13. **`src/optimization/gates.py::check_clifford_closure`** - 101 lines
14. **`src/pulses/composite.py::CompositePulse.__init__`** - 100 lines (est.)
15. **`src/visualization/reports.py::create_publication_figure`** - 85 lines

### Priority 3: MEDIUM (70-99 lines) - 14 functions

16. **`src/pulses/shapes.py::square_pulse`** - 84 lines
17. **`src/pulses/drag.py::create_drag_pulse_for_gate`** - 83 lines
18. **`src/visualization/dashboard.py::plot_heatmap`** - 81 lines
19. **`src/optimization/filter_functions.py::visualize_filter_function`** - 81 lines
20. **`src/pulses/adiabatic.py::adiabatic_condition`** - 78 lines
21. **`src/optimization/gates.py::euler_angles_from_unitary`** - 78 lines
22. **`src/visualization/dashboard.py::plot_states`** - 78 lines
23. **`src/visualization/dashboard.py::create_animation`** - 78 lines
24. **`src/hamiltonian/control.py::hamiltonian`** - 77 lines
25. **`src/pulses/shapes.py::drag_pulse`** - 77 lines
26. **`src/hamiltonian/lindblad.py::compare_with_unitary`** - 76 lines
27. **`src/optimization/compilation.py::_compile_joint`** - 75 lines
28. **`src/optimization/compilation.py::compile_circuit`** - 75 lines
29. **`src/io/export.py::export_pulse_json`** - 74 lines

### Priority 4: LOW (61-69 lines) - 19 functions

30-48. Functions between 61-69 lines (lower impact)

## Decomposition Schedule

### Day 1: Priority 1 Functions (5 functions, ~150-230 lines each)
- **Morning**: GRAPE optimize_unitary (230 lines → 4-5 helpers)
- **Afternoon**: Krotov optimize_unitary (132 lines → 3-4 helpers)

### Day 2: Priority 1 & Priority 2 (8 functions)
- **Morning**: GRAPE/Krotov __init__ (136, 123 lines → validation helpers)
- **Afternoon**: compare_with_gaussian, _optimize_gate, scan_beta_parameter

### Day 3: Priority 2 & Priority 3 (12 functions)
- **All day**: Medium-sized functions (85-112 lines)

### Day 4: Priority 3 & Priority 4 (remaining functions)
- **All day**: Smaller functions (61-78 lines)

## Testing Strategy

For each decomposition:
1. **Before**: Run existing tests for the module
   ```bash
   pytest tests/unit/test_MODULE.py -xvs
   ```

2. **During**: Write unit tests for new helper functions
   - Test helpers in isolation
   - Test edge cases
   
3. **After**: Verify no regressions
   ```bash
   pytest tests/unit/test_MODULE.py -xvs
   pytest tests/ -q --tb=no  # Full suite
   ```

4. **Compliance Check**:
   ```bash
   python scripts/compliance/power_of_10_checker.py src --verbose
   ```

## Decomposition Guidelines

### 1. Extract Logical Blocks
- **Setup/Initialization** → `_initialize_X()`
- **Validation** → `_validate_X()` (if not already done in Phase 2.1)
- **Core computation** → `_compute_X()`
- **Post-processing** → `_process_X()`, `_format_X()`
- **Plotting/Visualization** → `_plot_X()`, `_render_X()`

### 2. Helper Function Naming
- Prefix with `_` for internal helpers
- Use descriptive verb-noun names
- Examples:
  - `_compute_propagators()`
  - `_evaluate_gradient()`
  - `_update_controls()`
  - `_check_convergence()`
  - `_assemble_result()`

### 3. Preserve Behavior
- No functional changes
- Same inputs/outputs for main function
- Same side effects (logging, warnings, etc.)
- Same error handling

### 4. Documentation
- Each helper gets docstring
- Main function references helpers in its docstring
- Update module-level docs if needed

### 5. Type Hints
- Add type hints to all new helpers
- Helps with understanding and testing

## Example: GRAPE optimize_unitary Decomposition

### Before (230 lines)
```python
def optimize_unitary(self, U_target, u_init=None, step_decay=1.0):
    # Validation (20 lines)
    # Initialize controls (30 lines)
    # Main optimization loop (150 lines)
      # Compute propagators
      # Compute gradients
      # Update controls
      # Line search
      # Check convergence
    # Assemble result (30 lines)
```

### After (< 60 lines each)
```python
def optimize_unitary(self, U_target, u_init=None, step_decay=1.0):
    """Main optimization function - orchestrates the GRAPE algorithm."""
    self._validate_target_unitary(U_target)
    u = self._initialize_controls(u_init)
    opt_state = self._run_optimization_loop(U_target, u, step_decay)
    return self._assemble_optimization_result(opt_state)

def _validate_target_unitary(self, U_target):
    """Validate target unitary dimensions and properties."""
    # 15 lines

def _initialize_controls(self, u_init):
    """Initialize control pulse array."""
    # 25 lines

def _run_optimization_loop(self, U_target, u, step_decay):
    """Execute main GRAPE iteration loop."""
    # 50 lines (calls other helpers)
    
def _compute_iteration_step(self, U_target, u, iteration):
    """Compute one GRAPE iteration."""
    # 40 lines
    
def _update_controls_with_gradient(self, u, gradient, learning_rate):
    """Update controls using computed gradient."""
    # 25 lines

def _check_convergence(self, fidelity, gradient_norm, iteration):
    """Check if optimization has converged."""
    # 15 lines

def _assemble_optimization_result(self, opt_state):
    """Package optimization results into OptimizationResult."""
    # 30 lines
```

## Success Criteria

- [ ] All 48 functions reduced to ≤60 lines
- [ ] All tests still passing (609/635 minimum)
- [ ] No performance degradation (< 5% slowdown acceptable)
- [ ] Compliance checker shows 0 Rule 4 violations
- [ ] All new helpers have docstrings
- [ ] Code review approved

## Risk Mitigation

1. **Test Failures**: 
   - Commit after each successful decomposition
   - Easy rollback if needed
   
2. **Performance Degradation**:
   - Profile critical functions before/after
   - Inline small helpers if needed (with comments)
   
3. **Complexity Increase**:
   - Don't over-decompose (balance readability)
   - Keep related code together
   
4. **Time Overrun**:
   - Focus on Priority 1 & 2 first (15 functions = biggest impact)
   - Priority 3 & 4 can be addressed incrementally

## Progress Tracking

Track in this document:

### Completed (0/48)
- [ ] None yet

### In Progress (0/48)
- [ ] None yet

### Blocked (0/48)
- [ ] None yet

## Next Steps

1. Start with `src/optimization/grape.py::optimize_unitary` (230 lines)
2. Test thoroughly
3. Commit with descriptive message
4. Move to next Priority 1 function
5. Iterate

---

**Estimated Total Effort**: 3-4 days
**Target Completion**: 2025-01-30