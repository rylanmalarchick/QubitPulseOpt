"""
QubitPulseOpt Test Suite
=========================

This package contains unit and integration tests for the QubitPulseOpt
quantum controls simulation project.

Test Structure
--------------
unit/       : Component-level tests (individual classes/functions)
integration/: System-level tests (end-to-end workflows)

Running Tests
-------------
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src

# Run specific test file
pytest tests/unit/test_drift.py -v

# Run specific test class
pytest tests/unit/test_drift.py::TestDriftHamiltonian -v

Author: Orchestrator Agent
Project: QubitPulseOpt - Quantum Controls Simulation
SOW Reference: Week 1-4 Testing Strategy
"""

__version__ = "0.1.0"
