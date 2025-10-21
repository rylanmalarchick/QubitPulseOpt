"""
Unit Tests for Drift Hamiltonian Module
========================================

This module contains comprehensive unit tests for the drift Hamiltonian
implementation, validating physics correctness and numerical accuracy.

Test Coverage:
- DriftHamiltonian class initialization
- Energy eigenvalues and eigenstates
- Commutation relations
- Time evolution (periodicity, z-axis confinement)
- Analytical vs. numerical evolution comparison
- Edge cases and error handling

Author: Orchestrator Agent
Date: 2025-01-27
SOW Reference: Lines 150-161 (Week 1.2: Drift Dynamics)
"""

import pytest
import numpy as np
import qutip
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.hamiltonian import (
    DriftHamiltonian,
    create_drift_hamiltonian,
    TimeEvolution,
    bloch_coordinates,
    bloch_trajectory,
)


class TestDriftHamiltonianInitialization:
    """Tests for DriftHamiltonian initialization and basic properties."""

    def test_default_initialization(self):
        """Test initialization with default frequency."""
        drift = DriftHamiltonian()
        assert drift.omega_0 == 5.0
        assert isinstance(drift.H, qutip.Qobj)
        assert drift.H.dims == [[2], [2]]

    def test_custom_frequency(self):
        """Test initialization with custom frequency."""
        omega = 7.5
        drift = DriftHamiltonian(omega_0=omega)
        assert drift.omega_0 == omega

    def test_invalid_frequency(self):
        """Test that negative frequency raises ValueError."""
        with pytest.raises(ValueError, match="Frequency must be positive"):
            DriftHamiltonian(omega_0=-1.0)

    def test_zero_frequency(self):
        """Test that zero frequency raises ValueError."""
        with pytest.raises(ValueError, match="Frequency must be positive"):
            DriftHamiltonian(omega_0=0.0)

    def test_factory_function(self):
        """Test create_drift_hamiltonian factory function."""
        drift = create_drift_hamiltonian(omega_0=6.0)
        assert isinstance(drift, DriftHamiltonian)
        assert drift.omega_0 == 6.0


class TestDriftHamiltonianSpectrum:
    """Tests for energy eigenvalues and eigenstates."""

    def test_energy_eigenvalues(self):
        """Test that energy eigenvalues are ±ω₀/2."""
        omega = 5.0
        drift = DriftHamiltonian(omega_0=omega)
        E0, E1 = drift.energy_levels()

        assert np.isclose(E0, -omega / 2, atol=1e-10)
        assert np.isclose(E1, +omega / 2, atol=1e-10)

    def test_energy_splitting(self):
        """Test that energy splitting equals ω₀."""
        omega = 5.0
        drift = DriftHamiltonian(omega_0=omega)
        delta_E = drift.energy_splitting()

        assert np.isclose(delta_E, omega, atol=1e-10)

    def test_eigenstates_are_computational_basis(self):
        """Test that eigenstates are |0⟩ and |1⟩."""
        drift = DriftHamiltonian(omega_0=5.0)
        ket_g, ket_e = drift.eigenstates()

        ket0 = qutip.basis(2, 0)
        ket1 = qutip.basis(2, 1)

        # Ground state should match either |0⟩ or |1⟩
        fid_g0 = qutip.fidelity(ket_g, ket0)
        fid_g1 = qutip.fidelity(ket_g, ket1)
        fid_e0 = qutip.fidelity(ket_e, ket0)
        fid_e1 = qutip.fidelity(ket_e, ket1)

        # One eigenstate should match |0⟩, the other |1⟩
        assert (fid_g0 > 0.999 and fid_e1 > 0.999) or (
            fid_g1 > 0.999 and fid_e0 > 0.999
        )

    @pytest.mark.parametrize("omega", [1.0, 5.0, 10.0, 100.0])
    def test_energy_levels_different_frequencies(self, omega):
        """Test energy levels for various frequencies."""
        drift = DriftHamiltonian(omega_0=omega)
        E0, E1 = drift.energy_levels()

        assert np.isclose(E0, -omega / 2, atol=1e-10)
        assert np.isclose(E1, +omega / 2, atol=1e-10)


class TestDriftHamiltonianCommutators:
    """Tests for commutation relations."""

    def test_commutes_with_sigmaz(self):
        """Test that [H₀, σ_z] = 0."""
        drift = DriftHamiltonian(omega_0=5.0)
        comm = drift.commutator_with_sigmaz()

        assert comm.norm() < 1e-10

    def test_hamiltonian_is_diagonal(self):
        """Test that Hamiltonian is diagonal in computational basis."""
        drift = DriftHamiltonian(omega_0=5.0)
        H_matrix = drift.H.full()

        # Off-diagonal elements should be zero
        assert np.abs(H_matrix[0, 1]) < 1e-10
        assert np.abs(H_matrix[1, 0]) < 1e-10


class TestDriftHamiltonianPeriod:
    """Tests for precession period."""

    def test_precession_period_formula(self):
        """Test that period T = 2π/ω₀."""
        omega = 5.0
        drift = DriftHamiltonian(omega_0=omega)
        T = drift.precession_period()

        expected_T = 2 * np.pi / omega
        assert np.isclose(T, expected_T, atol=1e-10)

    @pytest.mark.parametrize("omega", [1.0, 5.0, 10.0])
    def test_period_scales_inversely(self, omega):
        """Test that period scales as 1/ω₀."""
        drift = DriftHamiltonian(omega_0=omega)
        T = drift.precession_period()
        assert np.isclose(T * omega, 2 * np.pi, atol=1e-10)


class TestDriftTimeEvolution:
    """Tests for time evolution under drift Hamiltonian."""

    def test_state_returns_after_one_period(self):
        """Test that |ψ(T)⟩ = |ψ(0)⟩ after one period."""
        drift = DriftHamiltonian(omega_0=5.0)
        psi0 = qutip.basis(2, 0)

        T = drift.precession_period()
        times = np.array([0, T])

        result = drift.evolve_state(psi0, times)
        psi_final = result.states[-1]

        fidelity = qutip.fidelity(psi0, psi_final)
        assert fidelity > 0.9999

    def test_no_x_rotation(self):
        """Test that ⟨σ_x⟩ remains zero (no x-axis rotation)."""
        drift = DriftHamiltonian(omega_0=5.0)
        psi0 = qutip.basis(2, 0)  # Start at north pole (z=+1)

        T = drift.precession_period()
        times = np.linspace(0, T, 50)

        sx = qutip.sigmax()
        result = drift.evolve_state(psi0, times)

        expect_x = [qutip.expect(sx, state) for state in result.states]
        max_x = np.abs(expect_x).max()

        assert max_x < 1e-10

    def test_no_y_rotation(self):
        """Test that ⟨σ_y⟩ remains zero (no y-axis rotation)."""
        drift = DriftHamiltonian(omega_0=5.0)
        psi0 = qutip.basis(2, 0)

        T = drift.precession_period()
        times = np.linspace(0, T, 50)

        sy = qutip.sigmay()
        result = drift.evolve_state(psi0, times)

        expect_y = [qutip.expect(sy, state) for state in result.states]
        max_y = np.abs(expect_y).max()

        assert max_y < 1e-10

    def test_z_expectation_conserved(self):
        """Test that ⟨σ_z⟩ is conserved (constant in time)."""
        drift = DriftHamiltonian(omega_0=5.0)
        psi0 = qutip.basis(2, 0)  # ⟨σ_z⟩ = +1

        T = drift.precession_period()
        times = np.linspace(0, T, 50)

        sz = qutip.sigmaz()
        result = drift.evolve_state(psi0, times)

        expect_z = [qutip.expect(sz, state) for state in result.states]

        # All z-expectations should be approximately +1
        assert np.allclose(expect_z, 1.0, atol=1e-10)

    @pytest.mark.parametrize(
        "initial_state",
        [
            qutip.basis(2, 0),  # |0⟩
            qutip.basis(2, 1),  # |1⟩
        ],
    )
    def test_computational_basis_states_stationary(self, initial_state):
        """Test that |0⟩ and |1⟩ are stationary (up to global phase)."""
        drift = DriftHamiltonian(omega_0=5.0)

        T = drift.precession_period()
        times = np.linspace(0, T, 50)

        result = drift.evolve_state(initial_state, times)

        # Check fidelity remains high (ignore global phase)
        for state in result.states:
            fidelity = qutip.fidelity(initial_state, state)
            assert fidelity > 0.9999


class TestTimeEvolutionEngine:
    """Tests for TimeEvolution class."""

    def test_numerical_evolution(self):
        """Test numerical evolution method."""
        H = 0.5 * 5.0 * qutip.sigmaz()
        evolver = TimeEvolution(H, method="numerical")

        psi0 = qutip.basis(2, 0)
        times = np.linspace(0, 1.0, 50)

        result = evolver.evolve(psi0, times)
        assert len(result.states) == 50

    def test_analytical_evolution(self):
        """Test analytical evolution method for drift Hamiltonian."""
        H = 0.5 * 5.0 * qutip.sigmaz()
        evolver = TimeEvolution(H, method="analytical")

        psi0 = qutip.basis(2, 0)
        times = np.linspace(0, 1.0, 50)

        result = evolver.evolve(psi0, times)
        assert len(result.states) == 50

    def test_propagator_is_unitary(self):
        """Test that propagator U(t) is unitary: U†U = I."""
        H = qutip.sigmaz()
        evolver = TimeEvolution(H)

        t = 1.0
        U = evolver.propagator(t)

        # Check U†U = I
        identity = U.dag() * U
        diff = (identity - qutip.qeye(2)).norm()

        assert diff < 1e-10

    def test_compare_methods(self):
        """Test comparison between analytical and numerical methods."""
        H = 0.5 * 5.0 * qutip.sigmaz()
        evolver = TimeEvolution(H)

        psi0 = qutip.basis(2, 0)
        times = np.linspace(0, 2.0, 100)

        comparison = evolver.compare_methods(psi0, times)

        # Methods should agree to high precision
        assert comparison["max_error"] < 1e-10
        assert np.all(comparison["fidelities"] > 0.999999)


class TestBlochSphereCoordinates:
    """Tests for Bloch sphere coordinate functions."""

    def test_bloch_coordinates_ket0(self):
        """Test Bloch coordinates for |0⟩ (north pole)."""
        psi = qutip.basis(2, 0)
        x, y, z = bloch_coordinates(psi)

        assert np.isclose(x, 0.0, atol=1e-10)
        assert np.isclose(y, 0.0, atol=1e-10)
        assert np.isclose(z, 1.0, atol=1e-10)

    def test_bloch_coordinates_ket1(self):
        """Test Bloch coordinates for |1⟩ (south pole)."""
        psi = qutip.basis(2, 1)
        x, y, z = bloch_coordinates(psi)

        assert np.isclose(x, 0.0, atol=1e-10)
        assert np.isclose(y, 0.0, atol=1e-10)
        assert np.isclose(z, -1.0, atol=1e-10)

    def test_bloch_coordinates_ket_plus(self):
        """Test Bloch coordinates for |+⟩ (x-axis)."""
        ket0 = qutip.basis(2, 0)
        ket1 = qutip.basis(2, 1)
        psi = (ket0 + ket1).unit()

        x, y, z = bloch_coordinates(psi)

        assert np.isclose(x, 1.0, atol=1e-10)
        assert np.isclose(y, 0.0, atol=1e-10)
        assert np.isclose(z, 0.0, atol=1e-10)

    def test_bloch_coordinates_ket_i(self):
        """Test Bloch coordinates for |i⟩ (y-axis)."""
        ket0 = qutip.basis(2, 0)
        ket1 = qutip.basis(2, 1)
        psi = (ket0 + 1j * ket1).unit()

        x, y, z = bloch_coordinates(psi)

        assert np.isclose(x, 0.0, atol=1e-10)
        assert np.isclose(y, 1.0, atol=1e-10)
        assert np.isclose(z, 0.0, atol=1e-10)

    def test_bloch_trajectory_shape(self):
        """Test that bloch_trajectory returns correct shape."""
        drift = DriftHamiltonian(omega_0=5.0)
        psi0 = qutip.basis(2, 0)

        times = np.linspace(0, 1.0, 50)
        result = drift.evolve_state(psi0, times)

        trajectory = bloch_trajectory(result.states)

        assert trajectory.shape == (50, 3)

    def test_bloch_trajectory_stays_on_sphere(self):
        """Test that all points on trajectory have |r| = 1."""
        drift = DriftHamiltonian(omega_0=5.0)
        psi0 = qutip.basis(2, 0)

        times = np.linspace(0, 1.0, 50)
        result = drift.evolve_state(psi0, times)

        trajectory = bloch_trajectory(result.states)
        radii = np.linalg.norm(trajectory, axis=1)

        assert np.allclose(radii, 1.0, atol=1e-10)


class TestStringRepresentations:
    """Tests for __repr__ and __str__ methods."""

    def test_drift_repr(self):
        """Test __repr__ for DriftHamiltonian."""
        drift = DriftHamiltonian(omega_0=5.0)
        repr_str = repr(drift)

        assert "DriftHamiltonian" in repr_str
        assert "5.0" in repr_str

    def test_drift_str(self):
        """Test __str__ for DriftHamiltonian."""
        drift = DriftHamiltonian(omega_0=5.0)
        str_repr = str(drift)

        assert "Drift Hamiltonian" in str_repr
        assert "ω₀" in str_repr
        assert "5.0 MHz" in str_repr

    def test_evolution_repr(self):
        """Test __repr__ for TimeEvolution."""
        H = qutip.sigmaz()
        evolver = TimeEvolution(H, method="numerical")
        repr_str = repr(evolver)

        assert "TimeEvolution" in repr_str
        assert "numerical" in repr_str


class TestPhysicsValidation:
    """High-level physics validation tests."""

    def test_full_evolution_cycle(self):
        """Integration test: Full evolution cycle with all components."""
        # Create drift Hamiltonian
        omega = 5.0
        drift = DriftHamiltonian(omega_0=omega)

        # Create evolver
        evolver = TimeEvolution(drift.to_qobj())

        # Initial state: |+⟩ (on equator)
        ket0 = qutip.basis(2, 0)
        ket1 = qutip.basis(2, 1)
        psi0 = (ket0 + ket1).unit()

        # Evolve for one period
        T = drift.precession_period()
        times = np.linspace(0, T, 100)

        result = evolver.evolve(psi0, times)

        # Final state should match initial (up to global phase)
        psi_final = result.states[-1]
        fidelity = qutip.fidelity(psi0, psi_final)

        assert fidelity > 0.9999

    def test_energy_conservation(self):
        """Test that energy expectation value is conserved."""
        drift = DriftHamiltonian(omega_0=5.0)
        psi0 = qutip.basis(2, 0)

        times = np.linspace(0, drift.precession_period(), 100)
        result = drift.evolve_state(psi0, times)

        # Compute energy at each time
        energies = [qutip.expect(drift.H, state) for state in result.states]

        # Energy should be constant
        energy_std = np.std(energies)
        assert energy_std < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
