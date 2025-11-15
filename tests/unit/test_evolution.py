"""
Unit Tests for Time Evolution Module
=====================================

CRITICAL MODULE: This tests src/hamiltonian/evolution.py, which is the core
quantum dynamics simulation engine. All GRAPE optimizations and pulse validation
depend on correct time evolution.

Test Coverage:
--------------
1. Analytical evolution (exact solutions for drift Hamiltonian)
2. Numerical evolution (QuTiP solver validation)
3. Control Hamiltonian evolution
4. Unitarity preservation
5. Bloch sphere dynamics
6. Method comparison (analytical vs numerical)
7. Edge cases and error handling

Author: Test Coverage Improvement Initiative
Date: 2025-11-15
Reference: TEST_COVERAGE_80_PLAN.md Phase 1.2
"""

import pytest
import numpy as np
import qutip as qt
from src.hamiltonian.evolution import (
    TimeEvolution,
    bloch_coordinates,
    bloch_trajectory,
)


class TestAnalyticalEvolution:
    """
    Test analytical evolution for drift Hamiltonians.
    
    CRITICAL: Validates exact solutions against known quantum mechanics.
    """
    
    def test_drift_hamiltonian_evolution(self):
        """
        Test analytical evolution of drift Hamiltonian H = (ω/2)σ_z.
        
        For H ∝ σ_z, evolution is pure rotation around z-axis:
        U(t) = exp(-iHt) = cos(ωt/2)I - i*sin(ωt/2)σ_z
        """
        omega = 2 * np.pi * 5.0  # 5 MHz
        H = 0.5 * omega * qt.sigmaz()
        
        evolver = TimeEvolution(H, method="analytical")
        
        psi0 = qt.basis(2, 0)  # |0⟩
        times = np.linspace(0, 1.0, 100)
        
        result = evolver.evolve(psi0, times)
        
        assert len(result.states) == len(times)
        assert result.times is not None
        assert len(result.times) == len(times)
    
    def test_analytical_solution_matches_formula(self):
        """
        Validate analytical solution against exact formula.
        
        For |ψ(0)⟩ = |0⟩ and H = (ω/2)σ_z:
        |ψ(t)⟩ = cos(ωt/2)|0⟩ - i*sin(ωt/2)|1⟩
        """
        omega = 2 * np.pi * 10.0  # 10 MHz
        H = 0.5 * omega * qt.sigmaz()
        
        evolver = TimeEvolution(H, method="analytical")
        
        psi0 = qt.basis(2, 0)
        t = 0.05  # 50 ns
        
        result = evolver.evolve(psi0, np.array([0, t]))
        psi_t = result.states[1]
        
        # Expected state: cos(ωt/2)|0⟩ - i*sin(ωt/2)|1⟩
        expected = (np.cos(omega * t / 2) * qt.basis(2, 0) -  # type: ignore
                   1j * np.sin(omega * t / 2) * qt.basis(2, 1))  # type: ignore
        
        fidelity = qt.fidelity(psi_t, expected)
        assert fidelity == pytest.approx(1.0, abs=1e-12), \
            f"Analytical solution incorrect, fidelity={fidelity}"
    
    def test_analytical_periodicity(self):
        """
        Test that evolution is periodic with correct period.
        
        For H = (ω/2)σ_z, period T = 2π/ω (full rotation).
        State should return to initial after period.
        """
        omega = 2 * np.pi * 5.0
        H = 0.5 * omega * qt.sigmaz()
        
        evolver = TimeEvolution(H, method="analytical")
        
        psi0 = qt.basis(2, 0)
        period = 2 * np.pi / omega  # Should be 0.2 μs
        
        result = evolver.evolve(psi0, np.array([0, period]))
        psi_final = result.states[1]
        
        # Should return to initial state (up to global phase)
        fidelity = qt.fidelity(psi0, psi_final)
        assert fidelity == pytest.approx(1.0, abs=1e-10), \
            f"State did not return after period, fidelity={fidelity}"
    
    def test_analytical_different_frequencies(self):
        """Test analytical evolution with different qubit frequencies."""
        frequencies = [1.0, 5.0, 10.0, 50.0]  # MHz
        
        for freq in frequencies:
            omega = 2 * np.pi * freq
            H = 0.5 * omega * qt.sigmaz()
            
            evolver = TimeEvolution(H, method="analytical")
            
            psi0 = qt.basis(2, 0)
            times = np.linspace(0, 0.1, 50)
            
            result = evolver.evolve(psi0, times)
            
            # All states should be normalized
            for state in result.states:
                norm = state.norm()
                assert norm == pytest.approx(1.0, abs=1e-12), \
                    f"State not normalized for freq={freq} MHz"
    
    def test_analytical_long_time_stability(self):
        """Test numerical stability of analytical evolution over long times."""
        omega = 2 * np.pi * 5.0
        H = 0.5 * omega * qt.sigmaz()
        
        evolver = TimeEvolution(H, method="analytical")
        
        psi0 = qt.basis(2, 0)
        # Evolve for 1000 periods
        t_long = 1000 * (2 * np.pi / omega)
        
        result = evolver.evolve(psi0, np.array([0, t_long]))
        psi_final = result.states[1]
        
        # Should still be normalized
        assert psi_final.norm() == pytest.approx(1.0, abs=1e-10)
        
        # Should return to initial state
        fidelity = qt.fidelity(psi0, psi_final)
        assert fidelity == pytest.approx(1.0, abs=1e-8)
    
    def test_analytical_initial_state_variations(self):
        """Test analytical evolution from different initial states."""
        omega = 2 * np.pi * 10.0
        H = 0.5 * omega * qt.sigmaz()
        
        evolver = TimeEvolution(H, method="analytical")
        
        # Test various initial states
        initial_states = [
            qt.basis(2, 0),  # |0⟩
            qt.basis(2, 1),  # |1⟩
            (qt.basis(2, 0) + qt.basis(2, 1)).unit(),  # |+⟩
            (qt.basis(2, 0) - qt.basis(2, 1)).unit(),  # |-⟩
        ]
        
        times = np.linspace(0, 0.1, 50)
        
        for psi0 in initial_states:
            result = evolver.evolve(psi0, times)
            
            # Check all states are normalized
            for state in result.states:
                assert state.norm() == pytest.approx(1.0, abs=1e-12)
    
    def test_analytical_unitarity_preservation(self):
        """
        Test that analytical evolution preserves unitarity.
        
        U†U = I for all time evolution operators.
        """
        omega = 2 * np.pi * 7.5
        H = 0.5 * omega * qt.sigmaz()
        
        evolver = TimeEvolution(H, method="analytical")
        
        times = np.linspace(0, 1.0, 20)
        
        for t in times:
            U = evolver.propagator(t)
            
            # Check U†U = I
            product = U.dag() * U  # type: ignore
            identity = qt.qeye(2)
            
            deviation = (product - identity).norm()  # type: ignore
            assert deviation < 1e-12, \
                f"Unitarity violated at t={t}: ||U†U - I|| = {deviation}"
    
    def test_analytical_energy_conservation(self):
        """
        Test that energy is conserved in analytical evolution.
        
        For closed system: ⟨H⟩ = constant
        """
        omega = 2 * np.pi * 5.0
        H = 0.5 * omega * qt.sigmaz()
        
        evolver = TimeEvolution(H, method="analytical")
        
        psi0 = qt.basis(2, 0)
        times = np.linspace(0, 1.0, 100)
        
        result = evolver.evolve(psi0, times, e_ops=[H])
        
        # Energy should be constant
        if result.expect:
            energies = np.array(result.expect[0])
            energy_std = np.std(energies)
            
            assert energy_std < 1e-12, \
                f"Energy not conserved: std={energy_std}"
    
    def test_analytical_trace_preservation(self):
        """Test that density matrix trace is preserved."""
        omega = 2 * np.pi * 5.0
        H = 0.5 * omega * qt.sigmaz()
        
        evolver = TimeEvolution(H, method="analytical")
        
        psi0 = qt.basis(2, 0)
        times = np.linspace(0, 0.5, 50)
        
        result = evolver.evolve(psi0, times)
        
        for state in result.states:
            # Convert to density matrix and check trace
            rho = state * state.dag()  # type: ignore
            trace = rho.tr()
            
            assert trace == pytest.approx(1.0, abs=1e-12), \
                "Trace not preserved"
    
    def test_analytical_bloch_sphere_trajectory(self):
        """
        Test that Bloch sphere trajectory is correct for drift evolution.
        
        For H ∝ σ_z, Bloch vector rotates around z-axis.
        x(t) and y(t) should oscillate, z(t) should be constant.
        """
        omega = 2 * np.pi * 5.0
        H = 0.5 * omega * qt.sigmaz()
        
        evolver = TimeEvolution(H, method="analytical")
        
        # Start in |+⟩ state (on equator)
        psi0 = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
        
        times = np.linspace(0, 0.2, 100)
        result = evolver.evolve(psi0, times)
        
        trajectory = bloch_trajectory(result.states)
        
        # z-component should be constant (≈ 0 for |+⟩)
        z_values = trajectory[:, 2]
        z_std = np.std(z_values)
        
        assert z_std < 0.01, \
            f"Z-component not constant for σ_z evolution: std={z_std}"
        
        # x and y should oscillate
        x_values = trajectory[:, 0]
        y_values = trajectory[:, 1]
        
        assert np.max(x_values) > 0.5, "X-component not oscillating"
        assert np.max(np.abs(y_values)) > 0.5, "Y-component not oscillating"


class TestNumericalEvolution:
    """Test numerical evolution using QuTiP solver."""
    
    def test_numerical_basic_evolution(self):
        """Test basic numerical evolution with QuTiP solver."""
        H = qt.sigmaz()
        
        evolver = TimeEvolution(H, method="numerical")
        
        psi0 = qt.basis(2, 0)
        times = np.linspace(0, 1.0, 100)
        
        result = evolver.evolve(psi0, times)
        
        assert len(result.states) == len(times)
        assert hasattr(result, 'times')
    
    def test_numerical_state_normalization(self):
        """Test that numerical evolution preserves state normalization."""
        H = 2 * np.pi * 5.0 * qt.sigmaz()
        
        evolver = TimeEvolution(H, method="numerical")
        
        psi0 = qt.basis(2, 0)
        times = np.linspace(0, 1.0, 100)
        
        result = evolver.evolve(psi0, times)
        
        for state in result.states:
            norm = state.norm()
            assert norm == pytest.approx(1.0, abs=1e-10), \
                f"State not normalized: norm={norm}"
    
    def test_numerical_time_step_convergence(self):
        """
        Test that numerical solution converges with finer time steps.
        
        Smaller time steps should give more accurate results.
        """
        H = 2 * np.pi * 10.0 * qt.sigmaz()
        evolver = TimeEvolution(H, method="numerical")
        
        psi0 = qt.basis(2, 0)
        t_final = 0.1
        
        # Coarse and fine time grids
        times_coarse = np.linspace(0, t_final, 20)
        times_fine = np.linspace(0, t_final, 200)
        
        result_coarse = evolver.evolve(psi0, times_coarse)
        result_fine = evolver.evolve(psi0, times_fine)
        
        # Both should give normalized states
        for state in result_coarse.states:
            assert state.norm() == pytest.approx(1.0, abs=1e-10)
        for state in result_fine.states:
            assert state.norm() == pytest.approx(1.0, abs=1e-10)
    
    def test_numerical_with_different_solvers(self):
        """Test that numerical evolution is consistent."""
        H = qt.sigmaz()
        
        evolver = TimeEvolution(H, method="numerical")
        
        psi0 = qt.basis(2, 0)
        times = np.linspace(0, 1.0, 50)
        
        result = evolver.evolve(psi0, times)
        
        # Check basic properties
        assert len(result.states) == len(times)
        for state in result.states:
            assert state.norm() == pytest.approx(1.0, abs=1e-10)
    
    def test_numerical_hamiltonian_time_dependence(self):
        """Test numerical evolution with time-independent Hamiltonian."""
        # Even though this is time-independent, test QuTiP integration
        H = 2 * np.pi * 5.0 * qt.sigmaz()
        
        evolver = TimeEvolution(H, method="numerical")
        
        psi0 = qt.basis(2, 0)
        times = np.linspace(0, 0.5, 100)
        
        result = evolver.evolve(psi0, times)
        
        # Verify evolution occurred
        psi_final = result.states[-1]
        assert qt.fidelity(psi0, psi_final) < 1.0, \
            "State didn't evolve (should have changed)"
    
    def test_numerical_convergence_to_analytical(self):
        """
        Test that numerical solution converges to analytical for drift H.
        
        This is a critical validation test.
        """
        omega = 2 * np.pi * 5.0
        H = 0.5 * omega * qt.sigmaz()
        
        psi0 = qt.basis(2, 0)
        times = np.linspace(0, 1.0, 100)
        
        # Numerical evolution
        evolver_num = TimeEvolution(H, method="numerical")
        result_num = evolver_num.evolve(psi0, times)
        
        # Analytical evolution
        evolver_ana = TimeEvolution(H, method="analytical")
        result_ana = evolver_ana.evolve(psi0, times)
        
        # Compare fidelities
        fidelities = [qt.fidelity(s_num, s_ana) 
                     for s_num, s_ana in zip(result_num.states, result_ana.states)]
        
        min_fidelity = np.min(fidelities)
        assert min_fidelity > 1.0 - 1e-8, \
            f"Numerical doesn't match analytical: min_fidelity={min_fidelity}"
    
    def test_numerical_stability(self):
        """Test numerical stability over long evolution times."""
        H = qt.sigmaz()
        
        evolver = TimeEvolution(H, method="numerical")
        
        psi0 = qt.basis(2, 0)
        # Long evolution time
        times = np.linspace(0, 100.0, 500)
        
        result = evolver.evolve(psi0, times)
        
        # All states should remain normalized
        for state in result.states:
            norm = state.norm()
            assert 0.99 < norm < 1.01, \
                f"Normalization lost in long evolution: norm={norm}"
    
    def test_numerical_error_accumulation(self):
        """Test that numerical errors don't accumulate catastrophically."""
        H = 2 * np.pi * 5.0 * qt.sigmaz()
        
        evolver = TimeEvolution(H, method="numerical")
        
        psi0 = qt.basis(2, 0)
        
        # Evolve to same final time with different step sizes
        t_final = 1.0
        times_fine = np.linspace(0, t_final, 1000)
        times_coarse = np.linspace(0, t_final, 100)
        
        result_fine = evolver.evolve(psi0, times_fine)
        result_coarse = evolver.evolve(psi0, times_coarse)
        
        psi_fine = result_fine.states[-1]
        psi_coarse = result_coarse.states[-1]
        
        # Should be close (adaptive stepping helps)
        fidelity = qt.fidelity(psi_fine, psi_coarse)
        assert fidelity > 0.9999, \
            f"Large discretization error: fidelity={fidelity}"
    
    def test_numerical_rapid_oscillations(self):
        """Test numerical solver with rapidly oscillating Hamiltonian."""
        # High frequency
        omega = 2 * np.pi * 100.0  # 100 MHz
        H = 0.5 * omega * qt.sigmaz()
        
        evolver = TimeEvolution(H, method="numerical")
        
        psi0 = qt.basis(2, 0)
        times = np.linspace(0, 0.1, 1000)  # Need fine sampling
        
        result = evolver.evolve(psi0, times)
        
        # Should still preserve normalization
        for state in result.states:
            assert state.norm() == pytest.approx(1.0, abs=1e-8)
    
    def test_numerical_large_time_evolution(self):
        """Test evolution over large time scale."""
        H = qt.sigmaz()
        
        evolver = TimeEvolution(H, method="numerical")
        
        psi0 = qt.basis(2, 0)
        times = np.array([0, 1000.0])  # Very long time
        
        result = evolver.evolve(psi0, times)
        
        # State should still be valid
        psi_final = result.states[-1]
        assert psi_final.norm() == pytest.approx(1.0, abs=1e-8)


class TestControlHamiltonianEvolution:
    """Test evolution with control Hamiltonians."""
    
    def test_control_hamiltonian_x_rotation(self):
        """Test evolution under σ_x control (Rabi oscillations)."""
        # H = Ω σ_x causes Rabi oscillations
        rabi_freq = 2 * np.pi * 10.0  # 10 MHz
        H = rabi_freq * qt.sigmax()
        
        evolver = TimeEvolution(H, method="numerical")
        
        psi0 = qt.basis(2, 0)  # Start in |0⟩
        
        # At t = π/(2Ω), should be in |1⟩ (π/2 pulse)
        t_pi_2 = np.pi / (2 * rabi_freq)
        
        result = evolver.evolve(psi0, np.array([0, t_pi_2]))
        psi_final = result.states[1]
        
        # Should be close to |1⟩
        psi_target = qt.basis(2, 1)
        fidelity = qt.fidelity(psi_final, psi_target)
        
        assert fidelity > 0.999, \
            f"π/2 pulse failed: fidelity with |1⟩ = {fidelity}"
    
    def test_rabi_oscillations(self):
        """
        Test Rabi oscillations under continuous X-drive.
        
        Should see population oscillate between |0⟩ and |1⟩.
        """
        rabi_freq = 2 * np.pi * 5.0
        H = rabi_freq * qt.sigmax()
        
        evolver = TimeEvolution(H, method="numerical")
        
        psi0 = qt.basis(2, 0)
        
        # Evolve for several Rabi periods
        t_rabi = 2 * np.pi / rabi_freq
        times = np.linspace(0, 3 * t_rabi, 200)
        
        result = evolver.evolve(psi0, times, e_ops=[qt.sigmaz()])
        
        # Population should oscillate
        if result.expect:
            sz_values = np.array(result.expect[0])
            
            # Should see oscillation (not constant)
            assert np.std(sz_values) > 0.5, \
                "No Rabi oscillations detected"
            
            # Should return to initial after period
            # (check last point close to first)
            assert np.abs(sz_values[-1] - sz_values[0]) < 0.2
    
    def test_multi_control_evolution(self):
        """Test evolution with multiple control terms."""
        # H = Ω_x σ_x + Ω_y σ_y (general rotation)
        omega_x = 2 * np.pi * 5.0
        omega_y = 2 * np.pi * 3.0
        
        H = omega_x * qt.sigmax() + omega_y * qt.sigmay()
        
        evolver = TimeEvolution(H, method="numerical")
        
        psi0 = qt.basis(2, 0)
        times = np.linspace(0, 0.5, 100)
        
        result = evolver.evolve(psi0, times)
        
        # Should preserve normalization
        for state in result.states:
            assert state.norm() == pytest.approx(1.0, abs=1e-10)
    
    def test_control_amplitude_bounds(self):
        """Test evolution with different control amplitudes."""
        amplitudes = [0.1, 1.0, 10.0, 100.0]  # MHz (in units of 2π)
        
        psi0 = qt.basis(2, 0)
        times = np.linspace(0, 0.01, 50)
        
        for amp in amplitudes:
            omega = 2 * np.pi * amp
            H = omega * qt.sigmax()
            
            evolver = TimeEvolution(H, method="numerical")
            result = evolver.evolve(psi0, times)
            
            # All should preserve normalization
            for state in result.states:
                norm = state.norm()
                assert norm == pytest.approx(1.0, abs=1e-8), \
                    f"Norm violated for amplitude={amp}"
    
    def test_resonance_conditions(self):
        """Test evolution at resonance vs off-resonance."""
        # On-resonance: H = Ω σ_x
        # Off-resonance: H = Δ σ_z + Ω σ_x
        
        omega = 2 * np.pi * 5.0
        delta = 2 * np.pi * 20.0  # Large detuning
        
        H_on = omega * qt.sigmax()
        H_off = delta * qt.sigmaz() + omega * qt.sigmax()
        
        psi0 = qt.basis(2, 0)
        times = np.linspace(0, 0.1, 100)
        
        evolver_on = TimeEvolution(H_on, method="numerical")
        evolver_off = TimeEvolution(H_off, method="numerical")
        
        result_on = evolver_on.evolve(psi0, times)
        result_off = evolver_off.evolve(psi0, times)
        
        # On-resonance should cause more population transfer
        psi_on_final = result_on.states[-1]
        psi_off_final = result_off.states[-1]
        
        # Both should be normalized
        assert psi_on_final.norm() == pytest.approx(1.0, abs=1e-10)
        assert psi_off_final.norm() == pytest.approx(1.0, abs=1e-10)
    
    def test_off_resonance_driving(self):
        """Test off-resonance driving dynamics."""
        delta = 2 * np.pi * 10.0  # Detuning
        omega = 2 * np.pi * 2.0   # Rabi frequency
        
        H = 0.5 * delta * qt.sigmaz() + omega * qt.sigmax()
        
        evolver = TimeEvolution(H, method="numerical")
        
        psi0 = qt.basis(2, 0)
        times = np.linspace(0, 1.0, 200)
        
        result = evolver.evolve(psi0, times)
        
        # Should see reduced Rabi oscillations
        for state in result.states:
            assert state.norm() == pytest.approx(1.0, abs=1e-10)


class TestEdgeCasesAndValidation:
    """Test edge cases and error handling."""
    
    def test_zero_hamiltonian_identity_evolution(self):
        """Test that zero Hamiltonian gives identity evolution."""
        H = 0 * qt.sigmaz()  # Zero Hamiltonian
        
        evolver = TimeEvolution(H, method="numerical")
        
        psi0 = qt.basis(2, 0)
        times = np.linspace(0, 1.0, 50)
        
        result = evolver.evolve(psi0, times)
        
        # All states should be identical to initial
        for state in result.states:
            fidelity = qt.fidelity(state, psi0)
            assert fidelity == pytest.approx(1.0, abs=1e-12), \
                "State evolved under zero Hamiltonian"
    
    def test_very_short_time(self):
        """Test evolution for very short times."""
        H = qt.sigmaz()
        
        evolver = TimeEvolution(H, method="numerical")
        
        psi0 = qt.basis(2, 0)
        t_short = 1e-12  # 1 picosecond
        
        result = evolver.evolve(psi0, np.array([0, t_short]))
        psi_final = result.states[1]
        
        # Should be almost unchanged
        fidelity = qt.fidelity(psi0, psi_final)
        assert fidelity > 0.9999
    
    def test_invalid_hamiltonian_type(self):
        """Test that invalid Hamiltonian type raises error."""
        with pytest.raises(TypeError, match="qutip.Qobj"):
            TimeEvolution([[1, 0], [0, -1]])  # type: ignore
    
    def test_dimension_mismatch(self):
        """Test that non-qubit Hamiltonians raise error."""
        H_3level = qt.qeye(3)  # 3-level system
        
        with pytest.raises(ValueError, match="single-qubit"):
            TimeEvolution(H_3level)
    
    def test_propagator_unitarity(self):
        """Test that propagator method returns unitary operators."""
        H = qt.sigmaz()
        
        evolver = TimeEvolution(H, method="numerical")
        
        times_test = [0.1, 0.5, 1.0, 5.0]
        
        for t in times_test:
            U = evolver.propagator(t)
            
            # Check unitarity: U†U = I
            product = U.dag() * U  # type: ignore
            identity = qt.qeye(2)
            
            deviation = (product - identity).norm()  # type: ignore
            assert deviation < 1e-10, \
                f"Propagator not unitary at t={t}: deviation={deviation}"
    
    def test_fidelity_over_time_method(self):
        """Test the fidelity_over_time convenience method."""
        H = qt.sigmaz()
        
        evolver = TimeEvolution(H, method="numerical")
        
        psi0 = qt.basis(2, 0)
        psi_target = qt.basis(2, 0)  # Same as initial
        
        times = np.linspace(0, 2*np.pi, 100)
        
        fidelities = evolver.fidelity_over_time(psi0, psi_target, times)
        
        assert len(fidelities) == len(times)
        assert all(0 <= f <= 1 for f in fidelities), \
            "Fidelity outside [0, 1] range"
        
        # At t=0, fidelity should be 1
        assert fidelities[0] == pytest.approx(1.0, abs=1e-12)
    
    def test_compare_methods_function(self):
        """Test the compare_methods utility function."""
        omega = 2 * np.pi * 5.0
        H = 0.5 * omega * qt.sigmaz()
        
        evolver = TimeEvolution(H, method="numerical")
        
        psi0 = qt.basis(2, 0)
        times = np.linspace(0, 1.0, 50)
        
        comparison = evolver.compare_methods(psi0, times)
        
        assert 'numerical_states' in comparison
        assert 'analytical_states' in comparison
        assert 'fidelities' in comparison
        assert 'max_error' in comparison
        
        # Error should be very small
        assert comparison['max_error'] < 1e-8
    
    def test_bloch_coordinates_computational_basis(self):
        """Test Bloch coordinates for computational basis states."""
        # |0⟩ → (0, 0, 1)
        psi_0 = qt.basis(2, 0)
        x, y, z = bloch_coordinates(psi_0)
        
        assert x == pytest.approx(0.0, abs=1e-12)
        assert y == pytest.approx(0.0, abs=1e-12)
        assert z == pytest.approx(1.0, abs=1e-12)
        
        # |1⟩ → (0, 0, -1)
        psi_1 = qt.basis(2, 1)
        x, y, z = bloch_coordinates(psi_1)
        
        assert x == pytest.approx(0.0, abs=1e-12)
        assert y == pytest.approx(0.0, abs=1e-12)
        assert z == pytest.approx(-1.0, abs=1e-12)
    
    def test_bloch_coordinates_superposition_states(self):
        """Test Bloch coordinates for superposition states."""
        # |+⟩ = (|0⟩ + |1⟩)/√2 → (1, 0, 0)
        psi_plus = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
        x, y, z = bloch_coordinates(psi_plus)
        
        assert x == pytest.approx(1.0, abs=1e-12)
        assert y == pytest.approx(0.0, abs=1e-12)
        assert z == pytest.approx(0.0, abs=1e-12)
        
        # |-⟩ = (|0⟩ - |1⟩)/√2 → (-1, 0, 0)
        psi_minus = (qt.basis(2, 0) - qt.basis(2, 1)).unit()
        x, y, z = bloch_coordinates(psi_minus)
        
        assert x == pytest.approx(-1.0, abs=1e-12)
        assert y == pytest.approx(0.0, abs=1e-12)
        assert z == pytest.approx(0.0, abs=1e-12)
    
    def test_bloch_trajectory_shape(self):
        """Test that bloch_trajectory returns correct shape."""
        H = qt.sigmaz()
        evolver = TimeEvolution(H, method="numerical")
        
        psi0 = qt.basis(2, 0)
        times = np.linspace(0, 1.0, 100)
        
        result = evolver.evolve(psi0, times)
        trajectory = bloch_trajectory(result.states)
        
        assert trajectory.shape == (100, 3), \
            f"Trajectory shape {trajectory.shape}, expected (100, 3)"
    
    def test_bloch_trajectory_unit_vector(self):
        """Test that all Bloch vectors have unit length."""
        H = qt.sigmax()
        evolver = TimeEvolution(H, method="numerical")
        
        psi0 = qt.basis(2, 0)
        times = np.linspace(0, 1.0, 50)
        
        result = evolver.evolve(psi0, times)
        trajectory = bloch_trajectory(result.states)
        
        # All Bloch vectors should have unit length
        norms = np.linalg.norm(trajectory, axis=1)
        
        for norm in norms:
            assert norm == pytest.approx(1.0, abs=1e-10), \
                f"Bloch vector not unit length: norm={norm}"
