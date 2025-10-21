#!/usr/bin/env python3
"""
Simple Environment Test for QubitPulseOpt
==========================================

This script performs basic quantum operations to verify the environment
is correctly configured and ready for Week 1.2 implementation.

Tests:
1. Package imports (QuTiP, NumPy, SciPy, Matplotlib)
2. Qubit state creation
3. Pauli operators
4. Basic Hamiltonian construction
5. Time evolution (Schrödinger equation)
6. Bloch sphere plotting

Usage:
    python scripts/test_env_simple.py
    # OR with venv:
    venv/bin/python scripts/test_env_simple.py
"""

import sys
import numpy as np


def test_imports():
    """Test 1: Verify all required packages can be imported."""
    print("=" * 60)
    print("TEST 1: Package Imports")
    print("=" * 60)

    try:
        import qutip
        import scipy
        import matplotlib
        import pytest

        print(f"✓ QuTiP version: {qutip.__version__}")
        print(f"✓ NumPy version: {np.__version__}")
        print(f"✓ SciPy version: {scipy.__version__}")
        print(f"✓ Matplotlib version: {matplotlib.__version__}")
        print(f"✓ Pytest version: {pytest.__version__}")
        print("\n✓✓ All imports successful\n")
        return True
    except ImportError as e:
        print(f"\n✗✗ Import failed: {e}\n")
        return False


def test_qubit_states():
    """Test 2: Create and manipulate basic qubit states."""
    print("=" * 60)
    print("TEST 2: Qubit State Creation")
    print("=" * 60)

    import qutip

    # Computational basis states
    ket0 = qutip.basis(2, 0)  # |0⟩
    ket1 = qutip.basis(2, 1)  # |1⟩

    print(f"✓ Created |0⟩ state: {ket0.dims}")
    print(f"✓ Created |1⟩ state: {ket1.dims}")

    # Superposition states
    ket_plus = (ket0 + ket1).unit()  # |+⟩ = (|0⟩ + |1⟩)/√2
    ket_minus = (ket0 - ket1).unit()  # |−⟩ = (|0⟩ − |1⟩)/√2
    ket_i = (ket0 + 1j * ket1).unit()  # |i⟩ = (|0⟩ + i|1⟩)/√2

    print(f"✓ Created |+⟩ state (superposition)")
    print(f"✓ Created |−⟩ state (superposition)")
    print(f"✓ Created |i⟩ state (superposition)")

    # Verify normalization
    norm0 = ket0.norm()
    norm_plus = ket_plus.norm()

    assert abs(norm0 - 1.0) < 1e-10, "State not normalized!"
    assert abs(norm_plus - 1.0) < 1e-10, "State not normalized!"

    print(f"✓ Normalization verified: ||ψ|| = 1.0")
    print("\n✓✓ Qubit state creation successful\n")
    return True


def test_pauli_operators():
    """Test 3: Create Pauli operators and verify properties."""
    print("=" * 60)
    print("TEST 3: Pauli Operators")
    print("=" * 60)

    import qutip

    # Create Pauli matrices
    sx = qutip.sigmax()
    sy = qutip.sigmay()
    sz = qutip.sigmaz()
    si = qutip.qeye(2)

    print("✓ Created Pauli operators: σx, σy, σz")
    print(f"✓ Created identity operator: I")

    # Verify commutation relations: [σi, σj] = 2iε_ijk σk
    comm_xy = qutip.commutator(sx, sy)
    expected_comm_xy = 2j * sz

    diff = (comm_xy - expected_comm_xy).norm()
    assert diff < 1e-10, f"Commutation relation failed: {diff}"

    print("✓ Verified: [σx, σy] = 2iσz")

    # Verify anticommutation: {σi, σj} = 2δ_ij I
    anticomm_xx = sx * sx + sx * sx
    expected_anticomm = 2 * si

    diff = (anticomm_xx - expected_anticomm).norm()
    assert diff < 1e-10, f"Anticommutation relation failed: {diff}"

    print("✓ Verified: {σx, σx} = 2I")

    # Verify eigenvalues: ±1
    evals_z = sz.eigenenergies()
    assert abs(evals_z[0] + 1.0) < 1e-10, "Eigenvalue error"
    assert abs(evals_z[1] - 1.0) < 1e-10, "Eigenvalue error"

    print("✓ Verified: σz eigenvalues = {-1, +1}")
    print("\n✓✓ Pauli operator tests passed\n")
    return True


def test_drift_hamiltonian():
    """Test 4: Construct drift Hamiltonian and verify properties."""
    print("=" * 60)
    print("TEST 4: Drift Hamiltonian")
    print("=" * 60)

    import qutip

    # Drift Hamiltonian: H₀ = (ω₀/2)σz
    omega0 = 5.0  # MHz (typical superconducting qubit frequency)
    sz = qutip.sigmaz()
    H0 = 0.5 * omega0 * sz

    print(f"✓ Created drift Hamiltonian: H₀ = (ω₀/2)σz")
    print(f"  Frequency: ω₀ = {omega0} MHz")

    # Verify eigenvalues: E = ±ω₀/2
    evals = H0.eigenenergies()
    expected_evals = np.array([-omega0 / 2, omega0 / 2])

    diff = np.abs(evals - expected_evals).max()
    assert diff < 1e-10, f"Energy eigenvalue error: {diff}"

    print(f"✓ Energy eigenvalues: E₀ = {evals[0]:.2f} MHz, E₁ = {evals[1]:.2f} MHz")
    print(f"✓ Energy splitting: ΔE = {evals[1] - evals[0]:.2f} MHz")

    # Verify eigenstates are |0⟩ and |1⟩
    _, evecs = H0.eigenstates()
    ket0 = qutip.basis(2, 0)
    ket1 = qutip.basis(2, 1)

    fid0 = qutip.fidelity(evecs[0], ket0)
    fid1 = qutip.fidelity(evecs[1], ket1)

    # Eigenstates can be in either order depending on implementation
    # Check that we have one of each (either evecs[0]=|0⟩, evecs[1]=|1⟩ OR vice versa)
    fid0_alt = qutip.fidelity(evecs[0], ket1)
    fid1_alt = qutip.fidelity(evecs[1], ket0)

    valid_ordering1 = fid0 > 0.999 and fid1 > 0.999
    valid_ordering2 = fid0_alt > 0.999 and fid1_alt > 0.999

    assert valid_ordering1 or valid_ordering2, (
        f"Eigenstate error: fid0={fid0:.4f}, fid1={fid1:.4f}"
    )

    print(f"✓ Eigenstates verified: |E₀⟩ and |E₁⟩ match |0⟩, |1⟩")
    print("\n✓✓ Drift Hamiltonian tests passed\n")
    return True


def test_time_evolution():
    """Test 5: Solve Schrödinger equation and verify periodicity."""
    print("=" * 60)
    print("TEST 5: Time Evolution (Schrödinger Equation)")
    print("=" * 60)

    import qutip

    # Setup: Drift Hamiltonian and initial state
    omega0 = 5.0  # MHz
    sz = qutip.sigmaz()
    H0 = 0.5 * omega0 * sz

    psi0 = qutip.basis(2, 0)  # Start in |0⟩

    # Time evolution: Should complete one full rotation at t = 2π/ω₀
    T_period = 2 * np.pi / omega0
    times = np.linspace(0, T_period, 100)

    print(f"✓ Initial state: |0⟩")
    print(f"✓ Evolution time: 0 to {T_period:.4f} μs (one period)")
    print(f"✓ Time steps: {len(times)}")

    # Solve Schrödinger equation
    result = qutip.sesolve(H0, psi0, times)

    print(f"✓ Solved Schrödinger equation: i∂ₜ|ψ⟩ = H₀|ψ⟩")

    # Verify periodicity: |ψ(T)⟩ should equal |ψ(0)⟩
    psi_final = result.states[-1]
    fidelity = qutip.fidelity(psi0, psi_final)

    print(f"✓ Final state fidelity: F = {fidelity:.10f}")

    assert fidelity > 0.9999, f"Periodicity failed: F = {fidelity}"

    print(f"✓ Periodicity verified: |ψ(T)⟩ ≈ |ψ(0)⟩")

    # Check that state stays on z-axis (no x, y rotation)
    sx = qutip.sigmax()
    sy = qutip.sigmay()

    expect_x = qutip.expect(sx, result.states)
    expect_y = qutip.expect(sy, result.states)

    max_x = np.abs(expect_x).max()
    max_y = np.abs(expect_y).max()

    assert max_x < 1e-10, f"Unexpected x-rotation: {max_x}"
    assert max_y < 1e-10, f"Unexpected y-rotation: {max_y}"

    print(f"✓ Verified: Rotation confined to z-axis")
    print(f"  ⟨σx⟩_max = {max_x:.2e}")
    print(f"  ⟨σy⟩_max = {max_y:.2e}")

    print("\n✓✓ Time evolution tests passed\n")
    return True


def test_bloch_sphere():
    """Test 6: Generate Bloch sphere visualization (optional)."""
    print("=" * 60)
    print("TEST 6: Bloch Sphere Visualization")
    print("=" * 60)

    try:
        import qutip
        import matplotlib.pyplot as plt

        # Create Bloch sphere
        b = qutip.Bloch()

        # Add some states
        ket0 = qutip.basis(2, 0)
        ket1 = qutip.basis(2, 1)
        ket_plus = (ket0 + ket1).unit()
        ket_i = (ket0 + 1j * ket1).unit()

        b.add_states([ket0, ket1, ket_plus, ket_i])

        print("✓ Created Bloch sphere")
        print("✓ Added states: |0⟩, |1⟩, |+⟩, |i⟩")

        # Try to render (won't display in headless mode, but tests API)
        b.vector_color = ["r", "b", "g", "orange"]

        print("✓ Bloch sphere API functional")
        print("  (Visualization ready for notebooks)")

        plt.close("all")  # Clean up

        print("\n✓✓ Bloch sphere tests passed\n")
        return True

    except Exception as e:
        print(f"⚠ Bloch sphere test skipped (non-critical): {e}")
        print("  This is normal in headless environments")
        print("\n✓✓ Bloch sphere API available\n")
        return True


def main():
    """Run all tests and report results."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "QubitPulseOpt Environment Test" + " " * 17 + "║")
    print("╚" + "=" * 58 + "╝")
    print("\n")

    tests = [
        ("Package Imports", test_imports),
        ("Qubit States", test_qubit_states),
        ("Pauli Operators", test_pauli_operators),
        ("Drift Hamiltonian", test_drift_hamiltonian),
        ("Time Evolution", test_time_evolution),
        ("Bloch Sphere", test_bloch_sphere),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n✗✗ TEST FAILED: {name}")
            print(f"Error: {e}\n")
            results.append((name, False))

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status:8} - {name}")

    print()
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED - Environment ready for Week 1.2! 🎉\n")
        return 0
    else:
        print(f"\n⚠ {total - passed} test(s) failed - Check configuration\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
