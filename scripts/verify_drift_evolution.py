#!/usr/bin/env python3
"""
Verify Drift Hamiltonian Evolution from First Principles
=========================================================

This script proves the analytical solution for drift evolution
by comparing it step-by-step with numerical solutions.
"""

import numpy as np
import qutip

print("=" * 70)
print("DRIFT HAMILTONIAN EVOLUTION: FIRST PRINCIPLES VERIFICATION")
print("=" * 70)

# Parameters
omega_0 = 5.0  # MHz
t = 0.5  # microseconds

print(f"\nParameters:")
print(f"  ω₀ = {omega_0} MHz")
print(f"  t = {t} μs")

# ============================================================================
# METHOD 1: Matrix Exponential (The "Right" Way)
# ============================================================================
print("\n" + "=" * 70)
print("METHOD 1: Direct Matrix Exponential")
print("=" * 70)

# Build Hamiltonian
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
H0 = 0.5 * omega_0 * sigma_z

print(f"\nHamiltonian H₀ = (ω₀/2)σz:")
print(H0)

# Compute U(t) = exp(-iHt) using scipy
from scipy.linalg import expm

U_direct = expm(-1j * H0 * t)
print(f"\nU(t) = exp(-iH₀t):")
print(U_direct)

# ============================================================================
# METHOD 2: Analytical Formula (What We Implemented)
# ============================================================================
print("\n" + "=" * 70)
print("METHOD 2: Analytical Formula")
print("=" * 70)

print(f"\nFor H₀ = (ω₀/2)σz, we can derive:")
print(f"  U(t) = cos(ω₀t/2)I - i·sin(ω₀t/2)σz")

cos_term = np.cos(omega_0 * t / 2)
sin_term = np.sin(omega_0 * t / 2)

I = np.eye(2, dtype=complex)
U_analytical = cos_term * I - 1j * sin_term * sigma_z

print(f"\ncos(ω₀t/2) = {cos_term:.6f}")
print(f"sin(ω₀t/2) = {sin_term:.6f}")
print(f"\nU(t) analytical:")
print(U_analytical)

# ============================================================================
# METHOD 3: Diagonal Form (Why This Works)
# ============================================================================
print("\n" + "=" * 70)
print("METHOD 3: Diagonal Form (Understanding Why)")
print("=" * 70)

print(f"\nSince σz is diagonal, exp(-i(ω₀/2)σz·t) is also diagonal:")

# For diagonal matrices, exponential is element-wise
U_diagonal = np.array([
    [np.exp(-1j * omega_0 * t / 2), 0],
    [0, np.exp(1j * omega_0 * t / 2)]
], dtype=complex)

print(f"\nU(t) diagonal form:")
print(U_diagonal)

print(f"\nExpanding with Euler's formula e^(iθ) = cos(θ) + i·sin(θ):")
print(f"  Top-left:  e^(-iω₀t/2) = {np.exp(-1j*omega_0*t/2):.6f}")
print(f"  Bottom-right: e^(+iω₀t/2) = {np.exp(1j*omega_0*t/2):.6f}")

# ============================================================================
# VERIFICATION: All Methods Agree
# ============================================================================
print("\n" + "=" * 70)
print("VERIFICATION: Do All Methods Agree?")
print("=" * 70)

diff_1_2 = np.linalg.norm(U_direct - U_analytical)
diff_1_3 = np.linalg.norm(U_direct - U_diagonal)
diff_2_3 = np.linalg.norm(U_analytical - U_diagonal)

print(f"\n||U_direct - U_analytical|| = {diff_1_2:.2e}")
print(f"||U_direct - U_diagonal||   = {diff_1_3:.2e}")
print(f"||U_analytical - U_diagonal|| = {diff_2_3:.2e}")

if diff_1_2 < 1e-10 and diff_1_3 < 1e-10 and diff_2_3 < 1e-10:
    print("\n✓✓✓ All methods agree to machine precision!")
else:
    print("\n✗ Methods disagree - something is wrong")

# ============================================================================
# UNITARITY CHECK
# ============================================================================
print("\n" + "=" * 70)
print("UNITARITY CHECK: U†U = I?")
print("=" * 70)

U_dagger = U_analytical.conj().T
product = U_dagger @ U_analytical

print(f"\nU†U =")
print(product)

diff_from_identity = np.linalg.norm(product - I)
print(f"\n||U†U - I|| = {diff_from_identity:.2e}")

if diff_from_identity < 1e-10:
    print("✓ Unitary property confirmed!")

# ============================================================================
# EXAMPLE: Evolve |0⟩ State
# ============================================================================
print("\n" + "=" * 70)
print("EXAMPLE: Time Evolution of |0⟩")
print("=" * 70)

ket0 = np.array([[1], [0]], dtype=complex)
print(f"\nInitial state |0⟩:")
print(ket0)

psi_t = U_analytical @ ket0
print(f"\n|ψ(t)⟩ = U(t)|0⟩:")
print(psi_t)

# Compute expectation values
def expect(operator, state):
    """Compute ⟨ψ|O|ψ⟩"""
    return (state.conj().T @ operator @ state)[0, 0].real

sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)

exp_x = expect(sigma_x, psi_t)
exp_y = expect(sigma_y, psi_t)
exp_z = expect(sigma_z, psi_t)

print(f"\nBloch sphere coordinates:")
print(f"  ⟨σx⟩ = {exp_x:.6f}")
print(f"  ⟨σy⟩ = {exp_y:.6f}")
print(f"  ⟨σz⟩ = {exp_z:.6f}")

print(f"\n✓ State stays on z-axis (no x, y rotation)")

# ============================================================================
# PERIODICITY
# ============================================================================
print("\n" + "=" * 70)
print("PERIODICITY: Return After One Period")
print("=" * 70)

T_period = 2 * np.pi / omega_0
print(f"\nPrecession period T = 2π/ω₀ = {T_period:.6f} μs")

# Evolve for one full period
U_full_period = expm(-1j * H0 * T_period)
psi_after_period = U_full_period @ ket0

print(f"\n|ψ(0)⟩:")
print(ket0)
print(f"\n|ψ(T)⟩:")
print(psi_after_period)

# Compute fidelity (overlaps can have global phase)
fidelity = np.abs((ket0.conj().T @ psi_after_period)[0, 0])**2
print(f"\nFidelity F = |⟨ψ(0)|ψ(T)⟩|² = {fidelity:.15f}")

if fidelity > 0.9999:
    print("✓ State returns to initial (up to global phase)")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY: What We Learned")
print("=" * 70)

print("""
1. The drift Hamiltonian H₀ = (ω₀/2)σz causes pure z-axis rotation
2. Time evolution operator: U(t) = exp(-iH₀t)
3. Three equivalent forms:
   - Matrix exponential (numerical)
   - Analytical: cos(ω₀t/2)I - i·sin(ω₀t/2)σz
   - Diagonal: diag(e^(-iω₀t/2), e^(+iω₀t/2))
4. |0⟩ and |1⟩ are stationary (up to global phase)
5. Period T = 2π/ω₀
6. Evolution is unitary: U†U = I

This is the foundation for quantum control - we understand the "drift"
before adding control pulses to steer the qubit where we want!
""")

print("=" * 70)

