#!/usr/bin/env python3
"""
Experiment 2: 3-Level Transmon Comparison
=========================================

Compares Gaussian, DRAG, and GRAPE pulses for X-gate in a 3-level transmon
system. This is the KEY differentiator for the PRA paper: in a 3-level system,
leakage to |2⟩ limits DRAG (first-order correction only), while GRAPE can
suppress it fully by optimizing over the full Hilbert space.

Measurements:
- Computational fidelity (|0⟩,|1⟩ subspace)
- Leakage probability P₂ = |⟨2|ψ_final⟩|²
- Gate time sweep from 10ns to 100ns

3-Level Transmon Hamiltonian (rotating frame):
    H_drift = (α/2) * n(n-1)   where n = a†a, α = anharmonicity
    H_x = 0.5 * (a + a†)       I-channel drive
    H_y = 0.5 * i(a† - a)      Q-channel drive

The qubit frequency ω_q is eliminated by going to the rotating frame.
In this frame, the 0→1 transition is resonant (zero detuning), and the
1→2 transition is detuned by α (the anharmonicity).

References:
-----------
[1] Motzoi et al., PRL 103, 110501 (2009)
[2] Gambetta et al., PRA 83, 012308 (2011)
[3] Chen et al., PRL 116, 020501 (2016)
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from time import perf_counter

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import qutip as qt

from src.pulses.drag import DRAGPulse, DRAGParameters
from src.optimization.grape import GRAPEOptimizer


# ── Transmon Hamiltonian ──────────────────────────────────────────────────


def build_transmon_drift(n_levels: int, anharmonicity_mhz: float) -> qt.Qobj:
    """
    Build 3-level transmon drift Hamiltonian in the rotating frame.

    In the frame rotating at the qubit frequency ω_q:
        H_drift = (α/2) * n̂(n̂ - 1)
    where n̂ = a†a is the number operator and α is the anharmonicity.

    This gives energy levels: E_0 = 0, E_1 = 0, E_2 = α
    so the 0→1 transition is resonant and 1→2 is detuned by α.

    Parameters
    ----------
    n_levels : int
        Number of transmon levels (typically 3).
    anharmonicity_mhz : float
        Anharmonicity α/2π in MHz (negative for transmon, e.g. -200).

    Returns
    -------
    H_drift : qt.Qobj
        Drift Hamiltonian in rad/ns units.
    """
    # Convert MHz → rad/ns: multiply by 2π × 1e-3
    alpha_radns = 2.0 * np.pi * anharmonicity_mhz * 1e-3

    n_op = qt.num(n_levels)
    # H = (α/2) * n(n-1) = (α/2) * (n² - n)
    H_drift = (alpha_radns / 2.0) * (n_op * n_op - n_op)

    return H_drift


def build_control_hamiltonians(n_levels: int) -> tuple:
    """
    Build control Hamiltonians for n-level transmon.

    H_x = 0.5 * (a + a†)   — in-phase (I) drive
    H_y = 0.5 * i(a† - a)  — quadrature (Q) drive

    The factor of 0.5 matches the Rabi convention H = (Ω/2)σ for 2-level.

    Returns
    -------
    H_x, H_y : qt.Qobj
        Control Hamiltonians.
    """
    a = qt.destroy(n_levels)
    a_dag = qt.create(n_levels)
    H_x = 0.5 * (a + a_dag)
    H_y = 0.5 * 1j * (a_dag - a)

    return H_x, H_y


def embed_target_unitary(U_2x2: qt.Qobj, n_levels: int) -> qt.Qobj:
    """
    Embed a 2×2 target unitary into n-level space.

    The target acts on the computational subspace {|0⟩, |1⟩} and is
    identity on higher levels.
    """
    U_full = np.eye(n_levels, dtype=complex)
    U_full[0:2, 0:2] = U_2x2.full()
    return qt.Qobj(U_full, dims=[[n_levels], [n_levels]])


# ── Fidelity and Leakage ─────────────────────────────────────────────────


def state_fidelity_and_leakage(psi_final: qt.Qobj, U_target_full: qt.Qobj,
                                psi0: qt.Qobj, n_levels: int) -> tuple:
    """
    Compute state fidelity and leakage for a given initial state.

    Returns
    -------
    fidelity : float
        |⟨ψ_target|ψ_final⟩|²
    leakage : float
        Σ_{j≥2} |⟨j|ψ_final⟩|² — population outside computational subspace.
    """
    psi_target = U_target_full * psi0
    fidelity = float(qt.fidelity(psi_target, psi_final) ** 2)

    # Leakage: population in levels |2⟩, |3⟩, ...
    psi_arr = psi_final.full().flatten()
    leakage = float(sum(np.abs(psi_arr[j]) ** 2 for j in range(2, n_levels)))

    return fidelity, leakage


def average_fidelity_and_leakage(U_achieved: qt.Qobj, U_target_full: qt.Qobj,
                                  n_levels: int) -> tuple:
    """
    Average state fidelity and leakage over computational basis states |0⟩ and |1⟩.
    """
    total_fid = 0.0
    total_leak = 0.0

    for k in range(2):  # |0⟩ and |1⟩
        psi0 = qt.basis(n_levels, k)
        psi_final = U_achieved * psi0
        fid, leak = state_fidelity_and_leakage(psi_final, U_target_full, psi0, n_levels)
        total_fid += fid
        total_leak += leak

    return total_fid / 2.0, total_leak / 2.0


# ── Simulation ────────────────────────────────────────────────────────────


def simulate_pulse_evolution_3level(
    pulse_I: np.ndarray,
    pulse_Q: np.ndarray,
    times: np.ndarray,
    H_drift: qt.Qobj,
    H_x: qt.Qobj,
    H_y: qt.Qobj,
    n_levels: int,
) -> qt.Qobj:
    """
    Simulate unitary evolution under I/Q pulse envelopes for n-level system.

    Uses piecewise-constant propagation:
        U(T) = ∏_k exp(-i * H_k * dt)
    where H_k = H_drift + pulse_I[k]*H_x + pulse_Q[k]*H_y
    """
    dt = times[1] - times[0]
    U = qt.qeye(n_levels)

    for i in range(len(times) - 1):
        H_t = H_drift + pulse_I[i] * H_x + pulse_Q[i] * H_y
        U = (-1j * H_t * dt).expm() * U

    return U


# ── Pulse Methods ─────────────────────────────────────────────────────────


def run_gaussian_3level(gate_time: float, n_points: int, params: dict) -> dict:
    """Run Gaussian pulse (β=0) in 3-level system."""
    start = perf_counter()

    n_levels = params['n_levels']
    sigma = gate_time / 8.0
    # Calibrate amplitude for π rotation: ∫Ω dt = π
    # For n-level, the 0→1 matrix element of (a+a†) is 1 (not √2),
    # so with H_x = 0.5*(a+a†), we get ⟨1|H_x|0⟩ = 0.5, same as (σx/2).
    # Therefore same calibration as 2-level.
    amplitude = np.pi / (sigma * np.sqrt(2 * np.pi))

    times = np.linspace(0, gate_time, n_points)
    t_center = gate_time / 2.0
    gaussian = amplitude * np.exp(-0.5 * ((times - t_center) / sigma) ** 2)

    pulse_I = gaussian
    pulse_Q = np.zeros_like(times)

    H_drift = build_transmon_drift(n_levels, params['anharmonicity_mhz'])
    H_x, H_y = build_control_hamiltonians(n_levels)

    U_achieved = simulate_pulse_evolution_3level(
        pulse_I, pulse_Q, times, H_drift, H_x, H_y, n_levels
    )

    U_target_full = embed_target_unitary(qt.sigmax(), n_levels)
    fidelity, leakage = average_fidelity_and_leakage(U_achieved, U_target_full, n_levels)

    elapsed = perf_counter() - start
    return {
        "method": "gaussian",
        "gate_time_ns": gate_time,
        "fidelity": float(fidelity),
        "infidelity": float(1 - fidelity),
        "leakage": float(leakage),
        "time_s": elapsed,
        "amplitude_radns": float(amplitude),
    }


def run_drag_3level(gate_time: float, n_points: int, params: dict) -> dict:
    """Run DRAG pulse with unit-corrected β in 3-level system."""
    start = perf_counter()

    n_levels = params['n_levels']
    sigma = gate_time / 8.0
    amplitude = np.pi / (sigma * np.sqrt(2 * np.pi))

    # Correct DRAG beta with proper unit conversion
    # DRAG beta = -1/(2*alpha) where alpha is anharmonicity in rad/ns
    # Motzoi et al. PRL 103, 110501 (2009): beta depends only on anharmonicity
    anharmonicity_radns = 2.0 * np.pi * params["anharmonicity_mhz"] * 1e-3
    optimal_beta = -1.0 / (2.0 * anharmonicity_radns)

    drag_params = DRAGParameters(
        amplitude=amplitude,
        sigma=sigma,
        beta=optimal_beta,
        anharmonicity=params['anharmonicity_mhz'],
    )
    drag = DRAGPulse(drag_params)

    times = np.linspace(0, gate_time, n_points)
    t_center = gate_time / 2.0
    omega_I, omega_Q = drag.envelope(times, t_center)

    H_drift = build_transmon_drift(n_levels, params['anharmonicity_mhz'])
    H_x, H_y = build_control_hamiltonians(n_levels)

    U_achieved = simulate_pulse_evolution_3level(
        omega_I, omega_Q, times, H_drift, H_x, H_y, n_levels
    )

    U_target_full = embed_target_unitary(qt.sigmax(), n_levels)
    fidelity, leakage = average_fidelity_and_leakage(U_achieved, U_target_full, n_levels)

    elapsed = perf_counter() - start
    return {
        "method": "drag",
        "gate_time_ns": gate_time,
        "fidelity": float(fidelity),
        "infidelity": float(1 - fidelity),
        "leakage": float(leakage),
        "time_s": elapsed,
        "amplitude_radns": float(amplitude),
        "beta": float(optimal_beta),
    }


def _grape_propagate(pulses: np.ndarray, n_slices: int, gate_time: float,
                     H_drift: qt.Qobj, H_x: qt.Qobj, H_y: qt.Qobj,
                     n_levels: int) -> qt.Qobj:
    """
    Reconstruct unitary exactly as GRAPE does internally.

    GRAPE uses n_slices piecewise-constant intervals with dt = gate_time/n_slices.
    This must match exactly to get consistent fidelity/leakage numbers.
    """
    dt = gate_time / n_slices
    U = qt.qeye(n_levels)
    for k in range(n_slices):
        H_t = H_drift + pulses[0, k] * H_x + pulses[1, k] * H_y
        U = (-1j * H_t * dt).expm() * U
    return U


def run_grape_3level(gate_time: float, n_slices: int, params: dict) -> dict:
    """
    Run GRAPE optimization in 3-level system.

    GRAPE optimizes over the full 3-level Hilbert space, so it can
    suppress leakage to |2⟩ that DRAG can only partially cancel.

    Uses tuned hyperparameters: lr=0.5, momentum=0.5, no step decay.
    """
    start = perf_counter()

    n_levels = params['n_levels']
    sigma = gate_time / 8.0
    amplitude = np.pi / (sigma * np.sqrt(2 * np.pi))

    H_drift = build_transmon_drift(n_levels, params['anharmonicity_mhz'])
    H_x, H_y = build_control_hamiltonians(n_levels)

    U_target_full = embed_target_unitary(qt.sigmax(), n_levels)

    # Initialize with Gaussian seed on I-channel
    t_slices = np.linspace(0, gate_time, n_slices)
    t_center = gate_time / 2.0
    gaussian = amplitude * np.exp(-0.5 * ((t_slices - t_center) / sigma) ** 2)

    u_init = np.zeros((2, n_slices))
    u_init[0, :] = gaussian  # I-channel: Gaussian seed
    # Add small random perturbation to help optimizer explore Q-channel
    rng = np.random.default_rng(42)
    u_init[1, :] = rng.normal(0, amplitude * 0.05, n_slices)  # small Q seed

    optimizer = GRAPEOptimizer(
        H_drift=H_drift,
        H_controls=[H_x, H_y],
        n_timeslices=n_slices,
        total_time=gate_time,
        u_limits=(-amplitude * 3.0, amplitude * 3.0),
        convergence_threshold=1e-10,
        max_iterations=params['grape_max_iter'],
        learning_rate=0.5,          # Higher LR for 3-level
        verbose=False,
        use_line_search=True,
        momentum=0.5,               # Momentum helps escape saddle points
    )

    # No step decay — critical for 3-level convergence
    result = optimizer.optimize_unitary(U_target_full, u_init=u_init, step_decay=1.0)

    # Verify with propagation that exactly matches GRAPE's internal convention
    U_grape = _grape_propagate(
        result.optimized_pulses, n_slices, gate_time,
        H_drift, H_x, H_y, n_levels
    )
    fidelity, leakage = average_fidelity_and_leakage(U_grape, U_target_full, n_levels)

    elapsed = perf_counter() - start
    return {
        "method": "grape",
        "gate_time_ns": gate_time,
        "fidelity": float(fidelity),
        "infidelity": float(1 - fidelity),
        "leakage": float(leakage),
        "grape_fidelity": float(result.final_fidelity),
        "iterations": result.n_iterations,
        "converged": result.converged,
        "time_s": elapsed,
    }


# ── Main Experiment ───────────────────────────────────────────────────────


def run_single_gate_time(gate_time: float, params: dict) -> dict:
    """Run all three methods at a single gate time."""
    n_points = max(200, int(gate_time * 10))  # At least 10 points/ns
    n_slices = max(50, int(gate_time * 2))    # At least 2 slices/ns for GRAPE

    print(f"\n  Gate time = {gate_time:.0f} ns (n_points={n_points}, n_slices={n_slices})")

    print("    Gaussian...", end=" ", flush=True)
    gauss = run_gaussian_3level(gate_time, n_points, params)
    print(f"F={gauss['fidelity']:.6f}, P2={gauss['leakage']:.2e}")

    print("    DRAG...", end=" ", flush=True)
    drag = run_drag_3level(gate_time, n_points, params)
    print(f"F={drag['fidelity']:.6f}, P2={drag['leakage']:.2e}, beta={drag['beta']:.3f}")

    print("    GRAPE...", end=" ", flush=True)
    grape = run_grape_3level(gate_time, n_slices, params)
    print(f"F={grape['fidelity']:.6f}, P2={grape['leakage']:.2e}, "
          f"iter={grape['iterations']}, conv={grape['converged']}")

    return {
        "gate_time_ns": gate_time,
        "gaussian": gauss,
        "drag": drag,
        "grape": grape,
    }


def run_experiment():
    """Run the full 3-level transmon comparison experiment."""
    print("=" * 70)
    print("Experiment 2: 3-Level Transmon Comparison")
    print("=" * 70)
    print("Comparing Gaussian, DRAG, and GRAPE for X-gate in 3-level transmon")
    print()

    params = {
        "n_levels": 3,
        "anharmonicity_mhz": -200.0,   # Typical transmon alpha/2pi
        "grape_max_iter": 1000,         # More iterations for 3-level
    }

    # Transmon parameters
    alpha_mhz = params['anharmonicity_mhz']
    alpha_radns = 2.0 * np.pi * alpha_mhz * 1e-3
    print(f"Transmon parameters:")
    print(f"  Anharmonicity: alpha/2pi = {alpha_mhz} MHz = {alpha_radns:.4f} rad/ns")
    print(f"  Number of levels: {params['n_levels']}")
    print(f"  GRAPE max iterations: {params['grape_max_iter']}")

    # Gate time sweep: 10, 15, 20, 25, 30, 40, 50, 60, 80, 100 ns
    gate_times = [10, 15, 20, 25, 30, 40, 50, 60, 80, 100]
    print(f"  Gate times: {gate_times} ns")
    print()

    results = {
        "timestamp": datetime.now().isoformat(),
        "experiment": "3-level transmon comparison",
        "parameters": params,
        "gate_times": gate_times,
        "sweeps": [],
    }

    for gt in gate_times:
        sweep_result = run_single_gate_time(float(gt), params)
        results["sweeps"].append(sweep_result)

    # ── Summary table ──
    print("\n" + "=" * 90)
    print("SUMMARY: 3-Level Transmon X-Gate Comparison")
    print("=" * 90)
    print(f"{'t_gate':>6}  {'Gaussian F':>12} {'Gauss P2':>10}  "
          f"{'DRAG F':>12} {'DRAG P2':>10}  "
          f"{'GRAPE F':>12} {'GRAPE P2':>10}")
    print("-" * 90)

    for s in results["sweeps"]:
        gt = s["gate_time_ns"]
        gf = s["gaussian"]["fidelity"]
        gl = s["gaussian"]["leakage"]
        df = s["drag"]["fidelity"]
        dl = s["drag"]["leakage"]
        pf = s["grape"]["fidelity"]
        pl = s["grape"]["leakage"]
        print(f"{gt:6.0f}  {gf:12.6f} {gl:10.2e}  "
              f"{df:12.6f} {dl:10.2e}  "
              f"{pf:12.6f} {pl:10.2e}")

    # ── Summary statistics ──
    gauss_fids = [s["gaussian"]["fidelity"] for s in results["sweeps"]]
    drag_fids = [s["drag"]["fidelity"] for s in results["sweeps"]]
    grape_fids = [s["grape"]["fidelity"] for s in results["sweeps"]]
    gauss_leaks = [s["gaussian"]["leakage"] for s in results["sweeps"]]
    drag_leaks = [s["drag"]["leakage"] for s in results["sweeps"]]
    grape_leaks = [s["grape"]["leakage"] for s in results["sweeps"]]

    print("-" * 90)
    print(f"{'Avg':>6}  {np.mean(gauss_fids):12.6f} {np.mean(gauss_leaks):10.2e}  "
          f"{np.mean(drag_fids):12.6f} {np.mean(drag_leaks):10.2e}  "
          f"{np.mean(grape_fids):12.6f} {np.mean(grape_leaks):10.2e}")

    results["summary"] = {
        "gaussian_avg_fidelity": float(np.mean(gauss_fids)),
        "gaussian_avg_leakage": float(np.mean(gauss_leaks)),
        "drag_avg_fidelity": float(np.mean(drag_fids)),
        "drag_avg_leakage": float(np.mean(drag_leaks)),
        "grape_avg_fidelity": float(np.mean(grape_fids)),
        "grape_avg_leakage": float(np.mean(grape_leaks)),
    }

    # ── Save results ──
    results_dir = Path(__file__).parent.parent / "results" / "multilevel_comparison"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_file = results_dir / f"3level_xgate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")
    return results


if __name__ == "__main__":
    run_experiment()
