#!/usr/bin/env python3
"""
DRAG vs GRAPE Pulse Optimization Benchmark
For IEEE TQE submission - comparing pulse optimization methods

Compares:
- Gaussian baseline (DRAG with β=0)
- DRAG pulses (with optimized β)
- GRAPE-optimized pulses

Gates: X, Y, H (single-qubit)
Uses proper pulse calibration and correct Hamiltonian H = (Ω/2)σ.
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


def create_results_dir():
    results_dir = Path(__file__).parent.parent / "results" / "drag_vs_grape"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def gate_fidelity(U_achieved: qt.Qobj, U_target: qt.Qobj) -> float:
    """Compute gate fidelity (Frobenius inner product)."""
    d = U_target.shape[0]
    overlap = np.abs((U_achieved.dag() * U_target).tr())
    return (overlap / d) ** 2


def simulate_pulse_evolution(
    pulse_I: np.ndarray,
    pulse_Q: np.ndarray,
    times: np.ndarray,
    detuning: float = 0.0,
) -> qt.Qobj:
    """
    Simulate unitary evolution under I/Q pulse envelopes.
    
    Uses proper Rabi Hamiltonian: H = (Δ/2)σz + (Ω_I/2)σx + (Ω_Q/2)σy
    where rotation angle = ∫Ω dt for resonant driving.
    """
    dt = times[1] - times[0]
    U = qt.qeye(2)
    
    sigma_x = qt.sigmax()
    sigma_y = qt.sigmay()
    sigma_z = qt.sigmaz()
    
    for i in range(len(times) - 1):
        # Proper Rabi Hamiltonian with factor of 1/2
        H_t = 0.5 * detuning * sigma_z + 0.5 * pulse_I[i] * sigma_x + 0.5 * pulse_Q[i] * sigma_y
        U = (-1j * H_t * dt).expm() * U
    
    return U


def run_gaussian_baseline(gate_type: str, U_target: qt.Qobj, params: dict) -> dict:
    """Run Gaussian (β=0) pulse with proper calibration."""
    start = perf_counter()
    
    gate_time = params['gate_time']
    n_points = params['n_samples']
    
    target_angle = np.pi if gate_type in ['X', 'Y'] else np.pi / 2
    sigma = gate_time / 8.0
    amplitude = target_angle / (sigma * np.sqrt(2 * np.pi))
    
    drag_params = DRAGParameters(
        amplitude=amplitude,
        sigma=sigma,
        beta=0.0,
        anharmonicity=params['anharmonicity'],
    )
    drag = DRAGPulse(drag_params)
    
    times = np.linspace(0, gate_time, n_points)
    t_center = gate_time / 2
    omega_I, omega_Q = drag.envelope(times, t_center)
    
    if gate_type == 'Y':
        omega_I, omega_Q = omega_Q, omega_I
    elif gate_type == 'H':
        # Hadamard: rotation about (X+Z)/√2 axis
        # Approximate as rotation about X axis (simplified)
        pass
    
    U_achieved = simulate_pulse_evolution(omega_I, omega_Q, times, params['detuning'])
    fidelity = gate_fidelity(U_achieved, U_target)
    elapsed = perf_counter() - start
    
    return {
        "method": "gaussian",
        "gate": gate_type,
        "fidelity": float(fidelity),
        "infidelity": float(1 - fidelity),
        "time_s": elapsed,
        "amplitude_MHz": float(amplitude),
        "sigma_ns": float(sigma),
        "beta": 0.0,
    }


def run_drag_optimized(gate_type: str, U_target: qt.Qobj, params: dict) -> dict:
    """Run DRAG pulse with optimized β."""
    start = perf_counter()
    
    gate_time = params['gate_time']
    n_points = params['n_samples']
    
    target_angle = np.pi if gate_type in ['X', 'Y'] else np.pi / 2
    sigma = gate_time / 8.0
    amplitude = target_angle / (sigma * np.sqrt(2 * np.pi))
    # DRAG beta = -1/(2*alpha) where alpha is anharmonicity in rad/ns
    # Motzoi et al. PRL 103, 110501 (2009): beta depends only on anharmonicity
    anharmonicity_radns = 2.0 * np.pi * params["anharmonicity"] * 1e-3  # MHz -> rad/ns
    optimal_beta = -1.0 / (2.0 * anharmonicity_radns)
    
    drag_params = DRAGParameters(
        amplitude=amplitude,
        sigma=sigma,
        beta=optimal_beta,
        anharmonicity=params['anharmonicity'],
    )
    drag = DRAGPulse(drag_params)
    
    times = np.linspace(0, gate_time, n_points)
    t_center = gate_time / 2
    omega_I, omega_Q = drag.envelope(times, t_center)
    
    if gate_type == 'Y':
        omega_I, omega_Q = omega_Q, omega_I
    
    U_achieved = simulate_pulse_evolution(omega_I, omega_Q, times, params['detuning'])
    fidelity = gate_fidelity(U_achieved, U_target)
    elapsed = perf_counter() - start
    
    return {
        "method": "drag",
        "gate": gate_type,
        "fidelity": float(fidelity),
        "infidelity": float(1 - fidelity),
        "time_s": elapsed,
        "amplitude_MHz": float(amplitude),
        "sigma_ns": float(sigma),
        "beta": float(optimal_beta),
    }


def _build_grape_init(gate_type: str, params: dict) -> tuple:
    """Build gate-specific GRAPE initialization.
    
    Returns (n_slices, u_init, amplitude) tuple.
    
    For X/Y gates: Gaussian on the appropriate channel.
    For H gate: Euler decomposition H = e^{iπ/2} Ry(π/2) Rz(π).
        In the rotating frame, Rz is a virtual-Z (free), so we seed
        with a π/2 rotation on the Y-channel plus a small X component
        to break symmetry. Use 100 timeslices for better convergence.
    For S/T gates: small-angle rotations, seed with scaled Gaussian on both channels.
    """
    sigma = params['gate_time'] / 8.0
    
    if gate_type in ['X', 'Y']:
        n_slices = params['n_grape_slices']
        target_angle = np.pi
        amplitude = target_angle / (sigma * np.sqrt(2 * np.pi))
        t_slices = np.linspace(0, params['gate_time'], n_slices)
        t_center = params['gate_time'] / 2
        gaussian = amplitude * np.exp(-0.5 * ((t_slices - t_center) / sigma) ** 2)
        u_init = np.zeros((2, n_slices))
        if gate_type == 'Y':
            u_init[1, :] = gaussian
        else:
            u_init[0, :] = gaussian
        return n_slices, u_init, amplitude
    
    elif gate_type == 'H':
        # H = e^{iπ/2} Ry(π/2) Rz(π) — need rotation about Y and phase
        # Use more timeslices for this harder optimization landscape
        n_slices = 100
        target_angle = np.pi  # amplitude scale reference
        amplitude = target_angle / (sigma * np.sqrt(2 * np.pi))
        t_slices = np.linspace(0, params['gate_time'], n_slices)
        t_center = params['gate_time'] / 2
        # Ry(π/2) needs half the area of a π pulse on Y-channel
        gaussian_half = 0.5 * amplitude * np.exp(-0.5 * ((t_slices - t_center) / sigma) ** 2)
        u_init = np.zeros((2, n_slices))
        # Seed both channels: Y for Ry(π/2), X for symmetry breaking
        u_init[0, :] = gaussian_half * 0.3  # small X component
        u_init[1, :] = gaussian_half         # main Y component for Ry(π/2)
        # Add small random perturbation to help escape saddle points
        rng = np.random.default_rng(42)
        u_init += rng.normal(0, amplitude * 0.02, u_init.shape)
        return n_slices, u_init, amplitude
    
    elif gate_type == 'S':
        # S = diag(1, i) = Rz(π/2) — virtual Z in rotating frame
        # But we optimize via physical X/Y drives
        n_slices = 100
        target_angle = np.pi / 2
        amplitude = np.pi / (sigma * np.sqrt(2 * np.pi))  # use π-pulse scale
        t_slices = np.linspace(0, params['gate_time'], n_slices)
        t_center = params['gate_time'] / 2
        gaussian = 0.3 * amplitude * np.exp(-0.5 * ((t_slices - t_center) / sigma) ** 2)
        u_init = np.zeros((2, n_slices))
        rng = np.random.default_rng(43)
        u_init[0, :] = gaussian * 0.5
        u_init[1, :] = gaussian * 0.5
        u_init += rng.normal(0, amplitude * 0.02, u_init.shape)
        return n_slices, u_init, amplitude
    
    elif gate_type == 'T':
        # T = diag(1, e^{iπ/4}) = Rz(π/4)
        n_slices = 100
        target_angle = np.pi / 4
        amplitude = np.pi / (sigma * np.sqrt(2 * np.pi))
        t_slices = np.linspace(0, params['gate_time'], n_slices)
        t_center = params['gate_time'] / 2
        gaussian = 0.2 * amplitude * np.exp(-0.5 * ((t_slices - t_center) / sigma) ** 2)
        u_init = np.zeros((2, n_slices))
        rng = np.random.default_rng(44)
        u_init[0, :] = gaussian * 0.3
        u_init[1, :] = gaussian * 0.3
        u_init += rng.normal(0, amplitude * 0.02, u_init.shape)
        return n_slices, u_init, amplitude
    
    else:
        raise ValueError(f"Unknown gate type: {gate_type}")


def run_grape_optimized(gate_type: str, U_target: qt.Qobj, params: dict) -> dict:
    """Run GRAPE pulse optimization with gate-specific initialization."""
    start = perf_counter()
    
    # GRAPE uses H = H_drift + Σ u_k * H_control[k]
    # For Rabi: H = (Δ/2)σz + (u_x/2)σx + (u_y/2)σy
    H_drift = 0.5 * params['detuning'] * qt.sigmaz()
    H_controls = [0.5 * qt.sigmax(), 0.5 * qt.sigmay()]  # Factor of 1/2
    
    n_slices, u_init, amplitude = _build_grape_init(gate_type, params)
    
    optimizer = GRAPEOptimizer(
        H_drift=H_drift,
        H_controls=H_controls,
        n_timeslices=n_slices,
        total_time=params['gate_time'],
        u_limits=(-amplitude * 3.0, amplitude * 3.0),
        convergence_threshold=1e-8,
        max_iterations=params['grape_max_iter'],
        learning_rate=0.1,
        verbose=False,
    )
    
    result = optimizer.optimize_unitary(U_target, u_init=u_init)
    
    fidelity = result.final_fidelity
    elapsed = perf_counter() - start
    
    return {
        "method": "grape",
        "gate": gate_type,
        "fidelity": float(fidelity),
        "infidelity": float(1 - fidelity),
        "time_s": elapsed,
        "iterations": result.n_iterations,
        "converged": result.converged,
    }


def run_benchmark():
    """Run full DRAG vs GRAPE comparison benchmark."""
    print("=" * 60)
    print("DRAG vs GRAPE Pulse Optimization Benchmark")
    print("=" * 60)
    print("For PRA submission — comparing pulse optimization methods")
    print("Using Rabi Hamiltonian H = (\u03a9/2)\u03c3 with calibrated pulses")
    print()
    
    params = {
        "gate_time": 40.0,
        "n_samples": 400,
        "detuning": 0.0,
        "anharmonicity": -200.0,
        "T1_us": 50.0,
        "T2_us": 70.0,
        "n_grape_slices": 50,
        "grape_max_iter": 500,
    }
    
    sigma = params['gate_time'] / 8.0
    amplitude_pi = np.pi / (sigma * np.sqrt(2 * np.pi))
    
    print("Parameters:")
    print(f"  Gate time: {params['gate_time']} ns")
    print(f"  \u03c3 (width): {sigma:.2f} ns")
    print(f"  \u03c0 pulse amplitude: {amplitude_pi:.4f} rad/ns")
    print(f"  Anharmonicity: {params['anharmonicity']} MHz")
    print()
    
    # Define target gates
    S_gate = qt.Qobj([[1, 0], [0, 1j]])
    T_gate = qt.Qobj([[1, 0], [0, np.exp(1j * np.pi / 4)]])
    
    gates = {
        "X": qt.sigmax(),
        "Y": qt.sigmay(),
        "H": qt.gates.hadamard_transform(),
        "S": S_gate,
        "T": T_gate,
    }
    
    # Gaussian/DRAG baselines only meaningful for X/Y axis rotations
    pulse_baseline_gates = {"X", "Y"}
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "parameters": params,
        "calibration": {"sigma_ns": sigma, "pi_amplitude_radns": float(amplitude_pi)},
        "gates": {},
    }
    
    all_data = []
    
    for gate_type, U_target in gates.items():
        print(f"\n{'='*40}")
        print(f"Optimizing {gate_type} gate...")
        print("=" * 40)
        
        gate_results = {}
        has_baseline = gate_type in pulse_baseline_gates
        
        if has_baseline:
            print("  [1/3] Gaussian baseline...", end=" ", flush=True)
            gaussian_result = run_gaussian_baseline(gate_type, U_target, params)
            gate_results["gaussian"] = gaussian_result
            print(f"F = {gaussian_result['fidelity']:.6f}")
            
            print("  [2/3] DRAG optimized...", end=" ", flush=True)
            drag_result = run_drag_optimized(gate_type, U_target, params)
            gate_results["drag"] = drag_result
            print(f"F = {drag_result['fidelity']:.6f} (\u03b2={drag_result['beta']:.4f})")
        else:
            print(f"  [--] Gaussian/DRAG: N/A for {gate_type} gate (not an X/Y rotation)")
        
        step = "3/3" if has_baseline else "1/1"
        print(f"  [{step}] GRAPE optimization...", end=" ", flush=True)
        grape_result = run_grape_optimized(gate_type, U_target, params)
        gate_results["grape"] = grape_result
        print(f"F = {grape_result['fidelity']:.6f} ({grape_result['iterations']} iter)")
        
        results["gates"][gate_type] = gate_results
        
        all_data.append({
            "gate": gate_type,
            "gaussian_fidelity": gaussian_result['fidelity'] if has_baseline else None,
            "drag_fidelity": drag_result['fidelity'] if has_baseline else None,
            "grape_fidelity": grape_result['fidelity'],
        })
    
    # Print summary table
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"{'Gate':<6} {'Gaussian':<14} {'DRAG':<14} {'GRAPE':<14}")
    print("-" * 48)
    for row in all_data:
        g_f = f"{row['gaussian_fidelity']:.6f}" if row['gaussian_fidelity'] is not None else "N/A"
        d_f = f"{row['drag_fidelity']:.6f}" if row['drag_fidelity'] is not None else "N/A"
        p_f = f"{row['grape_fidelity']:.6f}"
        print(f"{row['gate']:<6} {g_f:<14} {d_f:<14} {p_f:<14}")
    
    # Averages (only over gates that have baselines for Gaussian/DRAG)
    baseline_rows = [r for r in all_data if r['gaussian_fidelity'] is not None]
    if baseline_rows:
        avg_gaussian = np.mean([r['gaussian_fidelity'] for r in baseline_rows])
        avg_drag = np.mean([r['drag_fidelity'] for r in baseline_rows])
    else:
        avg_gaussian = avg_drag = float('nan')
    avg_grape = np.mean([r['grape_fidelity'] for r in all_data])
    
    print("-" * 48)
    print(f"{'Avg':<6} {avg_gaussian:<14.6f} {avg_drag:<14.6f} {avg_grape:<14.6f}")
    
    results["summary"] = {
        "avg_gaussian_fidelity": float(avg_gaussian),
        "avg_drag_fidelity": float(avg_drag),
        "avg_grape_fidelity": float(avg_grape),
        "n_gates": len(all_data),
        "n_baseline_gates": len(baseline_rows),
    }
    
    results_dir = create_results_dir()
    output_file = results_dir / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    return results


if __name__ == "__main__":
    run_benchmark()
