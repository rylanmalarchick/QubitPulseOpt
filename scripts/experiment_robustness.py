#!/usr/bin/env python3
"""
Experiment 4: Robustness Analysis
==================================

Tests robustness of Gaussian, DRAG, and GRAPE pulses against:
1. Detuning errors: +/-5 MHz frequency offset
2. Amplitude errors: +/-5% systematic calibration error

Uses the RobustnessTester from src/optimization/robustness.py.

Both 2-level and 3-level transmon systems are tested.

For the paper, the key question: does GRAPE's superior fidelity come at the
cost of reduced robustness? (Answer: typically no for moderate errors.)

References:
-----------
[1] Khaneja et al., J. Magn. Reson. 172, 296 (2005) -- GRAPE
[2] Motzoi et al., PRL 103, 110501 (2009) -- DRAG robustness
[3] Egger & Wilhelm, Phys. Rev. Applied 11, 014017 (2019) -- robust control
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
from src.optimization.robustness import RobustnessTester, RobustnessResult


# -- Parameters ------------------------------------------------------------

PARAMS = {
    "gate_time_ns": 20.0,
    "anharmonicity_mhz": -200.0,
    "n_slices": 50,              # Common resolution for all methods
    "grape_max_iter": 500,
    # Sweep ranges
    "detuning_mhz_range": 5.0,  # +/- 5 MHz
    "detuning_n_points": 41,
    "amplitude_error_range": 0.05,  # +/- 5%
    "amplitude_n_points": 41,
}


# -- Pulse Generation (piecewise-constant at n_slices resolution) -----------


def resample_to_piecewise_constant(pulse_I, pulse_Q, times, n_slices, gate_time):
    """
    Resample a dense pulse to piecewise-constant at GRAPE resolution.

    Takes the average of each timeslice interval.
    """
    dt_slice = gate_time / n_slices
    resampled_I = np.zeros(n_slices)
    resampled_Q = np.zeros(n_slices)

    for k in range(n_slices):
        t_start = k * dt_slice
        t_end = (k + 1) * dt_slice
        mask = (times >= t_start) & (times < t_end)
        if np.any(mask):
            resampled_I[k] = np.mean(pulse_I[mask])
            resampled_Q[k] = np.mean(pulse_Q[mask])
        else:
            # Interpolate at midpoint
            t_mid = (t_start + t_end) / 2.0
            idx = np.searchsorted(times, t_mid)
            idx = min(idx, len(times) - 1)
            resampled_I[k] = pulse_I[idx]
            resampled_Q[k] = pulse_Q[idx]

    return resampled_I, resampled_Q


def generate_all_2level_pulses(params):
    """Generate Gaussian, DRAG, and GRAPE pulses for 2-level X-gate."""
    gate_time = params["gate_time_ns"]
    n_slices = params["n_slices"]
    sigma = gate_time / 8.0
    amplitude = np.pi / (sigma * np.sqrt(2 * np.pi))
    n_dense = 400

    # Dense time array for Gaussian/DRAG
    times_dense = np.linspace(0, gate_time, n_dense)
    t_center = gate_time / 2.0

    # --- Gaussian ---
    gauss_I_dense = amplitude * np.exp(-0.5 * ((times_dense - t_center) / sigma) ** 2)
    gauss_Q_dense = np.zeros_like(times_dense)
    gauss_I, gauss_Q = resample_to_piecewise_constant(
        gauss_I_dense, gauss_Q_dense, times_dense, n_slices, gate_time
    )

    anharmonicity_radns = 2.0 * np.pi * params["anharmonicity_mhz"] * 1e-3
    optimal_beta = -1.0 / (2.0 * anharmonicity_radns)

    drag_params = DRAGParameters(
        amplitude=amplitude, sigma=sigma, beta=optimal_beta,
        anharmonicity=params["anharmonicity_mhz"],
    )
    drag = DRAGPulse(drag_params)
    drag_I_dense, drag_Q_dense = drag.envelope(times_dense, t_center)
    drag_I, drag_Q = resample_to_piecewise_constant(
        drag_I_dense, drag_Q_dense, times_dense, n_slices, gate_time
    )

    # --- GRAPE ---
    H_drift = 0.0 * qt.sigmaz()
    H_controls = [0.5 * qt.sigmax(), 0.5 * qt.sigmay()]

    t_slices = np.linspace(0, gate_time, n_slices)
    gaussian_seed = amplitude * np.exp(-0.5 * ((t_slices - t_center) / sigma) ** 2)

    u_init = np.zeros((2, n_slices))
    u_init[0, :] = gaussian_seed

    optimizer = GRAPEOptimizer(
        H_drift=H_drift, H_controls=H_controls,
        n_timeslices=n_slices, total_time=gate_time,
        u_limits=(-amplitude * 3.0, amplitude * 3.0),
        convergence_threshold=1e-10,
        max_iterations=params["grape_max_iter"],
        learning_rate=0.1, verbose=False,
    )
    result = optimizer.optimize_unitary(qt.sigmax(), u_init=u_init)
    grape_I = result.optimized_pulses[0, :]
    grape_Q = result.optimized_pulses[1, :]

    pulses = {
        "gaussian": np.stack([gauss_I, gauss_Q]),   # (2, n_slices)
        "drag": np.stack([drag_I, drag_Q]),
        "grape": np.stack([grape_I, grape_Q]),
    }

    return pulses, H_drift, H_controls


def generate_all_3level_pulses(params):
    """Generate Gaussian, DRAG, and GRAPE pulses for 3-level X-gate."""
    n_levels = 3
    gate_time = params["gate_time_ns"]
    n_slices = params["n_slices"]
    sigma = gate_time / 8.0
    amplitude = np.pi / (sigma * np.sqrt(2 * np.pi))
    n_dense = 400

    times_dense = np.linspace(0, gate_time, n_dense)
    t_center = gate_time / 2.0

    # 3-level Hamiltonians
    alpha_radns = 2.0 * np.pi * params["anharmonicity_mhz"] * 1e-3
    n_op = qt.num(n_levels)
    H_drift = (alpha_radns / 2.0) * (n_op * n_op - n_op)

    a = qt.destroy(n_levels)
    a_dag = qt.create(n_levels)
    H_x = 0.5 * (a + a_dag)
    H_y = 0.5 * 1j * (a_dag - a)
    H_controls = [H_x, H_y]

    # Target
    U_target_arr = np.eye(n_levels, dtype=complex)
    U_target_arr[0:2, 0:2] = qt.sigmax().full()
    U_target = qt.Qobj(U_target_arr, dims=[[n_levels], [n_levels]])

    # --- Gaussian ---
    gauss_I_dense = amplitude * np.exp(-0.5 * ((times_dense - t_center) / sigma) ** 2)
    gauss_Q_dense = np.zeros_like(times_dense)
    gauss_I, gauss_Q = resample_to_piecewise_constant(
        gauss_I_dense, gauss_Q_dense, times_dense, n_slices, gate_time
    )

    anharmonicity_radns = 2.0 * np.pi * params["anharmonicity_mhz"] * 1e-3
    optimal_beta = -1.0 / (2.0 * anharmonicity_radns)

    drag_params = DRAGParameters(
        amplitude=amplitude, sigma=sigma, beta=optimal_beta,
        anharmonicity=params["anharmonicity_mhz"],
    )
    drag = DRAGPulse(drag_params)
    drag_I_dense, drag_Q_dense = drag.envelope(times_dense, t_center)
    drag_I, drag_Q = resample_to_piecewise_constant(
        drag_I_dense, drag_Q_dense, times_dense, n_slices, gate_time
    )

    # --- GRAPE (3-level) ---
    t_slices = np.linspace(0, gate_time, n_slices)
    gaussian_seed = amplitude * np.exp(-0.5 * ((t_slices - t_center) / sigma) ** 2)

    u_init = np.zeros((2, n_slices))
    u_init[0, :] = gaussian_seed
    rng = np.random.default_rng(42)
    u_init[1, :] = rng.normal(0, amplitude * 0.05, n_slices)

    optimizer = GRAPEOptimizer(
        H_drift=H_drift, H_controls=H_controls,
        n_timeslices=n_slices, total_time=gate_time,
        u_limits=(-amplitude * 3.0, amplitude * 3.0),
        convergence_threshold=1e-10,
        max_iterations=1000,
        learning_rate=0.5, verbose=False,
        use_line_search=True, momentum=0.5,
    )
    result = optimizer.optimize_unitary(U_target, u_init=u_init, step_decay=1.0)
    grape_I = result.optimized_pulses[0, :]
    grape_Q = result.optimized_pulses[1, :]

    pulses = {
        "gaussian": np.stack([gauss_I, gauss_Q]),
        "drag": np.stack([drag_I, drag_Q]),
        "grape": np.stack([grape_I, grape_Q]),
    }

    return pulses, H_drift, H_controls, U_target


# -- Robustness Sweeps (manual, to handle 3-level properly) ----------------


def piecewise_constant_propagate(pulse_amps, H_drift, H_controls, gate_time):
    """
    Propagate piecewise-constant pulse to get unitary.

    Parameters
    ----------
    pulse_amps : np.ndarray, shape (n_controls, n_slices)
    """
    n_slices = pulse_amps.shape[1]
    dim = H_drift.shape[0]
    dt = gate_time / n_slices
    U = qt.qeye(dim)

    for k in range(n_slices):
        H_t = H_drift
        for j in range(len(H_controls)):
            H_t = H_t + pulse_amps[j, k] * H_controls[j]
        U = (-1j * H_t * dt).expm() * U

    return U


def gate_fidelity_unitary(U_achieved, U_target):
    """Compute gate fidelity |Tr(U_target^dag U)|^2 / d^2."""
    d = U_target.shape[0]
    overlap = np.abs((U_target.dag() * U_achieved).tr())
    return float((overlap / d) ** 2)


def avg_state_fidelity_3level(U_achieved, U_target, n_levels):
    """Average fidelity over |0> and |1> for 3-level system."""
    total_fid = 0.0
    for k in range(2):
        psi0 = qt.basis(n_levels, k)
        psi_final = U_achieved * psi0
        psi_target = U_target * psi0
        fid = float(qt.fidelity(psi_target, psi_final) ** 2)
        total_fid += fid
    return total_fid / 2.0


def sweep_detuning(pulse_amps, H_drift, H_controls, U_target, gate_time,
                   detuning_values_radns, n_levels):
    """
    Sweep frequency detuning and compute fidelity.

    Adds delta * (sigma_z / 2) for 2-level, or delta * n_hat for 3-level.
    """
    fidelities = []

    for delta in detuning_values_radns:
        if n_levels == 2:
            H_detuned = H_drift + 0.5 * delta * qt.sigmaz()
        else:
            # In rotating frame, detuning shifts 0->1 transition
            # H_det = delta * |1><1| + 2*delta * |2><2| (approx)
            # More precisely: delta * n_hat (number operator)
            H_detuned = H_drift + delta * qt.num(n_levels)

        U = piecewise_constant_propagate(pulse_amps, H_detuned, H_controls, gate_time)

        if n_levels == 2:
            fid = gate_fidelity_unitary(U, U_target)
        else:
            fid = avg_state_fidelity_3level(U, U_target, n_levels)

        fidelities.append(fid)

    return np.array(fidelities)


def sweep_amplitude_error(pulse_amps, H_drift, H_controls, U_target, gate_time,
                           error_values, n_levels):
    """
    Sweep amplitude scaling error and compute fidelity.

    u -> u * (1 + epsilon)
    """
    fidelities = []

    for eps in error_values:
        scaled_amps = pulse_amps * (1.0 + eps)
        U = piecewise_constant_propagate(scaled_amps, H_drift, H_controls, gate_time)

        if n_levels == 2:
            fid = gate_fidelity_unitary(U, U_target)
        else:
            fid = avg_state_fidelity_3level(U, U_target, n_levels)

        fidelities.append(fid)

    return np.array(fidelities)


# -- Main Experiment -------------------------------------------------------


def run_sweeps(pulses, H_drift, H_controls, U_target, gate_time, n_levels,
               detuning_values_radns, amplitude_errors, label):
    """Run detuning and amplitude sweeps for all methods."""
    results = {}

    for method_name, pulse_amps in pulses.items():
        print(f"    {method_name}...", end=" ", flush=True)
        start = perf_counter()

        det_fids = sweep_detuning(
            pulse_amps, H_drift, H_controls, U_target, gate_time,
            detuning_values_radns, n_levels,
        )
        amp_fids = sweep_amplitude_error(
            pulse_amps, H_drift, H_controls, U_target, gate_time,
            amplitude_errors, n_levels,
        )

        elapsed = perf_counter() - start
        nominal_det = det_fids[len(det_fids) // 2]
        nominal_amp = amp_fids[len(amp_fids) // 2]
        min_det = float(np.min(det_fids))
        min_amp = float(np.min(amp_fids))

        print(f"det: nom={nominal_det:.6f} min={min_det:.6f}  "
              f"amp: nom={nominal_amp:.6f} min={min_amp:.6f}  ({elapsed:.1f}s)")

        results[method_name] = {
            "detuning_fidelities": det_fids.tolist(),
            "detuning_nominal": float(nominal_det),
            "detuning_min": min_det,
            "detuning_mean": float(np.mean(det_fids)),
            "amplitude_fidelities": amp_fids.tolist(),
            "amplitude_nominal": float(nominal_amp),
            "amplitude_min": min_amp,
            "amplitude_mean": float(np.mean(amp_fids)),
        }

    return results


def run_experiment():
    """Run the full robustness analysis."""
    print("=" * 70)
    print("Experiment 4: Robustness Analysis")
    print("=" * 70)
    print("Testing pulse robustness against detuning and amplitude errors.")
    print()

    gate_time = PARAMS["gate_time_ns"]
    n_slices = PARAMS["n_slices"]

    # Detuning values: +/- 5 MHz in rad/ns
    det_mhz = np.linspace(
        -PARAMS["detuning_mhz_range"],
        PARAMS["detuning_mhz_range"],
        PARAMS["detuning_n_points"],
    )
    det_radns = 2.0 * np.pi * det_mhz * 1e-3  # MHz -> rad/ns

    # Amplitude errors: +/- 5%
    amp_errors = np.linspace(
        -PARAMS["amplitude_error_range"],
        PARAMS["amplitude_error_range"],
        PARAMS["amplitude_n_points"],
    )

    print(f"Parameters:")
    print(f"  Gate time: {gate_time} ns, n_slices: {n_slices}")
    print(f"  Detuning range: +/-{PARAMS['detuning_mhz_range']} MHz "
          f"({PARAMS['detuning_n_points']} points)")
    print(f"  Amplitude range: +/-{PARAMS['amplitude_error_range']*100:.0f}% "
          f"({PARAMS['amplitude_n_points']} points)")
    print()

    start_total = perf_counter()

    # ===== 2-Level =====
    print("--- 2-Level System ---")
    print("  Generating pulses...", end=" ", flush=True)
    pulses_2l, H_drift_2l, H_controls_2l = generate_all_2level_pulses(PARAMS)
    U_target_2l = qt.sigmax()
    print("done")

    print("  Running sweeps:")
    results_2level = run_sweeps(
        pulses_2l, H_drift_2l, H_controls_2l, U_target_2l, gate_time,
        n_levels=2, detuning_values_radns=det_radns,
        amplitude_errors=amp_errors, label="2-level",
    )

    # ===== 3-Level =====
    print("\n--- 3-Level Transmon ---")
    print("  Generating pulses (GRAPE may take a moment)...", end=" ", flush=True)
    pulses_3l, H_drift_3l, H_controls_3l, U_target_3l = generate_all_3level_pulses(PARAMS)
    print("done")

    print("  Running sweeps:")
    results_3level = run_sweeps(
        pulses_3l, H_drift_3l, H_controls_3l, U_target_3l, gate_time,
        n_levels=3, detuning_values_radns=det_radns,
        amplitude_errors=amp_errors, label="3-level",
    )

    elapsed_total = perf_counter() - start_total

    # ===== Summary =====
    print("\n" + "=" * 80)
    print("ROBUSTNESS SUMMARY")
    print("=" * 80)

    for system_label, results in [("2-Level", results_2level), ("3-Level", results_3level)]:
        print(f"\n  {system_label} System:")
        print(f"  {'Method':<12} {'Det Nominal':>12} {'Det Min':>12} {'Det Mean':>12}"
              f"  {'Amp Nominal':>12} {'Amp Min':>12} {'Amp Mean':>12}")
        print(f"  {'-'*84}")
        for method, data in results.items():
            print(f"  {method:<12} {data['detuning_nominal']:>12.6f} "
                  f"{data['detuning_min']:>12.6f} {data['detuning_mean']:>12.6f}  "
                  f"{data['amplitude_nominal']:>12.6f} "
                  f"{data['amplitude_min']:>12.6f} {data['amplitude_mean']:>12.6f}")

    print(f"\nTotal time: {elapsed_total:.1f}s")

    # ===== Save =====
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "experiment": "robustness_analysis",
        "parameters": PARAMS,
        "detuning_mhz": det_mhz.tolist(),
        "detuning_radns": det_radns.tolist(),
        "amplitude_errors": amp_errors.tolist(),
        "results_2level": results_2level,
        "results_3level": results_3level,
        "total_time_s": elapsed_total,
    }

    results_dir = Path(__file__).parent.parent / "results" / "robustness"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_file = results_dir / f"robustness_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")
    return all_results


if __name__ == "__main__":
    run_experiment()
