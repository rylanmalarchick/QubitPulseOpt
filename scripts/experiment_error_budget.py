#!/usr/bin/env python3
"""
Experiment 3: Error Budget Analysis
====================================

Decomposes gate error by source for Gaussian, DRAG, and GRAPE pulses:

1. Coherent error    — closed-system (unitary) infidelity
2. T1 error          — amplitude damping contribution
3. T2 error          — pure dephasing contribution
4. Combined T1 + T2  — total decoherence error
5. Control noise     — amplitude noise ±1-5%, averaged over realizations

Uses IQM Garnet parameters: T1 = 37 us, T2 = 9.6 us, gate_time = 20 ns.

All Hamiltonians are in rad/ns; time in ns.
T1/T2 must therefore be in ns: T1 = 37000 ns, T2 = 9600 ns.

The Lindblad master equation:
    drho/dt = -i[H, rho] + Sum_k (L_k rho L_k^dag - 1/2 {L_k^dag L_k, rho})

For a 20 ns gate with T1 = 37 us, the decoherence-limited infidelity is:
    ~t_gate/(2*T1) ~ 20/(2*37000) ~ 2.7e-4

References:
-----------
[1] Krantz et al., Appl. Phys. Rev. 6, 021318 (2019) -- SC qubit review
[2] Motzoi et al., PRL 103, 110501 (2009) -- DRAG
[3] Khaneja et al., J. Magn. Reson. 172, 296 (2005) -- GRAPE
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from time import perf_counter

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import qutip as qt
from scipy.interpolate import interp1d

from src.pulses.drag import DRAGPulse, DRAGParameters
from src.optimization.grape import GRAPEOptimizer
from src.hamiltonian.lindblad import DecoherenceParams, LindbladEvolution


# -- IQM Garnet Parameters ------------------------------------------------

IQM_PARAMS = {
    "gate_time_ns": 20.0,
    "T1_us": 37.0,
    "T2_us": 9.6,
    "anharmonicity_mhz": -200.0,
    "n_samples": 400,       # Pulse time points
    "n_grape_slices": 50,   # GRAPE timeslices
    "grape_max_iter": 500,
}

# Convert to ns (Hamiltonian units)
T1_NS = IQM_PARAMS["T1_us"] * 1e3   # 37000 ns
T2_NS = IQM_PARAMS["T2_us"] * 1e3   # 9600 ns


# -- Pulse Generation -----------------------------------------------------


def generate_gaussian_pulse(params: dict) -> tuple:
    """
    Generate calibrated Gaussian pulse (beta=0) for X-gate.

    Returns (pulse_I, pulse_Q, times) -- all in rad/ns units.
    """
    gate_time = params["gate_time_ns"]
    n_points = params["n_samples"]
    sigma = gate_time / 8.0
    amplitude = np.pi / (sigma * np.sqrt(2 * np.pi))

    times = np.linspace(0, gate_time, n_points)
    t_center = gate_time / 2.0
    pulse_I = amplitude * np.exp(-0.5 * ((times - t_center) / sigma) ** 2)
    pulse_Q = np.zeros_like(times)

    return pulse_I, pulse_Q, times


def generate_drag_pulse(params: dict) -> tuple:
    """
    Generate DRAG pulse with unit-corrected beta for X-gate.

    Returns (pulse_I, pulse_Q, times) -- all in rad/ns units.
    """
    gate_time = params["gate_time_ns"]
    n_points = params["n_samples"]
    sigma = gate_time / 8.0
    amplitude = np.pi / (sigma * np.sqrt(2 * np.pi))

    # DRAG beta = -1/(2*alpha), Motzoi et al. PRL 103, 110501 (2009)
    anharmonicity_radns = 2.0 * np.pi * params["anharmonicity_mhz"] * 1e-3
    optimal_beta = -1.0 / (2.0 * anharmonicity_radns)

    drag_params = DRAGParameters(
        amplitude=amplitude,
        sigma=sigma,
        beta=optimal_beta,
        anharmonicity=params["anharmonicity_mhz"],
    )
    drag = DRAGPulse(drag_params)

    times = np.linspace(0, gate_time, n_points)
    t_center = gate_time / 2.0
    pulse_I, pulse_Q = drag.envelope(times, t_center)

    return pulse_I, pulse_Q, times


def generate_grape_pulse(params: dict) -> tuple:
    """
    Run GRAPE optimization for X-gate and return optimized pulse arrays.

    Returns (pulse_I, pulse_Q, times, grape_result).
    pulse_I/Q are piecewise-constant arrays of length n_grape_slices.
    times is the array of timeslice centers.
    """
    gate_time = params["gate_time_ns"]
    n_slices = params["n_grape_slices"]
    sigma = gate_time / 8.0
    amplitude = np.pi / (sigma * np.sqrt(2 * np.pi))

    # 2-level Hamiltonian: H = (Omega_I/2)sigma_x + (Omega_Q/2)sigma_y
    H_drift = 0.0 * qt.sigmaz()
    H_controls = [0.5 * qt.sigmax(), 0.5 * qt.sigmay()]

    # Gaussian seed on I-channel
    t_slices = np.linspace(0, gate_time, n_slices)
    t_center = gate_time / 2.0
    gaussian = amplitude * np.exp(-0.5 * ((t_slices - t_center) / sigma) ** 2)

    u_init = np.zeros((2, n_slices))
    u_init[0, :] = gaussian

    optimizer = GRAPEOptimizer(
        H_drift=H_drift,
        H_controls=H_controls,
        n_timeslices=n_slices,
        total_time=gate_time,
        u_limits=(-amplitude * 3.0, amplitude * 3.0),
        convergence_threshold=1e-10,
        max_iterations=params["grape_max_iter"],
        learning_rate=0.1,
        verbose=False,
    )

    U_target = qt.sigmax()
    result = optimizer.optimize_unitary(U_target, u_init=u_init)

    pulse_I = result.optimized_pulses[0, :]
    pulse_Q = result.optimized_pulses[1, :]

    return pulse_I, pulse_Q, t_slices, result


# -- Fidelity Computation -------------------------------------------------


def compute_unitary_fidelity(pulse_I, pulse_Q, times, U_target):
    """
    Compute closed-system (unitary) gate fidelity via piecewise propagation.

    Returns fidelity in [0, 1].
    """
    dt = times[1] - times[0]
    U = qt.qeye(2)
    sigma_x = qt.sigmax()
    sigma_y = qt.sigmay()

    for i in range(len(times) - 1):
        H_t = 0.5 * pulse_I[i] * sigma_x + 0.5 * pulse_Q[i] * sigma_y
        U = (-1j * H_t * dt).expm() * U

    d = U_target.shape[0]
    overlap = np.abs((U.dag() * U_target).tr())
    return float((overlap / d) ** 2)


def _make_pulse_interp(pulse_arr, times):
    """
    Create a callable pulse function f(t, args) for QuTiP mesolve.

    Uses piecewise-linear interpolation with fill_value=0 outside bounds.
    """
    interp_func = interp1d(
        times, pulse_arr,
        kind='linear', bounds_error=False, fill_value=0.0,
    )

    def pulse_func(t, args=None):
        return float(interp_func(t))

    return pulse_func


def _make_piecewise_constant_interp(pulse_arr, n_slices, gate_time):
    """
    Create a piecewise-constant callable for GRAPE pulses.

    GRAPE uses n_slices intervals of width dt = gate_time / n_slices.
    """
    dt = gate_time / n_slices

    def pulse_func(t, args=None):
        idx = int(t / dt)
        idx = max(0, min(idx, n_slices - 1))
        return float(pulse_arr[idx])

    return pulse_func


def build_td_hamiltonian(pulse_I, pulse_Q, times, is_grape=False,
                         n_slices=None, gate_time=None):
    """
    Build time-dependent Hamiltonian for QuTiP mesolve.

    Format: [H0, [H_x, f_I(t)], [H_y, f_Q(t)]]

    For Gaussian/DRAG: uses interpolation over the dense time array.
    For GRAPE: uses piecewise-constant lookup.
    """
    H0 = 0.0 * qt.sigmaz()  # No detuning (resonant drive)
    H_x = 0.5 * qt.sigmax()
    H_y = 0.5 * qt.sigmay()

    if is_grape:
        f_I = _make_piecewise_constant_interp(pulse_I, n_slices, gate_time)
        f_Q = _make_piecewise_constant_interp(pulse_Q, n_slices, gate_time)
    else:
        f_I = _make_pulse_interp(pulse_I, times)
        f_Q = _make_pulse_interp(pulse_Q, times)

    return [H0, [H_x, f_I], [H_y, f_Q]]


def compute_lindblad_fidelity(H_td, U_target, gate_time, decoherence_params,
                               n_steps=2000):
    """
    Compute gate fidelity under Lindblad decoherence.

    Averages over computational basis states |0> and |1> (process fidelity
    approximation for single qubit).

    Parameters
    ----------
    H_td : list
        Time-dependent Hamiltonian in QuTiP format.
    U_target : qt.Qobj
        Target unitary (2x2).
    gate_time : float
        Gate duration in ns.
    decoherence_params : DecoherenceParams
        T1/T2 parameters (in ns).
    n_steps : int
        Number of time steps for mesolve.

    Returns
    -------
    float
        Average state fidelity.
    """
    lindblad = LindbladEvolution(H_td, decoherence_params)
    times_eval = np.linspace(0, gate_time, n_steps)

    total_fidelity = 0.0
    for k in range(2):
        psi0 = qt.basis(2, k)
        rho0 = psi0 * psi0.dag()

        result = lindblad.evolve(rho0, times_eval)
        rho_final = result.states[-1]

        rho_ideal = U_target * rho0 * U_target.dag()
        fid = qt.fidelity(rho_final, rho_ideal) ** 2
        total_fidelity += fid

    return float(total_fidelity / 2.0)


def compute_noisy_fidelity(pulse_I, pulse_Q, times, U_target, noise_level,
                            n_realizations=100, is_grape=False, n_slices=None,
                            gate_time=None, seed=0):
    """
    Compute fidelity with multiplicative amplitude noise (closed system).

    Noise model: Omega(t) -> Omega(t) * (1 + epsilon), epsilon ~ N(0, noise_level^2).
    Each realization draws a different epsilon (static noise per shot).

    Parameters
    ----------
    noise_level : float
        Fractional noise, e.g. 0.01 for 1%.
    n_realizations : int
        Number of noise realizations to average over.

    Returns
    -------
    mean_fidelity, std_fidelity : float, float
    """
    rng = np.random.default_rng(seed)
    fidelities = []

    for _ in range(n_realizations):
        epsilon = rng.normal(0, noise_level)
        noisy_I = pulse_I * (1.0 + epsilon)
        noisy_Q = pulse_Q * (1.0 + epsilon)

        if is_grape:
            dt = gate_time / n_slices
            U = qt.qeye(2)
            sx, sy = qt.sigmax(), qt.sigmay()
            for k in range(n_slices):
                H_t = 0.5 * noisy_I[k] * sx + 0.5 * noisy_Q[k] * sy
                U = (-1j * H_t * dt).expm() * U
        else:
            dt = times[1] - times[0]
            U = qt.qeye(2)
            sx, sy = qt.sigmax(), qt.sigmay()
            for i in range(len(times) - 1):
                H_t = 0.5 * noisy_I[i] * sx + 0.5 * noisy_Q[i] * sy
                U = (-1j * H_t * dt).expm() * U

        d = U_target.shape[0]
        overlap = np.abs((U.dag() * U_target).tr())
        fid = (overlap / d) ** 2
        fidelities.append(float(fid))

    return float(np.mean(fidelities)), float(np.std(fidelities))


# -- Error Budget Decomposition -------------------------------------------


def compute_error_budget(method_name, pulse_I, pulse_Q, times, U_target,
                          params, is_grape=False, n_slices=None):
    """
    Full error budget decomposition for one pulse method.

    Returns dict with infidelities from each error source.
    """
    gate_time = params["gate_time_ns"]
    print(f"\n  {method_name} error budget:")

    # 1. Coherent error (closed system)
    print("    [1/5] Coherent error...", end=" ", flush=True)
    if is_grape:
        # GRAPE: use piecewise-constant propagation matching GRAPE internals
        dt = gate_time / n_slices
        U = qt.qeye(2)
        sx, sy = qt.sigmax(), qt.sigmay()
        for k in range(n_slices):
            H_t = 0.5 * pulse_I[k] * sx + 0.5 * pulse_Q[k] * sy
            U = (-1j * H_t * dt).expm() * U
        d = U_target.shape[0]
        overlap = np.abs((U.dag() * U_target).tr())
        coherent_fidelity = float((overlap / d) ** 2)
    else:
        coherent_fidelity = compute_unitary_fidelity(pulse_I, pulse_Q, times, U_target)
    coherent_infidelity = 1.0 - coherent_fidelity
    print(f"infidelity = {coherent_infidelity:.2e}")

    # Build time-dependent Hamiltonian for Lindblad simulations
    H_td = build_td_hamiltonian(
        pulse_I, pulse_Q, times,
        is_grape=is_grape, n_slices=n_slices, gate_time=gate_time,
    )

    # 2. T1 error only (set T2 = 2*T1 to isolate T1 contribution)
    print("    [2/5] T1 error...", end=" ", flush=True)
    decoherence_t1_only = DecoherenceParams(T1=T1_NS, T2=2.0 * T1_NS)
    fid_t1 = compute_lindblad_fidelity(H_td, U_target, gate_time, decoherence_t1_only)
    t1_infidelity = 1.0 - fid_t1
    print(f"infidelity = {t1_infidelity:.2e}")

    # 3. T2 error only (set T1 very large, use actual T2)
    # Constraint: T2 <= 2*T1, so T1 >= T2/2 = 4800 ns
    # Use T1 = 1e8 ns (effectively infinite)
    print("    [3/5] T2 (dephasing) error...", end=" ", flush=True)
    T1_large = 1e8  # effectively infinite
    decoherence_t2_only = DecoherenceParams(T1=T1_large, T2=T2_NS)
    fid_t2 = compute_lindblad_fidelity(H_td, U_target, gate_time, decoherence_t2_only)
    t2_infidelity = 1.0 - fid_t2
    print(f"infidelity = {t2_infidelity:.2e}")

    # 4. Combined T1 + T2
    print("    [4/5] Combined T1+T2...", end=" ", flush=True)
    decoherence_full = DecoherenceParams(T1=T1_NS, T2=T2_NS)
    fid_full = compute_lindblad_fidelity(H_td, U_target, gate_time, decoherence_full)
    full_infidelity = 1.0 - fid_full
    print(f"infidelity = {full_infidelity:.2e}")

    # 5. Control noise (amplitude noise, closed system)
    noise_levels = [0.01, 0.02, 0.05]  # 1%, 2%, 5%
    noise_results = {}

    for nl in noise_levels:
        print(f"    [5/5] Control noise eps={nl*100:.0f}%...", end=" ", flush=True)
        mean_fid, std_fid = compute_noisy_fidelity(
            pulse_I, pulse_Q, times, U_target, nl,
            n_realizations=100, is_grape=is_grape,
            n_slices=n_slices, gate_time=gate_time,
        )
        noise_infidelity = 1.0 - mean_fid
        print(f"infidelity = {noise_infidelity:.2e} +/- {std_fid:.2e}")
        noise_results[f"{nl*100:.0f}pct"] = {
            "noise_level": nl,
            "mean_fidelity": mean_fid,
            "std_fidelity": std_fid,
            "mean_infidelity": float(noise_infidelity),
        }

    return {
        "method": method_name,
        "coherent_fidelity": coherent_fidelity,
        "coherent_infidelity": coherent_infidelity,
        "t1_only_fidelity": fid_t1,
        "t1_only_infidelity": t1_infidelity,
        "t2_only_fidelity": fid_t2,
        "t2_only_infidelity": t2_infidelity,
        "full_decoherence_fidelity": fid_full,
        "full_decoherence_infidelity": full_infidelity,
        "control_noise": noise_results,
    }


# -- Main Experiment -------------------------------------------------------


def run_experiment():
    """Run the full error budget analysis."""
    print("=" * 70)
    print("Experiment 3: Error Budget Analysis")
    print("=" * 70)
    print("Decomposing gate error by source: coherent, T1, T2, combined, noise")
    print()

    print("IQM Garnet Parameters:")
    print(f"  Gate time:        {IQM_PARAMS['gate_time_ns']} ns")
    print(f"  T1:               {IQM_PARAMS['T1_us']} us = {T1_NS:.0f} ns")
    print(f"  T2:               {IQM_PARAMS['T2_us']} us = {T2_NS:.0f} ns")
    print(f"  Anharmonicity:    {IQM_PARAMS['anharmonicity_mhz']} MHz")
    print(f"  t_gate / T1:      {IQM_PARAMS['gate_time_ns'] / T1_NS:.2e}")
    print(f"  t_gate / T2:      {IQM_PARAMS['gate_time_ns'] / T2_NS:.2e}")
    print(f"  Decoherence limit: ~{IQM_PARAMS['gate_time_ns'] / T1_NS / 2:.2e} "
          f"(t_gate / 2T1)")
    print()

    U_target = qt.sigmax()  # X-gate

    # -- Generate pulses --
    start_total = perf_counter()

    print("Generating pulses:")
    print("  Gaussian...", end=" ", flush=True)
    gauss_I, gauss_Q, gauss_times = generate_gaussian_pulse(IQM_PARAMS)
    print(f"done (amp={np.max(gauss_I):.4f} rad/ns)")

    print("  DRAG...", end=" ", flush=True)
    drag_I, drag_Q, drag_times = generate_drag_pulse(IQM_PARAMS)
    sigma = IQM_PARAMS["gate_time_ns"] / 8.0
    amplitude = np.pi / (sigma * np.sqrt(2 * np.pi))
    anharmonicity_radns = 2.0 * np.pi * IQM_PARAMS["anharmonicity_mhz"] * 1e-3
    beta_val = -1.0 / (2.0 * anharmonicity_radns)
    print(f"done (amp={np.max(drag_I):.4f} rad/ns, beta={beta_val:.3f})")

    print("  GRAPE...", end=" ", flush=True)
    grape_I, grape_Q, grape_times, grape_result = generate_grape_pulse(IQM_PARAMS)
    print(f"done (F={grape_result.final_fidelity:.8f}, "
          f"iter={grape_result.n_iterations})")

    # -- Error budget for each method --
    print("\n" + "-" * 70)
    print("Computing error budgets...")
    print("-" * 70)

    budget_gaussian = compute_error_budget(
        "Gaussian", gauss_I, gauss_Q, gauss_times, U_target, IQM_PARAMS,
    )

    budget_drag = compute_error_budget(
        "DRAG", drag_I, drag_Q, drag_times, U_target, IQM_PARAMS,
    )

    budget_grape = compute_error_budget(
        "GRAPE", grape_I, grape_Q, grape_times, U_target, IQM_PARAMS,
        is_grape=True, n_slices=IQM_PARAMS["n_grape_slices"],
    )

    elapsed_total = perf_counter() - start_total

    # -- Summary table --
    print("\n" + "=" * 80)
    print("ERROR BUDGET SUMMARY -- X-gate, IQM Garnet Parameters")
    print(f"Gate time: {IQM_PARAMS['gate_time_ns']} ns, "
          f"T1={IQM_PARAMS['T1_us']} us, T2={IQM_PARAMS['T2_us']} us")
    print("=" * 80)

    header = f"{'Error Source':<24} {'Gaussian':>14} {'DRAG':>14} {'GRAPE':>14}"
    print(header)
    print("-" * 66)

    rows = [
        ("Coherent (unitary)",
         budget_gaussian["coherent_infidelity"],
         budget_drag["coherent_infidelity"],
         budget_grape["coherent_infidelity"]),
        ("T1 only",
         budget_gaussian["t1_only_infidelity"],
         budget_drag["t1_only_infidelity"],
         budget_grape["t1_only_infidelity"]),
        ("T2 (dephasing) only",
         budget_gaussian["t2_only_infidelity"],
         budget_drag["t2_only_infidelity"],
         budget_grape["t2_only_infidelity"]),
        ("Full T1+T2",
         budget_gaussian["full_decoherence_infidelity"],
         budget_drag["full_decoherence_infidelity"],
         budget_grape["full_decoherence_infidelity"]),
    ]

    for label, g, d, p in rows:
        print(f"{label:<24} {g:>14.2e} {d:>14.2e} {p:>14.2e}")

    # Noise rows
    for noise_key in ["1pct", "2pct", "5pct"]:
        gn = budget_gaussian["control_noise"][noise_key]
        dn = budget_drag["control_noise"][noise_key]
        pn = budget_grape["control_noise"][noise_key]
        label = f"Noise eps={gn['noise_level']*100:.0f}%"
        print(f"{label:<24} {gn['mean_infidelity']:>14.2e} "
              f"{dn['mean_infidelity']:>14.2e} {pn['mean_infidelity']:>14.2e}")

    print("-" * 66)
    print(f"Total time: {elapsed_total:.1f}s")

    # -- Save results --
    results = {
        "timestamp": datetime.now().isoformat(),
        "experiment": "error_budget_analysis",
        "parameters": {
            **IQM_PARAMS,
            "T1_ns": T1_NS,
            "T2_ns": T2_NS,
            "U_target": "X (Pauli-X)",
        },
        "budgets": {
            "gaussian": budget_gaussian,
            "drag": budget_drag,
            "grape": budget_grape,
        },
        "total_time_s": elapsed_total,
    }

    results_dir = Path(__file__).parent.parent / "results" / "error_budget"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_file = results_dir / f"error_budget_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")
    return results




# =====================================================================
# 3-LEVEL TRANSMON ERROR BUDGET
# =====================================================================
# In 3-level transmon, GRAPE's advantage over Gaussian/DRAG is real
# because leakage to |2> is a significant error source that GRAPE
# can suppress but DRAG only partially cancels.
# =====================================================================


def build_3level_collapse_ops(T1_ns, T2_ns):
    """
    Build collapse operators for 3-level transmon with T1/T2 decoherence.

    For a 3-level system:
    - T1 (amplitude damping): L_1 = sqrt(gamma1) * a  (lowering operator)
      This gives |1>->|0> at rate gamma1 and |2>->|1> at rate 2*gamma1
      (because <1|a|2> = sqrt(2)).
    - Pure dephasing: L_phi = sqrt(gamma_phi) * n_hat
      where n_hat = diag(0, 1, 2) is the number operator.
      This dephases each level proportional to its excitation number.

    Parameters
    ----------
    T1_ns : float
        T1 in nanoseconds.
    T2_ns : float
        T2 in nanoseconds.

    Returns
    -------
    list of qt.Qobj
        Collapse operators for 3-level system.
    """
    n_levels = 3
    c_ops = []

    gamma1 = 1.0 / T1_ns
    a = qt.destroy(n_levels)
    c_ops.append(np.sqrt(gamma1) * a)

    # Pure dephasing rate: gamma_phi = 1/T2 - 1/(2*T1)
    gamma2 = 1.0 / T2_ns
    gamma_phi = gamma2 - gamma1 / 2.0

    if gamma_phi > 1e-15:
        # Dephasing via number operator for multi-level
        n_op = qt.num(n_levels)
        c_ops.append(np.sqrt(gamma_phi) * n_op)

    return c_ops


def generate_gaussian_3level(params):
    """Generate Gaussian pulse for 3-level X-gate."""
    gate_time = params["gate_time_ns"]
    n_points = params["n_samples"]
    sigma = gate_time / 8.0
    amplitude = np.pi / (sigma * np.sqrt(2 * np.pi))

    times = np.linspace(0, gate_time, n_points)
    t_center = gate_time / 2.0
    pulse_I = amplitude * np.exp(-0.5 * ((times - t_center) / sigma) ** 2)
    pulse_Q = np.zeros_like(times)

    return pulse_I, pulse_Q, times


def generate_drag_3level(params):
    """Generate DRAG pulse with unit-corrected beta for 3-level X-gate."""
    gate_time = params["gate_time_ns"]
    n_points = params["n_samples"]
    sigma = gate_time / 8.0
    amplitude = np.pi / (sigma * np.sqrt(2 * np.pi))

    anharmonicity_radns = 2.0 * np.pi * params["anharmonicity_mhz"] * 1e-3
    optimal_beta = -1.0 / (2.0 * anharmonicity_radns)

    drag_params = DRAGParameters(
        amplitude=amplitude,
        sigma=sigma,
        beta=optimal_beta,
        anharmonicity=params["anharmonicity_mhz"],
    )
    drag = DRAGPulse(drag_params)

    times = np.linspace(0, gate_time, n_points)
    t_center = gate_time / 2.0
    pulse_I, pulse_Q = drag.envelope(times, t_center)

    return pulse_I, pulse_Q, times


def generate_grape_3level(params):
    """
    Run GRAPE optimization for X-gate in 3-level transmon.

    Uses tuned hyperparameters: lr=0.5, momentum=0.5, step_decay=1.0.
    """
    n_levels = 3
    gate_time = params["gate_time_ns"]
    n_slices = params["n_grape_slices"]
    sigma = gate_time / 8.0
    amplitude = np.pi / (sigma * np.sqrt(2 * np.pi))

    alpha_radns = 2.0 * np.pi * params["anharmonicity_mhz"] * 1e-3
    n_op = qt.num(n_levels)
    H_drift = (alpha_radns / 2.0) * (n_op * n_op - n_op)

    a = qt.destroy(n_levels)
    a_dag = qt.create(n_levels)
    H_x = 0.5 * (a + a_dag)
    H_y = 0.5 * 1j * (a_dag - a)

    # Target: X-gate on computational subspace, identity on |2>
    U_target_full = np.eye(n_levels, dtype=complex)
    U_target_full[0:2, 0:2] = qt.sigmax().full()
    U_target = qt.Qobj(U_target_full, dims=[[n_levels], [n_levels]])

    # Gaussian seed
    t_slices = np.linspace(0, gate_time, n_slices)
    t_center = gate_time / 2.0
    gaussian = amplitude * np.exp(-0.5 * ((t_slices - t_center) / sigma) ** 2)

    u_init = np.zeros((2, n_slices))
    u_init[0, :] = gaussian
    rng = np.random.default_rng(42)
    u_init[1, :] = rng.normal(0, amplitude * 0.05, n_slices)

    optimizer = GRAPEOptimizer(
        H_drift=H_drift,
        H_controls=[H_x, H_y],
        n_timeslices=n_slices,
        total_time=gate_time,
        u_limits=(-amplitude * 3.0, amplitude * 3.0),
        convergence_threshold=1e-10,
        max_iterations=1000,
        learning_rate=0.5,
        verbose=False,
        use_line_search=True,
        momentum=0.5,
    )

    result = optimizer.optimize_unitary(U_target, u_init=u_init, step_decay=1.0)

    pulse_I = result.optimized_pulses[0, :]
    pulse_Q = result.optimized_pulses[1, :]

    return pulse_I, pulse_Q, t_slices, result, U_target


def propagate_3level(pulse_I, pulse_Q, times, H_drift, H_x, H_y,
                     n_levels, is_grape=False, n_slices=None, gate_time=None):
    """Propagate unitary in 3-level system."""
    if is_grape:
        dt = gate_time / n_slices
        U = qt.qeye(n_levels)
        for k in range(n_slices):
            H_t = H_drift + pulse_I[k] * H_x + pulse_Q[k] * H_y
            U = (-1j * H_t * dt).expm() * U
    else:
        dt = times[1] - times[0]
        U = qt.qeye(n_levels)
        for i in range(len(times) - 1):
            H_t = H_drift + pulse_I[i] * H_x + pulse_Q[i] * H_y
            U = (-1j * H_t * dt).expm() * U
    return U


def avg_fidelity_and_leakage_3level(U_achieved, U_target, n_levels):
    """Average fidelity and leakage over |0> and |1>."""
    total_fid = 0.0
    total_leak = 0.0
    for k in range(2):
        psi0 = qt.basis(n_levels, k)
        psi_final = U_achieved * psi0
        psi_target = U_target * psi0
        fid = float(qt.fidelity(psi_target, psi_final) ** 2)
        psi_arr = psi_final.full().flatten()
        leak = float(sum(np.abs(psi_arr[j]) ** 2 for j in range(2, n_levels)))
        total_fid += fid
        total_leak += leak
    return total_fid / 2.0, total_leak / 2.0


def lindblad_fidelity_3level(H_td, U_target, gate_time, c_ops, n_levels,
                              n_steps=2000):
    """
    Compute gate fidelity under Lindblad decoherence in 3-level system.

    Averages over |0> and |1> initial states.
    """
    times_eval = np.linspace(0, gate_time, n_steps)

    total_fidelity = 0.0
    total_leakage = 0.0

    for k in range(2):
        psi0 = qt.basis(n_levels, k)
        rho0 = psi0 * psi0.dag()

        result = qt.mesolve(H_td, rho0, times_eval, c_ops)
        rho_final = result.states[-1]

        # Ideal final state
        psi_ideal = U_target * psi0
        rho_ideal = psi_ideal * psi_ideal.dag()

        fid = qt.fidelity(rho_final, rho_ideal) ** 2
        total_fidelity += fid

        # Leakage: population in |2>
        leak = float(np.real(rho_final[2, 2]))
        total_leakage += leak

    return float(total_fidelity / 2.0), float(total_leakage / 2.0)


def build_3level_td_hamiltonian(pulse_I, pulse_Q, times, H_drift, H_x, H_y,
                                 is_grape=False, n_slices=None, gate_time=None):
    """Build time-dependent Hamiltonian for 3-level QuTiP mesolve."""
    if is_grape:
        f_I = _make_piecewise_constant_interp(pulse_I, n_slices, gate_time)
        f_Q = _make_piecewise_constant_interp(pulse_Q, n_slices, gate_time)
    else:
        f_I = _make_pulse_interp(pulse_I, times)
        f_Q = _make_pulse_interp(pulse_Q, times)

    return [H_drift, [H_x, f_I], [H_y, f_Q]]


def noisy_fidelity_3level(pulse_I, pulse_Q, times, U_target, H_drift, H_x, H_y,
                           n_levels, noise_level, n_realizations=100,
                           is_grape=False, n_slices=None, gate_time=None, seed=0):
    """
    Compute fidelity with amplitude noise in 3-level system (closed system).
    """
    rng = np.random.default_rng(seed)
    fidelities = []
    leakages = []

    for _ in range(n_realizations):
        epsilon = rng.normal(0, noise_level)
        noisy_I = pulse_I * (1.0 + epsilon)
        noisy_Q = pulse_Q * (1.0 + epsilon)

        U = propagate_3level(
            noisy_I, noisy_Q, times, H_drift, H_x, H_y, n_levels,
            is_grape=is_grape, n_slices=n_slices, gate_time=gate_time,
        )
        fid, leak = avg_fidelity_and_leakage_3level(U, U_target, n_levels)
        fidelities.append(fid)
        leakages.append(leak)

    return (float(np.mean(fidelities)), float(np.std(fidelities)),
            float(np.mean(leakages)), float(np.std(leakages)))


def compute_3level_error_budget(method_name, pulse_I, pulse_Q, times,
                                 U_target, params, H_drift, H_x, H_y,
                                 is_grape=False, n_slices=None):
    """
    Full error budget decomposition for one pulse method in 3-level transmon.
    """
    n_levels = 3
    gate_time = params["gate_time_ns"]
    print(f"\n  {method_name} (3-level) error budget:")

    # 1. Coherent error
    print("    [1/5] Coherent error...", end=" ", flush=True)
    U_achieved = propagate_3level(
        pulse_I, pulse_Q, times, H_drift, H_x, H_y, n_levels,
        is_grape=is_grape, n_slices=n_slices, gate_time=gate_time,
    )
    coherent_fid, coherent_leak = avg_fidelity_and_leakage_3level(
        U_achieved, U_target, n_levels
    )
    coherent_infid = 1.0 - coherent_fid
    print(f"infidelity = {coherent_infid:.2e}, leakage = {coherent_leak:.2e}")

    # Build time-dependent Hamiltonian
    H_td = build_3level_td_hamiltonian(
        pulse_I, pulse_Q, times, H_drift, H_x, H_y,
        is_grape=is_grape, n_slices=n_slices, gate_time=gate_time,
    )

    # 2. T1 only
    print("    [2/5] T1 error...", end=" ", flush=True)
    c_ops_t1 = build_3level_collapse_ops(T1_NS, 2.0 * T1_NS)
    fid_t1, leak_t1 = lindblad_fidelity_3level(
        H_td, U_target, gate_time, c_ops_t1, n_levels
    )
    print(f"infidelity = {1.0 - fid_t1:.2e}, leakage = {leak_t1:.2e}")

    # 3. T2 only
    print("    [3/5] T2 (dephasing) error...", end=" ", flush=True)
    T1_large = 1e8
    c_ops_t2 = build_3level_collapse_ops(T1_large, T2_NS)
    fid_t2, leak_t2 = lindblad_fidelity_3level(
        H_td, U_target, gate_time, c_ops_t2, n_levels
    )
    print(f"infidelity = {1.0 - fid_t2:.2e}, leakage = {leak_t2:.2e}")

    # 4. Combined T1 + T2
    print("    [4/5] Combined T1+T2...", end=" ", flush=True)
    c_ops_full = build_3level_collapse_ops(T1_NS, T2_NS)
    fid_full, leak_full = lindblad_fidelity_3level(
        H_td, U_target, gate_time, c_ops_full, n_levels
    )
    print(f"infidelity = {1.0 - fid_full:.2e}, leakage = {leak_full:.2e}")

    # 5. Control noise
    noise_levels = [0.01, 0.02, 0.05]
    noise_results = {}
    for nl in noise_levels:
        print(f"    [5/5] Noise eps={nl*100:.0f}%...", end=" ", flush=True)
        mean_fid, std_fid, mean_leak, std_leak = noisy_fidelity_3level(
            pulse_I, pulse_Q, times, U_target, H_drift, H_x, H_y,
            n_levels, nl, n_realizations=100,
            is_grape=is_grape, n_slices=n_slices, gate_time=gate_time,
        )
        print(f"infidelity = {1.0 - mean_fid:.2e} +/- {std_fid:.2e}")
        noise_results[f"{nl*100:.0f}pct"] = {
            "noise_level": nl,
            "mean_fidelity": mean_fid,
            "std_fidelity": std_fid,
            "mean_infidelity": float(1.0 - mean_fid),
            "mean_leakage": mean_leak,
            "std_leakage": std_leak,
        }

    return {
        "method": method_name,
        "n_levels": n_levels,
        "coherent_fidelity": coherent_fid,
        "coherent_infidelity": coherent_infid,
        "coherent_leakage": coherent_leak,
        "t1_only_fidelity": fid_t1,
        "t1_only_infidelity": 1.0 - fid_t1,
        "t1_only_leakage": leak_t1,
        "t2_only_fidelity": fid_t2,
        "t2_only_infidelity": 1.0 - fid_t2,
        "t2_only_leakage": leak_t2,
        "full_decoherence_fidelity": fid_full,
        "full_decoherence_infidelity": 1.0 - fid_full,
        "full_decoherence_leakage": leak_full,
        "control_noise": noise_results,
    }


def run_3level_error_budget():
    """Run 3-level transmon error budget analysis."""
    print("\n\n")
    print("=" * 70)
    print("3-LEVEL TRANSMON ERROR BUDGET")
    print("=" * 70)
    print("This is where GRAPE's advantage is real: leakage suppression.")
    print()

    n_levels = 3
    gate_time = IQM_PARAMS["gate_time_ns"]
    alpha_radns = 2.0 * np.pi * IQM_PARAMS["anharmonicity_mhz"] * 1e-3

    # Build 3-level Hamiltonians
    n_op = qt.num(n_levels)
    H_drift = (alpha_radns / 2.0) * (n_op * n_op - n_op)
    a = qt.destroy(n_levels)
    a_dag = qt.create(n_levels)
    H_x = 0.5 * (a + a_dag)
    H_y = 0.5 * 1j * (a_dag - a)

    # Target: X on {|0>,|1>}, identity on |2>
    U_target_arr = np.eye(n_levels, dtype=complex)
    U_target_arr[0:2, 0:2] = qt.sigmax().full()
    U_target = qt.Qobj(U_target_arr, dims=[[n_levels], [n_levels]])

    print(f"  alpha/2pi = {IQM_PARAMS['anharmonicity_mhz']} MHz")
    print(f"  alpha = {alpha_radns:.4f} rad/ns")
    print(f"  gate_time = {gate_time} ns")
    print(f"  T1 = {T1_NS:.0f} ns, T2 = {T2_NS:.0f} ns")

    start = perf_counter()

    # Generate pulses
    print("\nGenerating 3-level pulses:")

    print("  Gaussian...", end=" ", flush=True)
    gauss_I, gauss_Q, gauss_times = generate_gaussian_3level(IQM_PARAMS)
    print("done")

    print("  DRAG...", end=" ", flush=True)
    drag_I, drag_Q, drag_times = generate_drag_3level(IQM_PARAMS)
    print("done")

    print("  GRAPE (3-level)...", end=" ", flush=True)
    grape_I, grape_Q, grape_times, grape_result, _ = generate_grape_3level(IQM_PARAMS)
    print(f"done (F={grape_result.final_fidelity:.8f}, "
          f"iter={grape_result.n_iterations})")

    # Error budgets
    print("\n" + "-" * 70)
    print("Computing 3-level error budgets...")
    print("-" * 70)

    budget3_gaussian = compute_3level_error_budget(
        "Gaussian", gauss_I, gauss_Q, gauss_times, U_target, IQM_PARAMS,
        H_drift, H_x, H_y,
    )

    budget3_drag = compute_3level_error_budget(
        "DRAG", drag_I, drag_Q, drag_times, U_target, IQM_PARAMS,
        H_drift, H_x, H_y,
    )

    budget3_grape = compute_3level_error_budget(
        "GRAPE", grape_I, grape_Q, grape_times, U_target, IQM_PARAMS,
        H_drift, H_x, H_y, is_grape=True, n_slices=IQM_PARAMS["n_grape_slices"],
    )

    elapsed = perf_counter() - start

    # Summary table
    print("\n" + "=" * 90)
    print("3-LEVEL ERROR BUDGET SUMMARY -- X-gate, IQM Garnet Parameters")
    print(f"Gate time: {gate_time} ns, alpha/2pi = {IQM_PARAMS['anharmonicity_mhz']} MHz")
    print("=" * 90)

    header = (f"{'Error Source':<24} {'Gaussian':>14} {'DRAG':>14} {'GRAPE':>14}"
              f"  {'G-leak':>10} {'D-leak':>10} {'P-leak':>10}")
    print(header)
    print("-" * 98)

    budget_rows = [
        ("Coherent (unitary)",
         budget3_gaussian, budget3_drag, budget3_grape,
         "coherent_infidelity", "coherent_leakage"),
        ("T1 only",
         budget3_gaussian, budget3_drag, budget3_grape,
         "t1_only_infidelity", "t1_only_leakage"),
        ("T2 (dephasing) only",
         budget3_gaussian, budget3_drag, budget3_grape,
         "t2_only_infidelity", "t2_only_leakage"),
        ("Full T1+T2",
         budget3_gaussian, budget3_drag, budget3_grape,
         "full_decoherence_infidelity", "full_decoherence_leakage"),
    ]

    for label, bg, bd, bp, infid_key, leak_key in budget_rows:
        print(f"{label:<24} {bg[infid_key]:>14.2e} {bd[infid_key]:>14.2e} {bp[infid_key]:>14.2e}"
              f"  {bg[leak_key]:>10.2e} {bd[leak_key]:>10.2e} {bp[leak_key]:>10.2e}")

    # Noise rows
    for noise_key in ["1pct", "2pct", "5pct"]:
        gn = budget3_gaussian["control_noise"][noise_key]
        dn = budget3_drag["control_noise"][noise_key]
        pn = budget3_grape["control_noise"][noise_key]
        label = f"Noise eps={gn['noise_level']*100:.0f}%"
        print(f"{label:<24} {gn['mean_infidelity']:>14.2e} "
              f"{dn['mean_infidelity']:>14.2e} {pn['mean_infidelity']:>14.2e}"
              f"  {gn['mean_leakage']:>10.2e} {dn['mean_leakage']:>10.2e}"
              f" {pn['mean_leakage']:>10.2e}")

    print("-" * 98)
    print(f"3-level budget time: {elapsed:.1f}s")

    return {
        "gaussian": budget3_gaussian,
        "drag": budget3_drag,
        "grape": budget3_grape,
    }


# Patch the main to also run 3-level
if __name__ == "__main__":
    import sys

    # Check if we should only run 3-level
    only_3level = "--3level-only" in sys.argv

    if not only_3level:
        results_2level = run_experiment()
    else:
        results_2level = None

    results_3level = run_3level_error_budget()

    # Save combined results
    combined = {
        "timestamp": datetime.now().isoformat(),
        "experiment": "error_budget_analysis_combined",
        "parameters": {
            **IQM_PARAMS,
            "T1_ns": T1_NS,
            "T2_ns": T2_NS,
        },
    }
    if results_2level is not None:
        combined["budgets_2level"] = results_2level.get("budgets", {})
    combined["budgets_3level"] = results_3level

    results_dir = Path(__file__).parent.parent / "results" / "error_budget"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_file = results_dir / f"error_budget_combined_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(combined, f, indent=2, default=str)

    print(f"\nCombined results saved to: {output_file}")
