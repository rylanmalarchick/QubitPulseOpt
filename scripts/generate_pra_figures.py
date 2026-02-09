#!/usr/bin/env python3
"""
Generate publication-quality figures for PRA submission.

Figures:
    1. Pulse shape comparison (Gaussian I, DRAG I+Q, GRAPE I+Q) — 3-level, 20ns
    2. Gate-time fidelity sweep (3-level, 10-100ns)
    3. Robustness analysis (detuning + amplitude error, 3-level)
    4. Error budget decomposition (3-level, stacked bar)

All figures use consistent styling for PRA twocolumn format (3.375" single column).

Usage:
    .venv/bin/python scripts/generate_pra_figures.py
"""

import sys
import json
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import qutip as qt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.pulses.drag import DRAGPulse, DRAGParameters
from src.optimization.grape import GRAPEOptimizer


# -- Paths ------------------------------------------------------------------

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "paper" / "pra" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


# -- Global Style -----------------------------------------------------------

# PRA single-column width: 3.375", double: 6.75"
SINGLE_COL = 3.375
DOUBLE_COL = 6.75

# Color scheme: colorblind-friendly (Okabe-Ito palette)
COLORS = {
    "gaussian": "#E69F00",  # orange
    "drag": "#0072B2",      # blue
    "grape": "#009E73",     # green
    "drag_q": "#56B4E9",    # light blue (DRAG Q-channel)
    "grape_q": "#CC79A7",   # pink (GRAPE Q-channel)
}

LABELS = {
    "gaussian": "Gaussian",
    "drag": "DRAG",
    "grape": "GRAPE",
}


def setup_style():
    """Configure matplotlib for PRA publication figures."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "lines.linewidth": 1.0,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.8",
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "text.usetex": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_figure(fig, name):
    """Save figure as PDF and EPS for PRA submission."""
    fig.savefig(FIGURES / f"{name}.pdf", format="pdf")
    fig.savefig(FIGURES / f"{name}.eps", format="eps")
    print(f"  Saved: figures/{name}.pdf, .eps")


# -- Figure 1: Pulse Shape Comparison --------------------------------------


def generate_3level_pulses(gate_time=20.0, n_slices=50, anharmonicity_mhz=-200.0):
    """
    Generate Gaussian, DRAG, and GRAPE pulses for 3-level X-gate.

    Returns dense I/Q waveforms for Gaussian/DRAG (smooth curves)
    and piecewise-constant arrays for GRAPE.
    """
    n_levels = 3
    sigma = gate_time / 8.0
    amplitude = np.pi / (sigma * np.sqrt(2 * np.pi))
    n_dense = 400

    times_dense = np.linspace(0, gate_time, n_dense)
    t_center = gate_time / 2.0

    # Gaussian
    gauss_I = amplitude * np.exp(-0.5 * ((times_dense - t_center) / sigma) ** 2)
    gauss_Q = np.zeros_like(times_dense)

    # DRAG
    alpha_radns = 2.0 * np.pi * anharmonicity_mhz * 1e-3
    optimal_beta = -1.0 / (2.0 * alpha_radns)
    drag_params = DRAGParameters(
        amplitude=amplitude, sigma=sigma, beta=optimal_beta,
        anharmonicity=anharmonicity_mhz,
    )
    drag = DRAGPulse(drag_params)
    drag_I, drag_Q = drag.envelope(times_dense, t_center)

    # GRAPE (3-level)
    n_op = qt.num(n_levels)
    H_drift = (alpha_radns / 2.0) * (n_op * n_op - n_op)
    a = qt.destroy(n_levels)
    a_dag = qt.create(n_levels)
    H_x = 0.5 * (a + a_dag)
    H_y = 0.5 * 1j * (a_dag - a)
    H_controls = [H_x, H_y]

    U_target_arr = np.eye(n_levels, dtype=complex)
    U_target_arr[0:2, 0:2] = qt.sigmax().full()
    U_target = qt.Qobj(U_target_arr, dims=[[n_levels], [n_levels]])

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

    print(f"  GRAPE converged: {result.converged}, "
          f"F={result.final_fidelity:.10f}, "
          f"iter={result.n_iterations}")

    return {
        "times_dense": times_dense,
        "gauss_I": gauss_I, "gauss_Q": gauss_Q,
        "drag_I": drag_I, "drag_Q": drag_Q,
        "grape_times": t_slices,
        "grape_I": grape_I, "grape_Q": grape_Q,
        "gate_time": gate_time, "n_slices": n_slices,
    }


def plot_fig1_pulse_shapes(pulse_data):
    """
    Fig 1: Three-panel pulse shape comparison.

    (a) Gaussian: I-channel only
    (b) DRAG: I + Q channels
    (c) GRAPE: I + Q channels (piecewise-constant)
    """
    fig, axes = plt.subplots(3, 1, figsize=(SINGLE_COL, 3.5), sharex=True)

    t = pulse_data["times_dense"]
    gt = pulse_data["gate_time"]

    # (a) Gaussian
    ax = axes[0]
    ax.plot(t, pulse_data["gauss_I"], color=COLORS["gaussian"], label="I (Gaussian)")
    ax.axhline(0, color="0.7", linewidth=0.4, zorder=0)
    ax.set_ylabel(r"$\Omega$ (rad/ns)")
    ax.legend(loc="upper right")
    ax.text(0.02, 0.92, "(a)", transform=ax.transAxes, fontweight="bold", fontsize=8,
            verticalalignment="top")

    # (b) DRAG
    ax = axes[1]
    ax.plot(t, pulse_data["drag_I"], color=COLORS["drag"], label=r"I ($\Omega_x$)")
    ax.plot(t, pulse_data["drag_Q"], color=COLORS["drag_q"], linestyle="--",
            label=r"Q ($\Omega_y$)")
    ax.axhline(0, color="0.7", linewidth=0.4, zorder=0)
    ax.set_ylabel(r"$\Omega$ (rad/ns)")
    ax.legend(loc="upper right")
    ax.text(0.02, 0.92, "(b)", transform=ax.transAxes, fontweight="bold", fontsize=8,
            verticalalignment="top")

    # (c) GRAPE -- piecewise-constant (step plot)
    ax = axes[2]
    grape_t_edges = np.linspace(0, gt, pulse_data["n_slices"] + 1)
    ax.step(grape_t_edges[:-1], pulse_data["grape_I"], where="post",
            color=COLORS["grape"], label=r"I ($u_x$)")
    ax.step(grape_t_edges[:-1], pulse_data["grape_Q"], where="post",
            color=COLORS["grape_q"], linestyle="--", label=r"Q ($u_y$)")
    ax.axhline(0, color="0.7", linewidth=0.4, zorder=0)
    ax.set_ylabel(r"$\Omega$ (rad/ns)")
    ax.set_xlabel("Time (ns)")
    ax.legend(loc="upper right")
    ax.text(0.02, 0.92, "(c)", transform=ax.transAxes, fontweight="bold", fontsize=8,
            verticalalignment="top")

    for ax in axes:
        ax.set_xlim(0, gt)

    fig.tight_layout(h_pad=0.3)
    return fig


# -- Figure 2: Gate-Time Fidelity Sweep ------------------------------------


def load_multilevel_data():
    """Load 3-level gate-time sweep results."""
    path = RESULTS / "multilevel_comparison" / "3level_xgate_20260207_202907.json"
    with open(path) as f:
        data = json.load(f)
    return data


def plot_fig2_gatetime_sweep(data):
    """
    Fig 2: Fidelity and leakage vs gate time (3-level transmon).

    (a) Infidelity (log scale)
    (b) Leakage probability
    """
    gate_times = np.array(data["gate_times"], dtype=float)
    sweeps = data["sweeps"]

    fid_g = np.array([s["gaussian"]["fidelity"] for s in sweeps])
    fid_d = np.array([s["drag"]["fidelity"] for s in sweeps])
    fid_p = np.array([s["grape"]["fidelity"] for s in sweeps])

    leak_g = np.array([s["gaussian"]["leakage"] for s in sweeps])
    leak_d = np.array([s["drag"]["leakage"] for s in sweeps])
    leak_p = np.array([s["grape"]["leakage"] for s in sweeps])

    # Clamp GRAPE infidelity floor for log scale
    infid_g = np.maximum(1 - fid_g, 1e-16)
    infid_d = np.maximum(1 - fid_d, 1e-16)
    infid_p = np.maximum(1 - fid_p, 1e-16)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(SINGLE_COL, 3.2), sharex=True,
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.08},
    )

    # (a) Infidelity
    ax1.semilogy(gate_times, infid_g, "o-", color=COLORS["gaussian"],
                 label="Gaussian", markersize=3.5)
    ax1.semilogy(gate_times, infid_d, "s-", color=COLORS["drag"],
                 label="DRAG", markersize=3.5)
    ax1.semilogy(gate_times, infid_p, "^-", color=COLORS["grape"],
                 label="GRAPE", markersize=3.5)
    ax1.set_ylabel(r"Infidelity $1 - F$")
    ax1.set_ylim(1e-16, 1)
    ax1.legend(loc="upper right")
    ax1.text(0.02, 0.95, "(a)", transform=ax1.transAxes, fontweight="bold", fontsize=8,
             verticalalignment="top")
    ax1.axhline(1e-4, color="0.8", linewidth=0.4, linestyle=":")
    ax1.axhline(1e-2, color="0.8", linewidth=0.4, linestyle=":")

    # (b) Leakage
    ax2.semilogy(gate_times, np.maximum(leak_g, 1e-16), "o-",
                 color=COLORS["gaussian"], markersize=3.5)
    ax2.semilogy(gate_times, np.maximum(leak_d, 1e-16), "s-",
                 color=COLORS["drag"], markersize=3.5)
    ax2.semilogy(gate_times, np.maximum(leak_p, 1e-16), "^-",
                 color=COLORS["grape"], markersize=3.5)
    ax2.set_ylabel(r"Leakage $P_2$")
    ax2.set_xlabel("Gate time (ns)")
    ax2.set_ylim(1e-16, 1)
    ax2.text(0.02, 0.90, "(b)", transform=ax2.transAxes, fontweight="bold", fontsize=8,
             verticalalignment="top")

    fig.tight_layout()
    return fig


# -- Figure 3: Robustness Analysis -----------------------------------------


def load_robustness_data():
    """Load robustness sweep results."""
    path = RESULTS / "robustness" / "robustness_20260207_203855.json"
    with open(path) as f:
        data = json.load(f)
    return data


def plot_fig3_robustness(data):
    """
    Fig 3: Robustness to detuning and amplitude errors (3-level).

    (a) Fidelity vs detuning (MHz)
    (b) Fidelity vs amplitude error (%)
    """
    det_mhz = np.array(data["detuning_mhz"])
    amp_err = np.array(data["amplitude_errors"]) * 100  # to percent

    r3 = data["results_3level"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.4))

    # (a) Detuning robustness
    for method, marker in [("gaussian", "o"), ("drag", "s"), ("grape", "^")]:
        fids = np.array(r3[method]["detuning_fidelities"])
        ax1.plot(det_mhz, fids, f"{marker}-", color=COLORS[method],
                 label=LABELS[method], markersize=2.5, markevery=4)
    ax1.set_xlabel(r"Detuning $\Delta/2\pi$ (MHz)")
    ax1.set_ylabel("Fidelity")
    ax1.legend(loc="lower left", fontsize=6.5)
    ax1.set_xlim(det_mhz[0], det_mhz[-1])
    ax1.text(0.02, 0.95, "(a)", transform=ax1.transAxes, fontweight="bold", fontsize=8,
             verticalalignment="top")

    # (b) Amplitude robustness
    for method, marker in [("gaussian", "o"), ("drag", "s"), ("grape", "^")]:
        fids = np.array(r3[method]["amplitude_fidelities"])
        ax2.plot(amp_err, fids, f"{marker}-", color=COLORS[method],
                 label=LABELS[method], markersize=2.5, markevery=4)
    ax2.set_xlabel(r"Amplitude error $\delta\Omega/\Omega$ (%)")
    ax2.set_ylabel("Fidelity")
    ax2.legend(loc="lower left", fontsize=6.5)
    ax2.set_xlim(amp_err[0], amp_err[-1])
    ax2.text(0.02, 0.95, "(b)", transform=ax2.transAxes, fontweight="bold", fontsize=8,
             verticalalignment="top")

    fig.tight_layout(w_pad=1.5)
    return fig


# -- Figure 4: Error Budget ------------------------------------------------


def load_error_budget_data():
    """Load combined 2+3 level error budget."""
    path = RESULTS / "error_budget" / "error_budget_combined_20260207_203843.json"
    with open(path) as f:
        data = json.load(f)
    return data


def plot_fig4_error_budget(data):
    """
    Fig 4: Error budget decomposition (3-level transmon, 20ns X-gate).

    Grouped bar chart showing infidelity contributions:
    - Coherent error
    - T1 relaxation contribution
    - T2 dephasing contribution

    For each of Gaussian, DRAG, GRAPE.
    Leakage shown as markers on secondary axis.
    """
    budgets = data["budgets_3level"]

    methods = ["gaussian", "drag", "grape"]
    method_labels = ["Gaussian", "DRAG", "GRAPE"]

    # Extract infidelity components
    coherent = np.array([budgets[m]["coherent_infidelity"] for m in methods])
    t1_contrib = np.array([
        budgets[m]["t1_only_infidelity"] - budgets[m]["coherent_infidelity"]
        for m in methods
    ])
    t2_contrib = np.array([
        budgets[m]["t2_only_infidelity"] - budgets[m]["coherent_infidelity"]
        for m in methods
    ])
    full = np.array([budgets[m]["full_decoherence_infidelity"] for m in methods])

    # Clamp negative contributions to zero (numerical noise)
    t1_contrib = np.maximum(t1_contrib, 0)
    t2_contrib = np.maximum(t2_contrib, 0)

    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.4))

    x = np.arange(len(methods))
    bar_width = 0.6

    # Stacked bars
    ax.bar(x, coherent, bar_width, label="Coherent",
           color="#D55E00", edgecolor="white", linewidth=0.3)
    ax.bar(x, t1_contrib, bar_width, bottom=coherent,
           label=r"$T_1$ relaxation", color="#0072B2",
           edgecolor="white", linewidth=0.3)
    ax.bar(x, t2_contrib, bar_width, bottom=coherent + t1_contrib,
           label=r"$T_2$ dephasing", color="#009E73",
           edgecolor="white", linewidth=0.3)

    # Total infidelity annotation
    for i, f in enumerate(full):
        if f > 1e-3:
            label = f"{f:.1e}"
        else:
            label = f"{f:.2e}"
        ax.text(i, f * 1.15, label, ha="center", va="bottom", fontsize=6.5)

    ax.set_yscale("log")
    ax.set_ylabel(r"Infidelity $1 - F$")
    ax.set_xticks(x)
    ax.set_xticklabels(method_labels)
    ax.set_ylim(1e-5, 0.1)
    ax.legend(loc="upper right", fontsize=6.5)

    # Leakage on secondary axis
    leakage = np.array([budgets[m]["full_decoherence_leakage"] for m in methods])
    ax2 = ax.twinx()
    ax2.scatter(x, leakage, marker="x", color="black", s=20, zorder=5, label="Leakage")
    ax2.set_yscale("log")
    ax2.set_ylabel(r"Leakage $P_2$", fontsize=8)
    ax2.set_ylim(1e-5, 0.1)
    ax2.legend(loc="center right", fontsize=6.5)

    fig.tight_layout()
    return fig


# -- Main -------------------------------------------------------------------


def main():
    setup_style()
    t0 = perf_counter()

    # Figure 1: Pulse shapes (requires GRAPE optimization)
    print("Figure 1: Generating pulse shapes (3-level, 20ns X-gate)...")
    pulse_data = generate_3level_pulses()
    fig1 = plot_fig1_pulse_shapes(pulse_data)
    save_figure(fig1, "fig1_pulse_shapes")
    plt.close(fig1)

    # Figure 2: Gate-time sweep
    print("Figure 2: Gate-time fidelity sweep...")
    ml_data = load_multilevel_data()
    fig2 = plot_fig2_gatetime_sweep(ml_data)
    save_figure(fig2, "fig2_gatetime_sweep")
    plt.close(fig2)

    # Figure 3: Robustness
    print("Figure 3: Robustness analysis...")
    rob_data = load_robustness_data()
    fig3 = plot_fig3_robustness(rob_data)
    save_figure(fig3, "fig3_robustness")
    plt.close(fig3)

    # Figure 4: Error budget
    print("Figure 4: Error budget decomposition...")
    eb_data = load_error_budget_data()
    fig4 = plot_fig4_error_budget(eb_data)
    save_figure(fig4, "fig4_error_budget")
    plt.close(fig4)

    elapsed = perf_counter() - t0
    print(f"\nAll figures generated in {elapsed:.1f}s")
    print(f"Output directory: {FIGURES}")


if __name__ == "__main__":
    main()
