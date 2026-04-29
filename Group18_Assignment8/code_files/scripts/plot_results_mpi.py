#!/usr/bin/env python3
"""
plot_results_mpi.py — Generate Assignment 08 MPI+OpenMP plots.

Reads data_cluster/summary.csv and writes PNGs to results/.
"""
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(SCRIPT_DIR, "..", "..")
CSV_PATH = os.path.join(ROOT, "data_cluster", "summary.csv")
PLOT_DIR = os.path.join(ROOT, "results")
os.makedirs(PLOT_DIR, exist_ok=True)

CONFIG_LABELS = {
    "a": "Nx=250, Ny=100, 0.9M",
    "b": "Nx=250, Ny=100, 5M",
    "c": "Nx=500, Ny=200, 3.6M",
    "d": "Nx=500, Ny=200, 20M",
    "e": "Nx=1000, Ny=400, 14M",
}
BEST = "private_reduction"
CORE_COUNTS = [1, 2, 4, 8, 16, 32, 64]

plt.rcParams.update({
    "font.family": "serif",
    "figure.figsize": (8, 5),
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def load_data():
    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] {CSV_PATH} not found. Run run_all.sh first.")
        sys.exit(1)
    df = pd.read_csv(CSV_PATH)
    int_cols = [
        "mpi_ranks", "omp_threads", "total_cores", "Nx", "Ny",
        "N_particles", "local_particles", "Maxiter", "iteration",
        "n_active",
    ]
    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)
    for col in ["t_interp_s", "t_comm_s", "t_norm_s", "t_mover_s",
                "t_denorm_s", "t_total_s"]:
        if col in df.columns:
            df[col] = df[col].astype(float)
    if "t_comm_s" not in df.columns:
        df["t_comm_s"] = 0.0
    if "t_norm_s" not in df.columns:
        df["t_norm_s"] = 0.0
    if "t_denorm_s" not in df.columns:
        df["t_denorm_s"] = 0.0
    return df


def avg_time(df, config, cores):
    if cores == 1:
        sub = df[(df["config"] == config) & (df["total_cores"] == 1)]
    else:
        sub = df[(df["config"] == config) &
                 (df["variant"] == BEST) &
                 (df["total_cores"] == cores)]
    if sub.empty:
        return None
    return sub["t_total_s"].mean()


def plot_speedup(df):
    for cfg, label in CONFIG_LABELS.items():
        t1 = avg_time(df, cfg, 1)
        if t1 is None:
            continue
        cores, speedups = [], []
        for c in CORE_COUNTS:
            t = avg_time(df, cfg, c)
            if t is not None:
                cores.append(c)
                speedups.append(t1 / t)

        fig, ax = plt.subplots()
        ax.plot(cores, speedups, "o-", color="#2563eb", lw=2, ms=6,
                label="MPI+OpenMP")
        ax.plot([1, max(cores)], [1, max(cores)], "--",
                color="#9ca3af", lw=1, label="Ideal")
        ax.set_xscale("log", base=2)
        ax.set_xticks(cores)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlabel("Total cores")
        ax.set_ylabel("Speedup S(p)")
        ax.set_title(f"Speedup — Config ({cfg}): {label}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, f"speedup_{cfg}.png"), dpi=200)
        plt.close(fig)
        print(f"  [plot] speedup_{cfg}.png")


def plot_exec_time(df):
    for cfg, label in CONFIG_LABELS.items():
        cores, times = [], []
        for c in CORE_COUNTS:
            t = avg_time(df, cfg, c)
            if t is not None:
                cores.append(c)
                times.append(t)

        fig, ax = plt.subplots()
        ax.semilogy(cores, times, "s-", color="#dc2626", lw=2, ms=6)
        ax.set_xscale("log", base=2)
        ax.set_xticks(cores)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlabel("Total cores")
        ax.set_ylabel("Avg time/iteration (s)")
        ax.set_title(f"Execution Time — Config ({cfg}): {label}")
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, f"exectime_{cfg}.png"), dpi=200)
        plt.close(fig)
        print(f"  [plot] exectime_{cfg}.png")


def plot_efficiency(df):
    fig, ax = plt.subplots()
    markers = ["o", "s", "^", "D", "v"]
    colors = ["#2563eb", "#dc2626", "#16a34a", "#f59e0b", "#8b5cf6"]

    for idx, (cfg, label) in enumerate(CONFIG_LABELS.items()):
        t1 = avg_time(df, cfg, 1)
        if t1 is None:
            continue
        cores, effs = [], []
        for c in CORE_COUNTS:
            t = avg_time(df, cfg, c)
            if t is not None:
                cores.append(c)
                effs.append((t1 / t) / c)

        ax.plot(cores, effs, f"{markers[idx]}-", color=colors[idx],
                lw=1.5, ms=5, label=f"({cfg}) {label}")

    ax.axhline(1.0, color="#9ca3af", ls="--", lw=1)
    ax.set_xscale("log", base=2)
    ax.set_xticks(CORE_COUNTS)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("Total cores")
    ax.set_ylabel("Efficiency E(p)")
    ax.set_ylim(0, 1.2)
    ax.set_title("Parallel Efficiency — MPI+OpenMP")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "efficiency_all.png"), dpi=200)
    plt.close(fig)
    print("  [plot] efficiency_all.png")


def plot_phase_time(df):
    for cfg, label in CONFIG_LABELS.items():
        sub = df[(df["config"] == cfg) &
                 (df["variant"] == BEST) &
                 (df["total_cores"] == 64)]
        if sub.empty:
            sub = df[(df["config"] == cfg) & (df["variant"] == BEST)]
        if sub.empty:
            continue
        sub = sub.sort_values("iteration")

        fig, ax = plt.subplots()
        ax.plot(sub["iteration"], sub["t_interp_s"], "o-",
                color="#2563eb", lw=1.5, ms=5, label="Interpolation")
        ax.plot(sub["iteration"], sub["t_comm_s"], "^-",
                color="#7c3aed", lw=1.2, ms=4, label="MPI Allreduce")
        ax.plot(sub["iteration"], sub["t_norm_s"], "d-",
                color="#059669", lw=1.2, ms=4, label="Normalize")
        ax.plot(sub["iteration"], sub["t_mover_s"], "s-",
                color="#dc2626", lw=1.5, ms=5, label="Mover")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Time (s)")
        ax.set_title(f"Phase Time vs Iteration — ({cfg}): {label}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, f"phase_time_{cfg}.png"), dpi=200)
        plt.close(fig)
        print(f"  [plot] phase_time_{cfg}.png")


def plot_stacked_bar(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    cfgs = list(CONFIG_LABELS.keys())
    t_interps, t_comms, t_norms, t_movers, t_denorms, labels = [], [], [], [], [], []

    for cfg in cfgs:
        sub = df[(df["config"] == cfg) &
                 (df["variant"] == BEST) &
                 (df["total_cores"] == 64)]
        if sub.empty:
            sub = df[(df["config"] == cfg) & (df["variant"] == BEST)]
        t_interps.append(0.0 if sub.empty else sub["t_interp_s"].mean())
        t_comms.append(0.0 if sub.empty else sub["t_comm_s"].mean())
        t_norms.append(0.0 if sub.empty else sub["t_norm_s"].mean())
        t_movers.append(0.0 if sub.empty else sub["t_mover_s"].mean())
        t_denorms.append(0.0 if sub.empty else sub["t_denorm_s"].mean())
        labels.append(f"({cfg})\n{CONFIG_LABELS[cfg]}")

    x = np.arange(len(cfgs))
    b_interp = np.array(t_interps)
    b_comm = np.array(t_comms)
    b_norm = np.array(t_norms)
    b_mover = np.array(t_movers)
    b_denorm = np.array(t_denorms)
    ax.bar(x, b_interp, 0.55, label="Interpolation", color="#3b82f6")
    ax.bar(x, b_comm, 0.55, bottom=b_interp, label="MPI Allreduce",
           color="#7c3aed")
    ax.bar(x, b_norm, 0.55, bottom=b_interp + b_comm, label="Normalize",
           color="#10b981")
    ax.bar(x, b_mover, 0.55, bottom=b_interp + b_comm + b_norm,
           label="Mover", color="#f59e0b")
    ax.bar(x, b_denorm, 0.55,
           bottom=b_interp + b_comm + b_norm + b_mover,
           label="Denormalize", color="#64748b")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Avg time/iteration (s)")
    ax.set_title("Interpolation vs Mover — 64 Total Cores")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "stacked_bar_phases.png"), dpi=200)
    plt.close(fig)
    print("  [plot] stacked_bar_phases.png")


def plot_active_count(df):
    cfg = "e"
    sub = df[(df["config"] == cfg) &
             (df["variant"] == BEST) &
             (df["total_cores"] == df["total_cores"].max())]
    if sub.empty:
        return
    sub = sub.sort_values("iteration")

    fig, ax = plt.subplots()
    ax.plot(sub["iteration"], sub["n_active"], "o-",
            color="#16a34a", lw=2, ms=6)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Active particles")
    ax.set_title(f"Active Particles vs Iteration — Config (e): {CONFIG_LABELS[cfg]}")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "active_count_e.png"), dpi=200)
    plt.close(fig)
    print("  [plot] active_count_e.png")


def main():
    df = load_data()
    print(f"Loaded {len(df)} rows from {CSV_PATH}\n")
    plot_speedup(df)
    plot_exec_time(df)
    plot_efficiency(df)
    plot_phase_time(df)
    plot_stacked_bar(df)
    plot_active_count(df)
    print(f"\nPlots written to {PLOT_DIR}")


if __name__ == "__main__":
    main()
