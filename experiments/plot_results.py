import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

import config

# Set IEEE styling defaults
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "lines.linewidth": 1.8,
    "lines.markersize": 6
})


def generate_all_plots():
    data_path = config.BASE_DIR / "experiments" / "results" / "benchmark_data.json"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing benchmark data at {data_path}. Run run_benchmarks.py first.")

    with open(data_path, "r") as f:
        data = json.load(f)

    fig_dir = config.BASE_DIR / "paper" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    exp_dir = config.BASE_DIR / "experiments" / "results"
    exp_dir.mkdir(parents=True, exist_ok=True)

    def save_dual(fig, base_name):
        fig.tight_layout()
        fig.savefig(fig_dir / f"{base_name}.png", dpi=300)
        fig.savefig(fig_dir / f"{base_name}.pdf")
        fig.savefig(exp_dir / f"{base_name}.png", dpi=300)
        plt.close(fig)
        print(f"Saved: {base_name}.png & {base_name}.pdf")

    # =========================================================================
    # FIGURE 2: Dual Privacy Violations (Policy vs True Privacy Violations)
    # HEADLINE RESULT!
    # =========================================================================
    suite_b = data["suite_b_large_scale"]
    schedulers = list(suite_b.keys())
    # Take evaluation at N = 500 tasks
    idx = 2  # corresponding to N = 500
    n_tasks_eval = suite_b[schedulers[0]][idx]["num_tasks"]

    policy_viols = [suite_b[s][idx]["policy_violations_count"] for s in schedulers]
    true_viols = [suite_b[s][idx]["true_violations_count"] for s in schedulers]

    clean_names = ["Local-Only", "Edge-Only", "Cloud-Greedy", "PRISM-GA (Ours)"]
    x = np.arange(len(clean_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    rects1 = ax.bar(x - width/2, policy_viols, width, label="Policy Violations (Pred L2 $\\rightarrow$ Cloud)", color="#d62728", alpha=0.85)
    rects2 = ax.bar(x + width/2, true_viols, width, label="True Violations (True L2 $\\rightarrow$ Cloud)", color="#ff7f0e", alpha=0.85)

    ax.set_ylabel(f"Number of Task Violations (N = {n_tasks_eval})")
    ax.set_title("Dual Privacy Violation Analysis: Policy vs. End-to-End True Violations")
    ax.set_xticks(x)
    ax.set_xticklabels(clean_names)
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Add count labels on top of bars
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{int(h)}", xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{int(h)}", xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    save_dual(fig, "fig2_dual_privacy_violations")

    # =========================================================================
    # FIGURE 3: Optimality Gap Analysis (Exact ILP vs PRISM-GA)
    # =========================================================================
    suite_a = data["suite_a_ilp_vs_ga"]
    n_vals = [item["num_tasks"] for item in suite_a]
    ilp_costs = [item["ilp_cost"] for item in suite_a]
    ga_costs = [item["ga_cost"] for item in suite_a]
    ilp_times = [item["ilp_runtime_ms"] for item in suite_a]
    ga_times = [item["ga_runtime_ms"] for item in suite_a]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.8))

    # Left: Objective Cost Comparison
    ax1.plot(n_vals, ilp_costs, "o--", color="#2ca02c", label="Exact ILP ($J^*$ Optimum)")
    ax1.plot(n_vals, ga_costs, "s-", color="#1f77b4", label="PRISM-GA (Proposed)")
    ax1.set_xlabel("Task Batch Size ($N$)")
    ax1.set_ylabel("Total Objective Cost")
    ax1.set_title("(a) Solution Quality Comparison")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left")

    # Right: Algorithmic Runtime (Semi-log scale)
    ax2.plot(n_vals, ilp_times, "o--", color="#d62728", label="Exact ILP (Branch-and-Bound)")
    ax2.plot(n_vals, ga_times, "s-", color="#1f77b4", label="PRISM-GA Heuristic")
    ax2.set_yscale("log")
    ax2.set_xlabel("Task Batch Size ($N$)")
    ax2.set_ylabel("Solver Execution Time (ms) [Log Scale]")
    ax2.set_title("(b) Algorithmic Runtime Scaling")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left")

    save_dual(fig, "fig3_optimality_gap_ilp")

    # =========================================================================
    # FIGURE 4: Mobile Battery Energy vs. Latency Trade-off
    # =========================================================================
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    colors = {"Local-Only": "#7f7f7f", "Edge-Only": "#2ca02c", "Cloud-Greedy (Latency-First)": "#d62728", "PRISM-GA (Proposed Heuristic)": "#1f77b4"}
    markers = {"Local-Only": "^", "Edge-Only": "D", "Cloud-Greedy (Latency-First)": "x", "PRISM-GA (Proposed Heuristic)": "o"}

    for s_name in schedulers:
        series = suite_b[s_name]
        latencies = [item["avg_latency_ms"] for item in series]
        energies = [item["total_energy_j"] / item["num_tasks"] for item in series]  # Joules per task
        ax.plot(latencies, energies, marker=markers[s_name], color=colors[s_name], label=s_name, alpha=0.9)

    ax.set_xlabel("Average Execution Latency per Task (ms)")
    ax.set_ylabel("Average Mobile Battery Energy per Task (Joules)")
    ax.set_title("Pareto Trade-off: Mobile Battery Energy vs. Latency")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")

    save_dual(fig, "fig4_energy_latency_tradeoff")

    # =========================================================================
    # FIGURE 6: Weight Sensitivity Sweep (w_P from 0.0 to 1.0)
    # =========================================================================
    suite_c = data["suite_c_sensitivity"]
    wp_vals = [item["w_P"] for item in suite_c]
    local_p = [item["local_percent"] for item in suite_c]
    edge_p = [item["edge_percent"] for item in suite_c]
    cloud_p = [item["cloud_percent"] for item in suite_c]

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(wp_vals, local_p, "s-", color="#1f77b4", label="Local Device (%)")
    ax.plot(wp_vals, edge_p, "^-", color="#2ca02c", label="Edge MEC Server (%)")
    ax.plot(wp_vals, cloud_p, "v--", color="#d62728", label="Azure Public Cloud (%)")

    ax.set_xlabel("Privacy Objective Weight ($w_P$)")
    ax.set_ylabel("Tier Allocation Percentage (%)")
    ax.set_title("Workload Allocation Dynamics vs. Privacy Sensitivity Weight")
    ax.set_ylim(-2, 102)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="center right")

    save_dual(fig, "fig6_weight_sensitivity_sweep")

    # =========================================================================
    # FIGURE 7: Total System Cost Scaling (N = 100 to 1000)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    scale_n = [item["num_tasks"] for item in suite_b[schedulers[0]]]

    for s_name in schedulers:
        costs = [item["total_cost"] for item in suite_b[s_name]]
        ax.plot(scale_n, costs, marker=markers[s_name], color=colors[s_name], label=s_name)

    ax.set_xlabel("Task Stream Scale ($N$)")
    ax.set_ylabel("Total System Objective Cost ($J$)")
    ax.set_title("Total System Cost Scaling under High Workload")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left")

    save_dual(fig, "fig7_total_cost_scaling")

    print("\nAll publication figures successfully created!")


if __name__ == "__main__":
    generate_all_plots()

