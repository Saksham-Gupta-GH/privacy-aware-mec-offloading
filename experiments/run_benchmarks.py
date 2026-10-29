import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import json
import time
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

import config
from src.models.system_model import SystemModel
from src.simulator.discrete_event_sim import DiscreteEventSimulator
from src.optimizers.local_only import LocalOnlyScheduler
from src.optimizers.edge_only import EdgeOnlyScheduler
from src.optimizers.cloud_greedy import CloudGreedyScheduler
from src.optimizers.exact_ilp import ExactILPScheduler
from src.optimizers.genetic_algorithm import GeneticAlgorithmScheduler


def run_all_benchmarks():
    print("=" * 70)
    print("STARTING PRISM-MEC EXPERIMENTAL BENCHMARK SUITE")
    print("=" * 70)

    # Load calibrated system model parameters if available
    calib_file = config.BASE_DIR / "azure" / "calibrated_params.json"
    calib_params = None
    if calib_file.exists():
        with open(calib_file, "r") as f:
            calib_params = json.load(f)
        print(f"Loaded calibrated Azure parameters (WAN RTT = {calib_params.get('rtt_ms')} ms)")

    system_model = SystemModel(cloud_calibrated_params=calib_params)
    sim = DiscreteEventSimulator(system_model)

    results_dir = config.BASE_DIR / "experiments" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    benchmark_output = {}

    # =========================================================================
    # SUITE A: Exact ILP vs Proposed GA Optimality Gap Analysis (N <= 100)
    # =========================================================================
    print("\n[Suite A] Running Exact ILP vs PRISM-GA Optimality Gap Analysis...")
    ilp_sizes = [10, 25, 50, 75, 100]
    suite_a_results = []

    for n in ilp_sizes:
        print(f"  -> Testing instance size N = {n} tasks...")
        schedulers = [
            ExactILPScheduler(system_model, time_limit_sec=30),
            GeneticAlgorithmScheduler(system_model, pop_size=50, generations=60, random_seed=42)
        ]
        res = sim.run_benchmark(schedulers, n_tasks=n, seed=42 + n)

        ilp_res = res["Exact ILP (Global Optimum)"]
        ga_res = res["PRISM-GA (Proposed Heuristic)"]

        ilp_cost = ilp_res["total_objective_cost"]
        ga_cost = ga_res["total_objective_cost"]
        opt_gap = ((ga_cost - ilp_cost) / ilp_cost * 100.0) if ilp_cost > 0 else 0.0

        print(f"     ILP Cost: {ilp_cost:.3f} ({ilp_res['scheduler_runtime_ms']:.1f} ms) | "
              f"GA Cost: {ga_cost:.3f} ({ga_res['scheduler_runtime_ms']:.1f} ms) | "
              f"Gap: {opt_gap:.2f}%")

        suite_a_results.append({
            "num_tasks": n,
            "ilp_cost": round(ilp_cost, 4),
            "ilp_runtime_ms": round(ilp_res["scheduler_runtime_ms"], 2),
            "ga_cost": round(ga_cost, 4),
            "ga_runtime_ms": round(ga_res["scheduler_runtime_ms"], 2),
            "optimality_gap_percent": round(opt_gap, 2)
        })

    benchmark_output["suite_a_ilp_vs_ga"] = suite_a_results

    # =========================================================================
    # SUITE B: Large-Scale Benchmark Across All 4 Schedulers (N up to 1000)
    # =========================================================================
    print("\n[Suite B] Running Large-Scale Scaling Benchmark (N = 100 to 1000)...")
    scale_sizes = [100, 250, 500, 750, 1000]
    suite_b_results = {
        "Local-Only": [],
        "Edge-Only": [],
        "Cloud-Greedy (Latency-First)": [],
        "PRISM-GA (Proposed Heuristic)": []
    }

    large_schedulers = [
        LocalOnlyScheduler(system_model),
        EdgeOnlyScheduler(system_model),
        CloudGreedyScheduler(system_model),
        GeneticAlgorithmScheduler(system_model, pop_size=50, generations=50, random_seed=42)
    ]

    for n in scale_sizes:
        print(f"  -> Testing scale N = {n} tasks...")
        res = sim.run_benchmark(large_schedulers, n_tasks=n, seed=100 + n)
        for name, metrics in res.items():
            suite_b_results[name].append({
                "num_tasks": n,
                "total_cost": round(metrics["total_objective_cost"], 2),
                "avg_latency_ms": round(metrics["avg_latency_ms"], 2),
                "total_energy_j": round(metrics["total_energy_j"], 2),
                "total_cost_usd": round(metrics["total_cost_usd"], 5),
                "policy_violations_count": metrics["policy_violations_count"],
                "true_violations_count": metrics["true_violations_count"],
                "true_violation_rate": round(metrics["true_violation_rate"] * 100.0, 2),
                "deadline_miss_rate": round(metrics["deadline_miss_rate"] * 100.0, 2),
                "runtime_ms": round(metrics["scheduler_runtime_ms"], 2),
                "tier_distribution": metrics["tier_distribution"]
            })

    benchmark_output["suite_b_large_scale"] = suite_b_results

    # =========================================================================
    # SUITE C: Weight Sensitivity Analysis (Sweeping w_P from 0.0 to 1.0)
    # =========================================================================
    print("\n[Suite C] Running Privacy Weight Sensitivity Sweep (w_P in [0.0, 1.0])...")
    wp_values = [0.0, 0.1, 0.25, 0.40, 0.60, 0.80, 1.0]
    suite_c_results = []
    n_sweep = 250

    for wp in wp_values:
        # Normalize other weights proportionally
        remaining = 1.0 - wp
        custom_weights = {
            "w_L": round(remaining * 0.50, 4),
            "w_E": round(remaining * 0.40, 4),
            "w_C": round(remaining * 0.10, 4),
            "w_P": wp
        }
        sm_sweep = SystemModel(weights=custom_weights, cloud_calibrated_params=calib_params)
        sim_sweep = DiscreteEventSimulator(sm_sweep)
        ga_sweep = GeneticAlgorithmScheduler(sm_sweep, pop_size=40, generations=40, random_seed=42)

        res = sim_sweep.run_benchmark([ga_sweep], n_tasks=n_sweep, seed=777)
        m = res["PRISM-GA (Proposed Heuristic)"]
        tiers = m["tier_distribution"]
        total = sum(tiers.values())

        suite_c_results.append({
            "w_P": wp,
            "weights": custom_weights,
            "local_percent": round((tiers["local"] / total) * 100.0, 2),
            "edge_percent": round((tiers["edge"] / total) * 100.0, 2),
            "cloud_percent": round((tiers["cloud"] / total) * 100.0, 2),
            "policy_violations": m["policy_violations_count"],
            "true_violations": m["true_violations_count"],
            "avg_latency_ms": round(m["avg_latency_ms"], 2),
            "total_energy_j": round(m["total_energy_j"], 2)
        })

    benchmark_output["suite_c_sensitivity"] = suite_c_results

    # Save benchmark data
    out_file = results_dir / "benchmark_data.json"
    with open(out_file, "w") as f:
        json.dump(benchmark_output, f, indent=2)

    print("\n" + "=" * 70)
    print(f"BENCHMARKS COMPLETE! Results successfully saved to: {out_file}")
    print("=" * 70)

    return benchmark_output


if __name__ == "__main__":
    run_all_benchmarks()
