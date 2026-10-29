import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import json
import time
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import config
from src.models.system_model import SystemModel
from src.simulator.discrete_event_sim import DiscreteEventSimulator
from src.optimizers.local_only import LocalOnlyScheduler
from src.optimizers.edge_only import EdgeOnlyScheduler
from src.optimizers.cloud_greedy import CloudGreedyScheduler
from src.optimizers.exact_ilp import ExactILPScheduler
from src.optimizers.genetic_algorithm import GeneticAlgorithmScheduler

# Set Page Config
st.set_page_config(
    page_title="PRISM-MEC: Privacy-Aware Task Offloading",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.1rem; color: #4B5563; margin-bottom: 1.5rem; }
    .metric-card { background-color: #F8FAFC; border-radius: 8px; padding: 15px; border-left: 5px solid #3B82F6; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    .violation-card { background-color: #FEF2F2; border-radius: 8px; padding: 15px; border-left: 5px solid #EF4444; }
    .success-card { background-color: #ECFDF5; border-radius: 8px; padding: 15px; border-left: 5px solid #10B981; }
</style>
""", unsafe_allow_html=True)

# Title & Description
st.markdown('<div class="main-header">🏥 PRISM-MEC: Privacy-Aware Task Offloading</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">3-Tier Mobile Edge Computing Optimization (Local Mobile, Edge Gateway, Azure Cloud) for Smart Healthcare</div>', unsafe_allow_html=True)

# Load Azure Calibrated Parameters
calib_file = config.BASE_DIR / "azure" / "calibrated_params.json"
calib_data = {}
if calib_file.exists():
    with open(calib_file, "r") as f:
        calib_data = json.load(f)

# Sidebar: Controls & Weights
st.sidebar.header("⚙️ System Configuration")

st.sidebar.subheader("Objective Function Weights")
w_L = st.sidebar.slider("Execution Latency Weight ($w_L$)", 0.0, 1.0, 0.35, 0.05)
w_E = st.sidebar.slider("Mobile Battery Energy Weight ($w_E$)", 0.0, 1.0, 0.30, 0.05)
w_C = st.sidebar.slider("Financial Cloud Cost Weight ($w_C$)", 0.0, 1.0, 0.10, 0.05)
w_P = st.sidebar.slider("Privacy Risk Weight ($w_P$)", 0.0, 1.0, 0.25, 0.05)

# Weight normalization
total_w = w_L + w_E + w_C + w_P
if total_w > 0:
    weights = {
        "w_L": round(w_L / total_w, 3),
        "w_E": round(w_E / total_w, 3),
        "w_C": round(w_C / total_w, 3),
        "w_P": round(w_P / total_w, 3)
    }
else:
    weights = config.DEFAULT_WEIGHTS.copy()

st.sidebar.caption(f"Normalized Weights: $w_L$={weights['w_L']}, $w_E$={weights['w_E']}, $w_C$={weights['w_C']}, $w_P$={weights['w_P']}")

st.sidebar.subheader("Workload & Simulation")
n_tasks = st.sidebar.slider("Number of Healthcare Tasks ($N$)", min_value=10, max_value=250, value=50, step=10)
random_seed = st.sidebar.number_input("Random Seed", value=42, step=1)

# Initialize Model & Simulator
system_model = SystemModel(weights=weights, cloud_calibrated_params=calib_data)
sim = DiscreteEventSimulator(system_model)

# Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🚀 Live Offloading Engine",
    "📊 Comparative Benchmark",
    "☁️ Azure Testbed Calibration",
    "📄 Research Model & Formulations"
])

# =============================================================================
# TAB 1: Live Offloading Engine & Task Inspector
# =============================================================================
with tab1:
    st.subheader("Interactive Task Dispatcher")
    colA, colB = st.columns([3, 1])

    with colA:
        algo_choice = st.selectbox(
            "Select Offloading Algorithm:",
            [
                "PRISM-GA (Proposed Privacy-Aware Heuristic)",
                "Exact ILP (Global Optimum - PuLP)",
                "Cloud-Greedy (Latency-First, Privacy-Agnostic)",
                "Edge-Only Baseline",
                "Local-Only Baseline"
            ]
        )

    with colB:
        st.write("")
        st.write("")
        run_btn = st.button("⚡ Run Schedulers", use_container_width=True, type="primary")

    # Sample tasks and execute
    tasks = sim.sample_tasks(n_tasks=n_tasks, seed=int(random_seed))

    # Map algorithm choice to scheduler instance
    if "PRISM-GA" in algo_choice:
        scheduler = GeneticAlgorithmScheduler(system_model, pop_size=40, generations=40, random_seed=int(random_seed))
    elif "Exact ILP" in algo_choice:
        scheduler = ExactILPScheduler(system_model, time_limit_sec=15)
    elif "Cloud-Greedy" in algo_choice:
        scheduler = CloudGreedyScheduler(system_model)
    elif "Edge-Only" in algo_choice:
        scheduler = EdgeOnlyScheduler(system_model)
    else:
        scheduler = LocalOnlyScheduler(system_model)

    assignments, runtime_ms = scheduler.execute_and_time(tasks)
    metrics = sim.metrics_tracker.evaluate_schedule(tasks, assignments)

    # Top KPI Metrics Display
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("Avg Latency", f"{metrics['avg_latency_ms']:.1f} ms")
    kpi2.metric("Total Battery Drain", f"{metrics['total_energy_j']:.2f} J")
    kpi3.metric("Financial Cost", f"${metrics['total_cost_usd']:.4f}")
    
    # Violations metrics
    policy_v = metrics["policy_violations_count"]
    true_v = metrics["true_violations_count"]
    
    kpi4.metric("Policy Violations", f"{policy_v}", delta="0 (Compliant)" if policy_v == 0 else f"{policy_v} Leaks", delta_color="inverse")
    kpi5.metric("True Violations", f"{true_v}", delta=f"{metrics['true_violation_rate']*100:.1f}% rate", delta_color="inverse")

    st.markdown("---")

    # Tier Allocation Distribution Chart
    col_chart, col_details = st.columns([1, 1])

    with col_chart:
        st.subheader("Tier Allocation Breakdown")
        dist = metrics["tier_distribution"]
        labels = ["Local Device", "Edge Gateway", "Azure Cloud"]
        sizes = [dist.get("local", 0), dist.get("edge", 0), dist.get("cloud", 0)]
        colors = ["#3B82F6", "#10B981", "#EF4444"]

        fig, ax = plt.subplots(figsize=(5, 3.5))
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, autopct="%1.1f%%", colors=colors, startangle=140,
            textprops=dict(color="black", size=9)
        )
        ax.axis("equal")
        st.pyplot(fig)

    with col_details:
        st.subheader("Algorithm Diagnostics")
        st.write(f"**Scheduler**: `{scheduler.name}`")
        st.write(f"**Runtime Overhead**: `{runtime_ms:.2f} ms`")
        st.write(f"**Tasks Evaluated**: `{metrics['num_tasks']}`")
        st.write(f"**Deadline Miss Rate**: `{metrics['deadline_miss_rate']*100:.1f}%`")
        st.write(f"**Total Objective Cost ($J$)**: `{metrics['total_objective_cost']:.2f}`")

    st.subheader("📋 Detailed Task Stream Inspector")
    table_data = []
    tier_badges = {"local": "📱 Local", "edge": "⚡ Edge", "cloud": "☁️ Cloud"}
    level_names = {0: "L0: Public", 1: "L1: Restricted", 2: "L2: Critical PII"}

    for t in tasks:
        tier = assignments.get(t.task_id, "local")
        is_safe = not (t.true_privacy_level == 2 and tier == "cloud")
        table_data.append({
            "Task ID": f"#{t.task_id}",
            "Description": t.description[:65] + "..." if len(t.description) > 65 else t.description,
            "Compute (MFLOP)": t.cpu_cycles_mflop,
            "True Sensitivity": level_names.get(t.true_privacy_level),
            "ML Predicted": level_names.get(t.predicted_privacy_level),
            "Assigned Tier": tier_badges.get(tier, tier),
            "Compliance": "✅ Compliant" if is_safe else "🚨 PRIVACY VIOLATION"
        })

    st.dataframe(pd.DataFrame(table_data), use_container_width=True, height=280)


# =============================================================================
# TAB 2: Comparative Performance Benchmark
# =============================================================================
with tab2:
    st.subheader("Side-by-Side Benchmark across All Schedulers")
    st.write(f"Evaluating {n_tasks} tasks under identical Poisson arrival conditions.")

    if st.button("📊 Run Full Comparative Suite", type="primary"):
        bench_schedulers = [
            LocalOnlyScheduler(system_model),
            EdgeOnlyScheduler(system_model),
            CloudGreedyScheduler(system_model),
            GeneticAlgorithmScheduler(system_model, pop_size=40, generations=40, random_seed=int(random_seed))
        ]
        if n_tasks <= 100:
            bench_schedulers.append(ExactILPScheduler(system_model, time_limit_sec=15))

        res = sim.run_benchmark(bench_schedulers, n_tasks=n_tasks, seed=int(random_seed))

        # Comparison DataFrame
        comp_rows = []
        for name, m in res.items():
            comp_rows.append({
                "Scheduler": name,
                "Objective Cost (J)": round(m["total_objective_cost"], 2),
                "Avg Latency (ms)": round(m["avg_latency_ms"], 1),
                "Battery Drain (J)": round(m["total_energy_j"], 2),
                "Policy Violations": m["policy_violations_count"],
                "True Violations": m["true_violations_count"],
                "True Violation Rate (%)": round(m["true_violation_rate"] * 100.0, 1),
                "Runtime (ms)": round(m["scheduler_runtime_ms"], 1)
            })

        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True)

        # Plot Comparative Metrics
        col1, col2 = st.columns(2)
        sched_names = [r["Scheduler"].replace(" (Proposed Heuristic)", "").replace(" (Latency-First)", "") for r in comp_rows]

        with col1:
            fig, ax = plt.subplots(figsize=(6, 3.8))
            ax.bar(sched_names, [r["Policy Violations"] for r in comp_rows], color="#EF4444", alpha=0.85, label="Policy Violations")
            ax.bar(sched_names, [r["True Violations"] for r in comp_rows], color="#F59E0B", alpha=0.6, label="True Violations")
            ax.set_title("Privacy Violations Comparison (Lower is Better)")
            ax.set_ylabel("Violations Count")
            ax.legend()
            plt.xticks(rotation=15, ha="right")
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots(figsize=(6, 3.8))
            ax.bar(sched_names, [r["Battery Drain (J)"] for r in comp_rows], color="#3B82F6", alpha=0.85)
            ax.set_title("Total Mobile Battery Energy Drain (Joules)")
            ax.set_ylabel("Energy (J)")
            plt.xticks(rotation=15, ha="right")
            st.pyplot(fig)


# =============================================================================
# TAB 3: Azure Testbed Calibration
# =============================================================================
with tab3:
    st.subheader("Empirical Cloud Infrastructure & Telemetry")
    st.write("Real-world parameters captured from the Azure Cloud Worker node.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Azure Host", calib_data.get("azure_host", config.AZURE_HOST))
    c2.metric("Mean WAN RTT", f"{calib_data.get('rtt_ms', config.CLOUD_DEFAULT_RTT_MS)} ms")
    c3.metric("Uplink Bandwidth", f"{calib_data.get('uplink_mbps', config.CLOUD_DEFAULT_UPLINK_MBPS)} Mbps")
    c4.metric("Cloud CPU Spec", f"{calib_data.get('cpu_freq_ghz', config.CLOUD_CPU_FREQ_GHZ)} GHz")

    st.markdown("---")
    st.write("#### Empirical Latency Calibration Plot")
    calib_plot_path = config.BASE_DIR / "paper" / "figures" / "fig5_empirical_calibration.png"
    if calib_plot_path.exists():
        st.image(str(calib_plot_path), caption="Measured Azure WAN RTT Distribution vs. Simulated Edge MEC Latency", use_container_width=True)
    else:
        st.info("Run `azure/calibrate_azure.py` to generate the live calibration plot.")


# =============================================================================
# TAB 4: Research Paper & Theoretical Formulation
# =============================================================================
with tab4:
    st.subheader("Mathematical System Formulation (IEEE Paper Preview)")
    st.markdown(r"""
    ### 1. Multi-Objective Cost Formulation
    $$\min_{x_i \in \{\text{Local, Edge, Cloud}\}} \sum_{i=1}^N \left[ w_L \frac{T_i(x_i)}{T_{\text{norm}}} + w_E \frac{E_i(x_i)}{E_{\text{norm}}} + w_C \frac{C_i(x_i)}{C_{\text{norm}}} + w_P R_i(x_i) \right]$$

    ### 2. Strict Hard Privacy Constraint
    $$x_{i, \text{cloud}} = 0, \quad \forall i \in \{1, \dots, N\} \text{ where } \hat{P}_i = \text{Level 2 (Critical PII)}$$

    ### 3. Latency & Energy Equations
    - **Local Latency & Energy**:
      $$T_{i, \text{local}} = \frac{c_i}{f_{\text{local}}}, \quad E_{i, \text{local}} = P_{\text{active}} \cdot T_{i, \text{local}}$$
    - **Edge / Cloud Latency & Mobile Energy**:
      $$T_{i, \text{tx}} = \frac{d_i}{B_{\text{up}}}, \quad T_{i, \text{exec}} = \frac{c_i}{f_{\text{remote}}}, \quad T_{i, \text{rx}} = \frac{o_i}{B_{\text{down}}}$$
      $$E_{i, \text{offload}} = P_{\text{tx}} \cdot T_{i, \text{tx}} + P_{\text{idle}} \cdot (T_{i, \text{exec}} + \text{RTT})$$
    """)

    cm_path = config.BASE_DIR / "paper" / "figures" / "fig1_confusion_matrix.png"
    if cm_path.exists():
        st.image(str(cm_path), caption="Figure 1: Machine Learning Privacy Classifier Confusion Matrix", width=450)
