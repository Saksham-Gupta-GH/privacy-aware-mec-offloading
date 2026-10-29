# PRISM-MEC: Privacy-Aware Task Offloading in Mobile Edge Computing for Smart Healthcare

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit Dashboard](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![Cloud: Azure](https://img.shields.io/badge/Cloud-Microsoft%20Azure-0078D4.svg)](https://azure.microsoft.com/)

A research-grade framework and IEEE conference paper implementation for **Privacy-Aware Multi-Objective Task Offloading in Mobile Edge Computing (MEC)** tailored to **Smart Healthcare**.

PRISM-MEC jointly balances **execution latency**, **mobile battery drain**, **financial cloud metering**, and **privacy risk** across a 3-tier hierarchy (Local Mobile, Edge Gateway, and Azure Public Cloud).

---

## 🏛️ System Architecture

```
                                  ┌───────────────────────────┐
                                  │   Healthcare Task Stream  │
                                  │ (ECG, MRI, Vitals, Sensor)│
                                  └─────────────┬─────────────┘
                                                │
                                  ┌─────────────▼─────────────┐
                                  │ 1. ML Privacy Classifier  │
                                  │  (TF-IDF + ML Predictor)  │
                                  │  Outputs: L0 / L1 / L2    │
                                  └─────────────┬─────────────┘
                                                │
                                  ┌─────────────▼─────────────┐
                                  │ 2. PRISM-GA Optimization  │
                                  │  Min J = wL·T + wE·E + ...│
                                  │  Hard Constraint Repair   │
                                  └───────┬───────────┬───────┘
                                          │           │
                   ┌──────────────────────┼───────────┴──────────────────────┐
                   │                      │                                  │
            [Decision: Local]      [Decision: Edge]                  [Decision: Cloud]
                   ▼                      ▼                                  ▼
           ┌───────────────┐      ┌───────────────┐                  ┌───────────────┐
           │ Local Device  │      │ MEC Edge Node │                  │  Azure Cloud  │
           │ (100% Private,│      │ (Hospital GW, │                  │ (Empirically  │
           │ High Battery) │      │ Low Latency)  │                  │  Calibrated)  │
           └───────────────┘      └───────────────┘                  └───────────────┘
```

---

## 🚀 Key Research Contributions

1. **Supervised ML Privacy Classifier:** Real-time natural language processing (TF-IDF + Classifier) categorizing clinical tasks into $L_0$ (Public), $L_1$ (Restricted), and $L_2$ (Critical PII) with **97.08% accuracy and 100% sensitivity on critical health records**.
2. **Exact Global Optimum Benchmark (ILP):** Formulated as an Integer Linear Program in `PuLP` using Branch-and-Cut, proving that the proposed Genetic Algorithm achieves within **0.74% of the true mathematical global optimum**.
3. **Genetic Algorithm with Deterministic Repair:** Guarantees **0.0% policy privacy violations** by deterministically remapping any sensitive $L_2$ data assigned to the public cloud back to trusted edge/local tiers.
4. **Empirical Azure Testbed Calibration:** Measures live WAN RTT (mean $64.2 \text{ ms}$), bandwidth, and compute speedup on an active Microsoft Azure Linux VM, ensuring simulation models reflect reality.
5. **Interactive Streamlit Web Dashboard:** Interactive UI with live objective weight sliders ($w_L, w_E, w_C, w_P$), visual pie charts, comparative benchmarks, and task stream inspection.
6. **Complete IEEE Conference Paper:** Publication-ready 6-page IEEE two-column paper source in LaTeX (`paper/prism_mec.tex`) with publication figures and verified bibliography.

---

## 📁 Repository Structure

```
Cloud_project/
├── .env.example                  # Environment configuration template
├── README.md                     # Documentation and setup instructions
├── requirements.txt              # Pinned Python package dependencies
├── config.py                     # Global constants, hardware specs, network parameters
├── data/
│   ├── healthcare_tasks.csv      # 1,200 synthetic clinical tasks with label noise
│   └── trained_models/           # Serialized TF-IDF vectorizer and classifier
├── src/
│   ├── classifier/
│   │   ├── dataset_generator.py  # Generates realistic clinical task dataset
│   │   ├── train_classifier.py   # Trains ML model, outputs F1 and confusion matrix
│   │   └── privacy_scorer.py     # Inference engine returning sensitivity tier
│   ├── models/
│   │   ├── task.py               # HealthcareTask dataclass
│   │   └── system_model.py       # Formal mathematical equations (latency, energy, risk)
│   ├── optimizers/
│   │   ├── base_scheduler.py     # Base abstract class for offloading schedulers
│   │   ├── local_only.py         # Baseline: All-Local
│   │   ├── edge_only.py          # Baseline: All-Edge
│   │   ├── cloud_greedy.py       # Baseline: Cloud-Greedy (latency-first, privacy-agnostic)
│   │   ├── exact_ilp.py          # Exact solver via PuLP (for N <= 100)
│   │   └── genetic_algorithm.py  # Proposed PRISM-GA with constraint repair
│   └── simulator/
│       ├── discrete_event_sim.py # Realistic Poisson task-arrival simulator
│       └── metrics.py            # Dual violation tracker & optimality gap
├── azure/
│   ├── server.py                 # FastAPI task worker with X-API-Key security
│   ├── calibrate_azure.py        # Micro-benchmark probe for real WAN RTT & CPU speed
│   ├── calibrated_params.json    # Calibrated parameters from Azure deployment
│   └── deploy_azure.sh           # SSH automated deployment script for Azure VM
├── experiments/
│   ├── run_benchmarks.py         # Comprehensive benchmark suite (N = 10 to 1000)
│   ├── plot_results.py           # Generates IEEE-standard vector publication plots
│   └── results/                  # Benchmark JSON telemetry and figures
├── dashboard/
│   └── app.py                    # Interactive Streamlit Demo Application
└── paper/
    ├── prism_mec.tex             # IEEE 2-column LaTeX source
    ├── PAPER.md                  # Complete formatted markdown version of paper
    ├── figures/                  # Publication-ready vector figures (.png & .pdf)
    └── references.bib            # Curated and verified IEEE literature citations
```

---

## ⚡ Quick Start Guide

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/Saksham-Gupta-GH/privacy-aware-mec-offloading.git
cd privacy-aware-mec-offloading

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables
```bash
cp .env.example .env
# Edit .env with your Azure credentials (if deploying to live cloud)
```

### 3. Generate Dataset & Train Privacy Classifier
```bash
python3 src/classifier/dataset_generator.py
python3 src/classifier/train_classifier.py
```
*Outputs: Accuracy: 97.08%, F1: 0.9710, and `paper/figures/fig1_confusion_matrix.png`.*

### 4. Calibrate against Microsoft Azure Cloud
```bash
python3 azure/calibrate_azure.py
```
*Captures WAN RTT, uplink/downlink throughput, and generates `fig5_empirical_calibration.png`.*

### 5. Run Experimental Benchmark Suite
```bash
python3 experiments/run_benchmarks.py
python3 experiments/plot_results.py
```
*Executes all suites (ILP optimality gap, large-scale scaling up to $N=1000$, and privacy weight sensitivity sweep).*

### 6. Launch Interactive Streamlit Dashboard
```bash
streamlit run dashboard/app.py
```
*Opens the interactive web application at `http://localhost:8501`.*

---

## ☁️ Deploying Cloud Worker on Microsoft Azure

To deploy the compute worker on your Azure VM:
```bash
# Ensure your SSH key is authorized on the Azure VM
chmod +x azure/deploy_azure.sh
./azure/deploy_azure.sh
```
*The service starts automatically in the background using FastAPI and Uvicorn on port 8000, secured via `X-API-Key`.*

---

## 📊 Summary of Experimental Results

| Metric | Local-Only | Edge-Only | Cloud-Greedy | **PRISM-GA (Ours)** |
| :--- | :---: | :---: | :---: | :---: |
| **Optimality Gap ($N \le 100$)** | +71.2% | +33.8% | +28.4% | **$\le$ 0.74% of ILP** |
| **Policy Privacy Violations ($N=500$)** | 0 | 0 | 169 (100%) | **0 (0.0%)** |
| **True Privacy Violations ($N=500$)** | 0 | 0 | 169 (100%) | **0 (0.0%)** |
| **Average Latency** | 204.0 ms | 175.5 ms | 161.9 ms | **181.8 ms** |
| **Mobile Battery Drain** | 122.4 J | 26.8 J | 38.0 J | **39.0 J (-68.1%)** |
| **Solver Runtime ($N=500$)** | < 1 ms | < 1 ms | < 1 ms | **184 ms** |

---

## 📜 Academic Research Paper
The full paper source is located in `paper/prism_mec.tex`. You can compile it using any standard LaTeX compiler:
```bash
cd paper
pdflatex prism_mec.tex
bibtex prism_mec
pdflatex prism_mec.tex
pdflatex prism_mec.tex
```
Or simply upload the `paper/` directory to [Overleaf](https://www.overleaf.com/) for one-click compilation.

---

## 👤 Author
**Saksham Gupta**  
GitHub: [@Saksham-Gupta-GH](https://github.com/Saksham-Gupta-GH)

