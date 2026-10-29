# PRISM-MEC: Privacy-Aware Task Offloading and Multi-Objective Optimization in Mobile Edge Computing for Smart Healthcare

**Author:** Saksham Gupta  
*Department of Computer Science and Engineering, Cloud Computing Laboratory*  
**Repository:** [https://github.com/Saksham-Gupta-GH/privacy-aware-mec-offloading](https://github.com/Saksham-Gupta-GH/privacy-aware-mec-offloading)

---

## Abstract
Mobile Edge Computing (MEC) enables resource-constrained mobile and wearable health devices to offload computationally heavy tasks to edge and cloud infrastructures. However, conventional offloading schedulers focus exclusively on minimizing execution latency and device energy consumption, frequently offloading highly sensitive patient diagnostics (e.g., 12-lead electrocardiograms, brain MRI segmentations) to untrusted third-party public clouds, precipitating severe privacy vulnerabilities under regulations such as HIPAA and GDPR. In this paper, we propose **PRISM-MEC**, an end-to-end privacy-aware task offloading framework tailored for smart healthcare environments. PRISM-MEC incorporates a machine learning classifier that categorizes tasks into public telemetry ($L_0$), restricted operational data ($L_1$), and confidential protected health information ($L_2$). We formulate a multi-objective optimization problem that balances latency, mobile battery consumption, financial cloud metering, and privacy risk under strict deadlines and a hard policy constraint forbidding $L_2$ data from public cloud dispatch. To evaluate the framework, we solve the exact global optimum using Integer Linear Programming (ILP) and introduce a Genetic Algorithm (GA) equipped with a deterministic repair operator. We calibrate our simulation engine using empirical measurements gathered from an active Microsoft Azure cloud testbed. Benchmark results against Local-Only, Edge-Only, and Cloud-Greedy baselines demonstrate that PRISM-MEC completely eliminates policy privacy violations (0.0%), achieves an optimality gap under 0.75% relative to the exact ILP optimum, and reduces mobile energy consumption by 68.1% compared to local execution.

---

## I. Introduction
The proliferation of Internet of Medical Things (IoMT) devices—including wearable Holter monitors, continuous pulse oximeters, and portable ultrasound scanners—has enabled continuous remote patient monitoring. Despite their utility, mobile health (mHealth) devices are fundamentally constrained by compact form factors, limited battery capacity, and restricted on-device computational throughput.

To alleviate these hardware bottlenecks, Mobile Edge Computing (MEC) provides an elastic computing hierarchy comprising:
1. **Local Mobile Tier:** Immediate zero-network execution, total privacy containment, but severe battery depletion.
2. **MEC Edge Tier:** Low round-trip latency (~10 ms) and intermediate computing capacity within the trusted hospital intranet.
3. **Public Cloud Tier (e.g., Microsoft Azure):** Virtually infinite elastic compute resources accessible over wide-area networks (WAN), accompanied by financial metering and data privacy vulnerabilities.

Existing MEC scheduling literature predominantly treats task offloading as an engineering trade-off between execution delay and battery conservation. In smart healthcare, however, computation offloading intersects directly with patient confidentiality. Offloading raw biometric telemetry or diagnostic imaging to an untrusted public cloud risks violating statutory privacy mandates (e.g., HIPAA). Conversely, confining all computation to the local wearable causes excessive thermal throttling, battery exhaustion, and deadline violations for urgent diagnostic alarms.

---

## II. System Architecture & Mathematical Formulation

### 1. Task Model
Each task $t_i$ is represented by the 5-tuple:
$$t_i = \langle d_i, o_i, c_i, \tau_i^{\max}, P_i \rangle$$
- $d_i$: Input data payload size (KB)
- $o_i$: Output return payload size (KB)
- $c_i$: Computational complexity in MegaFLOPs (MFLOP)
- $\tau_i^{\max}$: Maximum permissible latency deadline (ms)
- $P_i \in \{0, 1, 2\}$: Privacy sensitivity level (Public, Restricted, or Critical PII)

### 2. Latency Model
- **Local:** $T_{i, \text{local}} = \frac{c_i \cdot 10^6}{f_{\text{local}} \cdot 10^9} \cdot 1000$ (ms)
- **Offload (Edge/Cloud):** $T_{i, j} = T_{i, j}^{\text{tx}} + T_{i, j}^{\text{exec}} + T_{i, j}^{\text{rx}} + \text{RTT}_j$

### 3. Mobile Device Battery Energy Model
- **Local:** $E_{i, \text{local}} = P_{\text{active}} \cdot \left(\frac{T_{i, \text{local}}}{1000}\right)$ (Joules)
- **Remote Offload:** $E_{i, j} = P_{\text{tx}} \cdot \left(\frac{T_{i, j}^{\text{tx}}}{1000}\right) + P_{\text{idle}} \cdot \left(\frac{T_{i, j}^{\text{exec}} + \text{RTT}_j}{1000}\right)$

### 4. Privacy Risk Formulation
$$R_{i, j} = S(P_i) \cdot \rho_j$$
Where:
- $S(L_0) = 0.05$ (Public Telemetry)
- $S(L_1) = 0.40$ (Operational Data)
- $S(L_2) = 1.00$ (Critical Protected Health Information)
- Tier Vulnerability: $\rho_{\text{local}} = 0.0$, $\rho_{\text{edge}} = 0.15$, $\rho_{\text{cloud}} = 0.95$

### 5. Multi-Objective Optimization Problem
$$\min_{\mathbf{X}} \sum_{i=1}^N \sum_{j} x_{i, j} \left[ w_L \frac{T_{i, j}}{T_{\text{norm}}} + w_E \frac{E_{i, j}}{E_{\text{norm}}} + w_C \frac{C_{i, j}}{C_{\text{norm}}} + w_P R_{i, j} \right] + \sum_{i=1}^N \Psi \cdot \mathbb{I}(T_i > \tau_i^{\max})$$
Subject to:
1. $\sum_{j} x_{i, j} = 1, \quad \forall i \in \{1, \dots, N\}$ (Single tier assignment)
2. **Hard Privacy Constraint:** $x_{i, \text{cloud}} = 0, \quad \forall i \text{ where } \hat{P}_i = 2$ (Critical PII strictly forbidden from cloud)

---

## III. Algorithmic Solutions

### 1. Exact Global Optimum via Integer Linear Programming (ILP)
Formulated in PuLP using binary assignment variables $x_{i, j} \in \{0, 1\}$ and solved via Branch-and-Cut (CBC solver). Provides the mathematically guaranteed lower bound $J^*$ for small-to-medium problem instances ($N \le 100$).

### 2. Proposed Genetic Algorithm (GA) with Hard Constraint Repair
- **Chromosome:** Integer vector of size $N$ where gene $\in \{0: \text{local}, 1: \text{edge}, 2: \text{cloud}\}$.
- **Deterministic Repair Operator:** For every gene $i$, if $\hat{P}_i = 2$ and gene $== 2$, the repair operator immediately remaps the assignment to $1$ (Edge).
- **Elitism & Selection:** Top 2 solutions preserved across generations; tournament selection ($k=3$) drives evolutionary progress.

---

## IV. Empirical Testbed Calibration (Microsoft Azure)

Parameters were measured against a live **Microsoft Azure Linux VM** (`20.219.130.205`):
- **Cloud WAN RTT:** Mean $64.20 \text{ ms}$ ($\sigma = 5.4 \text{ ms}$)
- **Uplink / Downlink Throughput:** $38.5 \text{ Mbps} / 82.0 \text{ Mbps}$
- **Cloud Compute Clock:** $4.1 \text{ GHz}$
- **Edge Node (Simulated):** $\text{RTT} = 12.0 \text{ ms}$, Uplink $= 100 \text{ Mbps}$, Clock $= 3.6 \text{ GHz}$
- **Local Mobile (Simulated):** Clock $= 1.8 \text{ GHz}$, Active Power $= 1.2 \text{ W}$, TX Power $= 0.8 \text{ W}$

---

## V. Experimental Results & Analysis

### 1. Privacy Classifier Performance
- **Accuracy (vs Labeled Data with 6% noise):** 97.08%
- **Weighted F1-Score:** 0.9710
- **Sensitivity on Critical PII ($L_2$):** 100.0%

### 2. Optimality Gap Analysis (Exact ILP vs. PRISM-GA)
| Tasks ($N$) | Exact ILP Cost ($J^*$) | ILP Runtime (ms) | PRISM-GA Cost | GA Runtime (ms) | Optimality Gap (%) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 10 | 5.778 | 51.4 | 5.778 | 70.9 | **0.00%** |
| 25 | 13.803 | 16.7 | 13.803 | 157.3 | **0.00%** |
| 50 | 17.874 | 17.0 | 17.877 | 301.6 | **0.01%** |
| 75 | 33.719 | 19.9 | 33.774 | 481.2 | **0.16%** |
| 100 | 49.535 | 19.9 | 49.900 | 593.8 | **0.74%** |

*Result:* PRISM-GA converges within **0.74% of the exact mathematical optimum** across all evaluated workloads.

### 3. Dual Privacy Violations ($N = 500$ Tasks)
| Algorithm | Policy Violations | True Violations | True Violation Rate (%) |
| :--- | :---: | :---: | :---: |
| **Local-Only** | 0 | 0 | 0.0% |
| **Edge-Only** | 0 | 0 | 0.0% |
| **Cloud-Greedy** | 169 | 169 | **100.0%** |
| **PRISM-GA (Ours)** | **0** | **0** | **0.0%** |

*Result:* Cloud-Greedy recklessly offloads every single heavy sensitive task to public Azure cloud. PRISM-GA achieves **zero policy violations and zero true violations**, fully respecting healthcare privacy regulations.

### 4. Energy & Latency Trade-Offs
- **Local-Only:** $204.0 \text{ ms}$ latency, $122.4 \text{ J}$ battery consumption (severe drain).
- **Edge-Only:** $175.5 \text{ ms}$ latency, $26.8 \text{ J}$ battery consumption.
- **PRISM-GA:** $181.8 \text{ ms}$ latency, $39.0 \text{ J}$ battery consumption (**68.1% energy reduction** vs local execution).

---

## VI. Conclusion
PRISM-MEC demonstrates that privacy awareness can be integrated into Mobile Edge Computing task offloading without sacrificing latency or battery efficiency. By combining natural language processing privacy classification, a multi-objective cost model, and a constraint-repaired genetic algorithm, PRISM-MEC guarantees 100% policy compliance while remaining within 0.74% of the true mathematical global optimum.
