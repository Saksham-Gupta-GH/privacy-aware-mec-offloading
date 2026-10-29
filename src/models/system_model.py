from typing import Dict, Tuple
import config
from src.models.task import HealthcareTask


class SystemModel:
    """
    Formal mathematical system model for 3-tier MEC offloading:
    Tiers:
      - 'local': Mobile Handheld / Wearable Device
      - 'edge': MEC Edge Server (Hospital Ward Gateway)
      - 'cloud': Azure Public Cloud VM
    """

    def __init__(self, weights: Dict[str, float] = None, cloud_calibrated_params: Dict = None):
        self.weights = weights or config.DEFAULT_WEIGHTS.copy()
        
        # Calibration overrides (if measured against real Azure VM)
        self.cloud_rtt_ms = config.CLOUD_DEFAULT_RTT_MS
        self.cloud_uplink_mbps = config.CLOUD_DEFAULT_UPLINK_MBPS
        self.cloud_downlink_mbps = config.CLOUD_DEFAULT_DOWNLINK_MBPS
        self.cloud_cpu_freq_ghz = config.CLOUD_CPU_FREQ_GHZ

        if cloud_calibrated_params:
            self.cloud_rtt_ms = cloud_calibrated_params.get("rtt_ms", self.cloud_rtt_ms)
            self.cloud_uplink_mbps = cloud_calibrated_params.get("uplink_mbps", self.cloud_uplink_mbps)
            self.cloud_downlink_mbps = cloud_calibrated_params.get("downlink_mbps", self.cloud_downlink_mbps)
            self.cloud_cpu_freq_ghz = cloud_calibrated_params.get("cpu_freq_ghz", self.cloud_cpu_freq_ghz)

        # Sensitivity coefficients for privacy levels
        self.privacy_sensitivity = {
            config.PRIVACY_LEVEL_PUBLIC: 0.05,
            config.PRIVACY_LEVEL_RESTRICTED: 0.40,
            config.PRIVACY_LEVEL_CRITICAL: 1.00
        }

        # Tier vulnerability factors (0=completely secure, 1=untrusted)
        self.tier_vulnerability = {
            "local": 0.0,
            "edge": config.EDGE_PRIVACY_RISK_FACTOR,
            "cloud": config.CLOUD_PRIVACY_RISK_FACTOR
        }

        # Normalization constants for objective balancing
        self.norm_latency_ms = 500.0   # Reference latency scale
        self.norm_energy_j = 0.50      # Reference energy scale
        self.norm_cost = 0.005         # Reference cost scale

    def compute_latency(self, task: HealthcareTask, tier: str) -> float:
        """
        Computes total completion latency in milliseconds (ms).
        Includes network transmission, RTT, and remote/local execution.
        """
        # Execution time: cycles / (freq * IPC), IPC assumed 1.0
        if tier == "local":
            exec_time_sec = (task.cpu_cycles_mflop * 1e6) / (config.LOCAL_CPU_FREQ_GHZ * 1e9)
            return exec_time_sec * 1000.0

        elif tier == "edge":
            tx_time_sec = (task.payload_size_kb * 8.0) / (config.EDGE_UPLINK_MBPS * 1e3)
            rx_time_sec = (task.output_size_kb * 8.0) / (config.EDGE_DOWNLINK_MBPS * 1e3)
            exec_time_sec = (task.cpu_cycles_mflop * 1e6) / (config.EDGE_CPU_FREQ_GHZ * 1e9)
            rtt_sec = config.EDGE_RTT_MS / 1000.0
            return (tx_time_sec + rx_time_sec + exec_time_sec + rtt_sec) * 1000.0

        elif tier == "cloud":
            tx_time_sec = (task.payload_size_kb * 8.0) / (self.cloud_uplink_mbps * 1e3)
            rx_time_sec = (task.output_size_kb * 8.0) / (self.cloud_downlink_mbps * 1e3)
            exec_time_sec = (task.cpu_cycles_mflop * 1e6) / (self.cloud_cpu_freq_ghz * 1e9)
            rtt_sec = self.cloud_rtt_ms / 1000.0
            return (tx_time_sec + rx_time_sec + exec_time_sec + rtt_sec) * 1000.0

        else:
            raise ValueError(f"Unknown tier: {tier}")

    def compute_mobile_energy(self, task: HealthcareTask, tier: str) -> float:
        """
        Computes energy consumed strictly by the MOBILE DEVICE in Joules (J).
        Demonstrates the energy conservation achieved by offloading.
        """
        if tier == "local":
            exec_time_sec = (task.cpu_cycles_mflop * 1e6) / (config.LOCAL_CPU_FREQ_GHZ * 1e9)
            return config.LOCAL_POWER_ACTIVE_W * exec_time_sec

        elif tier == "edge":
            tx_time_sec = (task.payload_size_kb * 8.0) / (config.EDGE_UPLINK_MBPS * 1e3)
            idle_time_sec = ((task.cpu_cycles_mflop * 1e6) / (config.EDGE_CPU_FREQ_GHZ * 1e9)) + (config.EDGE_RTT_MS / 1000.0)
            # Mobile device consumes active TX power during uplink, minimal idle power while awaiting results
            return (config.LOCAL_POWER_TX_W * tx_time_sec) + (0.05 * idle_time_sec)

        elif tier == "cloud":
            tx_time_sec = (task.payload_size_kb * 8.0) / (self.cloud_uplink_mbps * 1e3)
            idle_time_sec = ((task.cpu_cycles_mflop * 1e6) / (self.cloud_cpu_freq_ghz * 1e9)) + (self.cloud_rtt_ms / 1000.0)
            return (config.LOCAL_POWER_TX_W * tx_time_sec) + (0.05 * idle_time_sec)

        else:
            raise ValueError(f"Unknown tier: {tier}")

    def compute_financial_cost(self, task: HealthcareTask, tier: str) -> float:
        """
        Computes monetary cost of executing the task in USD ($).
        Local & Edge are zero-cost; Cloud is metered.
        """
        if tier in ("local", "edge"):
            return 0.0
        elif tier == "cloud":
            return task.cpu_cycles_mflop * config.CLOUD_FINANCIAL_COST_PER_MFLOP
        else:
            raise ValueError(f"Unknown tier: {tier}")

    def compute_privacy_risk(self, task: HealthcareTask, tier: str, use_predicted: bool = True) -> float:
        """
        Computes privacy risk score R_i in [0.0, 1.0].
        R_i = Sensitivity(P_i) * Vulnerability(Tier)
        """
        level = task.predicted_privacy_level if use_predicted and task.predicted_privacy_level is not None else task.true_privacy_level
        sensitivity = self.privacy_sensitivity.get(level, 0.5)
        vulnerability = self.tier_vulnerability[tier]
        return sensitivity * vulnerability

    def evaluate_task(self, task: HealthcareTask, tier: str, use_predicted: bool = True) -> Dict[str, float]:
        """
        Computes complete metric breakdown for an assignment of task to tier.
        """
        latency_ms = self.compute_latency(task, tier)
        energy_j = self.compute_mobile_energy(task, tier)
        cost_usd = self.compute_financial_cost(task, tier)
        privacy_risk = self.compute_privacy_risk(task, tier, use_predicted=use_predicted)
        
        # Hard constraint check: Predicted Level 2 forbidden on Cloud
        pred_level = task.predicted_privacy_level if task.predicted_privacy_level is not None else task.true_privacy_level
        is_hard_violation = (pred_level == config.PRIVACY_LEVEL_CRITICAL and tier == "cloud")
        
        # Deadline violation check
        missed_deadline = task.is_deadline_missed(latency_ms)

        # Weighted objective calculation
        w_L = self.weights["w_L"]
        w_E = self.weights["w_E"]
        w_C = self.weights["w_C"]
        w_P = self.weights["w_P"]

        cost_term = (
            w_L * (latency_ms / self.norm_latency_ms) +
            w_E * (energy_j / self.norm_energy_j) +
            w_C * (cost_usd / self.norm_cost) +
            w_P * privacy_risk
        )

        # Penalty additions
        if missed_deadline:
            cost_term += 2.0  # Significant penalty for failing deadline
        if is_hard_violation:
            cost_term += 100.0  # Extreme penalty for violating privacy policy

        return {
            "tier": tier,
            "latency_ms": latency_ms,
            "energy_j": energy_j,
            "cost_usd": cost_usd,
            "privacy_risk": privacy_risk,
            "missed_deadline": missed_deadline,
            "is_hard_violation": is_hard_violation,
            "objective_value": cost_term
        }
