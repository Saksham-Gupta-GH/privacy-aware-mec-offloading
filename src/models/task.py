from dataclasses import dataclass
from typing import Optional


@dataclass
class HealthcareTask:
    """
    Represents a discrete healthcare computation task in the MEC system.
    """
    task_id: int
    description: str
    payload_size_kb: float          # Input data size in Kilobytes (KB)
    output_size_kb: float           # Output result size in Kilobytes (KB)
    cpu_cycles_mflop: float         # Compute requirement in MegaFLOPs (MFLOP)
    deadline_ms: float              # Latency deadline in milliseconds (ms)
    true_privacy_level: int         # Ground truth sensitivity: 0=Public, 1=Restricted, 2=Critical
    predicted_privacy_level: Optional[int] = None   # Model prediction: 0, 1, or 2
    privacy_confidence: float = 1.0 # Prediction confidence score [0.0, 1.0]
    arrival_time_ms: float = 0.0    # Task arrival timestamp (ms)
    scheduled_tier: Optional[str] = None  # "local", "edge", or "cloud"

    def is_true_violation(self) -> bool:
        """
        True privacy violation occurs when a genuinely Critical task (L2)
        is dispatched to an untrusted public Cloud tier.
        Evaluates end-to-end vulnerability including ML misclassification errors.
        """
        return self.true_privacy_level == 2 and self.scheduled_tier == "cloud"

    def is_policy_violation(self) -> bool:
        """
        Policy violation occurs when a scheduler violates the explicit policy
        rule by sending a task it PREDICTED as Level 2 to Cloud.
        """
        return self.predicted_privacy_level == 2 and self.scheduled_tier == "cloud"

    def is_deadline_missed(self, total_latency_ms: float) -> bool:
        """Returns True if the task failed to finish before its deadline."""
        return total_latency_ms > self.deadline_ms
