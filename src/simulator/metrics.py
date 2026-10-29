from typing import List, Dict, Any
import numpy as np
from src.models.task import HealthcareTask
from src.models.system_model import SystemModel
import config


class MetricsTracker:
    """
    Computes rigorous performance and privacy violation metrics
    for evaluation and comparison in the IEEE research paper.
    """
    def __init__(self, system_model: SystemModel):
        self.system_model = system_model

    def evaluate_schedule(
        self,
        tasks: List[HealthcareTask],
        assignments: Dict[int, str]
    ) -> Dict[str, Any]:
        """
        Takes tasks and assignments dictionary {task_id: tier}
        and returns a comprehensive evaluation dictionary.
        """
        if not tasks:
            return {}

        n_tasks = len(tasks)
        latencies = []
        energies = []
        costs = []
        privacy_risks = []
        objective_values = []
        deadline_misses = 0

        policy_violations = 0
        true_violations = 0
        total_true_l2_tasks = 0

        tier_counts = {"local": 0, "edge": 0, "cloud": 0}

        for task in tasks:
            tier = assignments.get(task.task_id, "local")
            task.scheduled_tier = tier
            tier_counts[tier] = tier_counts.get(tier, 0) + 1

            eval_dict = self.system_model.evaluate_task(task, tier, use_predicted=True)
            latencies.append(eval_dict["latency_ms"])
            energies.append(eval_dict["energy_j"])
            costs.append(eval_dict["cost_usd"])
            privacy_risks.append(eval_dict["privacy_risk"])
            objective_values.append(eval_dict["objective_value"])

            if eval_dict["missed_deadline"]:
                deadline_misses += 1

            # Policy violation: scheduler violated its own policy rule
            if task.is_policy_violation():
                policy_violations += 1

            # True violation: genuine sensitive L2 data reached public cloud
            if task.true_privacy_level == config.PRIVACY_LEVEL_CRITICAL:
                total_true_l2_tasks += 1
                if task.is_true_violation():
                    true_violations += 1

        true_violation_rate = (true_violations / total_true_l2_tasks) if total_true_l2_tasks > 0 else 0.0

        return {
            "num_tasks": n_tasks,
            "avg_latency_ms": float(np.mean(latencies)),
            "max_latency_ms": float(np.max(latencies)),
            "total_latency_ms": float(np.sum(latencies)),
            "total_energy_j": float(np.sum(energies)),
            "avg_energy_j": float(np.mean(energies)),
            "total_cost_usd": float(np.sum(costs)),
            "avg_privacy_risk": float(np.mean(privacy_risks)),
            "total_objective_cost": float(np.sum(objective_values)),
            "deadline_misses_count": deadline_misses,
            "deadline_miss_rate": deadline_misses / n_tasks,
            "policy_violations_count": policy_violations,
            "policy_violation_rate": policy_violations / n_tasks,
            "true_violations_count": true_violations,
            "true_violation_rate": true_violation_rate,
            "total_true_l2_tasks": total_true_l2_tasks,
            "tier_distribution": tier_counts
        }
