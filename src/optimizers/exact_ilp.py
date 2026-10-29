from typing import List, Dict
import pulp
import config
from src.models.task import HealthcareTask
from src.optimizers.base_scheduler import BaseScheduler


class ExactILPScheduler(BaseScheduler):
    """
    Exact Global Optimum Solver via Integer Linear Programming (PuLP / CBC).
    Finds the provably optimal offloading assignment J* for small-to-medium
    task batches (N <= 100) subject to hard privacy constraints:
      x_{i, 'cloud'} == 0 for all tasks with predicted privacy Level 2.
    """
    def __init__(self, system_model, time_limit_sec: int = 15):
        super().__init__(system_model, name="Exact ILP (Global Optimum)")
        self.time_limit_sec = time_limit_sec

    def schedule(self, tasks: List[HealthcareTask]) -> Dict[int, str]:
        if not tasks:
            return {}

        tiers = ["local", "edge", "cloud"]
        prob = pulp.LpProblem("PRISM_MEC_Exact_Optimization", pulp.LpMinimize)

        # Binary decision variables x[i, tier]
        x = {}
        for task in tasks:
            for tier in tiers:
                x[task.task_id, tier] = pulp.LpVariable(
                    f"x_{task.task_id}_{tier}", cat=pulp.LpBinary
                )

        # Constraint 1: Exactly one tier per task
        for task in tasks:
            prob += (
                pulp.lpSum([x[task.task_id, tier] for tier in tiers]) == 1,
                f"AssignOneTier_{task.task_id}"
            )

        # Constraint 2: Hard Privacy Policy Constraint
        # Predicted Level 2 (Critical) tasks are strictly barred from Cloud
        for task in tasks:
            pred_level = (
                task.predicted_privacy_level
                if task.predicted_privacy_level is not None
                else task.true_privacy_level
            )
            if pred_level == config.PRIVACY_LEVEL_CRITICAL:
                prob += (
                    x[task.task_id, "cloud"] == 0,
                    f"HardPrivacyConstraint_{task.task_id}"
                )

        # Objective Function: Sum of weighted objective costs
        obj_terms = []
        for task in tasks:
            for tier in tiers:
                eval_dict = self.system_model.evaluate_task(task, tier, use_predicted=True)
                # Raw cost without hard constraint penalty (since constraint is enforced directly in ILP)
                w_L = self.system_model.weights["w_L"]
                w_E = self.system_model.weights["w_E"]
                w_C = self.system_model.weights["w_C"]
                w_P = self.system_model.weights["w_P"]

                base_cost = (
                    w_L * (eval_dict["latency_ms"] / self.system_model.norm_latency_ms) +
                    w_E * (eval_dict["energy_j"] / self.system_model.norm_energy_j) +
                    w_C * (eval_dict["cost_usd"] / self.system_model.norm_cost) +
                    w_P * eval_dict["privacy_risk"]
                )
                if eval_dict["missed_deadline"]:
                    base_cost += 2.0  # Penalty for deadline miss

                obj_terms.append(base_cost * x[task.task_id, tier])

        prob += pulp.lpSum(obj_terms)

        # Solve silently with CBC
        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=self.time_limit_sec)
        prob.solve(solver)

        assignments = {}
        for task in tasks:
            assigned_tier = "local"
            for tier in tiers:
                val = pulp.value(x[task.task_id, tier])
                if val is not None and val > 0.5:
                    assigned_tier = tier
                    break
            assignments[task.task_id] = assigned_tier

        return assignments

