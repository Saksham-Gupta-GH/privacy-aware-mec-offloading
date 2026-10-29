from typing import List, Dict
from src.models.task import HealthcareTask
from src.optimizers.base_scheduler import BaseScheduler


class EdgeOnlyScheduler(BaseScheduler):
    """
    Baseline 2: All-Edge Execution
    Offloads every incoming task to the near-premises MEC Edge Server.
    Achieves low RTT and zero cloud financial charge, but is constrained
    by edge server queue limits and wireless channel capacity.
    """
    def __init__(self, system_model):
        super().__init__(system_model, name="Edge-Only")

    def schedule(self, tasks: List[HealthcareTask]) -> Dict[int, str]:
        return {task.task_id: "edge" for task in tasks}

