from typing import List, Dict
from src.models.task import HealthcareTask
from src.optimizers.base_scheduler import BaseScheduler


class LocalOnlyScheduler(BaseScheduler):
    """
    Baseline 1: All-Local Execution
    Executes every task on the local mobile device.
    Guarantees zero privacy leakage and zero network latency,
    at the cost of heavy mobile battery depletion and potential deadline misses.
    """
    def __init__(self, system_model):
        super().__init__(system_model, name="Local-Only")

    def schedule(self, tasks: List[HealthcareTask]) -> Dict[int, str]:
        return {task.task_id: "local" for task in tasks}

