from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
import time
from src.models.task import HealthcareTask
from src.models.system_model import SystemModel


class BaseScheduler(ABC):
    """
    Abstract base class for all MEC offloading scheduling algorithms.
    """
    def __init__(self, system_model: SystemModel, name: str = "BaseScheduler"):
        self.system_model = system_model
        self.name = name

    @abstractmethod
    def schedule(self, tasks: List[HealthcareTask]) -> Dict[int, str]:
        """
        Takes a list of HealthcareTasks and returns an assignment dictionary:
        {task_id: "local" | "edge" | "cloud"}
        """
        pass

    def execute_and_time(self, tasks: List[HealthcareTask]) -> Tuple[Dict[int, str], float]:
        """
        Executes scheduling decision and measures algorithmic runtime overhead.
        Returns: (assignments, runtime_ms)
        """
        start = time.perf_counter()
        assignments = self.schedule(tasks)
        runtime_ms = (time.perf_counter() - start) * 1000.0
        return assignments, runtime_ms
