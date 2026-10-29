from typing import List, Dict
import config
from src.models.task import HealthcareTask
from src.optimizers.base_scheduler import BaseScheduler


class CloudGreedyScheduler(BaseScheduler):
    """
    Baseline 3: Cloud-Greedy Offloading (Latency/Compute-First, Privacy-Agnostic)
    
    Simulates traditional mobile cloud computing heuristics that offload
    compute-heavy workloads to high-capacity cloud infrastructure to minimize
    mobile computation strain and device battery depletion, without any
    regard for data privacy classification or HIPAA compliance.
    
    Decision Rule:
      - Heavy workloads (CPU cycles >= 120 MFLOPs) -> Dispatched to Azure Cloud
      - Medium workloads (50 <= CPU cycles < 120 MFLOPs) -> Dispatched to Edge MEC
      - Light workloads (< 50 MFLOPs) -> Retained on Local Device
    """
    def __init__(self, system_model, heavy_threshold_mflop: float = 120.0, medium_threshold_mflop: float = 50.0):
        super().__init__(system_model, name="Cloud-Greedy (Latency-First)")
        self.heavy_threshold_mflop = heavy_threshold_mflop
        self.medium_threshold_mflop = medium_threshold_mflop

    def schedule(self, tasks: List[HealthcareTask]) -> Dict[int, str]:
        assignments = {}
        for task in tasks:
            # Privacy-agnostic heuristic based strictly on computational intensity
            if task.cpu_cycles_mflop >= self.heavy_threshold_mflop:
                assignments[task.task_id] = "cloud"
            elif task.cpu_cycles_mflop >= self.medium_threshold_mflop:
                assignments[task.task_id] = "edge"
            else:
                assignments[task.task_id] = "local"

        return assignments
