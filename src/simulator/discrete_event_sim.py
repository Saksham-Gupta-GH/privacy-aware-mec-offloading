from typing import List, Dict, Any, Optional
import random
import pandas as pd
import numpy as np

import config
from src.models.task import HealthcareTask
from src.models.system_model import SystemModel
from src.classifier.privacy_scorer import PrivacyScorer
from src.simulator.metrics import MetricsTracker


class DiscreteEventSimulator:
    """
    Discrete-event simulator for evaluating MEC task offloading
    under Poisson arrival rates and dynamic network loads.
    """
    def __init__(self, system_model: SystemModel, privacy_scorer: Optional[PrivacyScorer] = None):
        self.system_model = system_model
        self.privacy_scorer = privacy_scorer or PrivacyScorer()
        self.metrics_tracker = MetricsTracker(system_model)
        self.dataset_df = pd.read_csv(config.BASE_DIR / "data" / "healthcare_tasks.csv")

    def sample_tasks(self, n_tasks: int, arrival_rate_lambda: float = 5.0, seed: int = 42) -> List[HealthcareTask]:
        """
        Samples n_tasks from the dataset and generates Poisson arrival times.
        Annotates tasks with ML predicted privacy levels.
        """
        rng = np.random.default_rng(seed)
        sampled_df = self.dataset_df.sample(n=min(n_tasks, len(self.dataset_df)), replace=(n_tasks > len(self.dataset_df)), random_state=seed)

        # Generate inter-arrival times using exponential distribution (Poisson process)
        inter_arrivals = rng.exponential(scale=1000.0 / arrival_rate_lambda, size=len(sampled_df))
        arrival_times = np.cumsum(inter_arrivals)

        tasks = []
        for i, (_, row) in enumerate(sampled_df.iterrows()):
            task = HealthcareTask(
                task_id=i,
                description=str(row["description"]),
                payload_size_kb=float(row["payload_size_kb"]),
                output_size_kb=float(row["output_size_kb"]),
                cpu_cycles_mflop=float(row["cpu_cycles_mflop"]),
                deadline_ms=float(row["deadline_ms"]),
                true_privacy_level=int(row["true_privacy_level"]),
                arrival_time_ms=float(arrival_times[i])
            )
            tasks.append(task)

        # Run ML Privacy Classifier batch annotation
        tasks = self.privacy_scorer.annotate_batch(tasks)
        return tasks

    def run_benchmark(self, schedulers: List[Any], n_tasks: int = 100, seed: int = 42) -> Dict[str, Any]:
        """
        Runs multiple schedulers against the exact same task sequence
        and returns comparative metrics.
        """
        tasks = self.sample_tasks(n_tasks=n_tasks, seed=seed)
        results = {}

        for scheduler in schedulers:
            # Make a clean copy of tasks
            task_copies = [
                HealthcareTask(
                    task_id=t.task_id,
                    description=t.description,
                    payload_size_kb=t.payload_size_kb,
                    output_size_kb=t.output_size_kb,
                    cpu_cycles_mflop=t.cpu_cycles_mflop,
                    deadline_ms=t.deadline_ms,
                    true_privacy_level=t.true_privacy_level,
                    predicted_privacy_level=t.predicted_privacy_level,
                    privacy_confidence=t.privacy_confidence,
                    arrival_time_ms=t.arrival_time_ms
                )
                for t in tasks
            ]

            assignments, runtime_ms = scheduler.execute_and_time(task_copies)
            metrics = self.metrics_tracker.evaluate_schedule(task_copies, assignments)
            metrics["scheduler_runtime_ms"] = runtime_ms
            metrics["scheduler_name"] = scheduler.name
            results[scheduler.name] = metrics

        return results
