import random
from typing import List, Dict, Tuple
import config
from src.models.task import HealthcareTask
from src.optimizers.base_scheduler import BaseScheduler

TIER_MAP = {0: "local", 1: "edge", 2: "cloud"}
INV_TIER_MAP = {"local": 0, "edge": 1, "cloud": 2}


class GeneticAlgorithmScheduler(BaseScheduler):
    """
    Proposed Method: Multi-Objective Genetic Algorithm Scheduler with Hard Constraint Repair.
    
    Key Features:
    - Chromosome: Vector of size N with integer gene values in {0: local, 1: edge, 2: cloud}.
    - Repair Operator: Immediately detects if any task with predicted privacy Level 2 is
      assigned to Cloud (gene 2) and deterministically repairs it to Edge (gene 1) or Local (gene 0).
    - Elitism: Retains the top 2 solutions unchanged across generations.
    - Tournament Selection: Balances exploration and exploitation.
    """
    def __init__(
        self,
        system_model,
        pop_size: int = 50,
        generations: int = 50,
        crossover_rate: float = 0.85,
        mutation_rate: float = 0.05,
        tournament_size: int = 3,
        random_seed: int = 42
    ):
        super().__init__(system_model, name="PRISM-GA (Proposed Heuristic)")
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.random = random.Random(random_seed)

    def _repair_chromosome(self, chromosome: List[int], tasks: List[HealthcareTask]) -> List[int]:
        """
        Hard Constraint Repair Operator:
        Enforces policy compliance by preventing predicted Level 2 tasks from being assigned to Cloud (gene 2).
        If gene == 2 (Cloud) for an L2 task, repairs it to gene 1 (Edge).
        """
        for i, task in enumerate(tasks):
            pred_level = (
                task.predicted_privacy_level
                if task.predicted_privacy_level is not None
                else task.true_privacy_level
            )
            if pred_level == config.PRIVACY_LEVEL_CRITICAL and chromosome[i] == 2:
                # Deterministically repair to Edge
                chromosome[i] = 1
        return chromosome

    def _evaluate_chromosome(self, chromosome: List[int], tasks: List[HealthcareTask]) -> float:
        """
        Computes total objective cost of the chromosome. Lower is better.
        """
        total_cost = 0.0
        for i, task in enumerate(tasks):
            tier = TIER_MAP[chromosome[i]]
            eval_dict = self.system_model.evaluate_task(task, tier, use_predicted=True)
            total_cost += eval_dict["objective_value"]
        return total_cost

    def _tournament_select(self, population: List[List[int]], costs: List[float]) -> List[int]:
        """
        Tournament selection of size k.
        """
        selected_indices = self.random.sample(range(len(population)), self.tournament_size)
        best_idx = min(selected_indices, key=lambda idx: costs[idx])
        return list(population[best_idx])

    def _crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """
        Single-point crossover.
        """
        if len(parent1) <= 1 or self.random.random() > self.crossover_rate:
            return list(parent1), list(parent2)

        point = self.random.randint(1, len(parent1) - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2

    def _mutate(self, chromosome: List[int]) -> List[int]:
        """
        Uniform gene mutation.
        """
        for i in range(len(chromosome)):
            if self.random.random() < self.mutation_rate:
                chromosome[i] = self.random.choice([0, 1, 2])
        return chromosome

    def schedule(self, tasks: List[HealthcareTask]) -> Dict[int, str]:
        if not tasks:
            return {}

        n_tasks = len(tasks)

        # 1. Initialize population with diverse heuristic seeds + random individuals
        population = []

        # Seed 1: All Local
        population.append(self._repair_chromosome([0] * n_tasks, tasks))
        # Seed 2: All Edge
        population.append(self._repair_chromosome([1] * n_tasks, tasks))

        # Fill remaining population randomly and apply repair
        for _ in range(self.pop_size - len(population)):
            ind = [self.random.choice([0, 1, 2]) for _ in range(n_tasks)]
            ind = self._repair_chromosome(ind, tasks)
            population.append(ind)

        # Evaluate initial population
        costs = [self._evaluate_chromosome(ind, tasks) for ind in population]

        # Best tracker
        best_cost = min(costs)
        best_idx = costs.index(best_cost)
        best_chromosome = list(population[best_idx])

        # 2. Evolutionary Loop
        for gen in range(self.generations):
            new_population = []

            # Elitism: retain top 2 best solutions
            sorted_indices = sorted(range(len(population)), key=lambda k: costs[k])
            new_population.append(list(population[sorted_indices[0]]))
            new_population.append(list(population[sorted_indices[1]]))

            # Generate next generation
            while len(new_population) < self.pop_size:
                p1 = self._tournament_select(population, costs)
                p2 = self._tournament_select(population, costs)

                c1, c2 = self._crossover(p1, p2)
                c1 = self._repair_chromosome(self._mutate(c1), tasks)
                c2 = self._repair_chromosome(self._mutate(c2), tasks)

                new_population.append(c1)
                if len(new_population) < self.pop_size:
                    new_population.append(c2)

            population = new_population
            costs = [self._evaluate_chromosome(ind, tasks) for ind in population]

            current_min = min(costs)
            if current_min < best_cost:
                best_cost = current_min
                best_idx = costs.index(current_min)
                best_chromosome = list(population[best_idx])

        # Convert best chromosome into final assignment dictionary
        assignments = {}
        for i, task in enumerate(tasks):
            assignments[task.task_id] = TIER_MAP[best_chromosome[i]]

        return assignments

