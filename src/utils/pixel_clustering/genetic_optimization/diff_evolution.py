from typing import Callable, Optional, Tuple

import torch
from torch import Tensor
from tqdm import tqdm


class DifferentialEvolution:
    def __init__(
        self,
        objective_func: Callable[[Tensor], float],
        bounds: list,
        mutation_strategy: str = "default",
        pop_size: int = 100,
        mutation_factor: float = 0.8,
        crossover_factor: float = 0.7,
        max_generations: int = 1000,
        tol: float = 1e-6,
        post_transforms: Optional[Callable[[Tensor], Tensor]] = None,
    ) -> None:
        self._objective_func = objective_func
        self._bounds = torch.Tensor(bounds)
        self._pop_size = pop_size
        self._F = mutation_factor
        self._CR = crossover_factor
        self._max_generations = max_generations
        self._tol = tol
        self._n_dims = self._bounds.shape[0]
        self._post_transforms = post_transforms
        self._lower_bounds, self._upper_bounds = self._bounds[:, 0], self._bounds[:, 1]
        self._population = self._initialize_population()
        self._fitness = self._evaluate_population(self._population)

        # Define available mutation strategies
        self._mutation_strategies = {"default": self._default_mutate}

        if mutation_strategy in self._mutation_strategies:
            self._mutate = self._mutation_strategies[mutation_strategy]
        else:
            raise ValueError(f"Unknown mutation strategy: {mutation_strategy}")

    def _initialize_population(self) -> Tensor:
        initialization = (
            torch.rand((self._pop_size, self._n_dims)) * (self._upper_bounds - self._lower_bounds)
            + self._lower_bounds
        )
        initialization = self._post_transforms(initialization)
        return initialization

    def _evaluate_population(self, population: Tensor) -> Tensor:
        return torch.tensor([self._objective_func(ind) for ind in population])

    def _default_mutate(self, i: int) -> Tuple[Tensor, float]:
        a, b, c = self._select_three_random_individuals(i)
        mutant = a + self._F * (b - c)
        cross_points = torch.rand(self._n_dims) < self._CR
        trial = torch.where(cross_points, mutant, self._population[i, :])

        if self._post_transforms is not None:
            trial = self._post_transforms(trial)

        trial_fitness = self._objective_func(trial)
        return trial, trial_fitness

    def optimize(self) -> Tuple[Tensor, float]:
        best_idx = torch.argmin(self._fitness)
        best_fitness = self._fitness[best_idx]
        best_individual = self._population[best_idx, :].clone()

        for generation in tqdm(range(self._max_generations)):
            for i in range(self._pop_size):
                trial, trial_fitness = self._mutate(i)

                # Selection
                if trial_fitness < self._fitness[i]:
                    self._fitness[i] = trial_fitness
                    self._population[i, :] = trial

                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_individual = trial.clone()

            # Convergence check
            if torch.std(self._fitness) < self._tol:
                print(f"Converged after {generation} generations.")
                break

        return best_individual, best_fitness.item()

    def _select_three_random_individuals(self, i: int) -> Tuple[Tensor, Tensor, Tensor]:
        idxs = [idx for idx in range(self._pop_size) if idx != i]
        a, b, c = self._population[torch.tensor(torch.randperm(len(idxs))[:3]), :]
        return a, b, c
