from abc import ABC, abstractmethod

import numpy as np


class CrossoverStrategy(ABC):

    @abstractmethod
    def crossover(self, parent1: list[int], parent2: list[int]) -> (list[int], list[int]):
        pass


class NoCrossover(CrossoverStrategy):

    def crossover(self, parent1: list[int], parent2: list[int]) -> (list[int], list[int]):
        return parent1, parent2


class PartiallyMappedCrossover(CrossoverStrategy):

    def crossover(self, parent1: list[int], parent2: list[int]) -> (list[int], list[int]):
        negatives1 = [-num for num in parent1 if num < 0]
        negatives2 = [-num for num in parent2 if num < 0]

        parent_arr1 = np.array([abs(num) for num in parent1])
        parent_arr2 = np.array([abs(num) for num in parent2])
        rng = np.random.default_rng()

        cutoff1, cutoff2 = np.sort(rng.choice(np.arange(len(parent_arr1) + 1), size=2, replace=False))

        def PMX_one_offspring(p1, p2):
            offspring = np.zeros(len(p1), dtype=p1.dtype)
            offspring[cutoff1:cutoff2] = p1[cutoff1:cutoff2]

            for j in np.concatenate([np.arange(0, cutoff1), np.arange(cutoff2, len(p1))]):
                candidate = p2[j]
                while candidate in p1[cutoff1:cutoff2]:
                    candidate = p2[np.where(p1 == candidate)[0][0]]
                offspring[j] = candidate
            return offspring

        child1 = PMX_one_offspring(parent_arr1, parent_arr2).tolist()
        child2 = PMX_one_offspring(parent_arr2, parent_arr1).tolist()

        for i in np.concatenate([np.arange(0, cutoff1), np.arange(cutoff2, len(child1))]):
            elem1 = child1[i]
            elem2 = child2[i]
            if elem1 in negatives2:
                child1[i] = -elem1
            if elem2 in negatives1:
                child2[i] = -elem2

        for i in range(cutoff1, cutoff2):
            elem1 = child1[i]
            elem2 = child2[i]
            if elem1 in negatives1:
                child1[i] = -elem1
            if elem2 in negatives2:
                child2[i] = -elem2

        return child1, child2


class SpecialCrossover(CrossoverStrategy):
    def crossover(self, parent1: list[int], parent2: list[int]) -> (list[int], list[int]):
        pass
