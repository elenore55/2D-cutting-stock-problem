import copy
import random
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

        parent1_absolutes = np.array([abs(num) for num in parent1])
        parent2_absolutes = np.array([abs(num) for num in parent2])
        rng = np.random.default_rng()

        cutoff1, cutoff2 = np.sort(rng.choice(np.arange(len(parent1_absolutes) + 1), size=2, replace=False))

        def PMX_one_child(p1, p2):
            child = np.zeros(len(p1), dtype=p1.dtype)
            child[cutoff1:cutoff2] = p1[cutoff1:cutoff2]

            for j in np.concatenate([np.arange(0, cutoff1), np.arange(cutoff2, len(p1))]):
                candidate = p2[j]
                while candidate in p1[cutoff1:cutoff2]:
                    candidate = p2[np.where(p1 == candidate)[0][0]]
                child[j] = candidate
            return child

        child1 = PMX_one_child(parent1_absolutes, parent2_absolutes).tolist()
        child2 = PMX_one_child(parent2_absolutes, parent1_absolutes).tolist()

        for i in np.concatenate([np.arange(0, cutoff1), np.arange(cutoff2, len(child1))]):
            gene1 = child1[i]
            gene2 = child2[i]
            if gene1 in negatives2:
                child1[i] = -gene1
            if gene2 in negatives1:
                child2[i] = -gene2

        for i in range(cutoff1, cutoff2):
            gene1 = child1[i]
            gene2 = child2[i]
            if gene1 in negatives1:
                child1[i] = -gene1
            if gene2 in negatives2:
                child2[i] = -gene2

        return child1, child2


class OrderCrossover(CrossoverStrategy):

    def crossover(self, parent1: list[int], parent2: list[int]) -> (list[int], list[int]):
        chromosome_len = len(parent1)
        child1, child2 = [-1] * chromosome_len, [-1] * chromosome_len

        index1, index2 = random.sample(range(chromosome_len), 2)
        if index1 > index2:
            index1, index2 = index2, index1

        child1_inherited = []
        child2_inherited = []
        for i in range(index1, index2 + 1):
            child1[i] = parent1[i]
            child2[i] = parent2[i]
            child1_inherited.append(parent1[i])
            child2_inherited.append(parent2[i])

        current_parent2_position, current_parent1_position = 0, 0

        fixed_positions = list(range(index1, index2 + 1))
        i = 0
        while i < chromosome_len:
            if i in fixed_positions:
                i += 1
                continue

            if child1[i] == -1:
                parent2_gene = parent2[current_parent2_position]
                while parent2_gene in child1_inherited or -parent2_gene in child1_inherited:
                    current_parent2_position += 1
                    parent2_gene = parent2[current_parent2_position]
                child1[i] = parent2_gene
                child1_inherited.append(parent2_gene)

            if child2[i] == -1:
                parent1_gene = parent1[current_parent1_position]
                while parent1_gene in child2_inherited or -parent1_gene in child2_inherited:
                    current_parent1_position += 1
                    parent1_gene = parent1[current_parent1_position]
                child2[i] = parent1_gene
                child2_inherited.append(parent1_gene)
            i += 1

        return child1, child2


class CycleCrossover(CrossoverStrategy):

    def crossover(self, parent1: list[int], parent2: list[int]) -> (list[int], list[int]):
        chromosome_len = len(parent1)
        child1, child2 = [-1] * chromosome_len, [-1] * chromosome_len

        parent1_copy = copy.deepcopy(parent1)
        parent2_copy = copy.deepcopy(parent2)
        parent1_abs = [abs(gene) for gene in parent1]
        parent2_abs = [abs(gene) for gene in parent2]
        swap = True
        count = 0
        position = 0

        while True:
            if count > chromosome_len:
                break
            for i in range(chromosome_len):
                if child1[i] == -1:
                    position = i
                    break
            if swap:
                while True:
                    child1[position] = parent1[position]
                    count += 1
                    position = parent2_abs.index(abs(parent1[position]))
                    if parent1_copy[position] == -1:
                        swap = False
                        break
                    parent1_copy[position] = -1
            else:
                while True:
                    child1[position] = parent2[position]
                    count += 1
                    position = parent1_abs.index(abs(parent2[position]))
                    if parent2_copy[position] == -1:
                        swap = True
                        break
                    parent2_copy[position] = -1

        for i in range(chromosome_len):
            if child1[i] == parent1[i]:
                child2[i] = parent2[i]
            else:
                child2[i] = parent1[i]

        for i in range(chromosome_len):
            if child1[i] == -1:
                if parent1_copy[i] == -1:
                    child1[i] = parent2[i]
                else:
                    child1[i] = parent1[i]
        return child1, child2


class SpecialCrossover(CrossoverStrategy):
    def crossover(self, parent1: list[int], parent2: list[int]) -> (list[int], list[int]):
        chromosome_len = len(parent1)
        child1, child2 = [-1] * chromosome_len, [-1] * chromosome_len

        index1, index2 = random.sample(range(1, chromosome_len + 1), 2)
        if index1 > index2:
            index1, index2 = index2, index1
        for i in range(index2):
            child1[i] = parent1[(index1 - 1 + i) % chromosome_len]
            child2[i] = parent2[(index1 - 1 + i) % chromosome_len]

        j = index2
        while j < chromosome_len:
            for gene in parent2:
                if gene not in child1 and -gene not in child1:
                    child1[j] = gene
                    break
            for gene in parent1:
                if gene not in child2 and -gene not in child2:
                    child2[j] = gene
                    break
            j += 1
        return child1, child2



if __name__ == '__main__':
    strategy = CycleCrossover()
    par1 = [-8, 4, 7, -3, 6, 2, 5, 1, -9, 0, 11, 12, 15, 80]
    par2 = [0, 1, 2, 3, -4, 5, -6, 7, 8, -9, -12, 15, -80, 11]
    c1, c2 = strategy.crossover(par1, par2)
    print(c1, c2)
