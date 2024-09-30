import copy
import random
from abc import ABC, abstractmethod

from util.util import Util


class MutationStrategy(ABC):
    OPERATORS = ['V', 'H']

    @abstractmethod
    def mutate(self, chromosome: list, **kwargs) -> list:
        pass


class NoMutation(MutationStrategy):

    def mutate(self, chromosome: list, **kwargs) -> list:
        return chromosome


class MutationWithOperators(MutationStrategy):

    def __init__(self, mutation_rate=0.5):
        self.mutation_rate = mutation_rate

    def mutate(self, chromosome: list, **kwargs) -> list:
        validation_fn = kwargs.get('validation_fn')

        result = chromosome
        if random.random() < self.mutation_rate:
            i1, i2 = random.sample(range(len(chromosome)), 2)
            if i1 > i2:
                i1, i2 = i2, i1
            p1 = chromosome[i1]
            p2 = chromosome[i2]
            if p1 not in self.OPERATORS and p2 in self.OPERATORS:
                chromosome_copy = copy.deepcopy(chromosome)
                chromosome_copy[i1], chromosome_copy[i2] = p2, p1
                if validation_fn(chromosome_copy):
                    result = chromosome_copy
            else:
                chromosome[i1], chromosome[i2] = p2, p1
                result = chromosome
        for i, elem in enumerate(result):
            if random.random() < 1 / len(chromosome):
                if elem not in self.OPERATORS:
                    result[i] = -elem
                else:
                    result[i] = Util.flip_operator(elem)
        return result


class MutationWithoutOperators(MutationStrategy):

    def __init__(self, mutation_rate=0.5):
        self.mutation_rate = mutation_rate

    def mutate(self, chromosome: list, **kwargs) -> list:
        if random.random() < self.mutation_rate:
            i1, i2 = random.sample(range(len(chromosome)), 2)
            chromosome[i1], chromosome[i2] = chromosome[i2], chromosome[i1]
        for i in range(len(chromosome)):
            if random.random() < (1 / len(chromosome)):
                chromosome[i] = -chromosome[i]
        return chromosome


class TabuSearch(MutationStrategy, ABC):

    def __init__(self, max_iters=3, tabu_list_size=100):
        self.max_iters = max_iters
        self.tabu_list_size = tabu_list_size

    def mutate(self, chromosome: list, **kwargs) -> list:
        cost_fn = kwargs.get('cost_fn')
        neighborhood_fn = kwargs.get('neighborhood_fn')

        best_solution = chromosome
        current_solution = chromosome
        tabu_list = []

        for _ in range(self.max_iters):
            neighbors = neighborhood_fn(current_solution)
            best_neighbor = None
            best_neighbor_fitness = float('inf')

            for neighbor in neighbors:
                if neighbor not in tabu_list:
                    neighbor_fitness = cost_fn(neighbor)
                    if neighbor_fitness < best_neighbor_fitness:
                        best_neighbor = neighbor
                        best_neighbor_fitness = neighbor_fitness

            if best_neighbor is None:
                break

            current_solution = best_neighbor
            tabu_list.append(best_neighbor)
            if len(tabu_list) > self.tabu_list_size:
                tabu_list.pop(0)

            if cost_fn(best_neighbor) < cost_fn(best_solution):
                best_solution = best_neighbor

        return best_solution
