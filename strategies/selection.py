from abc import ABC, abstractmethod
import random

from util.types import Chromosome


class SelectionStrategy(ABC):
    @abstractmethod
    def select(self, population: list[Chromosome]) -> (Chromosome, Chromosome):
        pass


class TournamentSelection(SelectionStrategy):

    def __init__(self, tournament_size=5):
        self.tournament_size = tournament_size

    def select(self, population: list[Chromosome]) -> (list, list):
        tournament = random.sample(population, self.tournament_size)
        parent1 = min(tournament, key=lambda chromosome: chromosome.cost)
        tournament = random.sample(population, self.tournament_size)
        parent2 = min(tournament, key=lambda chromosome: chromosome.cost)
        return parent1.chromosome, parent2.chromosome


class RouletteSelection(SelectionStrategy):
    def select(self, population: list[Chromosome]) -> (list, list):
        n = len(population)
        probabilities_sum = n * (n + 1) / 2
        probabilities = [i / probabilities_sum for i in range(n + 1, 1, -1)]
        return random.choices([ch.chromosome for ch in population], weights=probabilities, k=2)
