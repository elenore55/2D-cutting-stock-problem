import concurrent.futures
import random

import rectpack
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from rectpack import PackingMode, SORT_NONE, GuillotineBssfSas

from util.types import Chromosome, Piece
from strategies.crossover import CrossoverStrategy
from strategies.mutation import MutationStrategy
from strategies.selection import SelectionStrategy
from util.util import Util


class GeneticAlgorithmWithoutOperators(object):

    def __init__(
            self,
            selection_strategy: SelectionStrategy,
            crossover_strategy: CrossoverStrategy,
            mutation_strategy: MutationStrategy,
            population_size: int,
            max_iterations: int,
            elitism: int
    ):
        self.selection_strategy = selection_strategy
        self.crossover_strategy = crossover_strategy
        self.mutation_strategy = mutation_strategy
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.elitism = elitism

        self.SHEET_WIDTH = 0
        self.SHEET_HEIGHT = 0
        self.PIECES = []
        self.ROTATED_PIECES = []
        self.costs = {}

    def do(self, sheet_width: int, sheet_height: int, pieces: list[Piece], rotated_pieces: list[Piece]):
        self.SHEET_WIDTH = sheet_width
        self.SHEET_HEIGHT = sheet_height
        self.PIECES = pieces
        self.ROTATED_PIECES = rotated_pieces

        theoretical_minimum = Util.calculate_theoretical_minimum(sheet_width, sheet_height, pieces)
        best_results = []

        repetition_count = 0
        population = self.generate_initial_population()
        for iter_num in range(self.max_iterations):

            population_with_cost = []

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_chromosome = {executor.submit(self.calculate_cost, chrom): chrom for chrom in population}
                for future in concurrent.futures.as_completed(future_to_chromosome):
                    chrom = future_to_chromosome[future]
                    cost = future.result()
                    population_with_cost.append(Chromosome(chrom, cost))

            population_with_cost.sort(key=lambda x: x.cost)
            best_chromosome = population_with_cost[0]
            if len(best_results) > 0 and best_results[-1].cost == best_chromosome.cost:
                repetition_count += 1
            else:
                repetition_count = 0
            best_results.append(best_chromosome)

            if best_chromosome.cost - theoretical_minimum < 0.5 or repetition_count > 100:
                print('Stopping')
                return best_results

            print(iter_num, best_chromosome)
            next_generation = [ch.chromosome for ch in population_with_cost[:self.elitism]]

            for i in range(self.elitism, len(population_with_cost), 2):
                parent1, parent2 = self.selection_strategy.select(population_with_cost)
                child1, child2 = self.crossover_strategy.crossover(parent1, parent2)
                child1 = self.mutation_strategy.mutate(child1, cost_fn=self.calculate_cost, neighborhood_fn=self._get_neighbors)
                child2 = self.mutation_strategy.mutate(child2, cost_fn=self.calculate_cost, neighborhood_fn=self._get_neighbors)
                next_generation.append(child1)
                next_generation.append(child2)

            population = next_generation
        return best_results

    def generate_initial_population(self) -> list[list]:
        return [self._generate_chromosome() for _ in range(self.population_size)]

    def _generate_chromosome(self) -> list:
        indices = list(range(1, len(self.PIECES) + 1))
        random.shuffle(indices)
        for i, elem in enumerate(indices):
            if random.random() < 0.5:
                indices[i] = -elem
        return indices

    @staticmethod
    def _get_neighbors(chromosome):
        num_neighbors = 25
        neighbors = []
        chosen_indices = set()
        for i in range(num_neighbors):
            p1, p2 = random.sample(range(len(chromosome)), 2)
            if p1 > p2:
                p1, p2 = p2, p1
            while (p1, p2) in chosen_indices:
                p1, p2 = random.sample(range(len(chromosome)), 2)
                if p1 > p2:
                    p1, p2 = p2, p1
            chosen_indices.add((p1, p2))
            neighbor = chromosome[:]
            neighbor[p1], neighbor[p2] = neighbor[p2], neighbor[p1]
            neighbors.append(neighbor)
        return neighbors

    def calculate_cost(self, chromosome: list) -> float:
        key = Util.get_str(chromosome)
        if key in self.costs:
            return self.costs[key]

        rectangles = self._pack(chromosome)
        valid_cuts, unoccupied_area, total_height = self._get_valid_horizontal_cuts(rectangles)  # TODO: total height

        num_sheets, num_invalids = self._calculate_num_sheets(valid_cuts)
        invalids_percentage = (num_invalids / num_sheets) * 100
        cost = num_sheets + invalids_percentage + unoccupied_area / (10 ** len(str(unoccupied_area)))

        if len(self.costs) > 10000:
            self.costs.clear()
        self.costs[key] = cost
        return cost

    def _pack(self, order):
        packer = rectpack.newPacker(mode=PackingMode.Offline, sort_algo=SORT_NONE, rotation=False, pack_algo=GuillotineBssfSas)
        for num in order:
            piece = self.PIECES[num - 1] if num > 0 else self.ROTATED_PIECES[-num - 1]
            packer.add_rect(piece.width, piece.height)
        packer.add_bin(self.SHEET_WIDTH, float('inf'))
        packer.pack()
        result = []
        packed_bin = packer[0]
        for rect in packed_bin:
            result.append((rect.x, rect.y, rect.width, rect.height))
        return result

    def _get_valid_horizontal_cuts(self, rectangles):
        max_h = self._calculate_total_height(rectangles)
        rectangles.sort(key=lambda r: r[1])
        valid_y = []
        levels_and_max_heights = {}
        for rect in rectangles:
            y, h = rect[1], rect[3]
            if y not in levels_and_max_heights:
                levels_and_max_heights[y] = h
            else:
                levels_and_max_heights[y] = max(levels_and_max_heights[y], h)

        pairs = [(k, levels_and_max_heights[k]) for k in levels_and_max_heights]
        pairs.sort(key=lambda x: x[0])
        for i in range(1, len(pairs)):
            valid = True
            for j in range(i):
                prev_y = pairs[j][0]
                prev_max_h = pairs[j][1]
                if pairs[i][0] < prev_y + prev_max_h:
                    valid = False
                    break
            if valid:
                valid_y.append(pairs[i][0])
        valid_y.append(max_h)

        unoccupied = 0
        for rect in rectangles:
            _, y, w, h = rect
            if y in levels_and_max_heights:
                max_h_for_level = levels_and_max_heights[y]
                unoccupied += w * (max_h_for_level - h)
        return valid_y, unoccupied, max_h

    @staticmethod
    def _calculate_total_height(rectangles: list) -> list:
        result = 0
        for rectangle in rectangles:
            result = max(result, rectangle[1] + rectangle[3])
        return result

    def _calculate_num_sheets(self, valid_cuts: list) -> (int, int):
        if len(valid_cuts) == 1:
            return 1, 1

        sheet_counter = 1
        invalids_counter = 0
        score = 0
        valid_cuts = [0] + valid_cuts
        current_h = 0

        for i in range(1, len(valid_cuts)):
            h = valid_cuts[i] - valid_cuts[i - 1]
            if h + current_h < self.SHEET_HEIGHT:
                current_h += h
            else:
                sheet_counter += 1
                if h <= self.SHEET_HEIGHT:
                    current_h = h
                else:
                    current_h = 0
                    score += h - self.SHEET_HEIGHT
                    invalids_counter += 1
        return sheet_counter, invalids_counter

    def display_solution(self, chromosome):
        self._plot(chromosome)

    def _plot(self, chromosome):
        packer = rectpack.newPacker(mode=PackingMode.Offline, sort_algo=SORT_NONE, rotation=False, pack_algo=GuillotineBssfSas)
        for num in chromosome:
            piece = self.PIECES[num - 1] if num > 0 else self.ROTATED_PIECES[-num - 1]
            packer.add_rect(piece.width, piece.height)
        packer.add_bin(self.SHEET_WIDTH, float('inf'))
        packer.pack()

        fig, ax = plt.subplots()
        ax.plot([0, 10], [0, 10])
        packed_bin = packer[0]
        for rect in packed_bin:
            ax.add_patch(Rectangle((rect.x, rect.y), rect.width, rect.height, facecolor=Util.get_color()))
        plt.show()
