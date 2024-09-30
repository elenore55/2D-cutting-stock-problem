import copy
import random

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from util.types import *
from strategies.crossover import CrossoverStrategy
from strategies.mutation import MutationStrategy
from strategies.selection import SelectionStrategy
from util.util import Util


class GeneticAlgorithmWithOperators(object):

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

        self.OPERATORS = ['V', 'H']
        self.NUM_PIECES = 0
        self.CHROMOSOME_LEN = 0
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
        self.NUM_PIECES = len(pieces)
        self.CHROMOSOME_LEN = 2 * self.NUM_PIECES - 1

        theoretical_minimum = Util.calculate_theoretical_minimum(sheet_width, sheet_height, pieces)
        best_results = []

        population = self._generate_initial_population()
        repetition_count = 0

        for iter_num in range(self.max_iterations):
            population_with_cost = []
            for chromosome in population:
                cost = self.calculate_cost(chromosome)
                population_with_cost.append(Chromosome(chromosome, cost))

            population_with_cost.sort(key=lambda x: x.cost)
            best_chromosome = population_with_cost[0]
            if len(best_results) > 0 and best_results[-1].cost == best_chromosome.cost:
                repetition_count += 1
            else:
                repetition_count = 0
            best_results.append(best_chromosome)

            if best_chromosome.cost - theoretical_minimum < 0.5 or repetition_count > 100:
                print('Termination criteria reached')
                return best_results

            print(iter_num, best_chromosome)
            next_generation = [chromosome.chromosome for chromosome in population_with_cost[:self.elitism]]

            for i in range(self.elitism, len(population_with_cost), 2):
                parent1, parent2 = self.selection_strategy.select(population_with_cost)
                child1, child2 = self._crossover(parent1, parent2)
                child1 = self.mutation_strategy.mutate(
                    child1,
                    validation_fn=self._is_valid,
                    cost_fn=self.calculate_cost,
                    neighborhood_fn=self._get_neighbors
                )
                child2 = self.mutation_strategy.mutate(
                    child2,
                    validation_fn=self._is_valid,
                    cost_fn=self.calculate_cost,
                    neighborhood_fn=self._get_neighbors
                )
                next_generation.append(child1)
                next_generation.append(child2)
            population = next_generation
        return best_results[-1]

    def _generate_initial_population(self) -> list[Chromosome]:
        return [self._generate_chromosome() for _ in range(self.population_size)]

    def _generate_chromosome(self) -> list:
        pieces = list(range(1, self.NUM_PIECES + 1))
        operators = random.choices(self.OPERATORS, k=self.NUM_PIECES - 1)
        pieces_count = 0
        operators_count = 1
        result = []
        for i in range(self.CHROMOSOME_LEN):
            if operators_count > pieces_count - 1:
                rnd_index = random.randint(0, len(pieces) - 1)
                result.append(self._maybe_rotate(pieces.pop(rnd_index)))
                pieces_count += 1
            elif pieces_count == self.NUM_PIECES:
                result.append(random.choice(operators))
                operators_count += 1
            else:
                if random.random() < 0.5:
                    rnd_index = random.randint(0, len(pieces) - 1)
                    result.append(self._maybe_rotate(pieces.pop(rnd_index)))
                    pieces_count += 1
                else:
                    result.append(random.choice(operators))
                    operators_count += 1
        return result

    @staticmethod
    def _maybe_rotate(gene):
        if random.random() < 0.5:
            return -gene
        return gene

    def _crossover(self, parent1: list, parent2: list):
        pieces1 = [gene for gene in parent1 if gene not in self.OPERATORS]
        pieces2 = [gene for gene in parent2 if gene not in self.OPERATORS]
        child_pieces1, child_pieces2 = self.crossover_strategy.crossover(pieces1, pieces2)
        child1 = []
        child2 = []
        pieces_counter = 0
        for gene in parent1:
            if gene in self.OPERATORS:
                child1.append(gene)
            else:
                child1.append(child_pieces1[pieces_counter])
                pieces_counter += 1
        pieces_counter = 0
        for gene in parent2:
            if gene in self.OPERATORS:
                child2.append(gene)
            else:
                child2.append(child_pieces2[pieces_counter])
                pieces_counter += 1
        return child1, child2

    def calculate_cost(self, chromosome: list) -> float:
        key = Util.get_str(chromosome)
        if key in self.costs:
            return self.costs[key]

        bounding_box = self._calculate_bounding_box(chromosome)
        separated = self._separate_horizontally(bounding_box)
        total_width = sum([item.width for item in separated])  # TODO
        total_height = bounding_box.height
        num_sheets = len(separated)
        num_invalids = 0
        for item in separated:
            if item.width > self.SHEET_WIDTH:
                num_invalids += 1
        unoccupied_area = self._calculate_unoccupied_area(bounding_box)
        invalids_percentage = (num_invalids / num_sheets) * 100
        cost = (num_sheets + invalids_percentage + unoccupied_area / (10 ** len(str(unoccupied_area))) +
                total_width / (10 ** len(str(total_width))))
        if total_height > self.SHEET_HEIGHT:
            cost += (total_height - self.SHEET_HEIGHT) * 1000

        if len(self.costs) > 10000:
            self.costs.clear()
        self.costs[key] = cost
        return cost

    def _calculate_bounding_box(self, chromosome: list) -> BoundingBox:
        stack = []
        for gene in chromosome:
            if gene in self.OPERATORS:
                piece1 = stack.pop()
                piece2 = stack.pop()
                if gene == 'H':
                    box = BoundingBox(piece1, piece2, gene, piece1.width + piece2.width, max(piece1.height, piece2.height))
                    stack.append(box)
                else:
                    box = BoundingBox(piece1, piece2, gene, max(piece1.width, piece2.width), piece1.height + piece2.height)
                    stack.append(box)
            else:
                if gene > 0:
                    stack.append(self.PIECES[gene - 1])
                else:
                    stack.append(self.ROTATED_PIECES[-gene - 1])
        return stack.pop()

    @staticmethod
    def _separate_horizontally(root):
        queue = [root]
        result = []

        while len(queue) > 0:
            node = queue.pop(0)
            if isinstance(node, BoundingBox) and node.op == 'H':
                queue.append(node.elem1)
                queue.append(node.elem2)
            else:
                result.append(node)
        return result

    def _calculate_unoccupied_area(self, bounding_box):
        if isinstance(bounding_box, Piece):
            return 0
        result = 0
        elem1 = bounding_box.elem1
        elem2 = bounding_box.elem2
        op = bounding_box.op
        if op == 'H':
            if elem1.height > elem2.height:
                result = (elem1.height - elem2.height) * elem2.width
            elif elem1.height < elem2.height:
                result = (elem2.height - elem1.height) * elem1.width
        else:
            if elem1.width > elem2.width:
                result = (elem1.width - elem2.width) * elem2.height
            elif elem1.width < elem2.width:
                result = (elem2.width - elem1.width) * elem1.height
        result += self._calculate_unoccupied_area(elem1)
        result += self._calculate_unoccupied_area(elem2)
        return result

    def _is_valid(self, chromosome: list) -> bool:
        piece_count = 0
        operator_count = 1
        for gene in chromosome:
            if gene not in self.OPERATORS:
                piece_count += 1
            else:
                if operator_count > piece_count - 1:
                    return False
                operator_count += 1
        return True

    def _get_neighbors(self, solution):
        neighbors = []
        for i in range(self.CHROMOSOME_LEN):
            neighbor = solution[:]
            if solution[i] in self.OPERATORS:
                neighbor[i] = Util.flip_operator(solution[i])
            else:
                neighbor[i] = -solution[i]
            neighbors.append(neighbor)
        for i in range(self.CHROMOSOME_LEN):
            for j in range(i + 1, self.CHROMOSOME_LEN):
                p1 = solution[i]
                p2 = solution[j]
                neighbor = solution[:]
                if p1 not in self.OPERATORS and p2 in self.OPERATORS:
                    chromosome_copy = copy.deepcopy(solution)
                    chromosome_copy[i], chromosome_copy[j] = p2, p1
                    neighbor[i], neighbor[j] = p2, p1
                    if self._is_valid(neighbor):
                        neighbors.append(neighbor)
                else:
                    neighbor[i], neighbor[j] = p2, p1
                    neighbors.append(neighbor)
        if len(neighbors) > 80:
            return random.sample(neighbors, 80)
        return neighbors

    def display_solution(self, chromosome):
        bounding_box = self._calculate_bounding_box(chromosome)
        self._set_box_coords(bounding_box)
        pieces_list = []
        self._get_pieces_list(bounding_box, pieces_list)
        self._plot(pieces_list)

    def _set_box_coords(self, box):
        if isinstance(box, Piece):
            return
        op = box.op
        box.elem1.x = box.x
        box.elem1.y = box.y
        if op == 'H':
            box.elem2.x = box.x + box.elem1.width
            box.elem2.y = box.y
        else:
            box.elem2.x = box.x
            box.elem2.y = box.y + box.elem1.height
        self._set_box_coords(box.elem1)
        self._set_box_coords(box.elem2)

    def _get_pieces_list(self, box, result):
        if isinstance(box, Piece):
            result.append(box)
            return
        self._get_pieces_list(box.elem1, result)
        self._get_pieces_list(box.elem2, result)

    @staticmethod
    def _plot(pieces):
        fig, ax = plt.subplots()
        ax.plot([0, 10], [0, 10])
        for rect in pieces:
            color = '#%06x' % random.randint(0, 0xFFFFFF)
            ax.add_patch(Rectangle((rect.x, rect.y), rect.width, rect.height, facecolor=color))
        plt.show()
