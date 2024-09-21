import copy

from chromosome import Chromosome
from piece import Piece
import random

SHEET_H = 8
PIECES = [
    Piece(2, 7),
    Piece(3, 4),
    Piece(5, 3),
    Piece(2, 2),
    Piece(1, 6),
    Piece(7, 3),
    Piece(2, 4)
]
OPERATORS = ['V', 'H']
NUM_PIECES = len(PIECES)
POPULATION_SIZE = 30
CHROMOSOME_LEN = 2 * NUM_PIECES - 1
MAX_ITERS = 1000
ELITISM = 3
MUTATION_RATE = 0.5


class BoundingBox(object):
    def __init__(self, elem1, elem2, op, w, h):
        self.elem1 = elem1
        self.elem2 = elem2
        self.op = op
        self.width = w
        self.height = h


def main():
    population = generate_initial_population()

    for _ in range(MAX_ITERS):
        next_population = []
        # selection of parents
        population = next_population


def crossover(parent1: list, parent2: list):
    pieces1 = [gene for gene in parent1 if gene not in OPERATORS]
    pieces2 = [gene for gene in parent2 if gene not in OPERATORS]
    child_pieces1, child_pieces2 = crossover_pmx(pieces1, pieces2)
    child1 = []
    child2 = []
    pieces_counter = 0
    for gene in parent1:
        if gene in OPERATORS:
            child1.append(gene)
        else:
            child1.append(child_pieces1[pieces_counter])
            pieces_counter += 1
    pieces_counter = 0
    for gene in parent2:
        if gene in OPERATORS:
            child2.append(gene)
        else:
            child2.append(child_pieces2[pieces_counter])
            pieces_counter += 1
    return child1, child2


def crossover_pmx(parent1: list, parent2: list):
    assert len(parent1) == len(parent2)
    i1, i2 = random.sample(range(1, len(parent1)), 2)
    if i1 > i2:
        i1, i2 = i2, i1
    middle1 = parent1[i1:i2]
    middle2 = parent2[i1:i2]
    mapping = {}
    for i in range(len(middle1)):
        mapping[middle1[i]] = middle2[i]
    mapping[middle2[len(middle2) - 1]] = middle1[0]

    child1 = parent2[:i1] + middle1 + parent2[i2:]
    child2 = parent1[:i1] + middle2 + parent1[i2:]
    for i in range(i1):
        while child1[i] in mapping:
            if child1.count(child1[i]) == 1:
                break
            child1[i] = mapping[child1[i]]
        while child2[i] in mapping:
            if child2.count(child2[i]) == 1:
                break
            child2[i] = mapping[child2[i]]
    for i in range(i2, len(child1)):
        while child1[i] in mapping:
            if child1.count(child1[i]) == 1:
                break
            child1[i] = mapping[child1[i]]
        while child2[i] in mapping:
            if child2.count(child2[i]) == 1:
                break
            child2[i] = mapping[child2[i]]
    return child1, child2


def mutate(chromosome: list) -> list:
    if random.random() < MUTATION_RATE:
        i1, i2 = random.sample(range(CHROMOSOME_LEN), 2)
        p1 = chromosome[i1]
        p2 = chromosome[i2]
        if p1 not in OPERATORS and p2 in OPERATORS:
            chromosome_copy = copy.deepcopy(chromosome)
            chromosome_copy[i1], chromosome_copy[i2] = p2, p1
            if is_valid(chromosome_copy):
                return chromosome_copy
        else:
            chromosome[i1], chromosome[i2] = p2, p1
    return chromosome


def generate_initial_population() -> list[Chromosome]:
    population = [generate_chromosome() for _ in range(POPULATION_SIZE)]
    result = []
    for chromosome in population:
        result.append(Chromosome(chromosome, evaluate_cost(chromosome)))
    return result


def generate_chromosome() -> list:
    pieces = list(range(NUM_PIECES))
    operators = random.choices(OPERATORS, k=NUM_PIECES - 1)
    pieces_count = 0
    operators_count = 1
    result = []
    for i in range(CHROMOSOME_LEN):
        if operators_count > pieces_count - 1:
            rnd_index = random.randint(0, len(pieces) - 1)
            piece = pieces.pop(rnd_index)
            result.append(piece)
            pieces_count += 1
        elif pieces_count == NUM_PIECES:
            op = random.choice(operators)
            if op == 'H':
                result.append(op)
            else:
                chromosome_copy = copy.deepcopy(result)
                chromosome_copy.append(op)
                if calculate_height(chromosome_copy) <= SHEET_H:
                    result.append(op)
                else:
                    result.append('H')
            operators_count += 1
        else:
            if random.random() < 0.5:
                rnd_index = random.randint(0, len(pieces) - 1)
                piece = pieces.pop(rnd_index)
                result.append(piece)
                pieces_count += 1
            else:
                op = random.choice(operators)
                if op == 'H':
                    result.append(op)
                else:
                    chromosome_copy = copy.deepcopy(result)
                    chromosome_copy.append(op)
                    if calculate_height(chromosome_copy) <= SHEET_H:
                        result.append(op)
                    else:
                        result.append('H')
                operators_count += 1
    return result


def calculate_height(chromosome: list) -> int:
    return calculate_bounding_box(chromosome).height


def calculate_width(chromosome: list) -> float:
    return calculate_bounding_box(chromosome).width


def calculate_bounding_box(chromosome) -> BoundingBox:
    stack = []
    for gene in chromosome:
        if gene in OPERATORS:
            piece1 = stack.pop()
            piece2 = stack.pop()
            if gene == 'H':
                box = BoundingBox(piece1, piece2, gene, piece1.width + piece2.width, max(piece1.height, piece2.height))
                stack.append(box)
            else:
                box = BoundingBox(piece1, piece2, gene, max(piece1.width, piece2.width), piece1.height + piece2.height)
                stack.append(box)
        else:
            stack.append(PIECES[gene])
    return stack.pop()


def evaluate_cost(chromosome: list) -> float:
    return calculate_width(chromosome)


def is_valid(chromosome: list) -> bool:
    piece_count = 0
    operator_count = 1
    for gene in chromosome:
        if gene not in OPERATORS:
            piece_count += 1
        else:
            if operator_count > piece_count - 1:
                return False
            operator_count += 1
    return True
