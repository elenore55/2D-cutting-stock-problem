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
MUTATION_RATE = 0.2


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

        population = next_population


def crossover(parent1: list, parent2: list):

    pass


def mutate(chromosome: list):
    pass

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
