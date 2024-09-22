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
ROTATED_PIECES = [
    Piece(7, 2),
    Piece(4, 3),
    Piece(3, 5),
    Piece(2, 2),
    Piece(6, 1),
    Piece(3, 7),
    Piece(4, 2)
]
NUM_PIECES = len(PIECES)
POPULATION_SIZE = 10
CHROMOSOME_LEN = NUM_PIECES
MAX_ITERS = 1000
ELITISM = 3
MUTATION_RATE = 0.9


def main():
    population = generate_initial_population()
    print(population)


def generate_initial_population():
    return [generate_chromosome() for _ in range(POPULATION_SIZE)]


def generate_chromosome():
    indices = list(range(1, CHROMOSOME_LEN + 1))
    random.shuffle(indices)
    for i, elem in enumerate(indices):
        if random.random() < 0.5:
            indices[i] = -elem
    return indices
