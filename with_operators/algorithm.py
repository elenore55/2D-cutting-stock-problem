import copy

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from data_reader import DataReader
from piece import Piece
from with_operators.chromosome import Chromosome
import random

# SHEET_H = 200
# SHEET_W = 200
# PIECES, ROTATED_PIECES = DataReader.read('C:\\Users\\Milica\\Desktop\\Fakultet\\Master\\Rad\\2D-cutting-stock-problem\\data\\01.csv')
SHEET_H = 8
SHEET_W = 10
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
OPERATORS = ['V', 'H']
NUM_PIECES = len(PIECES)
POPULATION_SIZE = 250
CHROMOSOME_LEN = 2 * NUM_PIECES - 1
MAX_ITERS = 1000
ELITISM = 3
MUTATION_RATE = 0.3

costs = {}


# [1, -2, 'H', 4, 0, -5, 'H', 'H', -3, -6, 'H', 'V', 'V'] 84 6 14
# [-2, 1, 'H', -3, -6, 'H', 'V', 4, 0, -5, 'H', 'H', 'V'] 84 6 14
# [5, -3, 0, 'V', 2, 4, 1, -6, 'V', 'H', 'V', 'H', 'V'] 84 7 12
# [5, -3, 2, 'H', -7, -4, 'H', 'V', 'H', -6, 'H', 1, 'H'] 84 12 7

class BoundingBox(object):
    def __init__(self, elem1, elem2, op, w, h, x=0, y=0):
        self.elem1 = elem1
        self.elem2 = elem2
        self.op = op
        self.width = w
        self.height = h
        self.x = x
        self.y = y


def main():
    best_results = []
    population = generate_initial_population()

    for iter_num in range(MAX_ITERS):
        population_with_cost = []
        for chromosome in population:
            cost, w, h = evaluate_cost2(chromosome)
            population_with_cost.append(Chromosome(chromosome, cost, w, h))

        population_with_cost.sort(key=lambda x: x.cost)
        best_results.append(population_with_cost[0])
        print(iter_num, population_with_cost[0])
        next_generation = [ch.chromosome for ch in population_with_cost[:ELITISM]]

        for i in range(ELITISM, len(population_with_cost), 2):
            parent1, parent2 = select_parents(population_with_cost)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1)
            child1 = tabu_search(child1, 10, 100)
            child2 = mutate(child2)
            child2 = tabu_search(child2, 10, 100)
            next_generation.append(child1)
            next_generation.append(child2)
        population = next_generation


def select_parents(population: list[Chromosome]):
    n = len(population)
    probabilities_sum = n * (n + 1) / 2
    probabilities = [i / probabilities_sum for i in range(n + 1, 1, -1)]
    return random.choices([ch.chromosome for ch in population], weights=probabilities, k=2)


def crossover(parent1: list, parent2: list):
    pieces1 = [gene for gene in parent1 if gene not in OPERATORS]
    pieces2 = [gene for gene in parent2 if gene not in OPERATORS]
    child_pieces1, child_pieces2 = PMX_crossover(pieces1, pieces2)
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


def PMX_crossover(parent1, parent2):
    negatives1 = [-num for num in parent1 if num < 0]
    negatives2 = [-num for num in parent2 if num < 0]

    parent_arr1 = np.array([abs(num) for num in parent1])
    parent_arr2 = np.array([abs(num) for num in parent2])
    rng = np.random.default_rng()

    cutoff_1, cutoff_2 = np.sort(rng.choice(np.arange(len(parent_arr1) + 1), size=2, replace=False))

    def PMX_one_offspring(p1, p2):
        offspring = np.zeros(len(p1), dtype=p1.dtype)
        offspring[cutoff_1:cutoff_2] = p1[cutoff_1:cutoff_2]

        for j in np.concatenate([np.arange(0, cutoff_1), np.arange(cutoff_2, len(p1))]):
            candidate = p2[j]
            while candidate in p1[cutoff_1:cutoff_2]:
                candidate = p2[np.where(p1 == candidate)[0][0]]
            offspring[j] = candidate
        return offspring

    offspring1 = PMX_one_offspring(parent_arr1, parent_arr2)
    offspring2 = PMX_one_offspring(parent_arr2, parent_arr1)

    offspring1 = offspring1.tolist()
    offspring2 = offspring2.tolist()

    for i in np.concatenate([np.arange(0, cutoff_1), np.arange(cutoff_2, len(offspring1))]):
        elem1 = offspring1[i]
        elem2 = offspring2[i]
        if elem1 in negatives2:
            offspring1[i] = -elem1
        if elem2 in negatives1:
            offspring2[i] = -elem2

    for i in range(cutoff_1, cutoff_2):
        elem1 = offspring1[i]
        elem2 = offspring2[i]
        if elem1 in negatives1:
            offspring1[i] = -elem1
        if elem2 in negatives2:
            offspring2[i] = -elem2

    return offspring1, offspring2


def mutate(chromosome: list) -> list:
    cpy = copy.deepcopy(chromosome)
    result = chromosome
    if random.random() < MUTATION_RATE:
        i1, i2 = random.sample(range(CHROMOSOME_LEN), 2)
        p1 = chromosome[i1]
        p2 = chromosome[i2]
        if p1 not in OPERATORS and p2 in OPERATORS:
            chromosome_copy = []
            for c in chromosome:
                chromosome_copy.append(c)
            chromosome_copy[i1], chromosome_copy[i2] = p2, p1
            if is_valid(chromosome_copy):
                result = chromosome_copy
            else:
                result = chromosome
        else:
            chromosome[i1], chromosome[i2] = p2, p1
            result = chromosome
    if is_valid(result):
        ret_val = result
    else:
        ret_val = cpy
    for i, elem in enumerate(ret_val):
        if random.random() < 0.2:
            if elem not in OPERATORS:
                ret_val[i] = -elem
            else:
                ret_val[i] = flip(elem)
    return ret_val


def flip(operator):
    if operator == 'V':
        return 'H'
    return 'V'


def getStr(chromosome: list) -> str:
    return ' '.join(str(x) for x in chromosome)


def generate_initial_population() -> list[Chromosome]:
    return [generate_chromosome2() for _ in range(POPULATION_SIZE)]


def generate_chromosome() -> list:
    pieces = list(range(1, NUM_PIECES + 1))
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


def generate_chromosome2() -> list:
    pieces = list(range(1, NUM_PIECES + 1))
    operators = random.choices(OPERATORS, k=NUM_PIECES - 1)
    pieces_count = 0
    operators_count = 1
    result = []
    for i in range(CHROMOSOME_LEN):
        if operators_count > pieces_count - 1:
            rnd_index = random.randint(0, len(pieces) - 1)
            result.append(maybe_rotate(pieces.pop(rnd_index)))
            pieces_count += 1
        elif pieces_count == NUM_PIECES:
            result.append(random.choice(operators))
            operators_count += 1
        else:
            if random.random() < 0.5:
                rnd_index = random.randint(0, len(pieces) - 1)
                result.append(maybe_rotate(pieces.pop(rnd_index)))
                pieces_count += 1
            else:
                result.append(random.choice(operators))
                operators_count += 1
    return result


def maybe_rotate(gene):
    if random.random() < 0.5:
        return -gene
    return gene


def calculate_height(chromosome: list) -> int:
    return calculate_bounding_box(chromosome).height


def calculate_width(chromosome: list) -> float:
    return calculate_bounding_box(chromosome).width


def calculate_bounding_box(chromosome) -> BoundingBox:
    stack = []
    try:
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
                if gene > 0:
                    stack.append(PIECES[gene - 1])
                else:
                    stack.append(ROTATED_PIECES[-gene - 1])
        return stack.pop()
    except IndexError:
        print('ex')
        print(chromosome)
        raise Exception


def evaluate_cost(chromosome: list) -> float:
    return calculate_width(chromosome)


def evaluate_cost2(chromosome: list) -> (int, int, int):
    key = getStr(chromosome)
    if key in costs:
        return costs[key]
    box = calculate_bounding_box(chromosome)
    if box.height > SHEET_H:
        result = box.width * box.height, box.width, box.height
    else:
        result = box.width, box.width, box.height
    costs[key] = result
    return result


def evaluate_cost3(chromosome: list) -> int:
    # find coords for each piece
    pass


def set_box_coords(box):
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
    set_box_coords(box.elem1)
    set_box_coords(box.elem2)


def get_neighbors(solution):
    # rotate piece
    # flip operator
    # swap pieces
    neighbors = []
    for i in range(CHROMOSOME_LEN):
        neighbor = solution[:]
        if solution[i] in OPERATORS:
            neighbor[i] = flip(solution[i])
        else:
            neighbor[i] = -solution[i]
        neighbors.append(neighbor)
    for i in range(CHROMOSOME_LEN):
        for j in range(i + 1, CHROMOSOME_LEN):
            if solution[i] not in OPERATORS and solution[j] not in OPERATORS:
                neighbor = solution[:]
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                neighbors.append(neighbor)
    return neighbors


def tabu_search(initial_solution: list, max_iterations: int, tabu_list_size: int) -> list:
    best_solution = initial_solution
    current_solution = initial_solution
    tabu_list = []

    for _ in range(max_iterations):
        neighbors = get_neighbors(current_solution)
        best_neighbor = None
        best_neighbor_fitness = float('inf')

        for neighbor in neighbors:
            if neighbor not in tabu_list:
                neighbor_fitness = evaluate_cost2(neighbor)[0]
                if neighbor_fitness < best_neighbor_fitness:
                    best_neighbor = neighbor
                    best_neighbor_fitness = neighbor_fitness

        if best_neighbor is None:
            break

        current_solution = best_neighbor
        tabu_list.append(best_neighbor)
        if len(tabu_list) > tabu_list_size:
            tabu_list.pop(0)

        if evaluate_cost2(best_neighbor)[0] < evaluate_cost2(best_solution)[0]:
            best_solution = best_neighbor

    return best_solution


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


def get_pieces_list(box, result):
    if isinstance(box, Piece):
        result.append(box)
        return
    get_pieces_list(box.elem1, result)
    get_pieces_list(box.elem2, result)


def plot(pieces):
    fig, ax = plt.subplots()
    ax.plot([0, 10], [0, 10])
    for rect in pieces:
        color = "#%06x" % random.randint(0, 0xFFFFFF)
        ax.add_patch(Rectangle((rect.x, rect.y), rect.width, rect.height, facecolor=color))
    plt.show()


def sep_hor(root):
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


def calc_cost(chromosome: list) -> float:
    bounding_box = calculate_bounding_box(chromosome)
    separated = sep_hor(bounding_box)
    overall_width = sum([item.width for item in separated])
    num_sheets = len(separated)
    num_invalids = 0
    for item in separated:
        if item.width > SHEET_W:
            num_invalids += 1
    invalids_percentage = (num_invalids / num_sheets) * 100
    cost = num_sheets + invalids_percentage + 0.0
    return cost



if __name__ == '__main__':
    # main()
    # ch = [-13, -9, -12, 17, 'V', 3, 'V', 4, 'V', 'H', -11, 'H', -8, 15, 10, 'H', 6, -5, 'V', -2, -7, 'V', 'H', 'V', 'V', 14, 'V', 'H', 'H', 'H', -1, 16, 'H']
    # print(is_valid(ch))
    # calculate_bounding_box(ch)
    ch = [-5, -3, 2, 'H', -7, -4, 'H', 'V', 'V', -6, 'H', 1, 'H']
    print(is_valid(ch))
    print('-----')
    b = calculate_bounding_box(ch)
    res = sep_hor(b)
    for elem in res:
        print(elem.width)
    print('-----')
    print(b.width)
    print(b.height)
    set_box_coords(b)
    arr = []
    get_pieces_list(b, arr)
    plot(arr)
    # for piece in arr:
    #     print(f'({piece.x}, {piece.y}, {piece.width}, {piece.height})')
    #
    # arr.sort(key=lambda x: x.x)
    # w = 8
    # cnt = 1
    # curr_w = 0
    # for elem in arr:
    #     new_w = curr_w + elem.width
    #     if new_w > w:
    #         cnt += 1
    #         curr_w = elem.width
    #     else:
    #         curr_w += elem.width
    # print(cnt)
