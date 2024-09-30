import copy

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from data_readers.json_data_reader import JsonDataReader
from util.util import Util
from util.types import Piece, Chromosome
import random

SHEET_W, SHEET_H, PIECES, ROTATED_PIECES = JsonDataReader.read(f'../json/c/{sys.argv[1]}')
# SHEET_H = 8
# SHEET_W = 10
# PIECES = [
#     Piece(2, 7),
#     Piece(3, 4),
#     Piece(5, 3),
#     Piece(2, 2),
#     Piece(1, 6),
#     Piece(7, 3),
#     Piece(2, 4)
# ]
# ROTATED_PIECES = [
#     Piece(7, 2),
#     Piece(4, 3),
#     Piece(3, 5),
#     Piece(2, 2),
#     Piece(6, 1),
#     Piece(3, 7),
#     Piece(4, 2)
# ]
OPERATORS = ['V', 'H']
NUM_PIECES = len(PIECES)
POPULATION_SIZE = 350
CHROMOSOME_LEN = 2 * NUM_PIECES - 1
MAX_ITERS = 1000
ELITISM = 10
MUTATION_RATE = 0.4

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
    print(Util.calculate_theoretical_minimum(SHEET_W, SHEET_H, PIECES))
    best_results = []
    population = generate_initial_population()

    for iter_num in range(MAX_ITERS):
        population_with_cost = []
        for chromosome in population:
            cost = calculate_cost(chromosome)
            population_with_cost.append(Chromosome(chromosome, cost))

        population_with_cost.sort(key=lambda x: x.cost)
        best_results.append(population_with_cost[0])
        print(iter_num, population_with_cost[0])
        next_generation = [chromosome.chromosome for chromosome in population_with_cost[:ELITISM]]

        for i in range(ELITISM, len(population_with_cost), 2):
            parent1, parent2 = select_parents_roulette(population_with_cost)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1)
            child1 = tabu_search(child1, 3, 100)
            child2 = mutate(child2)
            child2 = tabu_search(child2, 3, 100)
            next_generation.append(child1)
            next_generation.append(child2)
        population = next_generation
    return best_results[-1]


def select_parents_roulette(population: list[Chromosome]):
    n = len(population)
    probabilities_sum = n * (n + 1) / 2
    probabilities = [i / probabilities_sum for i in range(n + 1, 1, -1)]
    return random.choices([chromosome.chromosome for chromosome in population], weights=probabilities, k=2)


def select_parents_tournament(population: list[Chromosome], tournament_size=5):
    tournament = random.sample(population, tournament_size)
    parent1 = min(tournament, key=lambda chromosome: calculate_cost(chromosome))
    tournament = random.sample(population, tournament_size)
    parent2 = min(tournament, key=lambda chromosome: calculate_cost(chromosome))
    return parent1, parent2


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
    result = chromosome
    if random.random() < MUTATION_RATE:
        i1, i2 = random.sample(range(CHROMOSOME_LEN), 2)
        if i1 > i2:
            i1, i2 = i2, i1
        p1 = chromosome[i1]
        p2 = chromosome[i2]
        if p1 not in OPERATORS and p2 in OPERATORS:
            chromosome_copy = copy.deepcopy(chromosome)
            chromosome_copy[i1], chromosome_copy[i2] = p2, p1
            if is_valid(chromosome_copy):
                result = chromosome_copy
        else:
            chromosome[i1], chromosome[i2] = p2, p1
            result = chromosome
    for i, elem in enumerate(result):
        if random.random() < 0.2:
            if elem not in OPERATORS:
                result[i] = -elem
            else:
                result[i] = flip(elem)
    return result


def flip(operator):
    if operator == 'V':
        return 'H'
    return 'V'


def get_str(chromosome: list) -> str:
    return ' '.join(str(x) for x in chromosome)


def generate_initial_population() -> list[Chromosome]:
    return [generate_chromosome() for _ in range(POPULATION_SIZE)]


def generate_chromosome() -> list:
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
            if gene > 0:
                stack.append(PIECES[gene - 1])
            else:
                stack.append(ROTATED_PIECES[-gene - 1])
    return stack.pop()


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
            p1 = solution[i]
            p2 = solution[j]
            neighbor = solution[:]
            if p1 not in OPERATORS and p2 in OPERATORS:
                chromosome_copy = copy.deepcopy(solution)
                chromosome_copy[i], chromosome_copy[j] = p2, p1
                neighbor[i], neighbor[j] = p2, p1
                if is_valid(neighbor):
                    neighbors.append(neighbor)
            else:
                neighbor[i], neighbor[j] = p2, p1
                neighbors.append(neighbor)
    if len(neighbors) > 80:
        return random.sample(neighbors, 80)
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
                neighbor_fitness = calculate_cost(neighbor)
                if neighbor_fitness < best_neighbor_fitness:
                    best_neighbor = neighbor
                    best_neighbor_fitness = neighbor_fitness

        if best_neighbor is None:
            break

        current_solution = best_neighbor
        tabu_list.append(best_neighbor)
        if len(tabu_list) > tabu_list_size:
            tabu_list.pop(0)

        if calculate_cost(best_neighbor) < calculate_cost(best_solution):
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
        color = '#%06x' % random.randint(0, 0xFFFFFF)
        ax.add_patch(Rectangle((rect.x, rect.y), rect.width, rect.height, facecolor=color))
    plt.show()


def separate_horizontally(root):
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


def calculate_cost(chromosome: list) -> float:
    key = get_str(chromosome)
    if key in costs:
        return costs[key]

    bounding_box = calculate_bounding_box(chromosome)
    separated = separate_horizontally(bounding_box)
    total_width = sum([item.width for item in separated])  # TODO
    total_height = calculate_height(chromosome)
    num_sheets = len(separated)
    num_invalids = 0
    for item in separated:
        if item.width > SHEET_W:
            num_invalids += 1
    unoccupied_area = calculate_unoccupied_area(bounding_box)
    invalids_percentage = (num_invalids / num_sheets) * 100
    cost = num_sheets + invalids_percentage + unoccupied_area / (10 ** len(str(unoccupied_area))) + total_width / (
            10 ** len(str(total_width)))
    if total_height > SHEET_H:
        cost += (total_height - SHEET_H) * 1000

    if len(costs) > 10000:
        costs.clear()
    costs[key] = cost
    return cost


def calculate_unoccupied_area(bounding_box):
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
    result += calculate_unoccupied_area(elem1)
    result += calculate_unoccupied_area(elem2)
    return result


if __name__ == '__main__':
    best = main()
    print(best.chromosome)
    b = calculate_bounding_box(best.chromosome)
    set_box_coords(b)
    arr = []
    get_pieces_list(b, arr)
    plot(arr)

    ch = [10, 5, 11, 'H', -12, 'V', 'V', -9, 6, -8, 'H', 'V', 7, 'H', 'H', -13, -16, 'H', -17, -15, -1, -4, 'V', -3, 'H', 'V', 14, 2, 'V',
          'V', 'H', 'V', 'H']
    print(is_valid(ch))
    b = calculate_bounding_box(ch)
    res = separate_horizontally(b)
    print('UNOCCUPIED')
    print(calculate_unoccupied_area(b))
    for r in res:
        print(r.width)
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
