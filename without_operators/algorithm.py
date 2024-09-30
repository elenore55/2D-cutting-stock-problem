import concurrent.futures
import random
import sys

import matplotlib.pyplot as plt
import rectpack
from matplotlib.patches import Rectangle
from rectpack import PackingMode, SORT_NONE, GuillotineBssfSas

from data_readers.json_data_reader import JsonDataReader
from util.util import Util
from util.types import Chromosome

SHEET_W, SHEET_H, PIECES, ROTATED_PIECES = JsonDataReader.read(f'../json/c/{sys.argv[1]}')
NUM_PIECES = len(PIECES)
POPULATION_SIZE = 250
CHROMOSOME_LEN = NUM_PIECES
MAX_ITERS = 500
ELITISM = 10
MUTATION_RATE = 0.5

costs = {}


def algorithm() -> list:
    theoretical_minimum = Util.calculate_theoretical_minimum(SHEET_W, SHEET_H, PIECES)
    print(theoretical_minimum)

    best_results = []
    repetition_count = 0
    population = generate_initial_population()
    for iter_num in range(MAX_ITERS):

        population_with_cost = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_chromosome = {executor.submit(calculate_cost, chrom): chrom for chrom in population}
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
        next_generation = [ch.chromosome for ch in population_with_cost[:ELITISM]]

        for i in range(ELITISM, len(population_with_cost), 2):
            parent1, parent2 = select_parents_roulette(population_with_cost)
            child1, child2 = crossover(parent1, parent2)
            # child1 = mutate(child1)
            # child2 = mutate(child2)
            child1 = tabu_search(child1, 2, 100)
            child2 = tabu_search(child2, 2, 100)
            next_generation.append(child1)
            next_generation.append(child2)

        population = next_generation
    return best_results


def generate_initial_population() -> list[list]:
    return [generate_chromosome() for _ in range(POPULATION_SIZE)]


def generate_chromosome() -> list:
    indices = list(range(1, CHROMOSOME_LEN + 1))
    random.shuffle(indices)
    for i, elem in enumerate(indices):
        if random.random() < 0.5:
            indices[i] = -elem
    return indices


def select_parents_roulette(population: list[Chromosome]):
    n = len(population)
    probabilities_sum = n * (n + 1) / 2
    probabilities = [i / probabilities_sum for i in range(n + 1, 1, -1)]
    return random.choices([ch.chromosome for ch in population], weights=probabilities, k=2)


def select_parents_tournament(population: list[Chromosome], tournament_size=5):
    tournament = random.sample(population, tournament_size)
    parent1 = min(tournament, key=lambda chromosome: chromosome.cost)
    tournament = random.sample(population, tournament_size)
    parent2 = min(tournament, key=lambda chromosome: chromosome.cost)
    return parent1.chromosome, parent2.chromosome


def crossover(parent1: list, parent2: list) -> (list, list):
    child1 = [0] * CHROMOSOME_LEN
    child2 = [0] * CHROMOSOME_LEN

    p1, p2 = random.sample(range(1, CHROMOSOME_LEN + 1), 2)
    if p1 > p2:
        p1, p2 = p2, p1
    for i in range(p2):
        child1[i] = parent1[(p1 - 1 + i) % CHROMOSOME_LEN]
        child2[i] = parent2[(p1 - 1 + i) % CHROMOSOME_LEN]

    j = p2
    while j < CHROMOSOME_LEN:
        for elem in parent2:
            if elem not in child1 and -elem not in child1:
                child1[j] = elem
                break
        for elem in parent1:
            if elem not in child2 and -elem not in child2:
                child2[j] = elem
                break
        j += 1
    return child1, child2


def mutate(chromosome: list) -> list:
    if random.random() < MUTATION_RATE:
        i1, i2 = random.sample(range(CHROMOSOME_LEN), 2)
        chromosome[i1], chromosome[i2] = chromosome[i2], chromosome[i1]
    for i in range(CHROMOSOME_LEN):
        if random.random() < (1 / CHROMOSOME_LEN):
            chromosome[i] = -chromosome[i]
    return chromosome


def get_neighbors(solution):
    num_neighbors = 25
    neighbors = []
    chosen_indices = set()
    for i in range(num_neighbors):
        p1, p2 = random.sample(range(len(solution)), 2)
        if p1 > p2:
            p1, p2 = p2, p1
        while (p1, p2) in chosen_indices:
            p1, p2 = random.sample(range(len(solution)), 2)
            if p1 > p2:
                p1, p2 = p2, p1
        chosen_indices.add((p1, p2))
        neighbor = solution[:]
        neighbor[p1], neighbor[p2] = neighbor[p2], neighbor[p1]
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

        neighbors_not_in_tabu_list = [neighbor for neighbor in neighbors if Util.get_str(neighbor) not in tabu_list]
        neighbors_with_cost = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_chromosome = {executor.submit(calculate_cost, chrom): chrom for chrom in neighbors_not_in_tabu_list}
            for future in concurrent.futures.as_completed(future_to_chromosome):
                chrom = future_to_chromosome[future]
                cost = future.result()
                neighbors_with_cost.append((chrom, cost))

        for neighbor in neighbors_with_cost:
            neighbor_fitness = neighbor[1]
            if neighbor_fitness < best_neighbor_fitness:
                best_neighbor = neighbor[0]
                best_neighbor_fitness = neighbor_fitness

        if best_neighbor is None:
            break

        current_solution = best_neighbor
        tabu_list.append(Util.get_str(best_neighbor))
        if len(tabu_list) > tabu_list_size:
            tabu_list.pop(0)

        if calculate_cost(best_neighbor) < calculate_cost(best_solution):
            best_solution = best_neighbor

    return best_solution


def pack(order):
    packer = rectpack.newPacker(mode=PackingMode.Offline, sort_algo=SORT_NONE, rotation=False, pack_algo=GuillotineBssfSas)
    for num in order:
        piece = PIECES[num - 1] if num > 0 else ROTATED_PIECES[-num - 1]
        packer.add_rect(piece.width, piece.height)
    packer.add_bin(SHEET_W, float('inf'))
    packer.pack()
    result = []
    packed_bin = packer[0]
    for rect in packed_bin:
        result.append((rect.x, rect.y, rect.width, rect.height))
    return result


def plot(order):
    packer = rectpack.newPacker(mode=PackingMode.Offline, sort_algo=SORT_NONE, rotation=False, pack_algo=GuillotineBssfSas)
    for num in order:
        piece = PIECES[num - 1] if num > 0 else ROTATED_PIECES[-num - 1]
        packer.add_rect(piece.width, piece.height)
    packer.add_bin(SHEET_W, float('inf'))
    packer.pack()

    fig, ax = plt.subplots()
    ax.plot([0, 10], [0, 10])
    packed_bin = packer[0]
    for rect in packed_bin:
        ax.add_patch(Rectangle((rect.x, rect.y), rect.width, rect.height, facecolor=Util.get_color()))
    plt.show()


def get_valid_horizontal_cuts(rectangles):
    max_h = calculate_total_height(rectangles)
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


def calculate_cost(chromosome: list) -> float:
    key = Util.get_str(chromosome)
    if key in costs:
        return costs[key]

    rectangles = pack(chromosome)
    valid_cuts, unoccupied_area, total_height = get_valid_horizontal_cuts(rectangles)  # TODO: total height

    num_sheets, num_invalids = calculate_num_sheets(valid_cuts)
    invalids_percentage = (num_invalids / num_sheets) * 100
    cost = num_sheets + invalids_percentage + unoccupied_area / (10 ** len(str(unoccupied_area)))

    if len(costs) > 10000:
        costs.clear()
    costs[key] = cost
    return cost


def calculate_total_height(rectangles: list) -> list:
    result = 0
    for rectangle in rectangles:
        result = max(result, rectangle[1] + rectangle[3])
    return result


def calculate_num_sheets(valid_cuts: list) -> (int, int):
    if len(valid_cuts) == 1:
        return 1, 1

    sheet_counter = 1
    invalids_counter = 0
    score = 0
    valid_cuts = [0] + valid_cuts
    current_h = 0

    for i in range(1, len(valid_cuts)):
        h = valid_cuts[i] - valid_cuts[i - 1]
        if h + current_h < SHEET_H:
            current_h += h
        else:
            sheet_counter += 1
            if h <= SHEET_H:
                current_h = h
            else:
                current_h = 0
                score += h - SHEET_H
                invalids_counter += 1
    return sheet_counter, invalids_counter


def main():
    best_results = algorithm()
    best_result = best_results[-1]
    print(best_result.chromosome)
    plot(best_result.chromosome)


if __name__ == '__main__':
    main()
    print(sys.argv[1])
    chrom0 = [-26, -16, -11, -4, -21, 33, -48, -12, -49, -25, -46, -44, 36, -45, 52, 18, 53, -32, -9, -20, -19, -10, -13, 39, -2, -29, -6,
              -28, 17, 50, 22, -30, -27, -34, 24, -47, -14, 35, 51, 40, -8, -1, -5, -3, -7, 42, 41, 23, 38, 31, 43, 37, -15]
    rects = pack(chrom0)
    v = get_valid_horizontal_cuts(rects)
    print(v)
    print(calculate_cost(chrom0))
    plot(chrom0)
    # plot1()
