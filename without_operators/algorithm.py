from rectpack import PackingMode, SORT_NONE
from rectpack.maxrects import MaxRectsBssf, MaxRectsBl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from data_reader import DataReader
import random
import rectpack

from without_operators.chromosome import Chromosome

SHEET_W = 200
PIECES, ROTATED_PIECES = DataReader.read('C:\\Users\\Milica\\Desktop\\Fakultet\\Master\\Rad\\2D-cutting-stock-problem\\data\\01.csv')
NUM_PIECES = len(PIECES)
POPULATION_SIZE = 100
CHROMOSOME_LEN = NUM_PIECES
MAX_ITERS = 1000
ELITISM = 10
MUTATION_RATE = 0.5

costs = {}


def main():
    best_results = []
    population = generate_initial_population()
    for iter_num in range(MAX_ITERS):
        population_with_cost = []
        for chromosome in population:
            cost, w, h = evaluate_cost(chromosome)
            population_with_cost.append(Chromosome(chromosome, cost, w, h))
        population_with_cost.sort(key=lambda x: x.cost)
        best_results.append(population_with_cost[0])
        print(iter_num, population_with_cost[0])
        next_generation = [ch.chromosome for ch in population_with_cost[:ELITISM]]

        for i in range(ELITISM, len(population_with_cost), 2):
            parent1, parent2 = select_parents(population_with_cost)
            child1, child2 = crossover(parent1, parent2)
            child1 = tabu_search(child1, 3, 10)
            child2 = tabu_search(child2, 3, 10)
            next_generation.append(child1)
            next_generation.append(child2)

        population = next_generation
    return best_results[-1]


def generate_initial_population() -> list[list]:
    return [generate_chromosome() for _ in range(POPULATION_SIZE)]


def generate_chromosome() -> list:
    indices = list(range(1, CHROMOSOME_LEN + 1))
    random.shuffle(indices)
    for i, elem in enumerate(indices):
        if random.random() < 0.5:
            indices[i] = -elem
    return indices


def select_parents(population: list[Chromosome]):
    n = len(population)
    probabilities_sum = n * (n + 1) / 2
    probabilities = [i / probabilities_sum for i in range(n + 1, 1, -1)]
    return random.choices([ch.chromosome for ch in population], weights=probabilities, k=2)


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
        if random.random() < 0.02:
            chromosome[i] = -chromosome[i]
    return chromosome


def get_neighbors(solution):
    num_neighbors = 30
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

        for neighbor in neighbors:
            if neighbor not in tabu_list:
                neighbor_fitness = evaluate_cost2(neighbor)
                if neighbor_fitness < best_neighbor_fitness:
                    best_neighbor = neighbor
                    best_neighbor_fitness = neighbor_fitness

        if best_neighbor is None:
            # No non-tabu neighbors found,
            # terminate the search
            break

        current_solution = best_neighbor
        tabu_list.append(best_neighbor)
        if len(tabu_list) > tabu_list_size:
            # Remove the oldest entry from the
            # tabu list if it exceeds the size
            tabu_list.pop(0)

        if evaluate_cost2(best_neighbor) < evaluate_cost2(best_solution):
            # Update the best solution if the
            # current neighbor is better
            best_solution = best_neighbor

    return best_solution


def evaluate_cost(chromosome: list) -> (int, int, int):
    key = getStr(chromosome)
    if key in costs:
        return costs[key]
    max_width = 0
    max_height = 0
    packer = rectpack.newPacker(mode=PackingMode.Offline, sort_algo=SORT_NONE, rotation=False, pack_algo=MaxRectsBl)
    for gene in chromosome:
        if gene > 0:
            piece = PIECES[gene - 1]
            packer.add_rect(piece.width, piece.height)
        else:
            piece = ROTATED_PIECES[-gene - 1]
            packer.add_rect(piece.width, piece.height)
    packer.add_bin(SHEET_W, float('inf'))
    packer.pack()
    packed_bin = packer[0]
    for rect in packed_bin:
        x = rect.x
        y = rect.y
        width = rect.width
        height = rect.height
        if y + height > max_height:
            max_height = y + height
        if x + width > max_width:
            max_width = x + width
    costs[key] = (max_height, max_width, max_height)
    return max_height, max_width, max_height

def getStr(chromosome: list) -> str:
    return ' '.join(str(x) for x in chromosome)


def evaluate_cost2(chromosome: list) -> (int, int, int):
    key = getStr(chromosome)
    if key in costs:
        return costs[key][0]
    max_width = 0
    max_height = 0
    packer = rectpack.newPacker(mode=PackingMode.Offline, sort_algo=SORT_NONE, rotation=False, pack_algo=MaxRectsBl)
    for gene in chromosome:
        if gene > 0:
            piece = PIECES[gene - 1]
            packer.add_rect(piece.width, piece.height)
        else:
            piece = ROTATED_PIECES[-gene - 1]
            packer.add_rect(piece.width, piece.height)
    packer.add_bin(SHEET_W, float('inf'))
    packer.pack()
    packed_bin = packer[0]
    for rect in packed_bin:
        x = rect.x
        y = rect.y
        width = rect.width
        height = rect.height
        if y + height > max_height:
            max_height = y + height
        if x + width > max_width:
            max_width = x + width
    costs[key] = (max_height, max_width, max_height)
    return max_height


def get_color():
    return "#%06x" % random.randint(0, 0xFFFFFF)


def plot(order):
    packer = rectpack.newPacker(mode=PackingMode.Offline, sort_algo=SORT_NONE, rotation=False, pack_algo=MaxRectsBl)
    for num in order:
        if num > 0:
            piece = PIECES[num - 1]
        else:
            piece = ROTATED_PIECES[-num - 1]
        packer.add_rect(piece.width, piece.height)
    packer.add_bin(SHEET_W, float('inf'))
    packer.pack()

    abin: MaxRectsBssf = packer[0]
    print(abin.used_area())
    print(abin.width, abin.height)

    fig, ax = plt.subplots()
    ax.plot([0, 10], [0, 10])
    for abin in packer:
        k = 0
        for rect in abin:
            print(rect.x, rect.y, rect.width, rect.height)
            color = get_color()
            ax.add_patch(Rectangle((rect.x, rect.y), rect.width, rect.height, facecolor=color))
            k += 1
    plt.show()


if __name__ == '__main__':
    best = main()
    print(best)
    plot(best.chromosome)
