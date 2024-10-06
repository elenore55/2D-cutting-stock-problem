import math
import sys
import time
from strategies.selection import RouletteSelection, TournamentSelection
from strategies.crossover import NoCrossover, PartiallyMappedCrossover, SpecialCrossover, OrderCrossover, CycleCrossover
from strategies.mutation import MutationWithOperators, TwoPointSwapWithoutOperators, TabuSearch, ThreePointSwapWithoutOperators
from algorithms.with_operators import GeneticAlgorithmWithOperators
from algorithms.without_operators import GeneticAlgorithmWithoutOperators
from data_io.json_data_reader import JsonDataReader
from data_io.data_writer import JsonDataWriter


def start_with_operators():
    POPULATION_SIZE = 300
    MAX_ITERATIONS = 1000
    ELITISM = 10

    TOURNAMENT_SIZE = 3
    MUTATION_RATE = 0.5
    TABU_SEARCH_MAX_ITERS = 3
    TABU_LIST_SIZE = 100

    selection = get_selection(TOURNAMENT_SIZE)
    crossover = get_crossover()
    mutation = get_mutation(MUTATION_RATE, TABU_SEARCH_MAX_ITERS, TABU_LIST_SIZE)

    for i in range(1, 31):
        algorithm = GeneticAlgorithmWithOperators(
            selection,
            crossover,
            mutation,
            POPULATION_SIZE,
            MAX_ITERATIONS,
            ELITISM
        )
        sheet_width, sheet_height, pieces, rotated_pieces = JsonDataReader.read(f'./json/c/{i}.json')

        time_start = time.time()
        best_chromosome, iter_num = algorithm.do(sheet_width, sheet_height, pieces, rotated_pieces)
        time_end = time.time()

        tournament_size = TOURNAMENT_SIZE if isinstance(selection, TournamentSelection) else -1
        write_data(
            f'with_operators_{sys.argv[1]}_{sys.argv[2]}_{sys.argv[3]}.json',
            f'c_{i}',
            selection,
            crossover,
            mutation,
            best_chromosome,
            iter_num,
            time_end - time_start,
            POPULATION_SIZE,
            ELITISM,
            MUTATION_RATE,
            tournament_size
        )
        # algorithm.display_solution(best_chromosome)


def start_without_operators():
    POPULATION_SIZE = 250
    MAX_ITERATIONS = 1000
    ELITISM = 10

    TOURNAMENT_SIZE = 3
    MUTATION_RATE = 0.5
    TABU_SEARCH_MAX_ITERS = 2
    TABU_LIST_SIZE = 100

    selection = get_selection(TOURNAMENT_SIZE)
    crossover = get_crossover()
    mutation = get_mutation(MUTATION_RATE, TABU_SEARCH_MAX_ITERS, TABU_LIST_SIZE)

    for i in range(1, 31):
        algorithm = GeneticAlgorithmWithoutOperators(
            selection,
            crossover,
            mutation,
            POPULATION_SIZE,
            MAX_ITERATIONS,
            ELITISM
        )
        sheet_width, sheet_height, pieces, rotated_pieces = JsonDataReader.read(f'./json/c/{i}.json')

        time_start = time.time()
        best_chromosome, iter_num = algorithm.do(sheet_width, sheet_height, pieces, rotated_pieces)
        time_end = time.time()

        tournament_size = TOURNAMENT_SIZE if isinstance(selection, TournamentSelection) else -1
        write_data(
            f'without_operators_{sys.argv[1]}_{sys.argv[2]}_{sys.argv[3]}.json',
            f'c_{i}',
            selection,
            crossover,
            mutation,
            best_chromosome,
            iter_num,
            time_end - time_start,
            POPULATION_SIZE,
            ELITISM,
            MUTATION_RATE,
            tournament_size
        )
        # algorithm.display_solution(best_chromosome)


def get_selection(tournament_size):
    selection_name = sys.argv[1]
    if selection_name == 'roulette':
        return RouletteSelection()
    return TournamentSelection(tournament_size=tournament_size)


def get_crossover():
    crossover_name = sys.argv[2]
    if crossover_name == 'none':
        return NoCrossover()
    if crossover_name == 'pmx':
        return PartiallyMappedCrossover()
    if crossover_name == 'order':
        return OrderCrossover()
    if crossover_name == 'cycle':
        return CycleCrossover()
    return SpecialCrossover()


def get_mutation(mutation_rate, tabu_max_iters, tabu_list_size):
    mutation_name = sys.argv[3]
    if mutation_name == 'with_ops':
        return MutationWithOperators(mutation_rate=mutation_rate)
    if mutation_name == '2ps':
        return TwoPointSwapWithoutOperators(mutation_rate=mutation_rate)
    if mutation_name == '3ps':
        return ThreePointSwapWithoutOperators(mutation_rate=mutation_rate)
    return TabuSearch(max_iters=tabu_max_iters, tabu_list_size=tabu_list_size)


def write_data(
        file_name,
        dataset,
        selection,
        crossover,
        mutation,
        best_chromosome,
        iter_num,
        elapsed,
        population_size,
        elitism,
        mutation_rate,
        tournament_size=-1
):
    unoccupied_percentage, num_sheets = math.modf(best_chromosome.cost)
    data = {
        'dataset': dataset,
        'selection': selection.__class__.__name__,
        'crossover': crossover.__class__.__name__,
        'mutation': mutation.__class__.__name__,
        'best_chromosome': best_chromosome.chromosome,
        'best_cost': best_chromosome.cost,
        'num_sheets': int(num_sheets),
        'unoccupied_percentage': unoccupied_percentage,
        'iter_num': iter_num,
        'elapsed': elapsed,
        'population_size': population_size,
        'elitism': elitism,
        'mutation_rate': mutation_rate,
        'tournament_size': tournament_size
    }
    data_writer = JsonDataWriter()
    data_writer.write(data, file_name)


# args: selection, crossover, mutation

if __name__ == '__main__':
    # print('helo')
    start_without_operators()
