import sys
from strategies.selection import RouletteSelection, TournamentSelection
from strategies.crossover import NoCrossover, PartiallyMappedCrossover, SpecialCrossover
from strategies.mutation import NoMutation, MutationWithOperators, TwoPointSwapWithoutOperators, TabuSearch
from algorithms.with_operators import GeneticAlgorithmWithOperators
from algorithms.without_operators import GeneticAlgorithmWithoutOperators
from data_readers.json_data_reader import JsonDataReader


def start_with_operators():
    POPULATION_SIZE = 300
    MAX_ITERATIONS = 1000
    ELITISM = 10

    TOURNAMENT_SIZE = 3
    MUTATION_RATE = 0.5
    TABU_SEARCH_MAX_ITERS = 3
    TABU_LIST_SIZE = 100

    roulette_selection = RouletteSelection()
    tournament_selection = TournamentSelection(tournament_size=TOURNAMENT_SIZE)

    no_crossover = NoCrossover()
    pmx = PartiallyMappedCrossover()
    special_crossover = SpecialCrossover()

    no_mutation = NoMutation()
    mutation_with_operators = MutationWithOperators(mutation_rate=MUTATION_RATE)
    mutation_without_operators = TwoPointSwapWithoutOperators(mutation_rate=MUTATION_RATE)
    tabu_search = TabuSearch(max_iters=TABU_SEARCH_MAX_ITERS, tabu_list_size=TABU_LIST_SIZE)

    algorithm = GeneticAlgorithmWithOperators(
        roulette_selection,
        pmx,
        tabu_search,
        300,
        600,
        10
    )
    sheet_width, sheet_height, pieces, rotated_pieces = JsonDataReader.read(f'./json/c/{sys.argv[1]}')
    best_chromosome = algorithm.do(sheet_width, sheet_height, pieces, rotated_pieces)
    algorithm.display_solution(best_chromosome)


def start_without_operators():
    POPULATION_SIZE = 250
    MAX_ITERATIONS = 1000
    ELITISM = 10

    TOURNAMENT_SIZE = 3
    MUTATION_RATE = 0.5
    TABU_SEARCH_MAX_ITERS = 3
    TABU_LIST_SIZE = 100

    roulette_selection = RouletteSelection()
    tournament_selection = TournamentSelection(tournament_size=TOURNAMENT_SIZE)

    no_crossover = NoCrossover()
    pmx = PartiallyMappedCrossover()
    special_crossover = SpecialCrossover()

    no_mutation = NoMutation()
    mutation_with_operators = MutationWithOperators(mutation_rate=MUTATION_RATE)
    mutation_without_operators = TwoPointSwapWithoutOperators(mutation_rate=MUTATION_RATE)
    tabu_search = TabuSearch(max_iters=TABU_SEARCH_MAX_ITERS, tabu_list_size=TABU_LIST_SIZE)

    algorithm = GeneticAlgorithmWithoutOperators(
        roulette_selection,
        pmx,
        tabu_search,
        POPULATION_SIZE,
        MAX_ITERATIONS,
        ELITISM
    )
    sheet_width, sheet_height, pieces, rotated_pieces = JsonDataReader.read(f'./json/c/{sys.argv[1]}')
    best_chromosome = algorithm.do(sheet_width, sheet_height, pieces, rotated_pieces)
    algorithm.display_solution(best_chromosome)


if __name__ == '__main__':
    start_without_operators()
