import sys
from strategies.selection import RouletteSelection, TournamentSelection
from strategies.crossover import NoCrossover, PartiallyMappedCrossover, SpecialCrossover
from strategies.mutation import NoMutation, MutationWithOperators, MutationWithoutOperators, TabuSearch
from algorithms.with_operators import GeneticAlgorithmWithOperators
from algorithms.without_operators import GeneticAlgorithmWithoutOperators
from data_readers.json_data_reader import JsonDataReader


def start_with_operators():
    roulette_selection = RouletteSelection()
    tournament_selection = TournamentSelection(tournament_size=10)

    no_crossover = NoCrossover()
    pmx = PartiallyMappedCrossover()
    special_crossover = SpecialCrossover()

    no_mutation = NoMutation()
    mutation_with_operators = MutationWithOperators(mutation_rate=0.55)
    mutation_without_operators = MutationWithoutOperators(mutation_rate=0.55)
    tabu_search = TabuSearch(max_iters=2, tabu_list_size=150)

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
    roulette_selection = RouletteSelection()
    tournament_selection = TournamentSelection(tournament_size=7)

    no_crossover = NoCrossover()
    pmx = PartiallyMappedCrossover()
    special_crossover = SpecialCrossover()

    no_mutation = NoMutation()
    mutation_with_operators = MutationWithOperators(mutation_rate=0.45)
    mutation_without_operators = MutationWithoutOperators(mutation_rate=0.45)
    tabu_search = TabuSearch(max_iters=3, tabu_list_size=150)

    algorithm = GeneticAlgorithmWithoutOperators(
        roulette_selection,
        pmx,
        tabu_search,
        250,
        600,
        10
    )
    sheet_width, sheet_height, pieces, rotated_pieces = JsonDataReader.read(f'./json/c/{sys.argv[1]}')
    best_chromosome = algorithm.do(sheet_width, sheet_height, pieces, rotated_pieces)
    algorithm.display_solution(best_chromosome)


if __name__ == '__main__':
    start_without_operators()
