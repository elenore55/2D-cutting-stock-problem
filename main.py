from with_operators import generate_initial_population, calculate_height, crossover_pmx

if __name__ == '__main__':
    initial = generate_initial_population()
    print(min(chromosome.cost for chromosome in initial))
    print(crossover_pmx([1, 4, 2, 5, 3], [4, 2, 5, 1, 3]))
    # for chromosome in initial:
    #     print(chromosome, end=' ')
    #     print(calculate_height(chromosome.chromosome))
