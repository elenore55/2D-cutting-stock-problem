from with_operators import generate_initial_population, calculate_height

if __name__ == '__main__':
    initial = generate_initial_population()
    print(min(chromosome.cost for chromosome in initial))
    # for chromosome in initial:
    #     print(chromosome, end=' ')
    #     print(calculate_height(chromosome.chromosome))
