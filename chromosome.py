class Chromosome(object):
    def __init__(self, chromosome: list, cost: int):
        self.chromosome = chromosome
        self.cost = cost

    def __str__(self):
        return f'{self.chromosome} {self.cost}'
