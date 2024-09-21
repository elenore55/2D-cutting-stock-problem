class Chromosome(object):
    def __init__(self, chromosome: list, cost: int, w=0, h=0):
        self.chromosome = chromosome
        self.cost = cost
        self.w = w
        self.h = h

    def __str__(self):
        return f'{self.chromosome} {self.cost} {self.w} {self.h}'
