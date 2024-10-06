class Chromosome(object):
    def __init__(self, chromosome: list, cost: float, w=0, h=0):
        self.chromosome = chromosome
        self.cost = cost
        self.w = w
        self.h = h

    def __str__(self):
        return f'{self.chromosome} {self.cost}'


class Piece(object):
    def __init__(self, width, height, x=0, y=0):
        self.width = width
        self.height = height
        self.x = x
        self.y = y

    def __str__(self):
        return f'({self.width}, {self.height}), ({self.x}, {self.y})'


class BoundingBox(object):
    def __init__(self, elem1, elem2, op, w, h, x=0, y=0):
        self.elem1 = elem1
        self.elem2 = elem2
        self.op = op
        self.width = w
        self.height = h
        self.x = x
        self.y = y
