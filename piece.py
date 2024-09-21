class Piece(object):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def __str__(self):
        return f'({self.width}, {self.height})'
