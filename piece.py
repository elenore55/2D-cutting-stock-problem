class Piece(object):
    def __init__(self, width, height, x=0, y=0):
        self.width = width
        self.height = height
        self.x = x
        self.y = y

    def __str__(self):
        return f'({self.width}, {self.height}), ({self.x}, {self.y})'
