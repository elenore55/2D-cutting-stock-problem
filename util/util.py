import random

from util.types import Piece


class Util(object):

    @staticmethod
    def get_str(chromosome: list) -> str:
        return ' '.join(str(x) for x in chromosome)

    @staticmethod
    def get_color():
        return '#%06x' % random.randint(0, 0xFFFFFF)

    @staticmethod
    def calculate_theoretical_minimum(sheet_width: int, sheet_height: int, pieces: list[Piece]) -> float:
        sheet_area = sheet_width * sheet_height
        pieces_area = sum(piece.width * piece.height for piece in pieces)
        return pieces_area / sheet_area

    @staticmethod
    def flip_operator(operator):
        return 'V' if operator == 'H' else 'H'
