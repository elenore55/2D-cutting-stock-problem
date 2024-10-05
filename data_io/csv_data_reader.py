import csv

from util.types import Piece


class DataReader(object):

    @staticmethod
    def read(path):
        pieces = []
        rotated_pieces = []
        with open(path) as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                if len(row) > 0:
                    width, height = int(row[0]), int(row[1])
                    pieces.append(Piece(width, height))
                    rotated_pieces.append(Piece(height, width))
        return pieces, rotated_pieces
