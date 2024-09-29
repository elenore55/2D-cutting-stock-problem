import json
from piece import Piece


class JsonDataReader(object):

    @staticmethod
    def read(path):
        with open(path) as file:
            data = json.load(file)

            obj = data['Objects'][0]
            stock_width = obj['Length']
            stock_height = obj['Height']

            pieces = []
            rotated_pieces = []

            for item in data['Items']:
                item_width = item['Length']
                item_height = item['Height']
                demand = item['Demand']
                for _ in range(demand):
                    pieces.append(Piece(item_width, item_height))
                    rotated_pieces.append(Piece(item_height, item_width))
            return stock_width, stock_height, pieces, rotated_pieces
