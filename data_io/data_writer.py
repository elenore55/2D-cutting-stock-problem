# results by iteration (chromosome, fitness)
# which input file
# which algorithm
# which strategies and parameters
# percentage of unused space

import json
import os
from pathlib import Path


class JsonDataWriter(object):

    @staticmethod
    def write(data, file_name):
        directory = Path('../output')
        directory.mkdir(parents=True, exist_ok=True)
        path = f'../output/{file_name}'
        existing_arr = []
        if os.path.exists(path):
            with open(path, 'r') as file:
                existing_arr = json.load(file)
        existing_arr.append(data)
        with open(path, 'w') as file:
            json.dump(existing_arr, file, indent=4)
