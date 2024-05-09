import numpy as np
from numpy.random import default_rng

from python_code import conf


class Generator:
    def __init__(self, block_length: int, pilots_length: int):
        self._bits_generator = default_rng(seed=conf.seed)
        self.block_length = block_length
        self.pilots_length = pilots_length

    def generate(self):
        pilots = self._bits_generator.integers(0, 2, size=(self.pilots_length, conf.n_user))
        data = self._bits_generator.integers(0, 2, size=(self.block_length - self.pilots_length, conf.n_user))
        mx = np.concatenate([pilots, data])
        return mx
