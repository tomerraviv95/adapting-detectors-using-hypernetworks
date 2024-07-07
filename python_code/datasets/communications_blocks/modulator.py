import numpy as np
import torch

from python_code.utils.constants import HALF


class BPSKModulator:
    @staticmethod
    def modulate(m: np.ndarray) -> np.ndarray:
        """
        BPSK modulation 0->1, 1->-1
        :param m: the binary codeword
        :return: binary modulated signal
        """
        return 1 - 2 * m

    @staticmethod
    def demodulate(s: torch.Tensor) -> torch.Tensor:
        """
        symbol_to_prob(x:PyTorch/Numpy Tensor/Array)
        Converts BPSK Symbols to Probabilities: '-1' -> 0, '+1' -> '1.'
        :param s: symbols vector
        :return: probabilities vector
        """
        return HALF * (s + 1)
