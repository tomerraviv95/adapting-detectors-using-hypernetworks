from typing import Tuple

import torch


def calculate_error_rate(prediction: torch.Tensor, target: torch.Tensor) -> Tuple[float, int]:
    """
    Returns the calculated ber of the prediction and the target (ground truth transmitted word)
    """
    prediction = prediction.long()
    target = target.long()
    equal_bits = torch.eq(prediction, target).float()
    bits_acc = torch.mean(equal_bits).item()
    return 1 - bits_acc
