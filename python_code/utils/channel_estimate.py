import torch


def ls_channel_estimation(mx_pilot: torch.Tensor, rx_pilot: torch.Tensor) -> torch.Tensor:
    x, y = mx_pilot, rx_pilot.float()
    A = torch.linalg.inv(torch.matmul(x.T, x))
    B = torch.matmul(x.T, y)
    unnorm_H_hat = torch.abs(torch.matmul(A, B))
    return unnorm_H_hat
