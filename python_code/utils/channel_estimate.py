import torch


def ls_channel_estimation(mx_pilot: torch.Tensor, rx_pilot: torch.Tensor) -> torch.Tensor:
    x, y = mx_pilot, rx_pilot.float()
    A = torch.linalg.inv(torch.matmul(x.T, x))
    B = torch.matmul(x.T, y)
    unnorm_H_hat = torch.abs(torch.matmul(A, B))
    H_hat = unnorm_H_hat / torch.amax(unnorm_H_hat)
    snr_hat = torch.amax(unnorm_H_hat, dim=1)
    est = torch.cat([H_hat, snr_hat.reshape(-1, 1)], dim=1)
    return est
