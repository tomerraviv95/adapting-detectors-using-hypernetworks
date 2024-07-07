import torch


def ls_channel_estimation(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Least squares channel estimation
    find solution to min||y = out * x||^2
    A = (x^t * x), B = (x^t * y)
    out_hat = A * B
    """
    A = torch.linalg.inv(torch.matmul(x.T, x))
    B = torch.matmul(x.T, y.float())
    out_hat = torch.abs(torch.matmul(A, B))
    return out_hat
