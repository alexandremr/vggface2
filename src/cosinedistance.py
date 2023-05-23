import torch
from torch import nn
import torch.nn.functional as F


class CosineDistance(nn.Module):
    __constants__ = ['dim', 'eps']
    dim: int
    eps: float

    def __init__(self, dim: int = 1, eps: float = 1e-8) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return 1 - F.cosine_similarity(x1, x2, self.dim, self.eps)
