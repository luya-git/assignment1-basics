import torch
import math
import torch.nn as nn
from einops import einsum, reduce, rearrange

class RSMNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        # hidden dimension of the model
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.g = torch.ones(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        row_sum = reduce(x * x, "b s d_model -> b s", reduction="sum")
        rms = 1 / (row_sum * (1/self.d_model) + self.eps).sqrt()
        result = einsum(x * self.g, rms, "b s d_model, b s -> b s d_model")
        return result.to(in_dtype)