import torch
import math
import torch.nn as nn
from einops import einsum

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()

        self.d_in = in_features
        self.d_out = out_features

        w = torch.empty(self.d_out, self.d_in)
        sd = math.sqrt(2.0/(self.d_in+self.d_in))
        self.w = nn.Parameter(nn.init.trunc_normal_(w, mean=0, std=sd, a=-3*sd, b=3*sd))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.w, x, "d_out d_in, ... d_in -> ... d_out")