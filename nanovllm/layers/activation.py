import torch
from torch import nn
import torch.nn.functional as F
from functools import lru_cache

class SiluAndMul(nn.Module):
    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y

@lru_cache(1)
def get_silu_and_mul():
    return SiluAndMul()
