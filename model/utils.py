from itertools import pairwise

import torch
from torch import nn

DEVICE = "cuda" if torch.cuda.is_available() else "mps"
# Dataset is ish 250M tokens, so 250/20 is around 12M, so lets do 12 layers with ~1M each so D_MODEL=256
D_MODEL = 2**8
NUM_HEADS = 8
assert D_MODEL % NUM_HEADS == 0
D_HEAD = D_MODEL // NUM_HEADS

NUM_LAYERS = 12
HIDDEN_MUL = 4


def p(a=D_MODEL, b=D_MODEL, s: float | None = None):
    s = s if s is not None else a**-0.5
    return nn.Parameter(torch.randn(a, b) * s)


class LayerNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = nn.Parameter(torch.zeros(1, 1, D_MODEL))
        self.std = nn.Parameter(torch.ones(1, 1, D_MODEL))

    def forward(self, input: torch.Tensor):
        # input is B, T, D
        a = {"dim": -1, "keepdim": True}
        mu, var = torch.mean(input, **a), torch.var(input, unbiased=False, **a)
        eps = 1e-5
        normalized = (input - mu) / (var + eps) ** 0.5
        rescaled = (normalized * self.std) + self.mean
        return rescaled


class Lin(nn.Module):
    def __init__(self, i: int, o: int):
        super().__init__()
        self.w = p(i, o)

    def forward(self, input: torch.Tensor):
        return input @ self.w


class MLP(nn.Module):
    def __init__(self, sizes: list[int], activation=nn.GELU):
        super().__init__()
        self.layers = nn.ModuleList([Lin(i, o) for (i, o) in pairwise(sizes)])
        self.activation = activation()

    def forward(self, input: torch.Tensor):
        res = input
        for i, l in enumerate(self.layers):
            f = self.activation if i < len(self.layers) - 1 else lambda a: a
            res = f(l(res))

        return res
