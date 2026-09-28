from bpe import VOCAB_SIZE
import torch
from torch import nn
import numpy as np
from itertools import pairwise


DEVICE = "cuda" if torch.cuda.is_available() else "mps"
# Dataset is ish 250M tokens, so 250/20 is around 12M, so lets do 12 layers with ~1M each so D_MODEL=256
D_MODEL = 2**8
NUM_LAYERS = 12
HIDDEN_MUL = 4


def p(a=D_MODEL, b=D_MODEL):
    return nn.Parameter(torch.randn(a, b) * a**-0.5)


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


class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.WK, self.WQ, self.WV = p(), p(), p()

    def forward(self, input: torch.Tensor):
        _, T, D = input.shape
        assert D == D_MODEL

        K = input @ self.WK  # B, T, D
        Q = input @ self.WQ  # B, T, D
        V = input @ self.WV  # B, T, D
        K_T = K.transpose(2, 1)  # B, D, T

        OUT = Q @ K_T  # B, T, T

        temporal_mask = (torch.arange(T)[:, None] < torch.arange(T)[None, :]).to(
            DEVICE
        )  # T,T
        OUT = OUT.masked_fill(temporal_mask, float("-inf"))

        OUT = torch.softmax(OUT / D_MODEL**0.5, dim=-1)  # B, T, T
        OUT = OUT @ V  # B, T, D

        return OUT


class AttnBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = Attn()
        self.mlp = MLP([D_MODEL, HIDDEN_MUL * D_MODEL, D_MODEL])
        self.layernorm1, self.layernorm2 = LayerNorm(), LayerNorm()

    def forward(self, input):
        res = input
        res = res + self.attn(self.layernorm1(res))
        res = res + self.mlp(self.layernorm2(res))
        return res


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_blocks = nn.ModuleList(AttnBlock() for _ in range(NUM_LAYERS))
        self.proj = Lin(D_MODEL, VOCAB_SIZE)
        self.final_ln = LayerNorm()

    def forward(self, input):
        res = input
        for l in self.attn_blocks:
            res = l(res)
        return self.proj(self.final_ln(res))


loss = nn.CrossEntropyLoss()


def run(corpus: np.ndarray, epochs=10):
    dataset = torch.from_numpy(corpus).to(DEVICE)
    model = Model().to(DEVICE)
    for e in range(epochs):
        for sample in dataset:
            model(sample)
