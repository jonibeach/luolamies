from bpe import VOCAB_SIZE
import torch
from torch import nn
import numpy as np
from itertools import pairwise


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


class RoPE(nn.Module):
    def __init__(self, b: float = 1e4):
        super().__init__()
        self.b = b
        freqs = self.b ** (
            -torch.arange(0, D_HEAD, step=2, device=DEVICE) / D_HEAD
        )  # theta = b^-2i/D_m. D_h/2
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, input: torch.Tensor):
        *_, T, _ = input.shape
        first, second = input.chunk(2, dim=-1)  # each B, H, T, D_h/2
        half_rotated = torch.cat((-second, first), dim=-1)  # B, H, T, D_h
        i = torch.arange(T, device=input.device)[:, None]  # T, 1
        angles = i * self.freqs  # m*theta. T, D_h/2
        angles = torch.cat((angles, angles), dim=1)  # T, D_h

        return input * angles.cos() + half_rotated * angles.sin()


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
        self.WK, self.WQ, self.WV, self.WO = p(), p(), p(), p()
        self.rope = RoPE()

    def forward(self, input: torch.Tensor):
        B, T, D = input.shape
        assert D == D_MODEL

        C = input @ torch.cat([self.WK, self.WQ, self.WV], dim=1)  # B, T, 3D
        C = C.reshape(B, T, 3, NUM_HEADS, D_HEAD)
        C = C.permute(2, 0, 3, 1, 4)  # 3, B, H, T, D_h
        K, Q, V = C.unbind(0)  # each B, H, T, D_h
        K, Q = self.rope(K), self.rope(Q)

        K_T = K.transpose(3, 2)  # B, H, D_h, T

        OUT = Q @ K_T  # B, H, T, T

        temporal_mask = (
            torch.arange(T, device=input.device)[:, None]
            < torch.arange(T, device=input.device)[None, :]
        )  # T,T
        OUT = OUT.masked_fill(temporal_mask, float("-inf"))

        OUT = torch.softmax(OUT / D_HEAD**0.5, dim=-1)  # B, H, T, T
        OUT = OUT @ V  # B, H, T, D_h
        OUT = OUT.transpose(1, 2)  # B, T, H, D_h
        OUT = OUT.reshape(B, T, D)  # B, T, D
        OUT = OUT @ self.WO  # B, T, D

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
        self.emb = p(VOCAB_SIZE, D_MODEL, 2e-2)
        self.attn_blocks = nn.ModuleList(AttnBlock() for _ in range(NUM_LAYERS))
        self.proj = Lin(D_MODEL, VOCAB_SIZE)
        self.final_ln = LayerNorm()

    def forward(self, input):
        input = self.emb[input]
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
