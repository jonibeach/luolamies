import torch
from torch import nn
from .utils import D_HEAD, D_MODEL, NUM_HEADS, HIDDEN_MUL, p, MLP, LayerNorm
from .rope import RoPE


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
