import torch
from torch import nn
from .utils import D_HEAD, DEVICE


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
