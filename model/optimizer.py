from collections.abc import Iterable

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[dict[str, torch.Tensor]],
        lr=1e-4,
        weight_decay=2e-2,
        betas=(1 - 1e-1, 1 - 1e-3),
        eps=1e-8,
    ):
        defaults = {"lr": lr, "weight_decay": weight_decay, "betas": betas, "eps": eps}
        super().__init__(params, defaults)
        for g in self.param_groups:
            for p in g["params"]:
                s = self.state[p]
                s["m"] = torch.zeros_like(p)
                s["v"] = torch.zeros_like(p)
                s["step"] = 0

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            beta1, beta2 = g["betas"]
            lr = g["lr"]
            eps = g["eps"]
            weight_decay = g["weight_decay"]
            for _p in g["params"]:
                p: torch.Tensor = _p
                if p.grad is None:
                    continue

                s = self.state[p]
                s["step"] += 1

                m: torch.Tensor = s["m"]
                v: torch.Tensor = s["v"]
                step = s["step"]

                m.mul_(beta1).add_(p.grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)
                corr_m = m / (1 - beta1**step)
                corr_v = v / (1 - beta2**step)

                p.mul_(1 - lr * weight_decay)
                p.addcdiv_(corr_m, corr_v.sqrt() + eps, value=-lr)
