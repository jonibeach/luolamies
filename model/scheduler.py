from torch.optim.lr_scheduler import LRScheduler
from torch.optim import Optimizer
import math


class CosineDecay(LRScheduler):
    def __init__(self, optimizer: Optimizer, steps: int, min_ratio=0.1):
        self.steps = steps
        self.min_ratio = min_ratio
        super().__init__(optimizer)

    def get_lr(self):
        t = self.last_epoch

        f = self.min_ratio
        if t < self.steps:
            ratio = t / (self.steps)
            f += 0.5 * (1 - self.min_ratio) * (1 + math.cos(math.pi * ratio))

        return [f * b for b in self.base_lrs]


class Warmup(LRScheduler):
    def __init__(self, optimizer: Optimizer, steps: int, initial_ratio=0.1):
        self.steps = steps
        self.initial_ratio = initial_ratio
        super().__init__(optimizer)

    def get_lr(self):
        t = self.last_epoch
        if t < self.steps:
            f = self.initial_ratio + (1 - self.initial_ratio) * t / (self.steps)
        else:
            f = 1

        return [f * b for b in self.base_lrs]
