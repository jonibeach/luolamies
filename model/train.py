import numpy as np
import torch
from torch import nn

from .utils import DEVICE
from .main import Model
from .optimizer import AdamW


BATCH_SIZE = 2**10
loss = nn.CrossEntropyLoss()


def train(corpus: np.ndarray, starts: np.ndarray, epochs=5):
    dataset = torch.from_numpy(corpus).to(DEVICE)
    article_starts = torch.from_numpy(starts).to(DEVICE)
    model = Model().to(DEVICE)
    optim = AdamW(model.parameters())
    for e in range(epochs):
        for sample in dataset:
            model(sample)
