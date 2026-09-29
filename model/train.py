import numpy as np
import torch
from torch import nn

from .utils import DEVICE
from .main import Model
from .optimizer import AdamW


BATCH_SIZE = 2**4
CONTEXT_LEN = 2**8
loss = nn.CrossEntropyLoss()


def train(corpus: np.ndarray, epochs=5):
    dataset = torch.from_numpy(corpus.astype(np.int32)).to(DEVICE)
    (L,) = dataset.shape
    overflow = L % (CONTEXT_LEN + 1)
    dataset = dataset[: L - overflow].reshape(-1, CONTEXT_LEN + 1)
    (N, _) = dataset.shape

    model = Model().to(DEVICE)
    optim = AdamW(model.parameters())

    print(dataset.dtype)

    for e in range(epochs):
        p = torch.randperm(N, device=DEVICE)
        batches = dataset[p].split(BATCH_SIZE)
        num_batches = len(batches)

        for b, batch in enumerate(batches):
            x, y = batch[:, :-1], batch[:, 1:]
            logits: torch.Tensor = model(x)  # B, T, V
            l = loss(logits.flatten(0, 1), y.flatten().long())  # B*T, V, and BT
            print(
                f"Epoch {e}, batch {b + 1}/{num_batches}, loss {l}", l.item(), type(l)
            )
            optim.zero_grad()
            l.backward()
            optim.step()

    return model
