import numpy as np
import torch
from torch import nn
import os
from pathlib import Path

from .utils import DEVICE
from .main import Model
from .optimizer import AdamW


BATCH_SIZE, CONTEXT_LEN = (2**7, 2**10) if DEVICE == "cuda" else (2**3, 2**8)
NUM_MICROBATCHES = 2**2 if DEVICE == "cuda" else 1
MICROBATCH = BATCH_SIZE // NUM_MICROBATCHES


loss = nn.CrossEntropyLoss()

CHECKPOINT_FILE = os.environ.get("CHECKPOINT_FILE", "checkpoint.pt")


def checkpoint(m: Model, o: AdamW, epoch: int, batch: int):
    tmp = CHECKPOINT_FILE + ".tmp"
    torch.save(
        {
            "model": m.state_dict(),
            "optim": o.state_dict(),
            "epoch": epoch,
            "batch": batch,
        },
        tmp,
    )
    os.replace(tmp, CHECKPOINT_FILE)


def restore_or_new(f: str = CHECKPOINT_FILE):
    p = Path(f)
    model = Model().to(DEVICE)
    optim = AdamW(model.parameters())

    if not p.exists():
        return (model, optim, 0, -1)

    c = torch.load(f, map_location=DEVICE)
    model.load_state_dict(c["model"])
    optim.load_state_dict(c["optim"])
    return (model, optim, int(c["epoch"]), int(c["batch"]))


def train(corpus: np.ndarray, epochs=5):
    dataset = torch.from_numpy(corpus.astype(np.int32)).to(DEVICE)
    (L,) = dataset.shape
    overflow = L % (CONTEXT_LEN + 1)
    dataset = dataset[: L - overflow].reshape(-1, CONTEXT_LEN + 1)
    (N, _) = dataset.shape

    (model, optim, initial_epoch, initial_batch) = restore_or_new()

    for e in range(initial_epoch, epochs):
        p = torch.randperm(N, device=DEVICE)
        batches = dataset[p].split(BATCH_SIZE)
        num_batches = len(batches)

        for b, batch in enumerate(batches):
            if e == initial_epoch and b <= initial_batch:
                continue

            optim.zero_grad()

            for mb in batch.split(MICROBATCH):
                x, y = mb[:, :-1], mb[:, 1:]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits: torch.Tensor = model(x)  # B, T, V
                    l = (
                        loss(logits.flatten(0, 1), y.flatten().long())
                        / NUM_MICROBATCHES
                    )  # B*T, V, and BT
                l.backward()

            optim.step()

            if b % 100 == 0:
                checkpoint(model, optim, e, b)
                print(
                    f"Epoch {e}, batch {b + 1}/{num_batches}, loss {l.item() * NUM_MICROBATCHES}",
                )

    checkpoint(model, optim, epochs, 0)

    return model
