import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .main import Model
from .optimizer import AdamW
from .utils import DEVICE

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


def run_microbatched(model: Model, batch: torch.Tensor):
    batches = batch.split(MICROBATCH)
    for mb in batches:
        x, y = mb[:, :-1], mb[:, 1:]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits: torch.Tensor = model(x)  # B, T, V
            yield (
                loss(logits.flatten(0, 1), y.flatten().long()) / len(batches)
            )  # B*T, V, and BT


def train(corpus: np.ndarray, epochs=5):
    dataset = torch.from_numpy(corpus.astype(np.int32)).to(DEVICE)
    (L,) = dataset.shape
    overflow = L % (CONTEXT_LEN + 1)
    dataset = dataset[: L - overflow].reshape(-1, CONTEXT_LEN + 1)
    (N, _) = dataset.shape

    test_val_size = int(N * 0.01)
    TRAIN = N - 2 * test_val_size
    val_end = N - test_val_size
    _TEST, _VAL = test_val_size, test_val_size
    train, val, _test = (
        dataset[:TRAIN, ...],
        dataset[TRAIN:val_end, ...],
        dataset[val_end:, ...],
    )

    (model, optim, initial_epoch, initial_batch) = restore_or_new()

    for e in range(initial_epoch, epochs):
        p = torch.randperm(TRAIN, device=DEVICE)
        batches = train[p].split(BATCH_SIZE)
        num_batches = len(batches)

        total_train_loss = 0
        num_loss_batches = 0

        for b, batch in enumerate(batches):
            if e == initial_epoch and b <= initial_batch:
                continue

            optim.zero_grad()

            for l in run_microbatched(model, batch):
                total_train_loss += l.item()
                l.backward()

            num_loss_batches += 1

            optim.step()

            if b % 100 == 0:
                checkpoint(model, optim, e, b)
                with torch.no_grad():
                    val_loss = next(run_microbatched(model, val[:MICROBATCH]))
                    print(
                        f"Epoch {e}, batch {b + 1}/{num_batches}, avg_train_loss: {total_train_loss / num_loss_batches}, val_loss: {val_loss.item()}",
                    )
                    num_loss_batches = 0
                    total_train_loss = 0

        with torch.no_grad():
            val_loss = 0
            for mb_val_loss in run_microbatched(model, val):
                val_loss += mb_val_loss.item()
            print(
                f"Epoch {e}, batch {b + 1}/{num_batches}, val_loss: {val_loss}",
            )

    checkpoint(model, optim, epochs, 0)

    return model
