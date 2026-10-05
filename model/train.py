from dataclasses import dataclass
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.lr_scheduler import SequentialLR
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .main import Model
from .optimizer import AdamW
from .utils import DEVICE
from .scheduler import Warmup, CosineDecay
from .utils import clip_grad_norm

BATCH_SIZE, CONTEXT_LEN = (2**7, 2**10) if DEVICE == "cuda" else (2**3, 2**8)
NUM_MICROBATCHES = 2**2 if DEVICE == "cuda" else 1
MICROBATCH = BATCH_SIZE // NUM_MICROBATCHES


loss = nn.CrossEntropyLoss()

CHECKPOINT_FILE = os.environ.get("CHECKPOINT_FILE", "checkpoint.pt")


def checkpoint(m: Model, o: AdamW, sched: LRScheduler, epoch: int, batch: int):
    tmp = CHECKPOINT_FILE + ".tmp"
    torch.save(
        {
            "model": m.state_dict(),
            "optim": o.state_dict(),
            "sched": sched.state_dict(),
            "epoch": epoch,
            "batch": batch,
        },
        tmp,
    )
    os.replace(tmp, CHECKPOINT_FILE)


def restore_or_new(total_steps: int, f: str = CHECKPOINT_FILE):
    p = Path(f)
    model = Model().to(DEVICE)
    optim = AdamW(model.parameters())
    W = int(1e3)
    warmup = Warmup(optim, W)
    cosine = CosineDecay(optim, total_steps - W)
    sched = SequentialLR(optim, [warmup, cosine], [W])

    if not p.exists():
        return (model, optim, sched, 0, -1)

    c = torch.load(f, map_location=DEVICE)
    model.load_state_dict(c["model"])
    optim.load_state_dict(c["optim"])
    sched.load_state_dict(c["sched"])
    return (model, optim, sched, int(c["epoch"]), int(c["batch"]))


def run_microbatched(model: Model, batch: torch.Tensor):
    batches = batch.split(MICROBATCH)
    for mb in batches:
        x, y = mb[:, :-1], mb[:, 1:]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits: torch.Tensor = model(x)  # B, T, V
            yield (
                loss(logits.flatten(0, 1), y.flatten().long()) / len(batches)
            )  # B*T, V, and BT


@dataclass
class Dataset:
    train: torch.Tensor
    val: torch.Tensor
    test: torch.Tensor
    total_steps: int


def dataset(corpus: np.ndarray, epochs: int = 5):
    dataset = torch.from_numpy(corpus.astype(np.int32)).to(DEVICE)
    (L,) = dataset.shape
    overflow = L % (CONTEXT_LEN + 1)
    dataset = dataset[: L - overflow].reshape(-1, CONTEXT_LEN + 1)
    (N, _) = dataset.shape

    test_val_size = int(N * 0.01)
    TRAIN = N - 2 * test_val_size
    val_end = N - test_val_size
    train, val, test = (
        dataset[:TRAIN, ...],
        dataset[TRAIN:val_end, ...],
        dataset[val_end:, ...],
    )

    TOTAL_STEPS = epochs * TRAIN // BATCH_SIZE

    return Dataset(train, val, test, TOTAL_STEPS)


def train(corpus: np.ndarray, epochs: int = 5):
    ds = dataset(corpus, epochs)
    (model, optim, sched, initial_epoch, initial_batch) = restore_or_new(ds.total_steps)

    for e in range(initial_epoch, epochs):
        p = torch.randperm(len(ds.train), device=DEVICE)
        batches = ds.train[p].split(BATCH_SIZE)
        num_batches = len(batches)

        train_losses = []
        grad_norms = []

        for b, batch in enumerate(batches):
            if e == initial_epoch and b <= initial_batch:
                continue

            optim.zero_grad()

            batch_loss = 0
            for l in run_microbatched(model, batch):
                batch_loss += l.detach()
                l.backward()
            train_losses.append(batch_loss)

            grad_norms.append(clip_grad_norm(model.parameters()))
            optim.step()
            sched.step()

            if b % 100 == 0:
                checkpoint(model, optim, sched, e, b)
                with torch.no_grad():
                    val_loss = next(run_microbatched(model, ds.val[:MICROBATCH]))
                    t = torch.stack(train_losses)
                    g = torch.stack(grad_norms)
                    print(
                        f"Epoch {e}, batch {b + 1}/{num_batches}",
                        f"train_loss: {t.mean().item()}+-{t.std().item()}",
                        f"grad_norms:{g.mean().item()}+-{g.std().item()} (max={g.max().item()})",
                        f"val_loss: {val_loss.item()}",
                    )
                    train_losses = []
                    grad_norms = []

        with torch.no_grad():
            val_loss = 0
            for mb_val_loss in run_microbatched(model, ds.val):
                val_loss += mb_val_loss.item()
            print(
                f"Epoch {e}, batch {b + 1}/{num_batches}, val_loss: {val_loss}",
            )

    checkpoint(model, optim, sched, epochs, 0)

    return model
