import argparse
from enum import StrEnum
from pathlib import Path

import numpy as np
import torch

import model
import poletti
from bpe import bpe
from preprocess import preprocess


class Mode(StrEnum):
    Train = "train"
    Run = "run"


parser = argparse.ArgumentParser()
modes = parser.add_subparsers(dest="mode", required=True)

train = modes.add_parser(Mode.Train)
run = modes.add_parser(Mode.Run)
run.add_argument("input")


def main():
    args = parser.parse_args()
    corpus = preprocess()
    p = str(corpus.absolute())
    tokenizer = bpe(p)

    corpus_path = Path("./data/corpus.npy")
    if corpus_path.exists():
        tokenized_corpus = np.load(corpus_path)
    else:
        corpus = poletti.Corpus(p)
        tokenized_corpus = tokenizer.encode_corpus(corpus)
        np.save(corpus_path, tokenized_corpus)

    mode = Mode(args.mode)

    if mode == Mode.Train:
        model.train(tokenized_corpus)
    else:
        with torch.no_grad():
            ds = model.dataset(tokenized_corpus)
            m, *_ = model.restore_or_new(ds.total_steps)
            input = tokenizer.encode_text(args.input)
            output = m(torch.from_numpy(input.astype(np.int32)))
            top10 = torch.topk(output, 10, dim=-1).indices
            top10 = top10[:, -1, :]
            text = tokenizer.decode(top10[None].cpu().numpy().astype(np.uint16))
            print(text)


if __name__ == "__main__":
    main()
