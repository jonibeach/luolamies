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
    Train = ("train",)
    Run = "run"


parser = argparse.ArgumentParser()
modes = parser.add_subparsers(dest="mode", required=True)

train = modes.add_parser(Mode.Train)
run = modes.add_parser(Mode.Run)
run.add_argument("input")

parser.add_argument("mode", type=Mode, choices=list(Mode))


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

    print(len(tokenized_corpus))
    mode = Mode(args.mode)

    if mode == Mode.Train:
        model.train(tokenized_corpus)
    else:
        m, *_ = model.restore_or_new()
        input = tokenizer.encode_text(args.input)
        output = m(torch.from_numpy(input.astype(np.int32)))
        output = tokenizer.decode(output.cpu().numpy())
        print(output)


if __name__ == "__main__":
    main()
