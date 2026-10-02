from bpe import bpe, VOCAB_SIZE
import poletti
import model
import numpy as np
from pathlib import Path

from preprocess import preprocess


def main():
    corpus = preprocess()
    p = str(corpus.absolute())
    merges = bpe(p)
    corpus_path = Path("./data/corpus.npy")
    if corpus_path.exists():
        tokenized_corpus = np.load(corpus_path)
    else:
        tokenized_corpus = poletti.encode_corpus(p, merges, VOCAB_SIZE)
        np.save(corpus_path, tokenized_corpus)

    print(len(tokenized_corpus))
    model.train(tokenized_corpus)


if __name__ == "__main__":
    main()
