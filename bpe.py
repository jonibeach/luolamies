import json
from pathlib import Path

import poletti

VOCAB_SIZE = 2**13


def bpe(corpus_path):
    bpe_path = Path("./data/tokenizer.json")
    if not bpe_path.exists():
        corpus = poletti.Corpus(corpus_path)
        tokenizer = poletti.Tokenizer.train(corpus, VOCAB_SIZE)
        with open(bpe_path, "w") as file:
            file.write(json.dumps(tokenizer.merges))
    else:
        with open(bpe_path, "rb") as file:
            merges = json.load(file)
            merges = [tuple(m) for m in merges]
            tokenizer = poletti.Tokenizer(merges, VOCAB_SIZE)

    return tokenizer
