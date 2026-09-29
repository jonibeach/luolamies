import json
from pathlib import Path

import poletti

VOCAB_SIZE = 2**15


def bpe(corpus_path):
    bpe_path = Path("./data/tokenizer.json")
    if not bpe_path.exists():
        merges = poletti.train_bpe(corpus_path, VOCAB_SIZE)
        with open(bpe_path, "w") as file:
            file.write(json.dumps(merges))
    else:
        with open(bpe_path, "rb") as file:
            merges = json.load(file)
            merges = [tuple(m) for m in merges]

    vocab = [bytes([i]) for i in range(256)]
    for a, b in merges:
        vocab.append(vocab[a] + vocab[b])
    print([v.decode("utf-8", "backslashreplace") for v in vocab[256:1000]])

    return merges
