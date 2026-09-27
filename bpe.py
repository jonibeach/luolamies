import json
import poletti
from pathlib import Path

VOCAB_SIZE = 2**15


def bpe(corpus_path):
    bpe_path = Path("./data/tokenizer.json")
    if not bpe_path.exists():
        bpe_table = poletti.train_bpe(corpus_path, VOCAB_SIZE)
        with open(bpe_path, "w") as file:
            file.write(json.dumps(bpe_table))
    else:
        with open(bpe_path, "rb") as file:
            bpe_table = json.load(file)

    vocab = [bytes([i]) for i in range(256)]
    for a, b in bpe_table:
        vocab.append(vocab[a] + vocab[b])
    print([v.decode("utf-8", "backslashreplace") for v in vocab[256:1000]])
