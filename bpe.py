from collections import Counter
import poletti

VOCAB_SIZE = 2**15


def bpe(corpus_path):
    counts = Counter(poletti.count_pretokenized(corpus_path))
    print(counts.most_common(100))
