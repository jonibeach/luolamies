from bpe import bpe
from preprocess import preprocess


def main():
    corpus = preprocess()
    p = str(corpus.absolute())
    merges = bpe(p)
    # tokenized_corpus = poletti.encode_corpus(p, merges)


if __name__ == "__main__":
    main()
