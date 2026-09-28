from bpe import bpe
import poletti

from preprocess import preprocess


def main():
    corpus = preprocess()
    p = str(corpus.absolute())
    merges = bpe(p)
    tokenized_corpus = poletti.encode_corpus(p, merges)
    print(len(tokenized_corpus))


if __name__ == "__main__":
    main()
