from preprocess import preprocess
from bpe import bpe


def main():
    corpus = preprocess()
    bpe(str(corpus.absolute()))


if __name__ == "__main__":
    main()
