from bpe import bpe
import poletti
import model

from preprocess import preprocess


def main():
    corpus = preprocess()
    p = str(corpus.absolute())
    merges = bpe(p)
    starts, tokenized_corpus = poletti.encode_corpus(p, merges)
    print(len(tokenized_corpus))
    model.train(tokenized_corpus, starts)


if __name__ == "__main__":
    main()
