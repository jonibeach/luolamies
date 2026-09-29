from bpe import bpe, VOCAB_SIZE
import poletti
import model

from preprocess import preprocess


def main():
    corpus = preprocess()
    p = str(corpus.absolute())
    merges = bpe(p)
    tokenized_corpus = poletti.encode_corpus(p, merges, VOCAB_SIZE)
    print(len(tokenized_corpus))
    model.train(tokenized_corpus)


if __name__ == "__main__":
    main()
