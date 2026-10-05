use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::{
    corpus::Corpus,
    pairs::{Pairs, TokenPair},
    util::{BASE_TOKEN, Token, WordCounts, pretokens},
};

pub(crate) struct Encoder {
    kv: FxHashMap<Vec<u8>, Vec<Token>>,
}

impl Encoder {
    pub(crate) fn new(word_counts: WordCounts, merge_table: &[TokenPair]) -> Self {
        let mut words = word_counts.flat();
        words
            .par_chunks_mut(8192)
            .map(Pairs::from_words)
            .for_each(|mut p| {
                merge_table
                    .iter()
                    .enumerate()
                    .for_each(|(i, m)| p.merge(*m, BASE_TOKEN + i as Token))
            });

        let kv: FxHashMap<_, _> = word_counts
            .0
            .into_keys()
            .zip(words.into_iter().map(|w| w.tokens))
            .collect();

        Self { kv }
    }

    pub(crate) fn from_iter<'a>(
        i: impl IntoIterator<Item = &'a str>,
        merge_table: &[TokenPair],
    ) -> Self {
        Self::new(WordCounts::from(i), merge_table)
    }

    pub(crate) fn from_corpus(corpus: &Corpus, merge_table: &[TokenPair]) -> anyhow::Result<Self> {
        corpus.word_counts().map(|w| Self::new(w, merge_table))
    }

    pub(crate) fn encode(&self, t: &str, out: &mut Vec<Token>) -> anyhow::Result<()> {
        for pt in pretokens(t) {
            out.extend_from_slice(&self.kv[pt]);
        }
        Ok(())
    }
}
