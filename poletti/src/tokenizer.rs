use std::collections::BinaryHeap;
use std::ops::Deref;

use pyo3::pyclass;
use rayon::iter::ParallelIterator;

use crate::corpus::Corpus;
use crate::encoder::Encoder;
use crate::pairs::{Pairs, TokenPair};
use crate::util::{BASE_TOKEN, Token};

fn merge_range(vocab_size: u16) -> std::ops::Range<u16> {
    BASE_TOKEN..vocab_size - 1
}

#[pyclass]
pub(crate) struct Tokenizer {
    pub(crate) merge_table: Vec<TokenPair>,
    vocab_size: u16,
}

impl Tokenizer {
    fn encode_text(&self, text: &str) -> anyhow::Result<Vec<Token>> {
        let encoder = Encoder::from_iter([text], &self.merge_table);
        let mut out = Vec::new();
        encoder.encode(text, &mut out)?;
        Ok(out)
    }

    fn encode_corpus(&self, corpus: &Corpus) -> anyhow::Result<Vec<Token>> {
        let encoder = Encoder::from_corpus(corpus, &self.merge_table)?;

        let tokens = corpus
            .row_groups()
            .map(|rg| {
                let mut out = Vec::new();
                for t in rg.texts()? {
                    for t in t?.iter().flatten() {
                        encoder.encode(t, &mut out)?;
                        out.push(self.vocab_size - 1);
                    }
                }
                Ok(out)
            })
            .collect::<anyhow::Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect();

        Ok(tokens)
    }

    fn train(corpus: &Corpus, vocab_size: u16) -> anyhow::Result<Self> {
        let mut merge_table = Vec::with_capacity(merge_range(vocab_size).len());
        let mut pair_ranks = BinaryHeap::new();

        let mut words = corpus.word_counts()?.flat();
        let pair_counts = Pairs::from_words(&mut words);

        for (ids, count) in pair_counts.deref() {
            pair_ranks.push((count.val, *ids));
        }

        let mut pair_counts = pair_counts.to_logged();

        for next_id in merge_range(vocab_size) {
            let argmax = loop {
                let Some((count, ids)) = pair_ranks.pop() else {
                    break None;
                };
                let Some(canon_count) = pair_counts.get_mut(&ids) else {
                    continue;
                };
                if count == canon_count.val {
                    break Some(ids);
                }
            };

            let Some(ids) = argmax else {
                break;
            };

            pair_counts.merge(ids, next_id);

            for p in pair_counts.drain_log() {
                if let Some(count) = pair_counts.get(&p) {
                    pair_ranks.push((count.val, p));
                }
            }

            merge_table.push(ids);
        }

        Ok(Self {
            merge_table,
            vocab_size,
        })
    }
}
