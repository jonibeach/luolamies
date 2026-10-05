use std::collections::BinaryHeap;
use std::ops::Deref;

use numpy::{PyArray1, PyArrayMethods};
use pyo3::{Bound, Python, pyclass, pymethods};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use rayon::iter::ParallelIterator;

use crate::corpus::Corpus;
use crate::encoder::Encoder;
use crate::pairs::{Pairs, TokenPair};
use crate::util::{BASE_TOKEN, Token};

type PyArr1<'py, T> = Bound<'py, PyArray1<T>>;

fn merge_range(vocab_size: u16) -> std::ops::Range<u16> {
    BASE_TOKEN..vocab_size - 1
}

fn vocab(merge_table: &[TokenPair], vocab_size: u16) -> Vec<u8> {
    let mut vocab = vec![0; vocab_size as usize];
    vocab.extend(0..u8::MAX);

    for TokenPair(a, b) in merge_table.iter() {
        vocab.push(vocab[*a as usize] + vocab[*b as usize])
    }

    vocab
}

#[gen_stub_pyclass]
#[pyclass]
pub(crate) struct Tokenizer {
    pub(crate) merge_table: Vec<TokenPair>,
    vocab: Vec<u8>,
    vocab_size: u16,
}

#[gen_stub_pymethods]
#[pymethods]
impl Tokenizer {
    #[new]
    fn new(merge_table: Vec<TokenPair>, vocab_size: u16) -> Self {
        let vocab = vocab(&merge_table, vocab_size);
        Self {
            merge_table,
            vocab_size,
            vocab,
        }
    }

    #[getter]
    fn merges(&self) -> Vec<TokenPair> {
        self.merge_table.clone()
    }

    fn decode<'py>(&self, py: Python<'py>, tokens: PyArr1<'py, Token>) -> anyhow::Result<String> {
        let tokens = tokens.to_vec()?;
        py.detach(|| {
            Ok(String::from_utf8(
                tokens.iter().map(|t| self.vocab[*t as usize]).collect(),
            )?)
        })
    }

    fn encode_text<'py>(&self, py: Python<'py>, text: &str) -> anyhow::Result<PyArr1<'py, Token>> {
        let tokens = py.detach(|| -> anyhow::Result<_> {
            let encoder = Encoder::from_iter([text], &self.merge_table);
            let mut out = Vec::new();
            encoder.encode(text, &mut out)?;
            Ok(out)
        })?;

        Ok(PyArray1::from_vec(py, tokens))
    }

    fn encode_corpus<'py>(
        &self,
        py: Python<'py>,
        corpus: &Bound<'py, Corpus>,
    ) -> anyhow::Result<PyArr1<'py, Token>> {
        let corpus = corpus.get();
        let tokens = py.detach(|| -> anyhow::Result<_> {
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
        })?;

        Ok(PyArray1::from_vec(py, tokens))
    }

    #[staticmethod]
    fn train(py: Python<'_>, corpus: &Bound<'_, Corpus>, vocab_size: u16) -> anyhow::Result<Self> {
        let corpus = corpus.get();
        py.detach(|| {
            let mut merge_table = Vec::with_capacity(merge_range(vocab_size).len());
            let mut pair_ranks = BinaryHeap::new();

            let mut words = corpus.word_counts()?.flat();
            let pair_counts = Pairs::from_words(&mut words);

            for (ids, count) in pair_counts.deref() {
                pair_ranks.push((count.val, *ids));
            }

            let mut pair_counts = pair_counts.into_logged();

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

            let vocab = vocab(&merge_table, vocab_size);

            Ok(Self {
                merge_table,
                vocab_size,
                vocab,
            })
        })
    }
}
