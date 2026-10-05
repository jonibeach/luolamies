use std::collections::BinaryHeap;
use std::ops::Deref;

use ndarray::{ArrayView1, ArrayView2, Axis};
use numpy::ndarray::Zip;
use numpy::{Ix3, PyArray1, PyReadonlyArrayDyn};
use pyo3::types::PyList;
use pyo3::{Bound, IntoPyObject, PyAny, Python, pyclass, pymethods};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use pyo3_stub_gen::impl_stub_type;
use rayon::iter::ParallelIterator;

use crate::corpus::Corpus;
use crate::encoder::Encoder;
use crate::pairs::{Pairs, TokenPair};
use crate::util::{BASE_TOKEN, Token};

type PyArr1<'py, T> = Bound<'py, PyArray1<T>>;

fn merge_range(vocab_size: u16) -> std::ops::Range<u16> {
    BASE_TOKEN..vocab_size - 1
}

fn vocab(merge_table: &[TokenPair], vocab_size: u16) -> Vec<Vec<u8>> {
    let mut vocab = Vec::with_capacity(vocab_size as usize);
    vocab.extend((0..=u8::MAX).map(|char| vec![char]));

    for TokenPair(a, b) in merge_table.iter() {
        let merged = [vocab[*a as usize].as_slice(), vocab[*b as usize].as_slice()].concat();
        vocab.push(merged);
    }

    vocab
}

#[gen_stub_pyclass]
#[pyclass]
pub(crate) struct Tokenizer {
    pub(crate) merge_table: Vec<TokenPair>,
    vocab: Vec<Vec<u8>>,
    vocab_size: u16,
}

impl Tokenizer {
    fn decode_seq(&self, seq: ArrayView1<'_, Token>) -> String {
        let mut bytes = Vec::new();
        for &t in seq {
            bytes.extend_from_slice(&self.vocab[t as usize]);
        }
        match String::from_utf8(bytes) {
            Ok(s) => s,
            Err(e) => String::from_utf8_lossy(e.as_bytes()).to_string(),
        }
    }

    fn decode_batch(&self, batch: ArrayView2<'_, Token>) -> Vec<String> {
        batch.outer_iter().map(|seq| self.decode_seq(seq)).collect()
    }
}

#[derive(IntoPyObject)]
enum Decoded {
    Seq(String),
    Batch(Vec<String>),
    BatchTopK(Vec<Vec<String>>),
}
impl_stub_type!(Decoded = String | Vec<String> | Vec<Vec<String>>);

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

    fn decode<'py>(
        &self,
        py: Python<'py>,
        tokens: PyReadonlyArrayDyn<'py, Token>,
    ) -> anyhow::Result<Decoded> {
        let tokens = tokens.as_array();
        py.detach(|| {
            // Lets hope this is (T,), (B, T) or (B, T, K)
            let res = match tokens.ndim() {
                1 => Decoded::Seq(self.decode_seq(tokens.into_dimensionality()?)),
                2 => Decoded::Batch(self.decode_batch(tokens.into_dimensionality()?)),
                3 => Decoded::BatchTopK(
                    tokens
                        .into_dimensionality::<Ix3>()?
                        .outer_iter()
                        .map(|seqs| self.decode_batch(seqs.reversed_axes()))
                        .collect(),
                ),
                _ => anyhow::bail!(
                    "Expected input of shape (T, ), (B, T) or (B, T, K). Got {:?}",
                    tokens.shape()
                ),
            };

            Ok(res)
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
                    let Some(canon_count) = pair_counts.get(&ids) else {
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
