use numpy::PyArray1;
use pyo3::{Bound, Python, pyfunction};
use pyo3_stub_gen::derive::gen_stub_pyfunction;
use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::{
    train::{
        BASE_TOKEN, apply_merge, build_pair_counts, count_pretokenized, handle_count,
        iter_row_groups, word, words,
    },
    util::{Pretoken, Token, pretokens, rowgroup_pretokens_foreach},
};

fn encode_words(words: &mut [(Vec<Token>, usize)], merge_table: &[(Token, Token)]) {
    let mut pc = build_pair_counts(words);
    for (id, pair) in merge_table.iter().enumerate() {
        let Some((_, indices)) = pc.get_mut(pair) else {
            continue;
        };
        let indices = std::mem::take(indices);
        apply_merge(
            *pair,
            &indices,
            words,
            &mut pc,
            None,
            BASE_TOKEN + id as Token,
        );
    }
}

type PyArr1<'py, T> = Bound<'py, PyArray1<T>>;

fn build_byte_token_kv(
    counts: FxHashMap<Vec<u8>, usize>,
    merge_table: &[(Token, Token)],
) -> FxHashMap<Vec<u8>, Vec<Token>> {
    let (keys, mut words): (Vec<_>, Vec<_>) = counts
        .into_iter()
        .map(|(bytes, n)| {
            let w = word(&bytes);
            (bytes, (w, n))
        })
        .unzip();

    words
        .par_chunks_mut(8192)
        .for_each(|c| encode_words(c, merge_table));

    let byte_token_kv: FxHashMap<_, _> = keys
        .into_iter()
        .zip(words.into_iter().map(|w| w.0))
        .collect();

    byte_token_kv
}

fn add_token(
    pt: Pretoken<'_>,
    tokens: &mut Vec<Token>,
    byte_token_kv: &FxHashMap<Vec<u8>, Vec<Token>>,
    vocab_size: u16,
) {
    match pt {
        Pretoken::Regular(pt) => tokens.extend_from_slice(&byte_token_kv[pt]),
        Pretoken::EndOfText => tokens.push(vocab_size - 1),
    }
}

#[gen_stub_pyfunction]
#[pyfunction]
pub(crate) fn encode_corpus<'py>(
    py: Python<'py>,
    parquet_path: &str,
    merge_table: Vec<(Token, Token)>,
    vocab_size: u16,
) -> anyhow::Result<PyArr1<'py, Token>> {
    let tokens = py.detach(|| -> anyhow::Result<_> {
        let counts = count_pretokenized(parquet_path)?;

        let byte_token_kv = build_byte_token_kv(counts, &merge_table);

        let tokens = iter_row_groups(parquet_path, |i, meta| -> anyhow::Result<_> {
            let mut tokens = Vec::new();
            rowgroup_pretokens_foreach(parquet_path, meta, i, |pt| {
                add_token(pt, &mut tokens, &byte_token_kv, vocab_size)
            })?;
            Ok(tokens)
        })?
        .collect::<anyhow::Result<Vec<_>>>()?;

        let tokens = tokens.into_iter().flatten().collect();

        Ok(tokens)
    })?;

    Ok(PyArray1::from_vec(py, tokens))
}

#[gen_stub_pyfunction]
#[pyfunction]
pub(crate) fn encode_text<'py>(
    py: Python<'py>,
    input: String,
    merge_table: Vec<(Token, Token)>,
    vocab_size: u16,
) -> anyhow::Result<PyArr1<'py, Token>> {
    let tokens = py.detach(|| {
        let mut counts = FxHashMap::default();
        pretokens(&input).for_each(|pt| handle_count(&mut counts, pt));
        let byte_token_kv = build_byte_token_kv(counts, &merge_table);
        let mut tokens = Vec::new();
        pretokens(&input).for_each(|pt| add_token(pt, &mut tokens, &byte_token_kv, vocab_size));
        tokens
    });

    Ok(PyArray1::from_vec(py, tokens))
}
