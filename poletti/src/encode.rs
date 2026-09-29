use numpy::PyArray1;
use pyo3::{Bound, Python, pyfunction};
use pyo3_stub_gen::derive::gen_stub_pyfunction;
use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::{
    train::{
        BASE_TOKEN, apply_merge, build_pair_counts, count_pretokenized, iter_row_groups, word,
    },
    util::{Pretoken, Token, rowgroup_pretokens_foreach},
};

fn encode(words: &mut [(Vec<Token>, usize)], merge_table: &[(Token, Token)]) {
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

        let (keys, mut words): (Vec<_>, Vec<_>) = counts
            .into_iter()
            .map(|(bytes, n)| {
                let w = word(&bytes);
                (bytes, (w, n))
            })
            .unzip();

        words
            .par_chunks_mut(8192)
            .for_each(|c| encode(c, &merge_table));

        let byte_token_kv: FxHashMap<_, _> = keys
            .into_iter()
            .zip(words.into_iter().map(|w| w.0))
            .collect();

        let tokens = iter_row_groups(parquet_path, |i, meta| -> anyhow::Result<_> {
            let mut tokens = Vec::new();
            rowgroup_pretokens_foreach(parquet_path, meta, i, |pt| match pt {
                Pretoken::Regular(pt) => tokens.extend_from_slice(&byte_token_kv[pt]),
                Pretoken::EndOfText => tokens.push(vocab_size - 1),
            })?;
            Ok(tokens)
        })?
        .collect::<anyhow::Result<Vec<_>>>()?;

        let tokens = tokens.into_iter().flatten().collect();

        Ok(tokens)
    })?;

    Ok(PyArray1::from_vec(py, tokens))
}
