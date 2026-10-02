use std::{collections::BinaryHeap, fs::File};

use parquet::arrow::arrow_reader::ArrowReaderMetadata;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::gen_stub_pyfunction;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::util::{Pretoken, Token, rowgroup_pretokens_foreach};

type PairCounts = FxHashMap<(Token, Token), (usize, FxHashSet<usize>)>;
pub(crate) const BASE_TOKEN: Token = u8::MAX as Token + 1;

pub(crate) fn iter_row_groups<T: Send, O: Send + Sync + Fn(usize, ArrowReaderMetadata) -> T>(
    parquet_path: &str,
    op: O,
) -> anyhow::Result<impl ParallelIterator<Item = T>> {
    let file = File::open(parquet_path)?;

    let meta = ArrowReaderMetadata::load(&file, Default::default())?;

    Ok((0..meta.metadata().num_row_groups())
        .into_par_iter()
        .map(move |i| op(i, meta.clone())))
}

pub(crate) fn handle_count<'a>(counts: &mut FxHashMap<Vec<u8>, usize>, pretoken: Pretoken<'a>) {
    match pretoken {
        Pretoken::Regular(bytes) => match counts.get_mut(bytes) {
            Some(c) => *c += 1,
            None => {
                counts.insert(bytes.to_vec(), 1);
            }
        },
        Pretoken::EndOfText => {}
    }
}

fn count_row_group(
    parquet_path: &str,
    i: usize,
    meta: ArrowReaderMetadata,
) -> anyhow::Result<FxHashMap<Vec<u8>, usize>> {
    let mut counts = FxHashMap::default();

    rowgroup_pretokens_foreach(parquet_path, meta, i, |pt| handle_count(&mut counts, pt))?;

    Ok(counts)
}

pub(crate) fn count_pretokenized(parquet_path: &str) -> anyhow::Result<FxHashMap<Vec<u8>, usize>> {
    let counts = iter_row_groups(parquet_path, |i, meta| {
        count_row_group(parquet_path, i, meta)
    })?
    .try_reduce(FxHashMap::default, |a, b| {
        let (mut a, b) = if a.len() >= b.len() { (a, b) } else { (b, a) };
        for (k, v) in b {
            *a.entry(k).or_insert(0) += v;
        }
        Ok(a)
    })?;

    Ok(counts)
}

enum Mode {
    Incr,
    Decr,
}

pub(crate) fn build_pair_counts(words: &[(Vec<Token>, usize)]) -> PairCounts {
    let mut pc = PairCounts::default();

    for (idx, w) in words.iter().enumerate() {
        for i in 1..w.0.len() {
            record_pair((idx, w), (i - 1, i), Mode::Incr, &mut pc, None);
        }
    }

    pc
}

fn record_pair(
    (word_idx, (word, count)): (usize, &(Vec<Token>, usize)),
    (a, b): (usize, usize),
    mode: Mode,
    pair_counts: &mut PairCounts,
    tracker: Option<&mut FxHashSet<(Token, Token)>>,
) {
    let ids = (word[a], word[b]);
    let p = pair_counts.entry(ids).or_insert((0, FxHashSet::default()));
    match mode {
        Mode::Incr => {
            p.0 += count;
            p.1.insert(word_idx);
        }
        Mode::Decr => {
            p.0 -= count;
            if p.0 == 0 {
                pair_counts.remove(&ids);
            }
        }
    }
    let Some(t) = tracker else { return };
    t.insert(ids);
}

pub(crate) fn apply_merge(
    ids: (Token, Token),
    word_indices: &FxHashSet<usize>,
    words: &mut [(Vec<Token>, usize)],
    pair_counts: &mut PairCounts,
    mut changed: Option<&mut FxHashSet<(Token, Token)>>,
    next_id: Token,
) {
    for w_idx in word_indices {
        let Some(w) = words.get_mut(*w_idx) else {
            continue;
        };

        let mut i = 1;

        while i < w.0.len() {
            if (w.0[i - 1], w.0[i]) == ids {
                let pair = (i - 1, i);

                let mut rec = |w: &(Vec<Token>, usize), d: isize, mode: Mode| {
                    if d < 0 && i < 2 {
                        return;
                    }
                    if (i as isize + d) as usize >= w.0.len() {
                        return;
                    }
                    let p = (
                        (pair.0 as isize + d) as usize,
                        (pair.1 as isize + d) as usize,
                    );

                    record_pair((*w_idx, w), p, mode, pair_counts, changed.as_deref_mut());
                };

                for p in [-1, 0, 1] {
                    rec(w, p, Mode::Decr)
                }

                w.0[i - 1] = next_id;
                w.0.remove(i);

                for p in [-1, 0] {
                    rec(w, p, Mode::Incr)
                }

                continue;
            }

            i += 1;
        }
    }
}

pub(crate) fn word(bytes: &[u8]) -> Vec<Token> {
    bytes.iter().copied().map(Token::from).collect()
}

pub(crate) fn words(counts: FxHashMap<Vec<u8>, usize>) -> Vec<(Vec<Token>, usize)> {
    counts
        .into_iter()
        .map(|(bytes, n)| (word(&bytes), n))
        .collect()
}

#[gen_stub_pyfunction]
#[pyfunction]
pub(crate) fn train_bpe(
    py: Python<'_>,
    parquet_path: &str,
    vocab_size: u16,
) -> anyhow::Result<Vec<(Token, Token)>> {
    py.detach(|| {
        let counts = count_pretokenized(parquet_path)?;
        let mut words = words(counts);

        let mut pair_counts = build_pair_counts(&words);
        let mut pair_ranks = BinaryHeap::new();
        let mut merges = Vec::new();

        for (ids, (count, _)) in &pair_counts {
            pair_ranks.push((*count, *ids));
        }

        for next_id in BASE_TOKEN..vocab_size - 1 {
            let argmax = loop {
                let Some((count, ids)) = pair_ranks.pop() else {
                    break None;
                };
                let Some((canon_count, indices)) = pair_counts.get_mut(&ids) else {
                    continue;
                };
                if count == *canon_count {
                    break Some((ids, std::mem::take(indices)));
                }
            };

            let Some((ids, word_indices)) = argmax else {
                break;
            };

            let mut changed = FxHashSet::default();

            apply_merge(
                ids,
                &word_indices,
                &mut words,
                &mut pair_counts,
                Some(&mut changed),
                next_id,
            );

            for p in changed {
                if let Some((count, _)) = pair_counts.get(&p) {
                    pair_ranks.push((*count, p));
                }
            }

            merges.push(ids);
        }

        Ok(merges)
    })
}
