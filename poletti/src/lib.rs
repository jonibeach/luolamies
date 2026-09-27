use std::{collections::BinaryHeap, fs::File, sync::LazyLock};

use arrow::array::AsArray;
use fancy_regex::Regex;
use parquet::arrow::arrow_reader::{ArrowReaderMetadata, ParquetRecordBatchReaderBuilder};
use pyo3::prelude::*;
use pyo3_stub_gen::{define_stub_info_gatherer, derive::gen_stub_pyfunction};
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

const PAT: &str =
    r"[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
static RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(PAT).unwrap());

fn count(
    parquet_path: &str,
    i: usize,
    meta: ArrowReaderMetadata,
) -> anyhow::Result<FxHashMap<Vec<u8>, usize>> {
    let file = File::open(parquet_path)?;
    let reader = ParquetRecordBatchReaderBuilder::new_with_metadata(file, meta)
        .with_row_groups(vec![i])
        .with_batch_size(8192)
        .build()?;

    let mut counts = FxHashMap::default();
    for batch in reader {
        let batch = batch?;
        let texts = batch
            .column_by_name("text")
            .ok_or_else(|| anyhow::anyhow!("no 'text' col"))?
            .as_string::<i32>();
        for text in texts {
            let Some(text) = text else { continue };

            let pretokens = RE.find_iter(text);
            for pretoken in pretokens {
                let bytes = pretoken?.as_str().as_bytes();
                match counts.get_mut(bytes) {
                    Some(c) => *c += 1,
                    None => {
                        counts.insert(bytes.to_vec(), 1);
                    }
                }
            }
        }
    }

    Ok(counts)
}

fn count_pretokenized(parquet_path: &str) -> anyhow::Result<FxHashMap<Vec<u8>, usize>> {
    let file = File::open(parquet_path)?;

    let meta = ArrowReaderMetadata::load(&file, Default::default())?;

    let counts = (0..meta.metadata().num_row_groups())
        .into_par_iter()
        .map(|i| count(parquet_path, i, meta.clone()))
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

fn record_pair(
    (word_idx, (word, count)): (usize, &(Vec<u32>, usize)),
    (a, b): (usize, usize),
    mode: Mode,
    pair_counts: &mut FxHashMap<(u32, u32), (usize, FxHashSet<usize>)>,
    tracker: Option<&mut FxHashSet<(u32, u32)>>,
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

#[gen_stub_pyfunction]
#[pyfunction]
fn train_bpe(
    py: Python<'_>,
    parquet_path: &str,
    num_merges: usize,
) -> anyhow::Result<Vec<(u32, u32)>> {
    py.detach(|| {
        let counts = count_pretokenized(parquet_path)?;
        let mut words: Vec<_> = counts
            .into_iter()
            .map(|(bytes, n)| (bytes.into_iter().map(u32::from).collect::<Vec<_>>(), n))
            .collect();

        let mut pair_counts = FxHashMap::default();
        let mut pair_ranks = BinaryHeap::new();
        let mut merges = Vec::new();
        let mut next_id = u8::MAX as u32 + 1;

        for (idx, w) in words.iter().enumerate() {
            for i in 1..w.0.len() {
                record_pair((idx, w), (i - 1, i), Mode::Incr, &mut pair_counts, None);
            }
        }

        for (ids, (count, _)) in &pair_counts {
            pair_ranks.push((*count, *ids));
        }

        for _ in 0..num_merges {
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
            for w_idx in word_indices {
                let Some(w) = words.get_mut(w_idx) else {
                    continue;
                };

                let mut i = 1;

                while i < w.0.len() {
                    if (w.0[i - 1], w.0[i]) == ids {
                        let pair = (i - 1, i);

                        let mut rec = |w: &(Vec<u32>, usize), d: isize, mode: Mode| {
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
                            record_pair((w_idx, w), p, mode, &mut pair_counts, Some(&mut changed));
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

            for p in changed {
                if let Some((count, _)) = pair_counts.get(&p) {
                    pair_ranks.push((*count, p));
                }
            }

            merges.push(ids);
            next_id += 1;
        }

        Ok(merges)
    })
}

#[pymodule]
fn poletti(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(train_bpe, m)?)?;
    Ok(())
}

define_stub_info_gatherer!(stub_info);
