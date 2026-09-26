use std::{fs::File, sync::LazyLock};

use arrow::array::AsArray;
use fancy_regex::Regex;
use parquet::arrow::arrow_reader::{ArrowReaderMetadata, ParquetRecordBatchReaderBuilder};
use pyo3::prelude::*;
use pyo3_stub_gen::{define_stub_info_gatherer, derive::gen_stub_pyfunction};
use rayon::prelude::*;
use rustc_hash::FxHashMap;

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

#[gen_stub_pyfunction]
#[gen_stub(override_return_type(type_repr = "dict[bytes, int]"))]
#[pyfunction]
fn count_pretokenized(
    py: Python<'_>,
    parquet_path: &str,
) -> anyhow::Result<FxHashMap<Vec<u8>, usize>> {
    py.detach(|| {
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
    })
}

#[pymodule]
fn poletti(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(count_pretokenized, m)?)?;
    Ok(())
}

define_stub_info_gatherer!(stub_info);
