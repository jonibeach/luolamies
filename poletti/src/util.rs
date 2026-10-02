use std::fs::File;

use arrow::array::AsArray;
use fancy_regex::Regex;
use parquet::arrow::arrow_reader::{ArrowReaderMetadata, ParquetRecordBatchReaderBuilder};

const PAT: &str =
    r"[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
pub type Token = u16;

thread_local! {
    static RE: &'static Regex = Box::leak(Box::new(Regex::new(PAT).unwrap()));
}

pub(crate) enum Pretoken<'a> {
    Regular(&'a [u8]),
    EndOfText,
}

pub(crate) fn pretokens<'a>(text: &'a str) -> impl Iterator<Item = Pretoken<'a>> {
    let re = RE.with(|re| *re);
    re.find_iter(text)
        .filter_map(|p| p.ok())
        .map(|p| p.as_str().as_bytes())
        .map(Pretoken::Regular)
}

pub(crate) fn rowgroup_pretokens_foreach(
    parquet_path: &str,
    meta: ArrowReaderMetadata,
    i: usize,
    mut cb: impl for<'a> FnMut(Pretoken<'a>),
) -> anyhow::Result<()> {
    let file = File::open(parquet_path)?;
    let reader = ParquetRecordBatchReaderBuilder::new_with_metadata(file, meta)
        .with_row_groups(vec![i])
        .with_batch_size(8192)
        .build()?;

    for batch in reader {
        let batch = batch?;
        let texts = batch
            .column_by_name("text")
            .ok_or_else(|| anyhow::anyhow!("no 'text' col"))?
            .as_string::<i32>();
        for text in texts {
            let Some(text) = text else { continue };
            pretokens(text).for_each(&mut cb);
            cb(Pretoken::EndOfText)
        }
    }

    Ok(())
}
