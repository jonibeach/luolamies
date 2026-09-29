use std::fs::File;

use arrow::array::AsArray;
use fancy_regex::Regex;
use parquet::arrow::arrow_reader::{ArrowReaderMetadata, ParquetRecordBatchReaderBuilder};

const PAT: &str =
    r"[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
pub type Token = u16;

thread_local! { static RE: Regex = Regex::new(PAT).unwrap();}

pub(crate) enum Pretoken<'a> {
    Regular(&'a [u8]),
    EndOfText,
}

pub(crate) fn rowgroup_pretokens_foreach(
    parquet_path: &str,
    meta: ArrowReaderMetadata,
    i: usize,
    mut cb: impl for<'a> FnMut(Pretoken<'a>),
) -> anyhow::Result<()> {
    RE.with(|re| {
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

                let pretokens = re.find_iter(text);
                for pretoken in pretokens {
                    let bytes = pretoken?.as_str().as_bytes();
                    cb(Pretoken::Regular(bytes))
                }

                cb(Pretoken::EndOfText)
            }
        }

        Ok(())
    })
}
