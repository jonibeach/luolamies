use std::{fs::File, hash::Hash, ops::AddAssign};

use arrow::array::{AsArray, StringArray};
use fancy_regex::Regex;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use rustc_hash::FxHashMap;

use crate::corpus::Corpus;

const PAT: &str =
    r"[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
pub type Token = u16;
pub(crate) const BASE_TOKEN: Token = u8::MAX as Token + 1;

thread_local! {
    static RE: &'static Regex = Box::leak(Box::new(Regex::new(PAT).unwrap()));
}

pub(crate) struct RowGroup<'a> {
    corpus: &'a Corpus,
    index: usize,
}

impl<'a> RowGroup<'a> {
    pub(crate) fn new(corpus: &'a Corpus, index: usize) -> Self {
        Self { corpus, index }
    }

    pub(crate) fn texts(
        &self,
    ) -> anyhow::Result<impl Iterator<Item = anyhow::Result<StringArray>>> {
        let f = File::open(&self.corpus.path)?;
        let reader =
            ParquetRecordBatchReaderBuilder::new_with_metadata(f, self.corpus.meta.clone())
                .with_row_groups(vec![self.index])
                .with_batch_size(8192)
                .build()?;

        Ok(reader.map(|b| {
            let b = b?;
            let texts = b
                .column_by_name("text")
                .ok_or_else(|| anyhow::anyhow!("no 'text' col"))?
                .as_string::<i32>()
                .clone();
            Ok(texts)
        }))
    }
}

#[derive(Default)]
pub(crate) struct WordCounts(pub(crate) FxHashMap<Vec<u8>, usize>);

impl<'a, T: IntoIterator<Item = &'a str>> From<T> for WordCounts {
    fn from(value: T) -> Self {
        let mut s = Self::default();
        s.extend(value);
        s
    }
}

impl<'a> Extend<&'a str> for WordCounts {
    fn extend<T: IntoIterator<Item = &'a str>>(&mut self, iter: T) {
        for pretoken in iter.into_iter().flat_map(pretokens) {
            match self.0.get_mut(pretoken) {
                Some(count) => *count += 1,
                None => {
                    self.0.insert(pretoken.to_vec(), 1);
                }
            }
        }
    }
}

impl TryFrom<RowGroup<'_>> for WordCounts {
    type Error = anyhow::Error;
    fn try_from(row_group: RowGroup<'_>) -> anyhow::Result<Self> {
        row_group
            .texts()?
            .try_fold(Self::default(), |mut c, t| -> anyhow::Result<_> {
                c.extend(t?.iter().flatten());
                Ok(c)
            })
    }
}

impl WordCounts {
    pub(crate) fn merge(Self(a): Self, Self(b): Self) -> Self {
        let (mut a, b) = if a.len() >= b.len() { (a, b) } else { (b, a) };
        for (k, v) in b {
            *a.entry(k).or_default() += v;
        }
        Self(a)
    }

    pub(crate) fn flat(&self) -> Vec<Word> {
        self.0
            .iter()
            .map(|(tokens, c)| Word {
                tokens: tokens.iter().copied().map(Token::from).collect(),
                count: *c,
            })
            .collect()
    }
}

#[derive(Default)]
pub(crate) struct Word {
    pub(crate) tokens: Vec<Token>,
    pub(crate) count: usize,
}

pub(crate) fn pretokens(text: &str) -> impl Iterator<Item = &[u8]> {
    let re = RE.with(|re| *re);
    re.find_iter(text)
        .filter_map(|p| p.ok())
        .map(|p| p.as_str().as_bytes())
}
