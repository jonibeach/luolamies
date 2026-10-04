use std::{fs::File, path::PathBuf};

use parquet::arrow::arrow_reader::ArrowReaderMetadata;
use pyo3::pyclass;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::util::{RowGroup, Word, WordCounts};

#[pyclass]
pub(crate) struct Corpus {
    pub(crate) path: PathBuf,
    pub(crate) meta: ArrowReaderMetadata,
}

impl Corpus {
    pub(crate) fn new(path: PathBuf) -> anyhow::Result<Self> {
        let f = File::open(&path)?;
        let meta = ArrowReaderMetadata::load(&f, Default::default())?;
        Ok(Self { path, meta })
    }

    pub(crate) fn row_groups(&self) -> impl ParallelIterator<Item = RowGroup<'_>> {
        (0..self.meta.metadata().num_row_groups())
            .into_par_iter()
            .map(move |i| RowGroup::new(&self, i))
    }

    pub(crate) fn word_counts(&self) -> anyhow::Result<WordCounts> {
        Ok(self
            .row_groups()
            .map(WordCounts::try_from)
            .try_reduce(WordCounts::default, |a, b| Ok(WordCounts::merge(a, b)))?)
    }
}
