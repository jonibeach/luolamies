use std::{fs::File, path::PathBuf};

use parquet::arrow::arrow_reader::ArrowReaderMetadata;
use pyo3::{pyclass, pymethods};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::util::{RowGroup, WordCounts};

#[gen_stub_pyclass]
#[pyclass(frozen)]
pub(crate) struct Corpus {
    pub(crate) path: PathBuf,
    pub(crate) meta: ArrowReaderMetadata,
}

#[gen_stub_pymethods]
#[pymethods]
impl Corpus {
    #[new]
    pub(crate) fn new(path: PathBuf) -> anyhow::Result<Self> {
        let f = File::open(&path)?;
        let meta = ArrowReaderMetadata::load(&f, Default::default())?;
        Ok(Self { path, meta })
    }
}

impl Corpus {
    pub(crate) fn row_groups(&self) -> impl ParallelIterator<Item = RowGroup<'_>> {
        (0..self.meta.metadata().num_row_groups())
            .into_par_iter()
            .map(move |i| RowGroup::new(self, i))
    }

    pub(crate) fn word_counts(&self) -> anyhow::Result<WordCounts> {
        self.row_groups()
            .map(WordCounts::try_from)
            .try_reduce(WordCounts::default, |a, b| Ok(WordCounts::merge(a, b)))
    }
}
