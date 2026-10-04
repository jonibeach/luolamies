mod corpus;
mod encoder;
mod pairs;
mod tokenizer;
mod util;

use pyo3::prelude::*;
use pyo3_stub_gen::define_stub_info_gatherer;

use crate::{corpus::Corpus, tokenizer::Tokenizer};

#[pymodule]
fn poletti(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Tokenizer>()?;
    m.add_class::<Corpus>()?;

    Ok(())
}

define_stub_info_gatherer!(stub_info);
