mod encode;
mod train;
mod util;

use pyo3::prelude::*;
use pyo3_stub_gen::define_stub_info_gatherer;

use encode::{encode_corpus, encode_text};
use train::train_bpe;

#[pymodule]
fn poletti(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(train_bpe, m)?)?;
    m.add_function(wrap_pyfunction!(encode_corpus, m)?)?;
    m.add_function(wrap_pyfunction!(encode_text, m)?)?;

    Ok(())
}

define_stub_info_gatherer!(stub_info);
