use numpy::PyReadonlyArray1;
use psnailder_core::{ln_likelihood_f64_for, ln_likelihood_f64_iter};
use pyo3::prelude::*;

#[pyfunction]
fn ln_likelihood_for(
    data: PyReadonlyArray1<f64>,
    prediction: PyReadonlyArray1<f64>,
    mask: PyReadonlyArray1<f64>,
) -> PyResult<f64> {
    let data = data.as_slice()?;
    let prediction = prediction.as_slice()?;
    let mask = mask.as_slice()?;

    Ok(ln_likelihood_f64_for(data, prediction, mask))
}

#[pyfunction]
fn ln_likelihood_iter(
    data: PyReadonlyArray1<f64>,
    prediction: PyReadonlyArray1<f64>,
    mask: PyReadonlyArray1<f64>,
) -> PyResult<f64> {
    let data = data.as_slice()?;
    let prediction = prediction.as_slice()?;
    let mask = mask.as_slice()?;

    Ok(ln_likelihood_f64_iter(data, prediction, mask))
}
#[pymodule]
fn _internal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ln_likelihood_for, m)?)?;
    m.add_function(wrap_pyfunction!(ln_likelihood_iter, m)?)?;
    Ok(())
}
