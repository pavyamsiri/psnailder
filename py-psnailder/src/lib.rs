use pyo3::prelude::*;

// Import from your actual crates
use psnailder_core::Foo;

#[pyclass]
pub struct PyFoo(Foo);

#[pymethods]
impl PyFoo {
    #[new]
    fn new(x: i64) -> Self {
        PyFoo(Foo::new(x))
    }

    fn compute(&self) -> i64 {
        self.0.compute()
    }
}

#[pymodule]
fn _internal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFoo>()?;
    Ok(())
}
