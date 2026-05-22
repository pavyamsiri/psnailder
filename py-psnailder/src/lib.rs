use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

#[pyclass]
pub struct PSpiralModel(psnailder_core::PSpiralModel);

#[pymethods]
impl PSpiralModel {
    #[new]
    fn new(components: Vec<PyRef<PSpiralComponent>>) -> Self {
        Self(psnailder_core::PSpiralModel {
            components: components.iter().map(|v| v.0.clone()).collect(),
        })
    }

    pub fn perturbation<'py>(
        &self,
        py: Python<'py>,
        z: PyReadonlyArray1<'py, f64>,
        vz: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Py<PyArray1<f64>>> {
        let z = z.as_slice().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err("z must be contiguous 1D array")
        })?;

        let vz = vz.as_slice().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err("vz must be contiguous 1D array")
        })?;

        if z.len() != vz.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "z and vz must have same length",
            ));
        }

        let mut out = vec![0.0; z.len()];

        self.0.perturbation_vec(z, vz, &mut out);

        Ok(PyArray1::from_vec(py, out).into())
    }

    pub fn evaluate_likelihood<'py>(
        &self,
        data: PyReadonlyArray1<'py, f64>,
        background: PyReadonlyArray1<'py, f64>,
        mask: PyReadonlyArray1<'py, f64>,
        z: PyReadonlyArray1<'py, f64>,
        vz: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<f64> {
        let z = z.as_slice().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err("z must be contiguous 1D array")
        })?;

        let vz = vz.as_slice().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err("vz must be contiguous 1D array")
        })?;

        if z.len() != vz.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "z and vz must have same length",
            ));
        }

        let data = data.as_slice()?;
        let mask = mask.as_slice()?;
        let background = background.as_slice()?;
        let mut out = vec![0.0; data.len()];
        for (zz, vzz, oo, current_background) in itertools::izip!(z, vz, &mut out, background) {
            let mut value = f64::NEG_INFINITY;

            for comp in self.0.components.iter() {
                let pert_value = comp.perturbation_scalar(*zz, *vzz);
                value = value.max(pert_value);
            }

            *oo = current_background * value;
        }

        Ok(psnailder_core::ln_likelihood_f64(data, &out, mask))
    }
}

#[pyclass]
pub struct PSpiralComponent(psnailder_core::PSpiralComponent);

#[pymethods]
impl PSpiralComponent {
    #[new]
    fn new(
        alpha: f64,
        lnb: f64,
        c: f64,
        theta0: f64,
        scale_factor: f64,
        rho: f64,
        winding: i8,
        flattening_strength: Option<f64>,
    ) -> Self {
        Self(psnailder_core::PSpiralComponent {
            alpha,
            lnb,
            c,
            theta0,
            scale_factor,
            rho,
            winding,
            flattening_strength: flattening_strength.unwrap_or(0.1),
        })
    }

    pub fn perturbation<'py>(
        &self,
        py: Python<'py>,
        z: PyReadonlyArray1<'py, f64>,
        vz: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Py<PyArray1<f64>>> {
        let z = z.as_slice().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err("z must be contiguous 1D array")
        })?;

        let vz = vz.as_slice().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err("vz must be contiguous 1D array")
        })?;

        if z.len() != vz.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "z and vz must have same length",
            ));
        }

        let mut out = vec![0.0; z.len()];

        self.0.perturbation_vec(z, vz, &mut out);

        Ok(PyArray1::from_vec(py, out).into())
    }
}

#[pyfunction]
fn ln_likelihood_f64(
    data: PyReadonlyArray1<f64>,
    prediction: PyReadonlyArray1<f64>,
    mask: PyReadonlyArray1<f64>,
) -> PyResult<f64> {
    let data = data.as_slice()?;
    let prediction = prediction.as_slice()?;
    let mask = mask.as_slice()?;

    Ok(psnailder_core::ln_likelihood_f64(data, prediction, mask))
}

#[pymodule]
fn _internal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ln_likelihood_f64, m)?)?;
    m.add_class::<PSpiralComponent>()?;
    m.add_class::<PSpiralModel>()?;
    Ok(())
}
