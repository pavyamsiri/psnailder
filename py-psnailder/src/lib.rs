use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

#[pyclass]
pub struct PSpiralModel(psnailder_core::PSpiralModel);

#[pymethods]
impl PSpiralModel {
    #[new]
    fn new(
        alpha1: f64,
        b1: f64,
        c1: f64,
        theta01: f64,
        scale_factor1: f64,
        rho1: f64,
        winding1: i8,
        flattening_strength1: Option<f64>,
        alpha2: f64,
        b2: f64,
        c2: f64,
        theta02: f64,
        scale_factor2: f64,
        rho2: f64,
        winding2: i8,
        flattening_strength2: Option<f64>,
    ) -> Self {
        let comp1 = psnailder_core::PSpiralComponent {
            alpha: alpha1,
            b: b1,
            c: c1,
            theta0: theta01,
            scale_factor: scale_factor1,
            rho: rho1,
            winding: winding1,
            flattening_strength: flattening_strength1.unwrap_or(0.1),
        };
        let comp2 = psnailder_core::PSpiralComponent {
            alpha: alpha2,
            b: b2,
            c: c2,
            theta0: theta02,
            scale_factor: scale_factor2,
            rho: rho2,
            winding: winding2,
            flattening_strength: flattening_strength2.unwrap_or(0.1),
        };
        Self(psnailder_core::PSpiralModel {
            components: vec![comp1, comp2],
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
        for (((zz, vzz), oo), current_background) in z
            .iter()
            .zip(vz.iter())
            .zip(out.iter_mut())
            .zip(background.iter())
        {
            let mut value = f64::NEG_INFINITY;

            for comp in self.0.components.iter() {
                value = value.max(comp.perturbation_scalar(*zz, *vzz));
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
        b: f64,
        c: f64,
        theta0: f64,
        scale_factor: f64,
        rho: f64,
        winding: i8,
        flattening_strength: Option<f64>,
    ) -> Self {
        Self(psnailder_core::PSpiralComponent {
            alpha,
            b,
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
