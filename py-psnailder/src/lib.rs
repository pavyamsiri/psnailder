use argmin::core::CostFunction;
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
#[derive(Debug)]
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
fn fit_spiral_rust<'py>(
    data: PyReadonlyArray1<'py, f64>,
    background: PyReadonlyArray1<'py, f64>,
    mask: PyReadonlyArray1<'py, f64>,
    z: PyReadonlyArray1<'py, f64>,
    vz: PyReadonlyArray1<'py, f64>,
    bounds: Vec<(f64, f64)>,
) -> PyResult<(Vec<f64>, f64, u64)> {
    let tiktak = psnailder_tiktak::TikTak::new(10, 128.0f32.recip(), 0.1, 0.995, 6);

    let data = data.as_slice()?;
    let background = background.as_slice()?;
    let mask = mask.as_slice()?;
    let z = z.as_slice()?;
    let vz = vz.as_slice()?;

    let res = tiktak
        .minimize(
            PSpiralModel1DProblem {
                data,
                background,
                mask,
                z,
                vz,
            },
            &bounds,
        )
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{}", e)))?;

    Ok((res.params, res.cost, res.nfev))
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

#[derive(Debug, Clone)]
struct PSpiralModel1DProblem<'py> {
    data: &'py [f64],
    background: &'py [f64],
    mask: &'py [f64],
    z: &'py [f64],
    vz: &'py [f64],
}

impl<'py> CostFunction for PSpiralModel1DProblem<'py> {
    type Param = Vec<f64>;

    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        let model = psnailder_core::PSpiralModel {
            components: vec![psnailder_core::PSpiralComponent {
                alpha: param[0],
                b: param[1],
                c: param[2],
                theta0: param[3],
                scale_factor: param[4],
                rho: param[5],
                winding: 1,
                flattening_strength: 0.1,
            }],
        };
        let mut out = vec![0.0; self.z.len()];
        model.perturbation_vec(self.z, self.vz, &mut out);

        let prediction: Vec<f64> = out
            .iter()
            .zip(self.background.iter())
            .map(|(oo, bb)| *bb * *oo)
            .collect();

        Ok(-psnailder_core::ln_likelihood_f64(
            self.data,
            &prediction,
            self.mask,
        ))
    }
}

#[pyfunction]
fn optimize_parameters<'py>(
    data: PyReadonlyArray1<'py, f64>,
    background: PyReadonlyArray1<'py, f64>,
    mask: PyReadonlyArray1<'py, f64>,
    z: PyReadonlyArray1<'py, f64>,
    vz: PyReadonlyArray1<'py, f64>,
) -> PyResult<PSpiralComponent> {
    let tiktak = psnailder_tiktak::TikTak::new(12, 128.0f32.recip(), 0.1, 0.995, 6);

    let data = data.as_slice()?;
    let background = background.as_slice()?;
    let mask = mask.as_slice()?;
    let z = z.as_slice()?;
    let vz = vz.as_slice()?;

    let res = tiktak
        .minimize(
            PSpiralModel1DProblem {
                data,
                background,
                mask,
                z,
                vz,
            },
            &[
                (0.0, 1.0),
                (0.005f64, 0.1f64),
                (0.0, 0.004),
                (-core::f64::consts::PI, core::f64::consts::PI),
                (30.00f64, 70.0f64),
                (0.0, 0.18),
            ],
        )
        .expect("no errors!");

    let best_model = PSpiralComponent(psnailder_core::PSpiralComponent {
        alpha: res.params[0],
        b: res.params[1],
        c: res.params[2],
        theta0: res.params[3],
        scale_factor: res.params[4],
        rho: res.params[5],
        winding: 1,
        flattening_strength: 0.1,
    });

    println!("best - {best_model:?}");
    Ok(best_model)
}

#[pymodule]
fn _internal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ln_likelihood_f64, m)?)?;
    m.add_function(wrap_pyfunction!(fit_spiral_rust, m)?)?;
    m.add_function(wrap_pyfunction!(optimize_parameters, m)?)?;
    m.add_class::<PSpiralComponent>()?;
    m.add_class::<PSpiralModel>()?;
    Ok(())
}
