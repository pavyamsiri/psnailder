use numpy::{PyArray1, PyReadonlyArray1};
use psnailder_core::{PSpiralComponent as RustComponent, PSpiralModel as RustModel};
use psnailder_fit::{PSpiralFitter as RustFitter, PSpiralFitterND};
use pyo3::prelude::*;

#[pyclass]
#[derive(Clone, Debug)]
pub struct PSpiralComponent(pub RustComponent);

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
        Self(RustComponent {
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

    #[getter]
    fn alpha(&self) -> f64 {
        self.0.alpha
    }
    #[getter]
    fn b(&self) -> f64 {
        self.0.b
    }
    #[getter]
    fn c(&self) -> f64 {
        self.0.c
    }
    #[getter]
    fn theta0(&self) -> f64 {
        self.0.theta0
    }
    #[getter]
    fn scale_factor(&self) -> f64 {
        self.0.scale_factor
    }
    #[getter]
    fn rho(&self) -> f64 {
        self.0.rho
    }
    #[getter]
    fn winding(&self) -> i8 {
        self.0.winding
    }

    pub fn perturbation<'py>(
        &self,
        py: Python<'py>,
        z: PyReadonlyArray1<'py, f64>,
        vz: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let z = z.as_slice()?;
        let vz = vz.as_slice()?;
        let mut out = vec![0.0; z.len()];
        self.0.perturbation_vec(z, vz, &mut out);
        Ok(PyArray1::from_vec(py, out))
    }

    fn __repr__(&self) -> String {
        format!(
            "PSpiralComponent(alpha={:.4}, b={:.4}, c={:.4}, theta0={:.4}, scale_factor={:.4}, rho={:.4}, winding={})",
            self.0.alpha,
            self.0.b,
            self.0.c,
            self.0.theta0,
            self.0.scale_factor,
            self.0.rho,
            self.0.winding
        )
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct PSpiralModel(pub RustModel);

#[pymethods]
impl PSpiralModel {
    #[new]
    fn new(components: Vec<PSpiralComponent>) -> Self {
        Self(RustModel {
            components: components.into_iter().map(|c| c.0).collect(),
        })
    }

    #[getter]
    fn components(&self) -> Vec<PSpiralComponent> {
        self.0
            .components
            .iter()
            .map(|c| PSpiralComponent(c.clone()))
            .collect()
    }

    fn __repr__(&self) -> String {
        let comps: Vec<String> = self.components().iter().map(|c| c.__repr__()).collect();
        format!("PSpiralModel(components=[{}])", comps.join(", "))
    }

    pub fn perturbation<'py>(
        &self,
        py: Python<'py>,
        z: PyReadonlyArray1<'py, f64>,
        vz: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let z = z.as_slice()?;
        let vz = vz.as_slice()?;
        let mut out = vec![0.0; z.len()];
        self.0.perturbation_vec(z, vz, &mut out);
        Ok(PyArray1::from_vec(py, out))
    }
}

#[pyclass]
pub struct PSpiralFitResult {
    #[pyo3(get)]
    pub initial_model: PSpiralModel,
    #[pyo3(get)]
    pub final_model: PSpiralModel,
    #[pyo3(get)]
    pub data: Py<PyArray1<f64>>,
    #[pyo3(get)]
    pub initial_background: Py<PyArray1<f64>>,
    #[pyo3(get)]
    pub final_background: Py<PyArray1<f64>>,
    #[pyo3(get)]
    pub num_iterations: usize,
    #[pyo3(get)]
    pub max_iterations: Option<usize>,
    #[pyo3(get)]
    pub converged: bool,
    #[pyo3(get)]
    pub lnl: f64,
}

#[pymethods]
impl PSpiralFitResult {
    fn __repr__(&self) -> String {
        format!(
            "PSpiralFitResult(num_iterations={}, converged={})",
            self.num_iterations, self.converged
        )
    }
}

#[pyclass]
pub struct PSpiralFitter {
    inner: RustFitter,
}

#[pymethods]
impl PSpiralFitter {
    #[new]
    #[pyo3(signature = (num_samples=4096, max_iterations=Some(50), smoothing_sigma=2.0))]
    fn new(num_samples: usize, max_iterations: Option<usize>, smoothing_sigma: f64) -> Self {
        let tiktak1 = psnailder_tiktak::TikTak::<6>::new(
            (num_samples as f64).log2() as u8,
            128.0f32.recip(),
            0.1,
            0.995,
        );
        let tiktak2 = psnailder_tiktak::TikTak::<12>::new(
            (num_samples as f64).log2() as u8,
            128.0f32.recip(),
            0.1,
            0.995,
        );

        let alpha_bounds = (0.0, 1.0);
        let b_bounds = (0.005, 0.1);
        let c_bounds = (0.0, 0.004);
        let theta0_bounds = (-std::f64::consts::PI, std::f64::consts::PI);
        let scale_factor_bounds = (30.0, 70.0);
        let rho_bounds = (0.0, 0.18);

        Self {
            inner: RustFitter {
                fitter_single: PSpiralFitterND {
                    tiktak: tiktak1,
                    alpha_bounds,
                    b_bounds,
                    c_bounds,
                    theta0_bounds,
                    scale_factor_bounds,
                    rho_bounds,
                },
                fitter_double: PSpiralFitterND {
                    tiktak: tiktak2,
                    alpha_bounds,
                    b_bounds,
                    c_bounds,
                    theta0_bounds,
                    scale_factor_bounds,
                    rho_bounds,
                },
                max_iterations,
                smoothing_sigma,
            },
        }
    }

    pub fn fit_spiral_with_background<'py>(
        &self,
        py: Python<'py>,
        initial_density: PyReadonlyArray1<'py, f64>,
        initial_background: PyReadonlyArray1<'py, f64>,
        mask: PyReadonlyArray1<'py, f64>,
        mesh_x: PyReadonlyArray1<'py, f64>,
        mesh_y: PyReadonlyArray1<'py, f64>,
        shape: (usize, usize),
    ) -> PyResult<PSpiralFitResult> {
        let initial_density = initial_density.as_slice()?;
        let initial_background = initial_background.as_slice()?;
        let mask = mask.as_slice()?;
        let mesh_x = mesh_x.as_slice()?;
        let mesh_y = mesh_y.as_slice()?;

        let res = self.inner.fit_spiral_with_background(
            initial_density,
            initial_background,
            mask,
            mesh_x,
            mesh_y,
            shape,
        );

        Ok(PSpiralFitResult {
            initial_model: PSpiralModel(res.initial_model),
            final_model: PSpiralModel(res.final_model),
            data: PyArray1::from_vec(py, res.data.to_vec()).into(),
            initial_background: PyArray1::from_vec(py, res.initial_background.to_vec()).into(),
            final_background: PyArray1::from_vec(py, res.final_background.to_vec()).into(),
            num_iterations: res.num_iterations,
            max_iterations: res.max_iterations,
            converged: res.converged,
            lnl: res.lnl,
        })
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
    m.add_class::<PSpiralFitter>()?;
    m.add_class::<PSpiralFitResult>()?;
    Ok(())
}
