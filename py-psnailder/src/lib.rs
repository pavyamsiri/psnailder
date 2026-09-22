use numpy::{PyArray1, PyReadonlyArray1};
use psnailder_core::{PSpiralComponent as RustComponent, PSpiralModel as RustModel};
use psnailder_fit::{PSpiralFitter as RustFitter, PSpiralFitterND};
use pyo3::{exceptions::PyValueError, prelude::*};
use statrs::distribution::ContinuousCDF;

#[pyclass(from_py_object)]
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
    ) -> PyResult<Self> {
        let Ok(winding) = winding.try_into() else {
            return Err(PyValueError::new_err(format!(
                "winding must be -1 or 1, got {winding}"
            )));
        };
        Ok(Self(RustComponent {
            alpha,
            b_winding: b,
            c_winding: c,
            theta0,
            scale_factor,
            rho,
            winding,
            flattening_strength: 0.1,
        }))
    }

    #[getter]
    fn alpha(&self) -> f64 {
        self.0.alpha
    }
    #[getter]
    fn b(&self) -> f64 {
        self.0.b_winding
    }
    #[getter]
    fn c(&self) -> f64 {
        self.0.c_winding
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
        self.0.winding as i8
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
            self.0.b_winding,
            self.0.c_winding,
            self.0.theta0,
            self.0.scale_factor,
            self.0.rho,
            self.0.winding
        )
    }
}

#[pyclass(from_py_object)]
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
    #[pyo3(get)]
    pub initial_pvalue: f64,
    #[pyo3(get)]
    pub final_pvalue: f64,
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
    #[pyo3(signature = (max_iterations=Some(50), atol=0.0, rtol=0.0, sigma_z=2.0, sigma_vz=2.0, bounds=None))]
    fn new(
        max_iterations: Option<usize>,
        atol: f64,
        rtol: f64,
        sigma_z: f64,
        sigma_vz: f64,
        bounds: Option<Vec<Vec<(f64, f64)>>>,
    ) -> PyResult<Self> {
        // TikTak sampling remains an internal implementation detail until the
        // runtime parameter-layout path replaces the fixed 1-/2-component path.
        let num_samples = 4096usize;
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

        let default_bounds = vec![
            (0.0, 1.0),
            (0.005, 0.1),
            (0.0, 0.004),
            (-std::f64::consts::PI, std::f64::consts::PI),
            (30.0, 70.0),
            (0.0, 0.18),
        ];
        let mut component_bounds = bounds.unwrap_or_else(|| vec![default_bounds.clone()]);
        if component_bounds.is_empty() || component_bounds.len() > 2 {
            return Err(PyValueError::new_err(
                "bounds must contain one or two component bound sets",
            ));
        }
        if component_bounds.len() == 1 {
            component_bounds.push(component_bounds[0].clone());
        }
        if component_bounds.iter().any(|component| {
            component.len() != 6
                || component
                    .iter()
                    .any(|(lower, upper)| !lower.is_finite() || !upper.is_finite() || lower > upper)
        }) {
            return Err(PyValueError::new_err(
                "each component must have six finite, ordered parameter bounds",
            ));
        }
        let bounds_for = |component: &[(f64, f64)]| {
            (
                component[0],
                component[1],
                component[2],
                component[3],
                component[4],
                component[5],
            )
        };
        let single = bounds_for(&component_bounds[0]);
        let double = bounds_for(&component_bounds[1]);

        Ok(Self {
            inner: RustFitter {
                fitter_single: PSpiralFitterND {
                    tiktak: tiktak1,
                    alpha_bounds: single.0,
                    b_bounds: single.1,
                    c_bounds: single.2,
                    theta0_bounds: single.3,
                    scale_factor_bounds: single.4,
                    rho_bounds: single.5,
                },
                fitter_double: PSpiralFitterND {
                    tiktak: tiktak2,
                    alpha_bounds: double.0,
                    b_bounds: double.1,
                    c_bounds: double.2,
                    theta0_bounds: double.3,
                    scale_factor_bounds: double.4,
                    rho_bounds: double.5,
                },
                max_iterations,
                sigma_z,
                sigma_vz,
                atol,
                rtol,
            },
        })
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "API would be overly complicated in order to reduce number of arguments."
    )]
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

        let dof = (6 * res.final_model.components.len()) as f64;
        let dist =
            statrs::distribution::ChiSquared::new(dof).expect("`freedom` is guaranteed positive.");
        let lnl_initial_null =
            psnailder_core::ln_likelihood(initial_density, &res.initial_background, mask);
        let lnl_final_null =
            psnailder_core::ln_likelihood(initial_density, &res.final_background, mask);
        let lnl_initial = res.initial_lnl;
        let lnl_final = res.final_lnl;
        let lambda_initial = -2.0 * (lnl_initial_null - lnl_initial);
        let lambda_final = -2.0 * (lnl_final_null - lnl_final);

        let initial_pvalue = dist.sf(lambda_initial);
        let final_pvalue = dist.sf(lambda_final);

        Ok(PSpiralFitResult {
            initial_model: PSpiralModel(res.initial_model),
            final_model: PSpiralModel(res.final_model),
            data: PyArray1::from_vec(py, res.data.to_vec()).into(),
            initial_background: PyArray1::from_vec(py, res.initial_background.to_vec()).into(),
            final_background: PyArray1::from_vec(py, res.final_background.to_vec()).into(),
            num_iterations: res.num_iterations,
            max_iterations: res.max_iterations,
            converged: res.converged,
            lnl: res.final_lnl,
            initial_pvalue,
            final_pvalue,
        })
    }
}

#[pyfunction]
fn ln_likelihood(
    data: PyReadonlyArray1<f64>,
    prediction: PyReadonlyArray1<f64>,
    mask: PyReadonlyArray1<f64>,
) -> PyResult<f64> {
    let data = data.as_slice()?;
    let prediction = prediction.as_slice()?;
    let mask = mask.as_slice()?;

    Ok(psnailder_core::ln_likelihood(data, prediction, mask))
}

#[pymodule]
fn _internal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ln_likelihood, m)?)?;
    m.add_class::<PSpiralComponent>()?;
    m.add_class::<PSpiralModel>()?;
    m.add_class::<PSpiralFitter>()?;
    m.add_class::<PSpiralFitResult>()?;
    Ok(())
}
