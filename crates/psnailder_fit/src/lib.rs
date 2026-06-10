use std::sync::Arc;

use basin::{BoxConstraints, CostFunction};
use psnailder_core::{PSpiralComponent, PSpiralModel, ln_likelihood};
use psnailder_tiktak::TikTak;

#[derive(Debug, Clone)]
struct PSpiralModelProblem<'prob, const NUM_COMPONENTS: u8> {
    data: &'prob [f64],
    background: &'prob [f64],
    mask: &'prob [f64],
    x: &'prob [f64],
    y: &'prob [f64],
    winding: i8,
    lb: &'prob Vec<f64>,
    ub: &'prob Vec<f64>,
}

impl<'prob> CostFunction for PSpiralModelProblem<'prob, 1> {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = core::convert::Infallible;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Self::Error> {
        let comp = PSpiralComponent {
            alpha: param[0],
            b_winding: param[1],
            c_winding: param[2],
            theta0: param[3],
            scale_factor: param[4],
            rho: param[5],
            winding: self.winding,
            flattening_strength: 0.1,
        };

        let res: f64 = itertools::izip!(self.data, self.x, self.y, self.background, self.mask)
            .map(|(d, x, y, bg, m)| {
                let p = comp.perturbation_scalar(*x, *y);
                let pred = p * bg;
                if pred <= 0.0 {
                    0.0
                } else {
                    let residual = m * (d - pred);
                    (residual * residual) / pred
                }
            })
            .sum();

        Ok(0.5 * res)
    }
}

impl<'prob> CostFunction for PSpiralModelProblem<'prob, 2> {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = core::convert::Infallible;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Self::Error> {
        let comp1 = PSpiralComponent {
            alpha: param[0],
            b_winding: param[1],
            c_winding: param[2],
            theta0: param[3],
            scale_factor: param[4],
            rho: param[5],
            winding: self.winding,
            flattening_strength: 0.1,
        };
        let comp2 = PSpiralComponent {
            alpha: param[6],
            b_winding: param[7],
            c_winding: param[8],
            theta0: param[9],
            scale_factor: param[10],
            rho: param[11],
            winding: self.winding,
            flattening_strength: 0.1,
        };

        let res: f64 = itertools::izip!(self.data, self.x, self.y, self.background, self.mask)
            .map(|(d, x, y, bg, m)| {
                let p1 = comp1.perturbation_scalar(*x, *y);
                let p2 = comp2.perturbation_scalar(*x, *y);
                let pred = p1.max(p2) * bg;
                if pred <= 0.0 {
                    0.0
                } else {
                    let residual = m * (d - pred);
                    (residual * residual) / pred
                }
            })
            .sum();

        Ok(0.5 * res)
    }
}

impl<'prob> BoxConstraints for PSpiralModelProblem<'prob, 1> {
    fn lower(&self) -> &Self::Param {
        self.lb
    }

    fn upper(&self) -> &Self::Param {
        self.ub
    }
}

impl<'prob> BoxConstraints for PSpiralModelProblem<'prob, 2> {
    fn lower(&self) -> &Self::Param {
        self.lb
    }

    fn upper(&self) -> &Self::Param {
        self.ub
    }
}

pub struct PSpiralFitterND<const N: usize> {
    pub tiktak: TikTak<N>,
    pub alpha_bounds: (f64, f64),
    pub b_bounds: (f64, f64),
    pub c_bounds: (f64, f64),
    pub theta0_bounds: (f64, f64),
    pub scale_factor_bounds: (f64, f64),
    pub rho_bounds: (f64, f64),
}

impl<const N: usize> Clone for PSpiralFitterND<N> {
    fn clone(&self) -> Self {
        Self {
            tiktak: TikTak {
                num_samples: self.tiktak.num_samples,
                num_star: self.tiktak.num_star,
                min_weight: self.tiktak.min_weight,
                max_weight: self.tiktak.max_weight,
                points: self.tiktak.points.clone(),
            },
            alpha_bounds: self.alpha_bounds,
            b_bounds: self.b_bounds,
            c_bounds: self.c_bounds,
            theta0_bounds: self.theta0_bounds,
            scale_factor_bounds: self.scale_factor_bounds,
            rho_bounds: self.rho_bounds,
        }
    }
}

#[derive(Clone)]
pub struct PSpiralFitter {
    pub fitter_single: PSpiralFitterND<6>,
    pub fitter_double: PSpiralFitterND<12>,
    pub max_iterations: Option<usize>,
    pub smoothing_sigma: f64,
}

#[derive(Debug, Clone)]
pub struct PSpiralFitResult {
    pub data: Arc<[f64]>,
    pub initial_model: PSpiralModel,
    pub initial_background: Arc<[f64]>,
    pub final_model: PSpiralModel,
    pub final_background: Arc<[f64]>,
    pub num_iterations: usize,
    pub max_iterations: Option<usize>,
    pub converged: bool,
    pub initial_lnl: f64,
    pub final_lnl: f64,
}

pub struct PSpiralFitterIterative<'a> {
    pub fitter: &'a PSpiralFitter,
    pub initial_density: Arc<[f64]>,
    pub initial_background: Arc<[f64]>,
    pub current_background: Arc<[f64]>,
    pub mask: Arc<[f64]>,
    pub mesh_x: Arc<[f64]>,
    pub mesh_y: Arc<[f64]>,
    pub shape: (usize, usize),
    pub num_components: usize,
    pub best_winding: Option<i8>,
    pub initial_quality: f64,
    pub best_quality: f64,
    pub best_model: Option<PSpiralModel>,
    pub initial_model: Option<PSpiralModel>,
    pub iteration_index: usize,
    pub max_iterations: Option<usize>,
    pub converged: bool,
    pub smoothing_sigma: f64,
    pub improve_background: bool,
    pub is_finished: bool,
}

fn gaussian_blur_2d(data: &[f64], shape: (usize, usize), sigma: f64) -> Arc<[f64]> {
    let (rows, cols) = shape;
    if sigma <= 0.0 {
        return Arc::from(data);
    }

    let kernel_size = (sigma * 4.0).ceil() as usize * 2 + 1;
    let mut kernel = vec![0.0; kernel_size];
    let half_size = (kernel_size / 2) as i32;
    let s2 = 2.0 * sigma * sigma;
    let mut sum = 0.0;
    for (i, kk) in kernel.iter_mut().enumerate() {
        let x = (i as i32 - half_size) as f64;
        *kk = (-x * x / s2).exp();
        sum += *kk;
    }
    for kk in kernel.iter_mut() {
        *kk /= sum;
    }

    let mut out = ndarray::Array2::from_shape_vec(shape, data.to_vec()).unwrap();
    let mut temp = ndarray::Array2::zeros(shape);

    // Horizontal pass
    for r in 0..rows {
        for c in 0..cols {
            let mut val = 0.0;
            for (i, kk) in kernel.iter().enumerate() {
                let cc = (c as i32 + i as i32 - half_size).clamp(0, cols as i32 - 1) as usize;
                val += out[[r, cc]] * kk;
            }
            temp[[r, c]] = val;
        }
    }

    // Vertical pass
    for r in 0..rows {
        for c in 0..cols {
            let mut val = 0.0;
            for (i, kk) in kernel.iter().enumerate() {
                let rr = (r as i32 + i as i32 - half_size).clamp(0, rows as i32 - 1) as usize;
                val += temp[[rr, c]] * kk;
            }
            out[[r, c]] = val;
        }
    }

    Arc::from(out.into_raw_vec_and_offset().0)
}

impl<'a> Iterator for PSpiralFitterIterative<'a> {
    type Item = PSpiralFitResult;

    fn next(&mut self) -> Option<Self::Item> {
        if self.is_finished {
            return None;
        }

        if let Some(max_iter) = self.max_iterations
            && self.iteration_index >= max_iter
        {
            self.is_finished = true;
            return None;
        }

        self.iteration_index += 1;

        // 1. Optimize parameters
        let (current_model, ll) = match self.num_components {
            1 => {
                let (comp, ll) = if let Some(w) = self.best_winding {
                    self.fitter
                        .fitter_single
                        .fit_spiral_with_background_with_winding(
                            &self.initial_density,
                            &self.current_background,
                            &self.mask,
                            &self.mesh_x,
                            &self.mesh_y,
                            w,
                        )
                } else {
                    let (comp, ll) = self.fitter.fitter_single.fit_spiral_with_background(
                        &self.initial_density,
                        &self.current_background,
                        &self.mask,
                        &self.mesh_x,
                        &self.mesh_y,
                    );
                    self.best_winding = Some(comp.winding);
                    (comp, ll)
                };
                (
                    PSpiralModel {
                        components: vec![comp],
                    },
                    ll,
                )
            }
            2 => {
                let (comp1, comp2, ll) = if let Some(w) = self.best_winding {
                    self.fitter
                        .fitter_double
                        .fit_spiral_with_background_with_winding(
                            &self.initial_density,
                            &self.current_background,
                            &self.mask,
                            &self.mesh_x,
                            &self.mesh_y,
                            w,
                        )
                } else {
                    let (c1, c2, ll) = self.fitter.fitter_double.fit_spiral_with_background(
                        &self.initial_density,
                        &self.current_background,
                        &self.mask,
                        &self.mesh_x,
                        &self.mesh_y,
                    );
                    self.best_winding = Some(c1.winding);
                    (c1, c2, ll)
                };
                (
                    PSpiralModel {
                        components: vec![comp1, comp2],
                    },
                    ll,
                )
            }
            _ => panic!("Unsupported num_components"),
        };

        // 2. Set initial model
        if self.initial_model.is_none() {
            self.initial_model = Some(current_model.clone());
            self.best_quality = ll;
            self.initial_quality = ll;
        }

        if !self.improve_background {
            self.best_model = Some(current_model.clone());
            self.converged = true;
            self.is_finished = true;
            return Some(PSpiralFitResult {
                data: Arc::clone(&self.initial_density),
                initial_model: self.initial_model.clone().unwrap(),
                initial_background: Arc::clone(&self.initial_background),
                final_model: self.best_model.clone().unwrap(),
                final_background: Arc::clone(&self.current_background),
                num_iterations: self.iteration_index,
                max_iterations: self.max_iterations,
                converged: self.converged,
                initial_lnl: self.initial_quality,
                final_lnl: self.best_quality,
            });
        }

        // 3. Update background
        let mut current_perturbation = vec![0.0; self.initial_density.len()];
        current_model.perturbation_vec(&self.mesh_x, &self.mesh_y, &mut current_perturbation);

        let next_background: Vec<f64> = self
            .initial_density
            .iter()
            .zip(current_perturbation.iter())
            .map(|(d, p)| d / p.max(1e-10))
            .collect();

        let blurred_background =
            gaussian_blur_2d(&next_background, self.shape, self.smoothing_sigma);
        let mut blurred_background_vec = blurred_background.to_vec();

        let next_bg_sum: f64 = blurred_background_vec.iter().sum();
        let density_sum: f64 = self.initial_density.iter().sum();
        if next_bg_sum > 0.0 {
            let scale = density_sum / next_bg_sum;
            for b in blurred_background_vec.iter_mut() {
                *b *= scale;
            }
        }
        let next_background_arc: Arc<[f64]> = Arc::from(blurred_background_vec);

        // 4. Check quality
        let mut new_data = vec![0.0; self.initial_density.len()];
        for i in 0..new_data.len() {
            new_data[i] = current_perturbation[i] * next_background_arc[i];
        }
        let quality = ln_likelihood(&self.initial_density, &new_data, &self.mask);

        if self.best_quality > quality {
            self.converged = self.best_model.is_some();
            if self.best_model.is_none() {
                self.best_model = self.initial_model.clone();
            }
            self.is_finished = true;
            return Some(PSpiralFitResult {
                data: Arc::clone(&self.initial_density),
                initial_model: self.initial_model.clone().unwrap(),
                initial_background: Arc::clone(&self.initial_background),
                final_model: self.best_model.clone().unwrap(),
                final_background: Arc::clone(&self.current_background),
                num_iterations: self.iteration_index,
                max_iterations: self.max_iterations,
                converged: self.converged,
                initial_lnl: self.initial_quality,
                final_lnl: self.best_quality,
            });
        }

        self.best_quality = quality;
        self.current_background = next_background_arc;
        self.best_model = Some(current_model.clone());

        Some(PSpiralFitResult {
            data: Arc::clone(&self.initial_density),
            initial_model: self.initial_model.clone().unwrap(),
            initial_background: Arc::clone(&self.initial_background),
            final_model: current_model,
            final_background: Arc::clone(&self.current_background),
            num_iterations: self.iteration_index,
            max_iterations: self.max_iterations,
            converged: self.converged,
            initial_lnl: self.initial_quality,
            final_lnl: self.best_quality,
        })
    }
}

impl PSpiralFitter {
    pub fn fit_spiral_with_background_iterative<'a>(
        &'a self,
        initial_density: &[f64],
        initial_background: &[f64],
        mask: &[f64],
        mesh_x: &[f64],
        mesh_y: &[f64],
        shape: (usize, usize),
        num_components: Option<usize>,
        winding: Option<i8>,
        improve_background: bool,
    ) -> PSpiralFitterIterative<'a> {
        let actual_num_components = if let Some(n) = num_components {
            n
        } else {
            // AIC comparison
            let (_, ll_single) = self.fitter_single.fit_spiral_with_background(
                initial_density,
                initial_background,
                mask,
                mesh_x,
                mesh_y,
            );
            let (_, _, ll_double) = self.fitter_double.fit_spiral_with_background(
                initial_density,
                initial_background,
                mask,
                mesh_x,
                mesh_y,
            );

            let ln_norm = initial_density.iter().sum::<f64>().ln();
            let bic_single = ln_norm * 6.0 - 2.0 * ll_single;
            let bic_double = ln_norm * 12.0 - 2.0 * ll_double;

            if bic_double < bic_single { 2 } else { 1 }
        };

        PSpiralFitterIterative {
            fitter: self,
            initial_density: Arc::from(initial_density),
            initial_background: Arc::from(initial_background),
            current_background: Arc::from(initial_background),
            mask: Arc::from(mask),
            mesh_x: Arc::from(mesh_x),
            mesh_y: Arc::from(mesh_y),
            shape,
            num_components: actual_num_components,
            best_winding: winding,
            initial_quality: f64::NEG_INFINITY,
            best_quality: f64::NEG_INFINITY,
            best_model: None,
            initial_model: None,
            iteration_index: 0,
            max_iterations: self.max_iterations,
            converged: false,
            smoothing_sigma: self.smoothing_sigma,
            improve_background,
            is_finished: false,
        }
    }

    pub fn fit_spiral_with_background(
        &self,
        initial_density: &[f64],
        initial_background: &[f64],
        mask: &[f64],
        mesh_x: &[f64],
        mesh_y: &[f64],
        shape: (usize, usize),
    ) -> PSpiralFitResult {
        let it = self.fit_spiral_with_background_iterative(
            initial_density,
            initial_background,
            mask,
            mesh_x,
            mesh_y,
            shape,
            None,
            None,
            true,
        );

        let Some(last) = it.last() else {
            panic!("should have at least one result");
        };

        last
    }
}

impl PSpiralFitterND<6> {
    pub fn fit_spiral_with_background(
        &self,
        initial_density: &[f64],
        initial_background: &[f64],
        mask: &[f64],
        mesh_x: &[f64],
        mesh_y: &[f64],
    ) -> (PSpiralComponent, f64) {
        let pos_winding = self.fit_spiral_with_background_with_winding(
            initial_density,
            initial_background,
            mask,
            mesh_x,
            mesh_y,
            1,
        );
        let neg_winding = self.fit_spiral_with_background_with_winding(
            initial_density,
            initial_background,
            mask,
            mesh_x,
            mesh_y,
            -1,
        );

        if pos_winding.1 >= neg_winding.1 {
            pos_winding
        } else {
            neg_winding
        }
    }

    pub fn fit_spiral_with_background_with_winding(
        &self,
        initial_density: &[f64],
        initial_background: &[f64],
        mask: &[f64],
        mesh_x: &[f64],
        mesh_y: &[f64],
        winding: i8,
    ) -> (PSpiralComponent, f64) {
        let lb = vec![
            self.alpha_bounds.0,
            self.b_bounds.0,
            self.c_bounds.0,
            self.theta0_bounds.0,
            self.scale_factor_bounds.0,
            self.rho_bounds.0,
        ];
        let ub = vec![
            self.alpha_bounds.1,
            self.b_bounds.1,
            self.c_bounds.1,
            self.theta0_bounds.1,
            self.scale_factor_bounds.1,
            self.rho_bounds.1,
        ];
        let res = self
            .tiktak
            .minimize(
                PSpiralModelProblem::<1> {
                    data: initial_density,
                    background: initial_background,
                    mask,
                    x: mesh_x,
                    y: mesh_y,
                    winding,
                    lb: &lb,
                    ub: &ub,
                },
                &[
                    self.alpha_bounds,
                    self.b_bounds,
                    self.c_bounds,
                    self.theta0_bounds,
                    self.scale_factor_bounds,
                    self.rho_bounds,
                ],
            )
            .expect("no errors!");

        let best_model = PSpiralComponent {
            alpha: res.params[0],
            b_winding: res.params[1],
            c_winding: res.params[2],
            theta0: res.params[3],
            scale_factor: res.params[4],
            rho: res.params[5],
            winding,
            flattening_strength: 0.1,
        };

        (best_model, -res.cost)
    }
}

impl PSpiralFitterND<12> {
    pub fn fit_spiral_with_background(
        &self,
        initial_density: &[f64],
        initial_background: &[f64],
        mask: &[f64],
        mesh_x: &[f64],
        mesh_y: &[f64],
    ) -> (PSpiralComponent, PSpiralComponent, f64) {
        let pos_winding = self.fit_spiral_with_background_with_winding(
            initial_density,
            initial_background,
            mask,
            mesh_x,
            mesh_y,
            1,
        );
        let neg_winding = self.fit_spiral_with_background_with_winding(
            initial_density,
            initial_background,
            mask,
            mesh_x,
            mesh_y,
            -1,
        );

        if pos_winding.2 >= neg_winding.2 {
            pos_winding
        } else {
            neg_winding
        }
    }

    pub fn fit_spiral_with_background_with_winding(
        &self,
        initial_density: &[f64],
        initial_background: &[f64],
        mask: &[f64],
        mesh_x: &[f64],
        mesh_y: &[f64],
        winding: i8,
    ) -> (PSpiralComponent, PSpiralComponent, f64) {
        let lb = vec![
            self.alpha_bounds.0,
            self.b_bounds.0,
            self.c_bounds.0,
            self.theta0_bounds.0,
            self.scale_factor_bounds.0,
            self.rho_bounds.0,
            self.alpha_bounds.0,
            self.b_bounds.0,
            self.c_bounds.0,
            self.theta0_bounds.0,
            self.scale_factor_bounds.0,
            self.rho_bounds.0,
        ];
        let ub = vec![
            self.alpha_bounds.1,
            self.b_bounds.1,
            self.c_bounds.1,
            self.theta0_bounds.1,
            self.scale_factor_bounds.1,
            self.rho_bounds.1,
            self.alpha_bounds.1,
            self.b_bounds.1,
            self.c_bounds.1,
            self.theta0_bounds.1,
            self.scale_factor_bounds.1,
            self.rho_bounds.1,
        ];
        let res = self
            .tiktak
            .minimize(
                PSpiralModelProblem::<2> {
                    data: initial_density,
                    background: initial_background,
                    mask,
                    x: mesh_x,
                    y: mesh_y,
                    winding,
                    lb: &lb,
                    ub: &ub,
                },
                &[
                    self.alpha_bounds,
                    self.b_bounds,
                    self.c_bounds,
                    self.theta0_bounds,
                    self.scale_factor_bounds,
                    self.rho_bounds,
                    self.alpha_bounds,
                    self.b_bounds,
                    self.c_bounds,
                    self.theta0_bounds,
                    self.scale_factor_bounds,
                    self.rho_bounds,
                ],
            )
            .expect("no errors!");

        let comp1 = PSpiralComponent {
            alpha: res.params[0],
            b_winding: res.params[1],
            c_winding: res.params[2],
            theta0: res.params[3],
            scale_factor: res.params[4],
            rho: res.params[5],
            winding,
            flattening_strength: 0.1,
        };
        let comp2 = PSpiralComponent {
            alpha: res.params[6],
            b_winding: res.params[7],
            c_winding: res.params[8],
            theta0: res.params[9],
            scale_factor: res.params[10],
            rho: res.params[11],
            winding,
            flattening_strength: 0.1,
        };

        (comp1, comp2, -res.cost)
    }
}
