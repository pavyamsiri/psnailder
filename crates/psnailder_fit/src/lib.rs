use argmin::core::CostFunction;
use psnailder_core::{ln_likelihood_f64, PSpiralComponent, PSpiralModel};
use psnailder_tiktak::TikTak;

#[derive(Debug, Clone)]
struct PSpiralModelProblem<'prob, const NUM_COMPONENTS: u8> {
    data: &'prob [f64],
    background: &'prob [f64],
    mask: &'prob [f64],
    x: &'prob [f64],
    y: &'prob [f64],
    winding: i8,
}

impl<'prob> CostFunction for PSpiralModelProblem<'prob, 1> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        let comp = PSpiralComponent {
            alpha: param[0],
            b: param[1],
            c: param[2],
            theta0: param[3],
            scale_factor: param[4],
            rho: param[5],
            winding: self.winding,
            flattening_strength: 0.1,
        };

        let mut out = vec![0.0; self.data.len()];
        comp.perturbation_vec(self.x, self.y, &mut out);
        let prediction: Vec<f64> = out
            .into_iter()
            .zip(self.background.iter())
            .map(|(p, b)| p * b)
            .collect();

        let res = -psnailder_core::ln_likelihood_f64(self.data, &prediction, self.mask);
        Ok(res)
    }
}

impl<'prob> CostFunction for PSpiralModelProblem<'prob, 2> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        let comp1 = PSpiralComponent {
            alpha: param[0],
            b: param[1],
            c: param[2],
            theta0: param[3],
            scale_factor: param[4],
            rho: param[5],
            winding: self.winding,
            flattening_strength: 0.1,
        };
        let comp2 = PSpiralComponent {
            alpha: param[6],
            b: param[7],
            c: param[8],
            theta0: param[9],
            scale_factor: param[10],
            rho: param[11],
            winding: self.winding,
            flattening_strength: 0.1,
        };

        let mut out1 = vec![0.0; self.data.len()];
        let mut out2 = vec![0.0; self.data.len()];
        comp1.perturbation_vec(self.x, self.y, &mut out1);
        comp2.perturbation_vec(self.x, self.y, &mut out2);

        let prediction: Vec<f64> =
            itertools::izip!(out1.into_iter(), out2.iter(), self.background.iter())
                .map(|(p1, p2, bb)| p1.max(*p2) * bb)
                .collect();

        let res = -psnailder_core::ln_likelihood_f64(self.data, &prediction, self.mask);
        Ok(res)
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
    pub data: Vec<f64>,
    pub initial_model: PSpiralModel,
    pub initial_background: Vec<f64>,
    pub final_model: PSpiralModel,
    pub final_background: Vec<f64>,
    pub num_iterations: usize,
    pub max_iterations: Option<usize>,
    pub converged: bool,
}

pub struct PSpiralFitterIterative<'a> {
    pub fitter: &'a PSpiralFitter,
    pub initial_density: Vec<f64>,
    pub initial_background: Vec<f64>,
    pub current_background: Vec<f64>,
    pub mask: Vec<f64>,
    pub mesh_x: Vec<f64>,
    pub mesh_y: Vec<f64>,
    pub shape: (usize, usize),
    pub num_components: usize,
    pub best_winding: Option<i8>,
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

fn gaussian_blur_2d(data: &[f64], shape: (usize, usize), sigma: f64) -> Vec<f64> {
    let (rows, cols) = shape;
    let mut out = data.to_vec();

    if sigma <= 0.0 {
        return out;
    }

    let kernel_size = (sigma * 4.0).ceil() as usize * 2 + 1;
    let mut kernel = vec![0.0; kernel_size];
    let half_size = (kernel_size / 2) as i32;
    let s2 = 2.0 * sigma * sigma;
    let mut sum = 0.0;
    for i in 0..kernel_size {
        let x = (i as i32 - half_size) as f64;
        kernel[i] = (-x * x / s2).exp();
        sum += kernel[i];
    }
    for i in 0..kernel_size {
        kernel[i] /= sum;
    }

    // Horizontal pass
    let mut temp = vec![0.0; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            let mut val = 0.0;
            for i in 0..kernel_size {
                let cc = (c as i32 + i as i32 - half_size).clamp(0, cols as i32 - 1) as usize;
                val += out[r * cols + cc] * kernel[i];
            }
            temp[r * cols + c] = val;
        }
    }

    // Vertical pass
    for r in 0..rows {
        for c in 0..cols {
            let mut val = 0.0;
            for i in 0..kernel_size {
                let rr = (r as i32 + i as i32 - half_size).clamp(0, rows as i32 - 1) as usize;
                val += temp[rr * cols + c] * kernel[i];
            }
            out[r * cols + c] = val;
        }
    }

    out
}

impl<'a> Iterator for PSpiralFitterIterative<'a> {
    type Item = PSpiralFitResult;

    fn next(&mut self) -> Option<Self::Item> {
        if self.is_finished {
            return None;
        }

        if let Some(max_iter) = self.max_iterations {
            if self.iteration_index >= max_iter {
                self.is_finished = true;
                return None;
            }
        }

        self.iteration_index += 1;

        // 1. Optimize parameters
        let (current_model, ll) = match self.num_components {
            1 => {
                let (comp, ll) = if let Some(w) = self.best_winding {
                    self.fitter.fitter_single.fit_spiral_with_background_with_winding(
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
                    self.fitter.fitter_double.fit_spiral_with_background_with_winding(
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
            if self.best_quality == f64::NEG_INFINITY {
                self.best_quality = ll;
            }
        }

        if !self.improve_background {
            self.best_model = Some(current_model.clone());
            self.converged = true;
            self.is_finished = true;
            return Some(PSpiralFitResult {
                data: self.initial_density.clone(),
                initial_model: self.initial_model.clone().unwrap(),
                initial_background: self.initial_background.clone(),
                final_model: self.best_model.clone().unwrap(),
                final_background: self.current_background.clone(),
                num_iterations: self.iteration_index,
                max_iterations: self.max_iterations,
                converged: self.converged,
            });
        }

        // 3. Update background
        let mut current_perturbation = vec![0.0; self.initial_density.len()];
        current_model.perturbation_vec(&self.mesh_x, &self.mesh_y, &mut current_perturbation);

        let mut next_background: Vec<f64> = self
            .initial_density
            .iter()
            .zip(current_perturbation.iter())
            .map(|(d, p)| d / p.max(1e-10))
            .collect();

        next_background = gaussian_blur_2d(&next_background, self.shape, self.smoothing_sigma);

        let next_bg_sum: f64 = next_background.iter().sum();
        let density_sum: f64 = self.initial_density.iter().sum();
        if next_bg_sum > 0.0 {
            for b in next_background.iter_mut() {
                *b = *b / next_bg_sum * density_sum;
            }
        }

        // 4. Check quality
        let mut new_data = vec![0.0; self.initial_density.len()];
        for i in 0..new_data.len() {
            new_data[i] = current_perturbation[i] * next_background[i];
        }
        let quality = ln_likelihood_f64(&self.initial_density, &new_data, &self.mask);

        if self.best_quality > quality {
            self.converged = self.best_model.is_some();
            if self.best_model.is_none() {
                self.best_model = self.initial_model.clone();
            }
            self.is_finished = true;
            return Some(PSpiralFitResult {
                data: self.initial_density.clone(),
                initial_model: self.initial_model.clone().unwrap(),
                initial_background: self.initial_background.clone(),
                final_model: self.best_model.clone().unwrap(),
                final_background: self.current_background.clone(),
                num_iterations: self.iteration_index,
                max_iterations: self.max_iterations,
                converged: self.converged,
            });
        }

        self.best_quality = quality;
        self.current_background = next_background;
        self.best_model = Some(current_model.clone());

        Some(PSpiralFitResult {
            data: self.initial_density.clone(),
            initial_model: self.initial_model.clone().unwrap(),
            initial_background: self.initial_background.clone(),
            final_model: current_model,
            final_background: self.current_background.clone(),
            num_iterations: self.iteration_index,
            max_iterations: self.max_iterations,
            converged: self.converged,
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

            let aic_single = 2.0 * 6.0 - 2.0 * ll_single;
            let aic_double = 2.0 * 12.0 - 2.0 * ll_double;

            if aic_double < aic_single {
                2
            } else {
                1
            }
        };

        PSpiralFitterIterative {
            fitter: self,
            initial_density: initial_density.to_vec(),
            initial_background: initial_background.to_vec(),
            current_background: initial_background.to_vec(),
            mask: mask.to_vec(),
            mesh_x: mesh_x.to_vec(),
            mesh_y: mesh_y.to_vec(),
            shape,
            num_components: actual_num_components,
            best_winding: winding,
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
        let mut iter = self.fit_spiral_with_background_iterative(
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

        let mut last_res = None;
        while let Some(res) = iter.next() {
            last_res = Some(res);
        }
        last_res.expect("Fitting failed to produce any result")
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
            b: res.params[1],
            c: res.params[2],
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

        let best_model1 = PSpiralComponent {
            alpha: res.params[0],
            b: res.params[1],
            c: res.params[2],
            theta0: res.params[3],
            scale_factor: res.params[4],
            rho: res.params[5],
            winding,
            flattening_strength: 0.1,
        };
        let best_model2 = PSpiralComponent {
            alpha: res.params[6],
            b: res.params[7],
            c: res.params[8],
            theta0: res.params[9],
            scale_factor: res.params[10],
            rho: res.params[11],
            winding,
            flattening_strength: 0.1,
        };

        (best_model1, best_model2, -res.cost)
    }
}
