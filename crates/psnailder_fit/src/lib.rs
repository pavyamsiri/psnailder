//! Fitting logic for phase spiral models.
//!
//! This crate provides tools for fitting one or two-component phase spiral models to 2D density data,
//! optionally refining the background density iteratively.

extern crate alloc;

use alloc::sync::Arc;
use basin::{BoxConstraints, CostFunction};
use core::convert;
use itertools::izip;
use psnailder_core::{PSpiralComponent, PSpiralModel, Winding, ln_likelihood};
use psnailder_tiktak::TikTak;
use wide::CmpLe as _;
use wide::f64x4;

/// A `basin` problem to optimise to find the best parameters for the phase spiral model.
///
/// `NUM_COMPONENTS` is a const generic that defines the number of component arms to solve for.
#[derive(Debug, Clone)]
struct PSpiralModelProblem<'prob, const NUM_COMPONENTS: u8> {
    /// The density grid; must be length `num_cells`.
    data: &'prob [f64],
    /// The estimated background grid; must be length `num_cells`.
    background: &'prob [f64],
    /// The corresponding mask; must be length `num_cells`.
    mask: &'prob [f64],
    /// The x coordinate of each grid point; must be length `num_cells`.
    x: &'prob [f64],
    /// The y coordinate of each grid point; must be length `num_cells`.
    y: &'prob [f64],
    /// The winding direction.
    winding: Winding,
    /// The lower bounds for the parameters; must be length `6 * NUM_COMPONENTS`.
    lb: &'prob Vec<f64>,
    /// The upper bounds for the parameters; must be length `6 * NUM_COMPONENTS`.
    ub: &'prob Vec<f64>,
}

impl CostFunction for PSpiralModelProblem<'_, 1> {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = convert::Infallible;

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

        let (data_chunks, data_remainder) = self.data.as_chunks::<4>();
        let (x_chunks, x_remainder) = self.x.as_chunks::<4>();
        let (y_chunks, y_remainder) = self.y.as_chunks::<4>();
        let (background_chunks, background_remainder) = self.background.as_chunks::<4>();
        let (mask_chunks, mask_remainder) = self.mask.as_chunks::<4>();

        let mut res = 0.0;
        let mut acc = f64x4::ZERO;

        for (current_data, x, y, bg, current_mask) in izip!(
            data_chunks,
            x_chunks,
            y_chunks,
            background_chunks,
            mask_chunks
        ) {
            let current_data = f64x4::from(*current_data);
            let x = f64x4::from(*x);
            let y = f64x4::from(*y);
            let bg = f64x4::from(*bg);
            let current_mask = f64x4::from(*current_mask);
            let pert = comp.perturbation_wide(x, y);
            let pred = pert * bg;
            let residual = current_mask * (current_data - pred);
            let term = (residual * residual) / pred;
            let pred_mask = pred.simd_le(f64x4::ZERO);
            let current_result = pred_mask.blend(f64x4::ZERO, term);

            acc += current_result;
        }

        for (current_data, x, y, bg, current_mask) in izip!(
            data_remainder,
            x_remainder,
            y_remainder,
            background_remainder,
            mask_remainder
        ) {
            let pert = comp.perturbation_scalar(*x, *y);
            let pred = pert * bg;
            res += if pred <= 0.0 {
                0.0
            } else {
                let residual = current_mask * (current_data - pred);
                (residual * residual) / pred
            };
        }

        res += acc.reduce_add();

        Ok(0.5 * res)
    }
}

impl CostFunction for PSpiralModelProblem<'_, 2> {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = convert::Infallible;

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

        let (data_chunks, data_remainder) = self.data.as_chunks::<4>();
        let (x_chunks, x_remainder) = self.x.as_chunks::<4>();
        let (y_chunks, y_remainder) = self.y.as_chunks::<4>();
        let (background_chunks, background_remainder) = self.background.as_chunks::<4>();
        let (mask_chunks, mask_remainder) = self.mask.as_chunks::<4>();

        let mut res = 0.0;
        let mut acc = f64x4::ZERO;

        for (current_data, x, y, bg, current_mask) in izip!(
            data_chunks,
            x_chunks,
            y_chunks,
            background_chunks,
            mask_chunks
        ) {
            let current_data = f64x4::from(*current_data);
            let x = f64x4::from(*x);
            let y = f64x4::from(*y);
            let bg = f64x4::from(*bg);
            let current_mask = f64x4::from(*current_mask);
            let pert1 = comp1.perturbation_wide(x, y);
            let pert2 = comp2.perturbation_wide(x, y);
            let pert = pert1.max(pert2);
            let pred = pert * bg;
            let residual = current_mask * (current_data - pred);
            let term = (residual * residual) / pred;
            let pred_mask = pred.simd_le(f64x4::ZERO);
            let current_result = pred_mask.blend(f64x4::ZERO, term);

            acc += current_result;
        }

        for (current_data, x, y, bg, current_mask) in izip!(
            data_remainder,
            x_remainder,
            y_remainder,
            background_remainder,
            mask_remainder
        ) {
            let p1 = comp1.perturbation_scalar(*x, *y);
            let p2 = comp2.perturbation_scalar(*x, *y);
            let pred = p1.max(p2) * bg;
            res += if pred <= 0.0 {
                0.0
            } else {
                let residual = current_mask * (current_data - pred);
                (residual * residual) / pred
            };
        }

        res += acc.reduce_add();

        Ok(0.5 * res)
    }
}

impl BoxConstraints for PSpiralModelProblem<'_, 1> {
    fn lower(&self) -> &Self::Param {
        self.lb
    }

    fn upper(&self) -> &Self::Param {
        self.ub
    }
}

impl BoxConstraints for PSpiralModelProblem<'_, 2> {
    fn lower(&self) -> &Self::Param {
        self.lb
    }

    fn upper(&self) -> &Self::Param {
        self.ub
    }
}

/// A fitter for phase spiral models with a fixed number of parameters.
///
/// This struct manages the optimization process for a specific dimensionality `N`,
/// which typically corresponds to the number of components multiplied by the six parameters for a single spiral component.
pub struct PSpiralFitterND<const N: usize> {
    /// The global optimization engine.
    pub tiktak: TikTak<N>,
    /// The bounds for `alpha`.
    pub alpha_bounds: (f64, f64),
    /// The bounds for `b`.
    pub b_bounds: (f64, f64),
    /// The bounds for `c`.
    pub c_bounds: (f64, f64),
    /// The bounds for `theta0`.
    pub theta0_bounds: (f64, f64),
    /// The bounds for `scale_factor`.
    pub scale_factor_bounds: (f64, f64),
    /// The bounds for `rho`.
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

type OneArmFitter = PSpiralFitterND<6>;
type TwoArmFitter = PSpiralFitterND<12>;

/// A high-level fitter that supports single and double component models with iterative background refinement.
#[derive(Clone)]
pub struct PSpiralFitter {
    /// Fitter used for one-arm models.
    pub fitter_single: OneArmFitter,
    /// Fitter used for two-arm models.
    pub fitter_double: TwoArmFitter,
    /// Maximum number of iterations for background refinement.
    pub max_iterations: Option<usize>,
    /// Sigma for Gaussian smoothing applied to the background during refinement.
    pub smoothing_sigma: f64,
}

/// The results of a spiral model fit.
#[derive(Debug, Clone)]
pub struct PSpiralFitResult {
    /// The original input density data.
    pub data: Arc<[f64]>,
    /// The model obtained after the first optimization step.
    pub initial_model: PSpiralModel,
    /// The initial background provided to the fitter.
    pub initial_background: Arc<[f64]>,
    /// The final optimized model.
    pub final_model: PSpiralModel,
    /// The final refined background.
    pub final_background: Arc<[f64]>,
    /// Number of iterations performed.
    pub num_iterations: usize,
    /// Maximum allowed iterations if set otherwise none.
    pub max_iterations: Option<usize>,
    /// Whether the background refinement converged.
    pub converged: bool,
    /// The log-likelihood of the initial fit.
    pub initial_lnl: f64,
    /// The log-likelihood of the final fit.
    pub final_lnl: f64,
}

/// An iterator that performs the fitting process step-by-step.
///
/// This allows for inspecting intermediate results or customizing the refinement loop.
pub struct PSpiralFitterIterative<'fit> {
    /// The composite one and two arm spiral fitter.
    pub fitter: &'fit PSpiralFitter,
    /// The initial density grid.
    pub initial_density: Arc<[f64]>,
    /// The initial estimated background grid.
    pub initial_background: Arc<[f64]>,
    /// The current estimated background grid.
    pub current_background: Arc<[f64]>,
    /// The mask used to evaluate quality.
    pub mask: Arc<[f64]>,
    /// The x coordinate at each grid point.
    pub mesh_x: Arc<[f64]>,
    /// The y coordinate at each grid point.
    pub mesh_y: Arc<[f64]>,
    /// The shape of the grid in row-major order.
    pub shape: (usize, usize),
    /// The number of components to fit.
    pub num_components: usize,
    /// The best fitting winding direction.
    pub best_winding: Option<Winding>,
    /// The quality or log-likelihood of the initial fit.
    pub initial_quality: f64,
    /// The quality or log-likelihood of the best fit so far.
    pub best_quality: f64,
    /// The model of the best fit so far.
    pub best_model: Option<PSpiralModel>,
    /// The model of the initial fit.
    pub initial_model: Option<PSpiralModel>,
    /// The current iteration index.
    pub iteration_index: usize,
    /// The maximum number of iterations if given otherwise none signifies no limit.
    pub max_iterations: Option<usize>,
    /// Whether the background refinement process has converged.
    pub converged: bool,
    /// The sigma used when smoothing the background during the refinement process.
    pub smoothing_sigma: f64,
    /// Whether to perform iterative background refinement.
    pub improve_background: bool,
    /// Whether the refinement process has terminated.
    pub is_finished: bool,
}

impl Iterator for PSpiralFitterIterative<'_> {
    type Item = PSpiralFitResult;

    fn next(&mut self) -> Option<Self::Item> {
        // Fitting is done so quit.
        if self.is_finished {
            return None;
        }

        // We ran out of allowed iterations.
        if let Some(max_iter) = self.max_iterations
            && self.iteration_index >= max_iter
        {
            self.is_finished = true;
            return None;
        }

        // Update the iteration index.
        self.iteration_index += 1;

        // Optimize parameters.
        let (current_model, ll) = match self.num_components {
            1 => {
                // Use the winding from a previous iteration or the given one.
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
                }
                // Determine the best winding by performing both fits.
                else {
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
                // Use the winding from a previous iteration or the given one.
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
                }
                // Determine the best winding by performing both fits.
                else {
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
            _ => panic!("Unsupported `num_components`"),
        };

        // Set initial model if this is the first iteration.
        if self.initial_model.is_none() {
            self.initial_model = Some(current_model.clone());
            self.best_quality = ll;
            self.initial_quality = ll;
        }

        // If we aren't performing background refinement then we return here.
        if !self.improve_background {
            self.best_model = Some(current_model);
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

        // Update background by the relation
        // ``new_background = smooth(data / pertubation)``

        // Calculate the perturbation
        let mut current_perturbation = vec![0.0; self.initial_density.len()];
        current_model.perturbation_vec(&self.mesh_x, &self.mesh_y, &mut current_perturbation);

        // Unsmoothed background
        let next_background: Vec<f64> = self
            .initial_density
            .iter()
            .zip(current_perturbation.iter())
            .map(|(current_data, current_pert)| current_data / current_pert.max(1e-10))
            .collect();

        // Smoothed background
        let blurred_background =
            gaussian_blur_2d(&next_background, self.shape, self.smoothing_sigma);
        let mut blurred_background_vec = blurred_background.to_vec();

        // NOTE: Do we need to do this? Normalising might lead to the background
        // absorbing the perturbation but not sure.
        // Normalise the background
        let next_bg_sum: f64 = blurred_background_vec.iter().sum();
        let density_sum: f64 = self.initial_density.iter().sum();
        if next_bg_sum > 0.0 {
            let scale = density_sum / next_bg_sum;
            for current_background in blurred_background_vec.iter_mut() {
                *current_background *= scale;
            }
        }
        let next_background_arc: Arc<[f64]> = Arc::from(blurred_background_vec);

        // Check quality
        let mut new_data = vec![0.0; self.initial_density.len()];
        for i in 0..new_data.len() {
            new_data[i] = current_perturbation[i] * next_background_arc[i];
        }
        let quality = ln_likelihood(&self.initial_density, &new_data, &self.mask);

        // The new background provides a worse fit so we are done with refinement.
        if self.best_quality > quality {
            // We only converge if the best fitting model (and hence background) was not the initial fit.
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

        // Update the quality, background and model.
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
    /// Creates an iterative fitter for refining both the spiral model and the background.
    ///
    /// # Arguments
    /// * `initial_density` - The observed density data.
    /// * `initial_background` - Initial guess for the background density.
    /// * `mask` - Mask for valid data points (1.0 for valid, 0.0 for invalid).
    /// * `mesh_x` - X-coordinates of the data points.
    /// * `mesh_y` - Y-coordinates of the data points.
    /// * `shape` - Dimensions of the 2D grid (rows, cols).
    /// * `num_components` - Optional override for the number of components (1 or 2). If None, BIC is used.
    /// * `winding` - Optional fixed winding direction (1 for CW, -1 for CCW).
    /// * `improve_background` - Whether to perform iterative background refinement.
    #[must_use]
    pub fn fit_spiral_with_background_iterative<'fit>(
        &'fit self,
        initial_density: &[f64],
        initial_background: &[f64],
        mask: &[f64],
        mesh_x: &[f64],
        mesh_y: &[f64],
        shape: (usize, usize),
        num_components: Option<usize>,
        winding: Option<Winding>,
        improve_background: bool,
    ) -> PSpiralFitterIterative<'fit> {
        let actual_num_components = num_components.unwrap_or_else(|| {
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
            let bic_single = ln_norm.mul_add(6.0, -2.0 * ll_single);
            let bic_double = ln_norm.mul_add(12.0, -2.0 * ll_double);

            if bic_double < bic_single { 2 } else { 1 }
        });

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

    /// Fits a spiral model with background refinement and returns the final result.
    ///
    /// # Panics
    /// This function can panic if there are no results from the fitting algorithm, however,
    /// this can only happen due to an internal error.
    #[must_use]
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

        it.last().expect("there will always be at least one result")
    }
}

impl PSpiralFitterND<6> {
    /// Fits a single-component spiral model, trying both winding directions.
    #[must_use]
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
            Winding::Positive,
        );
        let neg_winding = self.fit_spiral_with_background_with_winding(
            initial_density,
            initial_background,
            mask,
            mesh_x,
            mesh_y,
            Winding::Negative,
        );

        if pos_winding.1 >= neg_winding.1 {
            pos_winding
        } else {
            neg_winding
        }
    }

    /// Fits a single-component spiral model with a fixed winding direction.
    #[must_use]
    pub fn fit_spiral_with_background_with_winding(
        &self,
        initial_density: &[f64],
        initial_background: &[f64],
        mask: &[f64],
        mesh_x: &[f64],
        mesh_y: &[f64],
        winding: Winding,
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
                &PSpiralModelProblem::<1> {
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
    /// Fits a two-component spiral model, trying both winding directions.
    #[must_use]
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
            Winding::Positive,
        );
        let neg_winding = self.fit_spiral_with_background_with_winding(
            initial_density,
            initial_background,
            mask,
            mesh_x,
            mesh_y,
            Winding::Negative,
        );

        if pos_winding.2 >= neg_winding.2 {
            pos_winding
        } else {
            neg_winding
        }
    }

    /// Fits a two-component spiral model with a fixed winding direction.
    #[must_use]
    pub fn fit_spiral_with_background_with_winding(
        &self,
        initial_density: &[f64],
        initial_background: &[f64],
        mask: &[f64],
        mesh_x: &[f64],
        mesh_y: &[f64],
        winding: Winding,
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
                &PSpiralModelProblem::<2> {
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

/// Perform a Gaussian blur in 2D on the given data.
#[must_use]
fn gaussian_blur_2d(data: &[f64], shape: (usize, usize), sigma: f64) -> Arc<[f64]> {
    let (rows, cols) = shape;
    let num_cells = rows * cols;
    assert_eq!(data.len(), num_cells, "`data` must equal `num_cells`.");

    if sigma <= 0.0 {
        return Arc::from(data);
    }

    let kernel_size = (sigma * 4.0).ceil() as usize * 2 + 1;
    let mut kernel = vec![0.0; kernel_size];
    let half_size: i32 = (kernel_size / 2)
        .try_into()
        .expect("the kernel size will not exceed twice the limit of i32.");
    let s2 = 2.0 * sigma * sigma;
    let mut sum = 0.0;
    for (i, kk) in kernel.iter_mut().enumerate() {
        let idx_i32: i32 = i.try_into().expect("the index will not exceed i32.");
        let x = f64::from(idx_i32 - half_size);
        *kk = (-x * x / s2).exp();
        sum += *kk;
    }
    for kk in kernel.iter_mut() {
        *kk /= sum;
    }

    let mut out = ndarray::Array2::from_shape_vec(shape, data.to_vec()).unwrap();
    let mut temp = ndarray::Array2::zeros(shape);

    // Horizontal pass
    let max_col_idx: i32 = cols
        .try_into()
        .expect("the number of columns will not exceed i32.");
    for row_idx in 0..rows {
        for col_idx in 0..cols {
            let col_idx_i32: i32 = col_idx
                .try_into()
                .expect("the column index will not exceed i32.");
            let mut val = 0.0;
            for (i, kk) in kernel.iter().enumerate() {
                let idx_i32: i32 = i.try_into().expect("the index will not exceed i32.");
                let cc = (col_idx_i32 + idx_i32 - half_size).clamp(0, max_col_idx - 1) as usize;
                val += out[[row_idx, cc]] * kk;
            }
            temp[[row_idx, col_idx]] = val;
        }
    }

    // Vertical pass
    let max_row_idx: i32 = rows
        .try_into()
        .expect("the number of rows will not exceed i32.");
    for row_idx in 0..rows {
        let row_idx_i32: i32 = row_idx
            .try_into()
            .expect("the row index will not exceed i32.");
        for col_idx in 0..cols {
            let mut val = 0.0;
            for (i, kk) in kernel.iter().enumerate() {
                let idx_i32: i32 = i.try_into().expect("the index will not exceed i32.");
                let rr = (row_idx_i32 + idx_i32 - half_size).clamp(0, max_row_idx - 1) as usize;
                val += temp[[rr, col_idx]] * kk;
            }
            out[[row_idx, col_idx]] = val;
        }
    }

    Arc::from(out.into_raw_vec_and_offset().0)
}
