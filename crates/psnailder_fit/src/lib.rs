//! Fitting logic for phase spiral models.
//!
//! This crate provides tools for fitting one or two-component phase spiral models to 2D density data,
//! optionally refining the background density iteratively.

extern crate alloc;

pub mod parameter_layout;

pub use parameter_layout::{LayoutError, ParameterBound, ParameterLayout};

use alloc::sync::Arc;
use basin::{BoxConstraints, CostFunction};
use core::convert;
use itertools::izip;
use psnailder_core::{PSpiralComponent, PSpiralModel, Winding, ln_likelihood};
use psnailder_tiktak::{DynamicTikTak, TikTak};
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
            if pred.to_array().iter().any(|value| !value.is_finite()) {
                return Ok(f64::INFINITY);
            }
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
            if !pred.is_finite() {
                return Ok(f64::INFINITY);
            }
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
            // max can hide a NaN in one arm, so validate both first.
            if pert1
                .to_array()
                .iter()
                .chain(pert2.to_array().iter())
                .any(|value| !value.is_finite())
            {
                return Ok(f64::INFINITY);
            }
            let pert = pert1.max(pert2);
            let pred = pert * bg;
            if pred.to_array().iter().any(|value| !value.is_finite()) {
                return Ok(f64::INFINITY);
            }
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
            if !p1.is_finite() || !p2.is_finite() {
                return Ok(f64::INFINITY);
            }
            let pred = p1.max(p2) * bg;
            if !pred.is_finite() {
                return Ok(f64::INFINITY);
            }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn fixed_signal_fitter<const N: usize>() -> PSpiralFitterND<N> {
        // Fixed zero amplitude makes every prediction equal its background.
        // Exercise the real optimizer without depending on parameter recovery.
        PSpiralFitterND {
            tiktak: TikTak::new(1, 0.25, 0.1, 0.995),
            alpha_bounds: (0.0, 0.0),
            b_bounds: (0.05, 0.05),
            c_bounds: (0.002, 0.002),
            theta0_bounds: (0.0, 0.0),
            scale_factor_bounds: (40.0, 40.0),
            rho_bounds: (0.09, 0.09),
        }
    }

    fn refinement_fitter(max_iterations: usize, sigma_z: f64, sigma_vz: f64) -> PSpiralFitter {
        PSpiralFitter {
            fitter_single: fixed_signal_fitter(),
            fitter_double: fixed_signal_fitter(),
            max_iterations: Some(max_iterations),
            atol: 0.0,
            rtol: 0.0,
            sigma_z,
            sigma_vz,
        }
    }

    #[test]
    fn parameter_counts_follow_each_fitters_bounds() {
        let mut single = fixed_signal_fitter::<6>();
        let mut double = fixed_signal_fitter::<12>();
        assert_eq!(single.num_free_parameters(), 0);
        assert_eq!(double.num_free_parameters(), 0);
        single.b_bounds = (0.01, 0.1);
        double.rho_bounds = (0.01, 0.2);
        double.theta0_bounds = (0.0, 1.0);
        assert_eq!(single.num_free_parameters(), 1);
        assert_eq!(double.num_free_parameters(), 4);
        for bounds in [
            &mut single.alpha_bounds,
            &mut single.c_bounds,
            &mut single.theta0_bounds,
            &mut single.scale_factor_bounds,
            &mut single.rho_bounds,
        ] {
            bounds.1 = bounds.0 + 1.0;
        }
        assert_eq!(single.num_free_parameters(), 6);
        double = PSpiralFitterND {
            tiktak: TikTak::new(1, 0.25, 0.1, 0.995),
            alpha_bounds: single.alpha_bounds,
            b_bounds: single.b_bounds,
            c_bounds: single.c_bounds,
            theta0_bounds: single.theta0_bounds,
            scale_factor_bounds: single.scale_factor_bounds,
            rho_bounds: single.rho_bounds,
        };
        assert_eq!(double.num_free_parameters(), 12);
    }

    #[test]
    fn bic_prefers_fewer_free_parameters_and_single_on_ties() {
        let mut fitter = refinement_fitter(1, 0.0, 0.0);
        // Zero amplitude makes both likelihoods identical even when b is free.
        for (bounds, expected) in [((0.01, 0.1), 2), ((0.05, 0.05), 1)] {
            fitter.fitter_single.b_bounds = bounds;
            let iter = fitter.fit_spiral_with_background_iterative(
                &[2.0; 4],
                &[2.0; 4],
                &[1.0; 4],
                &[0.1; 4],
                &[0.1; 4],
                (2, 2),
                None,
                Some(Winding::Positive),
                false,
            );
            assert_eq!(iter.num_components, expected);
        }
    }

    fn assert_consistent_result(result: &PSpiralFitResult, coordinates: &[f64], mask: &[f64]) {
        for (model, background, score) in [
            (
                &result.initial_model,
                &result.initial_background,
                result.initial_lnl,
            ),
            (
                &result.final_model,
                &result.final_background,
                result.final_lnl,
            ),
        ] {
            let mut prediction = vec![0.0; coordinates.len()];
            model.perturbation_vec(coordinates, coordinates, &mut prediction);
            for (value, curr_background) in prediction.iter_mut().zip(background.iter()) {
                *value *= curr_background;
            }
            let recomputed = ln_likelihood(&result.data, &prediction, mask);
            assert!(score.is_finite());
            assert!(
                (score - recomputed).abs() < 1e-12,
                "stored {score}, recomputed {recomputed}"
            );
        }
    }

    #[test]
    fn refinement_accepts_update_at_iteration_limit() {
        let data = [1.0, 3.0, 2.0, 4.0, 2.0, 3.0];
        let background = [2.5; 6];
        let coordinates = [0.1; 6];
        let mask = [1.0; 6];
        let fitter = refinement_fitter(1, 0.0, 0.0);
        for count in [1, 2] {
            let mut iter = fitter.fit_spiral_with_background_iterative(
                &data,
                &background,
                &mask,
                &coordinates,
                &coordinates,
                (2, 3),
                Some(count),
                Some(Winding::Positive),
                true,
            );
            let result = iter.next().unwrap();
            assert_consistent_result(&result, &coordinates, &mask);
            assert_eq!(&*result.final_background, &data);
            assert_eq!(&*result.initial_background, &background);
            assert_eq!(result.final_model.components.len(), count);
            assert_eq!(result.num_iterations, 1);
            assert!(!result.converged);
            assert!(result.final_lnl > result.initial_lnl);
            assert!(iter.next().is_none());
            assert!(iter.next().is_none());
        }
    }

    #[test]
    fn refinement_rejects_first_update() {
        let data = [1.0, 3.0, 2.0, 4.0, 2.0, 3.0];
        let coordinates = [0.1; 6];
        let mask = [1.0; 6];
        let fitter = refinement_fitter(3, 1.0, 1.0);
        for count in [1, 2] {
            let mut iter = fitter.fit_spiral_with_background_iterative(
                &data,
                &data,
                &mask,
                &coordinates,
                &coordinates,
                (2, 3),
                Some(count),
                Some(Winding::Positive),
                true,
            );
            let result = iter.next().unwrap();
            assert_consistent_result(&result, &coordinates, &mask);
            assert_eq!(&*result.final_background, &data);
            assert_eq!(result.num_iterations, 1);
            assert!(!result.converged);
            assert!(iter.next().is_none());
        }
    }

    #[test]
    fn refinement_retains_accepted_state_after_rejection() {
        let data = [1.0, 3.0, 2.0, 4.0, 2.0, 3.0];
        let background = [2.5; 6];
        let coordinates = [0.1; 6];
        let mask = [1.0; 6];
        let fitter = refinement_fitter(3, 0.0, 0.0);
        for count in [1, 2] {
            let mut iter = fitter.fit_spiral_with_background_iterative(
                &data,
                &background,
                &mask,
                &coordinates,
                &coordinates,
                (2, 3),
                Some(count),
                Some(Winding::Positive),
                true,
            );
            let accepted = iter.next().unwrap();
            assert_eq!(&*accepted.final_background, &data);
            assert!(!accepted.converged);
            // Force a worse proposal next: smoothing moves away from the
            // exact fit. Only the proposal changes, not the accepted state.
            iter.sigma_z = 1.0;
            iter.sigma_vz = 1.0;
            let rejected = iter.next().unwrap();
            assert_consistent_result(&accepted, &coordinates, &mask);
            assert_consistent_result(&rejected, &coordinates, &mask);
            assert_eq!(&*rejected.final_background, &data);
            assert_eq!(&*accepted.initial_background, &background);
            assert_eq!(rejected.final_lnl, accepted.final_lnl);
            assert_eq!(rejected.num_iterations, 2);
            assert!(rejected.converged);
            assert!(iter.next().is_none());
        }
    }

    #[test]
    fn fixed_background_fit_skips_refinement() {
        let data = [1.0, 3.0, 2.0, 4.0, 2.0, 3.0];
        let background = [2.5; 6];
        let coordinates = [0.1; 6];
        let mask = [1.0; 6];
        let fitter = refinement_fitter(3, 0.0, 0.0);
        for count in [1, 2] {
            let mut iter = fitter.fit_spiral_with_background_iterative(
                &data,
                &background,
                &mask,
                &coordinates,
                &coordinates,
                (2, 3),
                Some(count),
                Some(Winding::Positive),
                false,
            );
            let result = iter.next().unwrap();
            assert_consistent_result(&result, &coordinates, &mask);
            assert_eq!(&*result.final_background, &background);
            assert_eq!(result.final_lnl, result.initial_lnl);
            assert_eq!(result.num_iterations, 1);
            assert!(result.converged);
            assert!(iter.next().is_none());
        }
    }

    #[test]
    fn objectives_reject_nonfinite_predictions() {
        let data = [2.0; 7];
        let coordinates = [0.1; 7];
        let bounds = vec![0.0; 12];
        let single = vec![0.5, 0.05, 0.002, 0.0, 40.0, 0.09];
        let double = single.repeat(2);
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for index in 0..7 {
                for weight in [0.0, 1.0] {
                    let mut background = [1.0; 7];
                    background[index] = invalid;
                    let mut mask = [1.0; 7];
                    mask[index] = weight;
                    let one = PSpiralModelProblem::<1> {
                        data: &data,
                        background: &background,
                        mask: &mask,
                        x: &coordinates,
                        y: &coordinates,
                        winding: Winding::Positive,
                        lb: &bounds,
                        ub: &bounds,
                    };
                    let two = PSpiralModelProblem::<2> {
                        data: &data,
                        background: &background,
                        mask: &mask,
                        x: &coordinates,
                        y: &coordinates,
                        winding: Winding::Positive,
                        lb: &bounds,
                        ub: &bounds,
                    };
                    assert_eq!(one.cost(&single).unwrap(), f64::INFINITY);
                    assert_eq!(two.cost(&double).unwrap(), f64::INFINITY);
                }
            }
        }
    }

    #[test]
    fn runtime_objective_matches_single_component_objective() {
        let data = [2.0; 7];
        let background = [1.0; 7];
        let coordinates = [0.1; 7];
        let full = [0.5, 0.05, 0.002, 0.0, 40.0, 0.09];
        let bounds: Vec<_> = full
            .iter()
            .map(|value| ParameterBound::Fixed(*value))
            .collect();
        let layout = ParameterLayout::from_bounds(&bounds).unwrap();
        let runtime = RuntimePSpiralModelProblem {
            data: &data,
            background: &background,
            mask: &[1.0; 7],
            x: &coordinates,
            y: &coordinates,
            winding: Winding::Positive,
            layout: &layout,
            num_components: 1,
        };
        let legacy = PSpiralModelProblem::<1> {
            data: &data,
            background: &background,
            mask: &[1.0; 7],
            x: &coordinates,
            y: &coordinates,
            winding: Winding::Positive,
            lb: &full.to_vec(),
            ub: &full.to_vec(),
        };
        assert_eq!(
            runtime.cost(&Vec::new()).unwrap(),
            legacy.cost(&full.to_vec()).unwrap()
        );
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

/// Runtime-sized spiral objective used when some parameters are fixed.
#[derive(Debug, Clone)]
pub struct RuntimePSpiralModelProblem<'prob> {
    /// The density grid.
    pub data: &'prob [f64],
    /// The estimated background grid.
    pub background: &'prob [f64],
    /// The evaluation mask.
    pub mask: &'prob [f64],
    /// Grid x coordinates.
    pub x: &'prob [f64],
    /// Grid y coordinates.
    pub y: &'prob [f64],
    /// Winding direction shared by all components.
    pub winding: Winding,
    /// Mapping between free and full parameter vectors.
    pub layout: &'prob ParameterLayout,
    /// Number of six-parameter components in the full vector.
    pub num_components: usize,
}

impl CostFunction for RuntimePSpiralModelProblem<'_> {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = convert::Infallible;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Self::Error> {
        let Ok(full) = self.layout.unpack(param) else {
            return Ok(f64::INFINITY);
        };
        if full.len() != self.num_components * 6 {
            return Ok(f64::INFINITY);
        }
        let components: Vec<_> = full
            .chunks_exact(6)
            .map(|values| PSpiralComponent {
                alpha: values[0],
                b_winding: values[1],
                c_winding: values[2],
                theta0: values[3],
                scale_factor: values[4],
                rho: values[5],
                winding: self.winding,
                flattening_strength: 0.1,
            })
            .collect();

        let (data_chunks, data_remainder) = self.data.as_chunks::<4>();
        let (x_chunks, x_remainder) = self.x.as_chunks::<4>();
        let (y_chunks, y_remainder) = self.y.as_chunks::<4>();
        let (background_chunks, background_remainder) = self.background.as_chunks::<4>();
        let (mask_chunks, mask_remainder) = self.mask.as_chunks::<4>();
        let mut result = 0.0;
        let mut accumulator = f64x4::ZERO;

        for (current_data, x, y, background, current_mask) in izip!(
            data_chunks,
            x_chunks,
            y_chunks,
            background_chunks,
            mask_chunks
        ) {
            let data = f64x4::from(*current_data);
            let x = f64x4::from(*x);
            let y = f64x4::from(*y);
            let background = f64x4::from(*background);
            let mask = f64x4::from(*current_mask);
            let mut perturbation = f64x4::ZERO;
            for component in &components {
                let current = component.perturbation_wide(x, y);
                if current.to_array().iter().any(|value| !value.is_finite()) {
                    return Ok(f64::INFINITY);
                }
                perturbation = perturbation.max(current);
            }
            let prediction = perturbation * background;
            if prediction.to_array().iter().any(|value| !value.is_finite()) {
                return Ok(f64::INFINITY);
            }
            let residual = mask * (data - prediction);
            let term = (residual * residual) / prediction;
            accumulator += prediction.simd_le(f64x4::ZERO).blend(f64x4::ZERO, term);
        }

        for (data, x, y, background, mask) in izip!(
            data_remainder,
            x_remainder,
            y_remainder,
            background_remainder,
            mask_remainder
        ) {
            let perturbation = components
                .iter()
                .map(|component| component.perturbation_scalar(*x, *y))
                .fold(0.0, f64::max);
            let prediction = perturbation * background;
            if !prediction.is_finite() {
                return Ok(f64::INFINITY);
            }
            if prediction > 0.0 {
                let residual = mask * (data - prediction);
                result += (residual * residual) / prediction;
            }
        }
        Ok(0.5 * (result + accumulator.reduce_add()))
    }
}

impl BoxConstraints for RuntimePSpiralModelProblem<'_> {
    fn lower(&self) -> &Self::Param {
        self.layout.lower()
    }

    fn upper(&self) -> &Self::Param {
        self.layout.upper()
    }
}

/// Fit a runtime-sized model using a compact parameter layout.
///
/// This retains the SIMD likelihood calculation while allowing arbitrary
/// fixed/free parameter combinations.
pub fn fit_with_parameter_layout(
    data: &[f64],
    background: &[f64],
    mask: &[f64],
    mesh_x: &[f64],
    mesh_y: &[f64],
    winding: Winding,
    layout: &ParameterLayout,
    num_components: usize,
    log_num_samples: u8,
    keep_ratio: f32,
) -> (PSpiralModel, f64) {
    let objective = RuntimePSpiralModelProblem {
        data,
        background,
        mask,
        x: mesh_x,
        y: mesh_y,
        winding,
        layout,
        num_components,
    };
    let full_params = if layout.free_len() == 0 {
        layout.unpack(&[]).expect("an all-fixed layout must unpack")
    } else {
        let optimizer =
            DynamicTikTak::new(layout.free_len(), log_num_samples, keep_ratio, 0.1, 0.995);
        let bounds: Vec<_> = layout
            .lower()
            .iter()
            .copied()
            .zip(layout.upper().iter().copied())
            .collect();
        let result = optimizer
            .minimize(&objective, &bounds)
            .expect("the objective is infallible");
        layout
            .unpack(&result.params)
            .expect("optimizer results satisfy layout bounds")
    };
    let components = full_params
        .chunks_exact(6)
        .map(|values| PSpiralComponent {
            alpha: values[0],
            b_winding: values[1],
            c_winding: values[2],
            theta0: values[3],
            scale_factor: values[4],
            rho: values[5],
            winding,
            flattening_strength: 0.1,
        })
        .collect();
    let cost = objective
        .cost(&if layout.free_len() == 0 {
            Vec::new()
        } else {
            layout
                .pack(&full_params)
                .expect("expanded parameters can be packed")
        })
        .expect("the objective is infallible");
    (PSpiralModel { components }, -cost)
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

impl<const N: usize> PSpiralFitterND<N> {
    /// Count free parameters using the bounds repeated for each fitted arm.
    fn num_free_parameters(&self) -> usize {
        let bounds: Vec<_> = [
            self.alpha_bounds,
            self.b_bounds,
            self.c_bounds,
            self.theta0_bounds,
            self.scale_factor_bounds,
            self.rho_bounds,
        ]
        .into_iter()
        .cycle()
        .take(N)
        .map(|(lower, upper)| {
            if lower == upper {
                ParameterBound::Fixed(lower)
            } else {
                ParameterBound::Interval { lower, upper }
            }
        })
        .collect();
        ParameterLayout::from_bounds(&bounds)
            .expect("fitter bounds must be finite and ordered")
            .free_len()
    }
}

/// A high-level fitter that supports single and double component models with iterative background refinement.
#[derive(Clone)]
pub struct PSpiralFitter {
    /// Fitter used for one-arm models.
    pub fitter_single: OneArmFitter,
    /// Fitter used for two-arm models.
    pub fitter_double: TwoArmFitter,
    /// Maximum number of iterations for background refinement.
    pub max_iterations: Option<usize>,
    /// Sigma for Gaussian smoothing applied to the background during refinement for z-axis.
    pub sigma_z: f64,
    /// Sigma for Gaussian smoothing applied to the background during refinement for Vz-axis.
    pub sigma_vz: f64,
    /// Absolute refinement improvement tolerance.
    pub atol: f64,
    /// Relative refinement improvement tolerance.
    pub rtol: f64,
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
    /// Total objective evaluations used by optimization attempts.
    pub nfev: u64,
    /// Number of background refinement attempts performed.
    pub nit: u64,
    /// Whether this checkpoint is terminal.
    pub terminal: bool,
}

/// An iterator that performs the fitting process step-by-step.
///
/// This allows for inspecting intermediate results or customizing the refinement loop.
pub struct PSpiralFitterIterative {
    /// The composite one and two arm spiral fitter.
    pub fitter: PSpiralFitter,
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
    /// Sigma for Gaussian smoothing applied to the background during refinement for z-axis.
    pub sigma_z: f64,
    /// Sigma for Gaussian smoothing applied to the background during refinement for Vz-axis.
    pub sigma_vz: f64,
    /// Absolute refinement improvement tolerance.
    pub atol: f64,
    /// Relative refinement improvement tolerance.
    pub rtol: f64,
    /// Whether to perform iterative background refinement.
    pub improve_background: bool,
    /// Whether the refinement process has terminated.
    pub is_finished: bool,
    /// Total objective evaluations across optimization attempts.
    pub total_nfev: u64,
}

impl PSpiralFitterIterative {
    /// Optimize the current background and preserve the selected winding.
    fn optimize_model(&mut self) -> (PSpiralModel, f64, u64) {
        match self.num_components {
            1 => {
                let (component, quality, nfev) = if let Some(winding) = self.best_winding {
                    self.fitter
                        .fitter_single
                        .fit_spiral_with_background_with_winding(
                            &self.initial_density,
                            &self.current_background,
                            &self.mask,
                            &self.mesh_x,
                            &self.mesh_y,
                            winding,
                        )
                } else {
                    let result = self.fitter.fitter_single.fit_spiral_with_background(
                        &self.initial_density,
                        &self.current_background,
                        &self.mask,
                        &self.mesh_x,
                        &self.mesh_y,
                    );
                    self.best_winding = Some(result.0.winding);
                    result
                };
                (
                    PSpiralModel {
                        components: vec![component],
                    },
                    quality,
                    nfev,
                )
            }
            2 => {
                let (first, second, quality, nfev) = if let Some(winding) = self.best_winding {
                    self.fitter
                        .fitter_double
                        .fit_spiral_with_background_with_winding(
                            &self.initial_density,
                            &self.current_background,
                            &self.mask,
                            &self.mesh_x,
                            &self.mesh_y,
                            winding,
                        )
                } else {
                    let result = self.fitter.fitter_double.fit_spiral_with_background(
                        &self.initial_density,
                        &self.current_background,
                        &self.mask,
                        &self.mesh_x,
                        &self.mesh_y,
                    );
                    self.best_winding = Some(result.0.winding);
                    result
                };
                (
                    PSpiralModel {
                        components: vec![first, second],
                    },
                    quality,
                    nfev,
                )
            }
            _ => panic!("Unsupported `num_components`"),
        }
    }

    /// Propose a smoothed, count-normalized background for a model.
    fn propose_background(&self, model: &PSpiralModel) -> Arc<[f64]> {
        let mut perturbation = vec![0.0; self.initial_density.len()];
        model.perturbation_vec(&self.mesh_x, &self.mesh_y, &mut perturbation);
        let unsmoothed: Vec<f64> = self
            .initial_density
            .iter()
            .zip(perturbation.iter())
            .map(|(data, factor)| data / factor.max(1e-10))
            .collect();
        let blurred = gaussian_blur_2d(&unsmoothed, self.shape, self.sigma_z, self.sigma_vz);
        let mut background = blurred.to_vec();
        let predicted_total: f64 = background
            .iter()
            .zip(perturbation.iter())
            .map(|(value, factor)| value * factor)
            .sum();
        let data_total: f64 = self.initial_density.iter().sum();
        if predicted_total.is_finite() && predicted_total > 0.0 {
            let scale = data_total / predicted_total;
            for value in &mut background {
                *value *= scale;
            }
        }
        Arc::from(background)
    }

    /// Evaluate the likelihood of a model with a proposed background.
    fn background_quality(&self, model: &PSpiralModel, background: &[f64]) -> f64 {
        let mut prediction = vec![0.0; background.len()];
        model.perturbation_vec(&self.mesh_x, &self.mesh_y, &mut prediction);
        prediction
            .iter_mut()
            .zip(background.iter())
            .for_each(|(value, background)| *value *= background);
        ln_likelihood(&self.initial_density, &prediction, &self.mask)
    }

    /// Materialize the currently accepted state as an immutable fit result.
    fn snapshot(&self) -> PSpiralFitResult {
        PSpiralFitResult {
            data: Arc::clone(&self.initial_density),
            initial_model: self
                .initial_model
                .clone()
                .expect("initial model is set before snapshot"),
            initial_background: Arc::clone(&self.initial_background),
            final_model: self
                .best_model
                .clone()
                .expect("best model is set before snapshot"),
            final_background: Arc::clone(&self.current_background),
            num_iterations: self.iteration_index,
            max_iterations: self.max_iterations,
            converged: self.converged,
            initial_lnl: self.initial_quality,
            final_lnl: self.best_quality,
            nfev: self.total_nfev,
            nit: self.iteration_index as u64,
            terminal: self.is_finished,
        }
    }
}

impl Iterator for PSpiralFitterIterative {
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

        let (current_model, ll, nfev) = self.optimize_model();
        self.total_nfev += nfev;

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
            return Some(self.snapshot());
        }

        let next_background = self.propose_background(&current_model);
        let quality = self.background_quality(&current_model, &next_background);

        // The new background provides a worse fit so we are done with refinement.
        let improvement = quality - self.best_quality;
        let tolerance = self.rtol.mul_add(self.best_quality.abs(), self.atol);
        if improvement <= tolerance {
            // We only converge if the best fitting model (and hence background) was not the initial fit.
            self.converged = self.best_model.is_some();
            if self.best_model.is_none() {
                self.best_model = self.initial_model.clone();
            }
            self.is_finished = true;
            return Some(self.snapshot());
        }

        // Update the quality, background and model.
        self.best_quality = quality;
        self.current_background = next_background;
        self.best_model = Some(current_model);

        Some(self.snapshot())
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
    pub fn fit_spiral_with_background_iterative(
        &self,
        initial_density: &[f64],
        initial_background: &[f64],
        mask: &[f64],
        mesh_x: &[f64],
        mesh_y: &[f64],
        shape: (usize, usize),
        num_components: Option<usize>,
        winding: Option<Winding>,
        improve_background: bool,
    ) -> PSpiralFitterIterative {
        let actual_num_components = num_components.unwrap_or_else(|| {
            // BIC comparison penalizes only parameters that remain free.
            let (_, ll_single, _) = match winding {
                Some(winding) => self.fitter_single.fit_spiral_with_background_with_winding(
                    initial_density,
                    initial_background,
                    mask,
                    mesh_x,
                    mesh_y,
                    winding,
                ),
                None => self.fitter_single.fit_spiral_with_background(
                    initial_density,
                    initial_background,
                    mask,
                    mesh_x,
                    mesh_y,
                ),
            };
            let (_, _, ll_double, _) = match winding {
                Some(winding) => self.fitter_double.fit_spiral_with_background_with_winding(
                    initial_density,
                    initial_background,
                    mask,
                    mesh_x,
                    mesh_y,
                    winding,
                ),
                None => self.fitter_double.fit_spiral_with_background(
                    initial_density,
                    initial_background,
                    mask,
                    mesh_x,
                    mesh_y,
                ),
            };

            let ln_norm = initial_density.iter().sum::<f64>().ln();
            let bic_single = ln_norm.mul_add(
                self.fitter_single.num_free_parameters() as f64,
                -2.0 * ll_single,
            );
            let bic_double = ln_norm.mul_add(
                self.fitter_double.num_free_parameters() as f64,
                -2.0 * ll_double,
            );

            if bic_double < bic_single { 2 } else { 1 }
        });

        PSpiralFitterIterative {
            fitter: self.clone(),
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
            sigma_z: self.sigma_z,
            sigma_vz: self.sigma_vz,
            atol: self.atol,
            rtol: self.rtol,
            improve_background,
            is_finished: false,
            total_nfev: 0,
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
    ) -> (PSpiralComponent, f64, u64) {
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
            (pos_winding.0, pos_winding.1, pos_winding.2 + neg_winding.2)
        } else {
            (neg_winding.0, neg_winding.1, pos_winding.2 + neg_winding.2)
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
    ) -> (PSpiralComponent, f64, u64) {
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

        (best_model, -res.cost, res.nfev)
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
    ) -> (PSpiralComponent, PSpiralComponent, f64, u64) {
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
            (
                pos_winding.0,
                pos_winding.1,
                pos_winding.2,
                pos_winding.3 + neg_winding.3,
            )
        } else {
            (
                neg_winding.0,
                neg_winding.1,
                neg_winding.2,
                pos_winding.3 + neg_winding.3,
            )
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
    ) -> (PSpiralComponent, PSpiralComponent, f64, u64) {
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

        (comp1, comp2, -res.cost, res.nfev)
    }
}

/// Perform a Gaussian blur in 2D on the given data.
#[must_use]
fn gaussian_blur_2d(
    data: &[f64],
    shape: (usize, usize),
    sigma_z: f64,
    sigma_vz: f64,
) -> Arc<[f64]> {
    let (rows, cols) = shape;
    let num_cells = rows * cols;
    assert_eq!(data.len(), num_cells, "`data` must equal `num_cells`.");

    let make_kernel = |sigma: f64| {
        if sigma <= 0.0 {
            return vec![1.0];
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
        for kk in &mut kernel {
            *kk /= sum;
        }
        kernel
    };
    let z_kernel = make_kernel(sigma_z);
    let vz_kernel = make_kernel(sigma_vz);
    let z_half_size: i32 = (z_kernel.len() / 2)
        .try_into()
        .expect("the kernel size will not exceed twice the limit of i32.");
    let vz_half_size: i32 = (vz_kernel.len() / 2)
        .try_into()
        .expect("the kernel size will not exceed twice the limit of i32.");

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
            for (i, kk) in z_kernel.iter().enumerate() {
                let idx_i32: i32 = i.try_into().expect("the index will not exceed i32.");
                let cc = (col_idx_i32 + idx_i32 - z_half_size).clamp(0, max_col_idx - 1) as usize;
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
            for (i, kk) in vz_kernel.iter().enumerate() {
                let idx_i32: i32 = i.try_into().expect("the index will not exceed i32.");
                let rr = (row_idx_i32 + idx_i32 - vz_half_size).clamp(0, max_row_idx - 1) as usize;
                val += temp[[rr, col_idx]] * kk;
            }
            out[[row_idx, col_idx]] = val;
        }
    }

    Arc::from(out.into_raw_vec_and_offset().0)
}
