use argmin::core::CostFunction;
use psnailder_core::{PSpiralComponent, PSpiralModel};
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

pub struct PSpiralFitterIterative {
    pub fitter: PSpiralFitter,
    pub best_background: Vec<f64>,
    pub iteration_index: usize,
}

pub struct PSpiralFitter {
    pub fitter_single: PSpiralFitterND<6>,
    pub fitter_double: PSpiralFitterND<12>,
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

impl PSpiralFitter {
    pub fn fit_spiral_with_background(
        &self,
        initial_density: &[f64],
        initial_background: &[f64],
        mask: &[f64],
        mesh_x: &[f64],
        mesh_y: &[f64],
    ) -> PSpiralFitResult {
        let (model_single, ll_single) = self.fitter_single.fit_spiral_with_background(
            initial_density,
            initial_background,
            mask,
            mesh_x,
            mesh_y,
        );
        let (model_double_a, model_double_b, ll_double) = self
            .fitter_double
            .fit_spiral_with_background(initial_density, initial_background, mask, mesh_x, mesh_y);

        let aic_single = 2.0 * 6.0 - 2.0 * ll_single;
        let aic_double = 2.0 * 6.0 - 2.0 * ll_double;

        if aic_double < aic_single {
            let model = PSpiralModel {
                components: vec![model_double_a, model_double_b],
            };
            PSpiralFitResult {
                data: initial_density.to_vec(),
                initial_model: model.clone(),
                initial_background: initial_background.to_vec(),
                final_model: model,
                final_background: initial_background.to_vec(),
                num_iterations: 1,
                max_iterations: Some(1),
                converged: true,
            }
        } else {
            let model = PSpiralModel {
                components: vec![model_single],
            };
            PSpiralFitResult {
                data: initial_density.to_vec(),
                initial_model: model.clone(),
                initial_background: initial_background.to_vec(),
                final_model: model,
                final_background: initial_background.to_vec(),
                num_iterations: 1,
                max_iterations: Some(1),
                converged: true,
            }
        }
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
