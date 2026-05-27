pub fn create_sigmoid_mask(x_scale: f64, y_scale: f64) -> impl Fn(f64, f64) -> f64 {
    move |x: f64, y: f64| {
        let xs = x / x_scale;
        let ys = y / y_scale;
        expit(- (xs * xs + ys * ys - 1.0))
    }
}

pub fn ln_likelihood_f64(data: &[f64], prediction: &[f64], mask: &[f64]) -> f64 {
    assert_eq!(data.len(), prediction.len());
    assert_eq!(data.len(), mask.len());

    let mut result = 0.0;
    for ((current_data, current_prediction), current_mask) in
        data.iter().zip(prediction.iter()).zip(mask.iter())
    {
        if *current_prediction <= 0.0 {
            continue;
        }
        let residual = current_mask * (current_data - current_prediction);
        let numer = residual * residual;
        result += numer / current_prediction;
    }

    -0.5 * result
}

#[derive(Clone, Debug)]
pub struct PSpiralModel {
    pub components: Vec<PSpiralComponent>,
}

impl PSpiralModel {
    pub fn perturbation_scalar(&self, z: f64, vz: f64) -> f64 {
        if self.components.is_empty() {
            return 1.0;
        }
        let mut value = f64::NEG_INFINITY;

        for comp in self.components.iter() {
            value = value.max(comp.perturbation_scalar(z, vz));
        }

        if value.is_finite() { value } else { 1.0 }
    }
    pub fn perturbation_vec(&self, z: &[f64], vz: &[f64], out: &mut [f64]) {
        assert_eq!(z.len(), vz.len());
        assert_eq!(z.len(), out.len());

        for ((zz, vzz), oo) in z.iter().zip(vz.iter()).zip(out.iter_mut()) {
            *oo = self.perturbation_scalar(*zz, *vzz);
        }
    }
}

#[derive(Clone, Debug)]
pub struct PSpiralComponent {
    pub alpha: f64,
    pub b: f64,
    pub c: f64,
    pub theta0: f64,
    pub scale_factor: f64,
    pub rho: f64,
    pub winding: i8,
    pub flattening_strength: f64,
}

#[inline]
pub fn expit(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

impl PSpiralComponent {
    #[inline]
    pub fn spiral_phase(&self, r: f64) -> f64 {
        let abs_c = self.c.abs();
        let abs_b = self.b.abs();
        if abs_c > 1e-10 {
            let half_b_over_c = 0.5 * abs_b / abs_c;
            -half_b_over_c + (half_b_over_c * half_b_over_c + r / abs_c).sqrt()
        } else {
            r / abs_b
        }
    }

    #[inline]
    pub fn perturbation_scalar(&self, z: f64, vz: f64) -> f64 {
        let scale_factor = self.scale_factor;
        
        let r = z.hypot(vz / scale_factor);
        let theta = vz.atan2(z * scale_factor);

        let phase = self.spiral_phase(r);
        let flattening = expit((r - self.rho) / self.flattening_strength);

        1.0 + self.alpha * flattening * (self.winding as f64 * theta - phase - self.theta0).cos()
    }

    pub fn perturbation_vec(&self, z: &[f64], vz: &[f64], out: &mut [f64]) {
        assert_eq!(z.len(), vz.len());
        assert_eq!(z.len(), out.len());

        for ((zz, vzz), oo) in z.iter().zip(vz.iter()).zip(out.iter_mut()) {
            *oo = self.perturbation_scalar(*zz, *vzz);
        }
    }
}
