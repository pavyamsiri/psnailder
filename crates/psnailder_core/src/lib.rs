pub fn ln_likelihood_f64(data: &[f64], prediction: &[f64], mask: &[f64]) -> f64 {
    assert_eq!(data.len(), prediction.len());
    assert_eq!(data.len(), mask.len());

    let mut result = 0.0;
    for ((current_data, current_prediction), current_mask) in
        data.iter().zip(prediction.iter()).zip(mask.iter())
    {
        let residual = current_mask * (current_data - current_prediction);
        let numer = residual * residual;
        let denom = current_prediction + ((*current_prediction == 0.0) as i32 as f64);
        result += numer / denom;
    }

    -0.5 * result
}

#[derive(Clone, Debug)]
pub struct PSpiralModel {
    pub components: Vec<PSpiralComponent>,
}

impl PSpiralModel {
    pub fn perturbation_scalar(&self, z: f64, vz: f64) -> f64 {
        let mut value = f64::NEG_INFINITY;

        for comp in self.components.iter() {
            value = value.max(comp.perturbation_scalar(z, vz));
        }

        value
    }
    pub fn perturbation_vec(&self, z: &[f64], vz: &[f64], out: &mut [f64]) {
        assert_eq!(z.len(), vz.len());
        assert_eq!(z.len(), out.len());

        for ((zz, vzz), oo) in z.iter().zip(vz.iter()).zip(out.iter_mut()) {
            let mut value = f64::NEG_INFINITY;

            for comp in self.components.iter() {
                value = value.max(comp.perturbation_scalar(*zz, *vzz));
            }

            *oo = value;
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
fn expit(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

impl PSpiralComponent {
    #[inline]
    pub fn spiral_phase(&self, r: f64) -> f64 {
        if self.c != 0.0 {
            let half_b_over_c = 0.5 * self.b / self.c;
            -half_b_over_c + (half_b_over_c * half_b_over_c + r / self.c).sqrt()
        } else {
            r / self.b
        }
    }

    #[inline]
    pub fn perturbation_scalar(&self, z: f64, vz: f64) -> f64 {
        let scaled_z = z * self.scale_factor;
        let scaled_vz = vz / self.scale_factor;

        let r = (z * z + scaled_vz * scaled_vz).sqrt();
        let theta = vz.atan2(scaled_z);

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
