use itertools::izip;

/// Create a sigmoid mask function given a scale for `x` and `y`.
pub fn create_sigmoid_mask(x_scale: f64, y_scale: f64) -> impl Fn(f64, f64) -> f64 {
    move |x: f64, y: f64| {
        let xs = x / x_scale;
        let ys = y / y_scale;
        -expit(-(xs.mul_add(xs, ys * ys) - 1.0)) + 1.0
    }
}

/// Calculate the ln likelihood with respect to the observed number count.
///
/// # Panics
/// This function assumes that `data`, `prediction` and `mask` have the same length
/// and will panic if this is not true.
#[must_use]
pub fn ln_likelihood(data: &[f64], prediction: &[f64], mask: &[f64]) -> f64 {
    ln_likelihood_wide(data, prediction, mask)
}

/// Calculate the ln likelihood with respect to the observed number count.
///
/// # Panics
/// This function assumes that `data`, `prediction` and `mask` have the same length
/// and will panic if this is not true.
#[must_use]
fn ln_likelihood_naive(data: &[f64], prediction: &[f64], mask: &[f64]) -> f64 {
    assert_eq!(
        data.len(),
        prediction.len(),
        "`data` and `prediction` must be the same length."
    );
    assert_eq!(
        data.len(),
        mask.len(),
        "`data` and `mask` must be the same length."
    );

    let mut result = 0.0;
    for (current_data, current_prediction, current_mask) in izip!(data, prediction, mask) {
        if *current_prediction <= 0.0 {
            continue;
        }
        let residual = current_mask * (current_data - current_prediction);
        let numer = residual * residual;
        result += numer / current_prediction;
    }

    -0.5 * result
}

/// Calculate the ln likelihood with respect to the observed number count implemented using `wide`.
///
/// # Panics
/// This function assumes that `data`, `prediction` and `mask` have the same length
/// and will panic if this is not true.
#[must_use]
fn ln_likelihood_wide(data: &[f64], prediction: &[f64], mask: &[f64]) -> f64 {
    use wide::CmpGt as _;
    use wide::f64x4;
    assert_eq!(
        data.len(),
        prediction.len(),
        "`data` and `prediction` must be the same length."
    );
    assert_eq!(
        data.len(),
        mask.len(),
        "`data` and `mask` must be the same length."
    );

    let mut accum = f64x4::ZERO;

    let (data_chunks, data_remainder) = data.as_chunks::<4>();
    let (prediction_chunks, prediction_remainder) = prediction.as_chunks::<4>();
    let (mask_chunks, mask_remainder) = mask.as_chunks::<4>();

    for (current_data, current_prediction, current_mask) in
        izip!(data_chunks, prediction_chunks, mask_chunks)
    {
        let current_data = f64x4::from(*current_data);
        let current_prediction = f64x4::from(*current_prediction);
        let current_mask = f64x4::from(*current_mask);

        let valid = current_prediction.simd_gt(f64x4::ZERO);

        let residual = current_mask * (current_data - current_prediction);
        let numer = residual * residual;

        let denom = valid.blend(current_prediction, f64x4::ONE);
        let contrib = valid.blend(numer / denom, f64x4::ZERO);

        accum += contrib;
    }

    let result = accum.reduce_add()
        + ln_likelihood_naive(data_remainder, prediction_remainder, mask_remainder);

    -0.5 * result
}

/// A phase spiral model that can contain multiple spiral components.
#[derive(Clone, Debug)]
pub struct PSpiralModel {
    /// The individual spiral components.
    pub components: Vec<PSpiralComponent>,
}

impl PSpiralModel {
    /// Calculate the perturbation coefficient at a point `(z, vz)`.
    #[must_use]
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

    /// Calculate the perturbation coefficient at a series of points `(z, vz)`, writing the result to `out`.
    ///
    /// # Panics
    /// This function assumes that `z`, `vz` and `out` have the same length
    /// and will panic if this is not true.
    pub fn perturbation_vec(&self, z: &[f64], vz: &[f64], out: &mut [f64]) {
        assert_eq!(z.len(), vz.len(), "`z` must be the same length as `vz`.");
        assert_eq!(z.len(), out.len(), "`z` must be the same length as `out`.");

        for (zz, vzz, oo) in izip!(z.iter(), vz.iter(), out.iter_mut()) {
            *oo = self.perturbation_scalar(*zz, *vzz);
        }
    }
}

/// A single spiral component.
///
/// The shape of the spiral is a log spiral with linear winding
/// and potentially quadratic winding.
#[derive(Clone, Debug)]
pub struct PSpiralComponent {
    /// The perturbation strength.
    pub alpha: f64,
    /// The linear winding parameter.
    pub b: f64,
    /// The quadratic winding parameter.
    pub c: f64,
    /// The angle offset.
    pub theta0: f64,
    /// The scale factor relating `vz` and `z`.
    pub scale_factor: f64,
    /// The flattening radius.
    pub rho: f64,
    /// The winding direction.
    pub winding: i8,
    /// The flattening scale.
    pub flattening_strength: f64,
}

/// The logistic sigmoid function defined as
///
/// `sigm(x) = 1 / (1 + exp(-x))`
#[inline]
#[must_use]
pub fn expit(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

impl PSpiralComponent {
    /// Calculate the log spiral's phase defined implicitly as
    ///
    /// `r = b * phi + c * phi^2`
    #[inline]
    #[must_use]
    pub fn spiral_phase(&self, radius: f64) -> f64 {
        let abs_c = self.c.abs();
        let abs_b = self.b.abs();
        if abs_c > 1e-10 {
            let half_b_over_c = 0.5 * abs_b / abs_c;
            -half_b_over_c + half_b_over_c.mul_add(half_b_over_c, radius / abs_c).sqrt()
        } else {
            radius / abs_b
        }
    }

    /// Calculate the perturbation coefficient at a point `(z, vz)`.
    #[inline]
    #[must_use]
    pub fn perturbation_scalar(&self, z: f64, vz: f64) -> f64 {
        let scale_factor = self.scale_factor;

        let radius = z.hypot(vz / scale_factor);
        let theta = vz.atan2(z * scale_factor);

        let phase = self.spiral_phase(radius);
        let flattening = expit((radius - self.rho) / self.flattening_strength);
        let geometric_term = theta
            .mul_add(f64::from(self.winding), -phase - self.theta0)
            .cos();

        // 1 + alpha * flattening * cos(theta - phi_s - theta0)
        (self.alpha * flattening).mul_add(geometric_term, 1.0)
    }

    /// Calculate the perturbation coefficient at a series of points `(z, vz)`, writing the result to `out`.
    ///
    /// # Panics
    /// This function assumes that `z`, `vz` and `out` have the same length
    /// and will panic if this is not true.
    pub fn perturbation_vec(&self, z: &[f64], vz: &[f64], out: &mut [f64]) {
        assert_eq!(z.len(), vz.len(), "`z` must be the same length as `vz`.");
        assert_eq!(z.len(), out.len(), "`z` must be the same length as `out`.");

        for (zz, vzz, oo) in izip!(z.iter(), vz.iter(), out.iter_mut()) {
            *oo = self.perturbation_scalar(*zz, *vzz);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::ln_likelihood_naive;
    use crate::ln_likelihood_wide;
    use rand::RngExt as _;
    use rand::SeedableRng as _;
    use rand::rngs::SmallRng;

    /// Make test data.
    fn make_data(n: usize, zero_fraction: f64, seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut rng = SmallRng::seed_from_u64(seed);

        let prediction: Vec<f64> = (0..n)
            .map(|_| {
                if rng.random::<f64>() < zero_fraction {
                    0.0
                } else {
                    rng.random_range(0.1..100.0)
                }
            })
            .collect();

        let data: Vec<f64> = prediction
            .iter()
            .map(|&pred| {
                if pred == 0.0 {
                    0.0
                } else {
                    pred * rng.random_range(0.8..1.2)
                }
            })
            .collect();

        let mask: Vec<f64> = (0..n).map(|_| rng.random_range(0.5..2.0)).collect();

        (data, prediction, mask)
    }

    /// Check that the SIMD implementation is correct by comparing to the scalar version.
    #[test]
    fn check_ln_likelihood_implementation() {
        let (data, prediction, mask) = make_data(100_000, 0.05, 1001);

        let naive_ll = ln_likelihood_naive(&data, &prediction, &mask);
        let wide_ll = ln_likelihood_wide(&data, &prediction, &mask);

        assert_float_eq::assert_float_absolute_eq!(naive_ll, wide_ll);
    }
}
