/// Routines to calculate the ln likelihood.
pub mod likelihood;

pub use likelihood::ln_likelihood;

use itertools::izip;

/// Create a sigmoid mask function given a scale for `x` and `y`.
pub fn create_sigmoid_mask(x_scale: f64, y_scale: f64) -> impl Fn(f64, f64) -> f64 {
    move |x: f64, y: f64| {
        let xs = x / x_scale;
        let ys = y / y_scale;
        -expit(-(xs.mul_add(xs, ys * ys) - 1.0)) + 1.0
    }
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
