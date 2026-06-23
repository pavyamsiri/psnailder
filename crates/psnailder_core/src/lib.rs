/// Routines to calculate the ln likelihood.
pub mod likelihood;

use core::fmt;
use itertools::izip;
use psnailder_math::{arctan2_wide, expit, expit_wide};
use wide::f64x4;

pub use likelihood::ln_likelihood;

/// Helper macro to convert a `usize` to `f64` by doing a checked
/// conversion through `u32` with an expect message.
#[macro_export]
macro_rules! usize_to_f64 {
    ($val:expr, $msg:literal) => {
        f64::from(u32::try_from($val).expect($msg))
    };
}

/// Create a sigmoid mask function given a scale for `x` and `y`.
pub fn create_sigmoid_mask(
    func: impl Fn(f64) -> f64,
    x_scale: f64,
    y_scale: f64,
) -> impl Fn(f64, f64) -> f64 {
    move |x: f64, y: f64| {
        let xs = x / x_scale;
        let ys = y / y_scale;
        -func(-(xs.mul_add(xs, ys * ys) - 1.0)) + 1.0
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

/// Represents the spiral winding direction.
#[derive(Debug, Clone, Copy)]
#[repr(i8)]
pub enum Winding {
    /// Positive winding.
    Positive = 1,
    /// Negative winding.
    Negative = -1,
}

impl fmt::Display for Winding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", *self as i8)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct WindingConversionError;

impl TryFrom<i8> for Winding {
    type Error = WindingConversionError;

    fn try_from(value: i8) -> Result<Self, Self::Error> {
        match value {
            -1 => Ok(Self::Negative),
            1 => Ok(Self::Positive),
            _ => Err(WindingConversionError),
        }
    }
}

impl From<Winding> for f64 {
    fn from(value: Winding) -> Self {
        f64::from(value as i8)
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
    pub b_winding: f64,
    /// The quadratic winding parameter.
    pub c_winding: f64,
    /// The angle offset.
    pub theta0: f64,
    /// The scale factor relating `vz` and `z`.
    pub scale_factor: f64,
    /// The flattening radius.
    pub rho: f64,
    /// The winding direction.
    pub winding: Winding,
    /// The flattening scale.
    pub flattening_strength: f64,
}

impl PSpiralComponent {
    /// Calculate the log spiral's phase defined implicitly as
    ///
    /// `r = b * phi + c * phi^2`
    #[inline]
    #[must_use]
    pub fn spiral_phase(&self, radius: f64) -> f64 {
        let abs_c = self.c_winding.abs();
        let abs_b = self.b_winding.abs();
        if abs_c > 1e-10 {
            let half_b_over_c = 0.5 * abs_b / abs_c;
            -half_b_over_c + half_b_over_c.mul_add(half_b_over_c, radius / abs_c).sqrt()
        } else {
            radius / abs_b
        }
    }

    /// Calculate the log spiral's phase defined implicitly as
    ///
    /// `r = b * phi + c * phi^2`
    ///
    /// for `wide`'s `f64x4` registers.
    #[inline]
    #[must_use]
    pub fn spiral_phase_wide(&self, radius: f64x4) -> f64x4 {
        let abs_c = self.c_winding.abs();
        let abs_b = self.b_winding.abs();

        // Quadratic branch: -b/(2c) + sqrt((b/(2c))^2 + r/c)
        let half_b_over_c = f64x4::splat(0.5 * abs_b / abs_c);
        let quadratic =
            (half_b_over_c * half_b_over_c + radius / f64x4::splat(abs_c)).sqrt() - half_b_over_c;

        // Linear branch: r / b
        let linear = radius / f64x4::splat(abs_b);

        // All lanes take the same branch since abs_c is scalar — mask is all-ones or all-zeros
        let mask = f64x4::splat(if abs_c > 1e-10 {
            f64::from_bits(u64::MAX)
        } else {
            0.0
        });
        mask.blend(quadratic, linear)
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

    /// Calculate the perturbation coefficient at a point `(z, vz)` implemented for `wide`'s `f64x4` registers.
    #[inline]
    #[must_use]
    pub fn perturbation_wide(&self, z: f64x4, vz: f64x4) -> f64x4 {
        const ONE: f64x4 = f64x4::splat(1.0);
        let winding = f64x4::splat(f64::from(self.winding));
        let scale_factor = self.scale_factor;

        let scaled_vz = vz / scale_factor;
        let radius = (z * z + scaled_vz * scaled_vz).sqrt();
        let theta = arctan2_wide(z, scaled_vz);

        let phase = self.spiral_phase_wide(radius);
        let flattening = expit_wide((radius - self.rho) / self.flattening_strength);
        let geometric_term = theta.mul_add(winding, -phase - self.theta0).cos();

        // 1 + alpha * flattening * cos(theta - phi_s - theta0)
        (self.alpha * flattening).mul_add(geometric_term, ONE)
    }
}
