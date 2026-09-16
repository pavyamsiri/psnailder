use core::f64::consts::{FRAC_PI_2, PI};
use itertools::izip;
use wide::bytemuck;
use wide::{CmpGt as _, CmpLt as _};
use wide::{f64x4, u64x4};

/// 11-degree polynomial approximation of atan(x) minimaxed in the range [-1, 1].
#[must_use]
pub const fn arctan_bound1(x: f64) -> f64 {
    const A1: f64 = 0.999_977_121_162;
    const A3: f64 = -0.332_620_905_180;
    const A5: f64 = 0.193_528_178_987;
    const A7: f64 = -0.116_394_929_844;
    const A9: f64 = 0.052_612_044_305;
    const A11: f64 = -0.011_704_925_903;
    let z = x * x;
    let px = A11
        .mul_add(z, A9)
        .mul_add(z, A7)
        .mul_add(z, A5)
        .mul_add(z, A3)
        .mul_add(z, A1);
    x * px
}

/// 11-degree polynomial approximation of atan(x) minimaxed in the range [-1, 1] accepting a SIMD register.
fn arctan_bound1_wide(x: f64x4) -> f64x4 {
    const A1: f64x4 = f64x4::splat(0.999_977_121_162);
    const A3: f64x4 = f64x4::splat(-0.332_620_905_180);
    const A5: f64x4 = f64x4::splat(0.193_528_178_987);
    const A7: f64x4 = f64x4::splat(-0.116_394_929_844);
    const A9: f64x4 = f64x4::splat(0.052_612_044_305);
    const A11: f64x4 = f64x4::splat(-0.011_704_925_903);
    let z = x * x;
    let px = A11
        .mul_add(z, A9)
        .mul_add(z, A7)
        .mul_add(z, A5)
        .mul_add(z, A3)
        .mul_add(z, A1);
    x * px
}

/// Fast atan2 approximation.
#[inline]
#[must_use]
pub fn arctan2_scalar(x: f64, y: f64) -> f64 {
    let swap = x.abs() < y.abs();
    let input = if swap { x / y } else { y / x };

    let mut res = input.atan();
    if swap {
        if input >= 0.0 {
            res = FRAC_PI_2 - res;
        } else {
            res = -FRAC_PI_2 - res;
        }
    }

    if x == 0.0 && y == 0.0 {
        res = 0.0;
    } else if x < 0.0 {
        if y >= 0.0 {
            res += PI;
        } else {
            res += -PI;
        }
    }
    res
}

/// Fast atan2 approximation for wide SIMD registers.
#[inline]
#[must_use]
pub fn arctan2_wide(x: f64x4, y: f64x4) -> f64x4 {
    const PI_REG: f64x4 = f64x4::PI;
    const FRAC_PI_2_REG: f64x4 = f64x4::FRAC_PI_2;
    const ABS_MASK_REG: u64x4 = u64x4::splat(0x7FFF_FFFF_FFFF_FFFF);
    const SIGN_MASK_REG: u64x4 = u64x4::splat(0x8000_0000_0000_0000);

    let y_abs: f64x4 = bytemuck::cast(bytemuck::cast::<f64x4, u64x4>(y) & ABS_MASK_REG);
    let x_abs: f64x4 = bytemuck::cast(bytemuck::cast::<f64x4, u64x4>(x) & ABS_MASK_REG);
    let swap_mask = y_abs.simd_gt(x_abs);

    let atan_input = swap_mask.blend(x, y) / swap_mask.blend(y, x);
    let result = arctan_bound1_wide(atan_input);

    // sign transfer onto pi/2: OR the sign bit of atan_input into pi/2
    let pi_2_bits = bytemuck::cast::<f64x4, u64x4>(FRAC_PI_2_REG);
    let input_sign = bytemuck::cast::<f64x4, u64x4>(atan_input) & SIGN_MASK_REG;
    let pi_2_signed: f64x4 = bytemuck::cast(pi_2_bits | input_sign);
    let result = swap_mask.blend(pi_2_signed - result, result);

    // quadrant adjustment: XOR pi with y's sign, AND with x<0 mask
    let x_neg_mask: u64x4 = bytemuck::cast(x.simd_lt(f64x4::ZERO)); // all-1s where x<0
    let pi_signed: u64x4 = bytemuck::cast::<f64x4, u64x4>(PI_REG)
        ^ (bytemuck::cast::<f64x4, u64x4>(y) & SIGN_MASK_REG);
    let adjustment: f64x4 = bytemuck::cast(x_neg_mask & pi_signed);
    result + adjustment
}

/// Fast atan2 approximation over arrays of points.
///
/// # Panics
/// This function assumes that `xs`, `ys` and `out` are the same length
/// and will panic if this is not true.
pub fn arctan2_vec(xs: &[f64], ys: &[f64], out: &mut [f64]) {
    assert_eq!(xs.len(), ys.len(), "`xs` and `ys` must be the same length.");
    assert_eq!(
        xs.len(),
        out.len(),
        "`xs` and `out` must be the same length."
    );

    for ((x, y), oo) in xs.iter().zip(ys.iter()).zip(out.iter_mut()) {
        *oo = arctan2_scalar(*x, *y);
    }
}

/// Fast atan2 approximation over arrays of points implemented with SIMD via wide.
///
/// # Panics
/// This function assumes that `xs`, `ys` and `out` are the same length
/// and will panic if this is not true.
pub fn arctan2_vec_simd(xs: &[f64], ys: &[f64], out: &mut [f64]) {
    assert_eq!(xs.len(), ys.len(), "`xs` and `ys` must be the same length.");
    assert_eq!(
        xs.len(),
        out.len(),
        "`xs` and `out` must be the same length."
    );

    let (xs_chunks, xs_remainder) = xs.as_chunks::<4>();
    let (ys_chunks, ys_remainder) = ys.as_chunks::<4>();
    let (out_chunks, out_remainder) = out.as_chunks_mut::<4>();

    for (current_x, current_y, current_out) in izip!(xs_chunks, ys_chunks, out_chunks) {
        let x = f64x4::from(*current_x);
        let y = f64x4::from(*current_y);

        *current_out = arctan2_wide(x, y).to_array();
    }

    arctan2_vec(xs_remainder, ys_remainder, out_remainder);
}

#[cfg(test)]
mod tests {
    use super::arctan_bound1;
    use super::arctan2_vec;
    use super::arctan2_vec_simd;
    use proptest::prelude::*;

    const MAX_LIMIT: f64 = 1e52;

    #[test]
    fn smoke() {}

    proptest! {
        #[test]
        fn check_atan_bound1_error(x in -1.0f64..=1.0f64) {
            let expected = x.atan();
            let actual = arctan_bound1(x);
            let abs_error = (actual - expected).abs();
            assert!(
                abs_error < 1.7e-6,
                "atan({x}): expected = {expected} vs actual = {actual}, abs error = {abs_error}"
            );
        }


        #[test]
        fn check_atan2_vec_error(x in -MAX_LIMIT..=MAX_LIMIT, y in -MAX_LIMIT..=MAX_LIMIT) {
            let expected = y.atan2(x);
            let mut out = [0.0; 1];
            arctan2_vec(&[x], &[y], &mut out);
            let actual = out[0];
            let abs_error = (actual- expected).abs();
            assert!(
                abs_error < 1.7e-6,
                "atan2({y}, {x}): expected = {expected} vs actual = {actual}, abs error = {abs_error}"
            );
        }


        #[test]
        fn check_atan2_vec_simd_error(x in -MAX_LIMIT..=MAX_LIMIT, y in -MAX_LIMIT..=MAX_LIMIT) {
            let expected = y.atan2(x);
            let mut out = [0.0; 4];
            arctan2_vec_simd(&[x; 4], &[y; 4], &mut out);
            let actual = out[0];
            let abs_error = (actual- expected).abs();
            assert!(
                abs_error < 1.7e-6,
                "atan2({y}, {x}): expected = {expected} vs actual = {actual}, abs error = {abs_error}"
            );
        }
    }
}
