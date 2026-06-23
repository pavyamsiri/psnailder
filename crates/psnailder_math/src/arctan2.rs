use core::f64::consts::{FRAC_PI_2, PI};

/// 11-degree polynomial approximation of atan(x) minimaxed in the range [-1, 1].
fn arctan_bound1(x: f64) -> f64 {
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
        let swap = x.abs() < y.abs();
        let input = if swap { x / y } else { y / x };

        let mut res = arctan_bound1(input);
        if swap {
            if input >= 0.0 {
                res = FRAC_PI_2 - res;
            } else {
                res = -FRAC_PI_2 - res;
            }
        }

        if *x == 0.0 && *y == 0.0 {
            res = 0.0;
        } else if *x < 0.0 {
            if *y >= 0.0 {
                res += PI;
            } else {
                res += -PI;
            }
        }

        *oo = res;
    }
}

#[cfg(test)]
mod tests {
    use super::arctan_bound1;
    use super::arctan2_vec;
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
    }
}
