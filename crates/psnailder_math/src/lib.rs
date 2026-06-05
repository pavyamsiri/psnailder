/// The logistic sigmoid function defined as
///
/// `sigm(x) = 1 / (1 + exp(-x))`
#[inline]
#[must_use]
pub fn expit(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// The fast logistic sigmoid function defined as
///
/// `sigm(x) = clamp(0.2 * x + 0.5, 0.0, 1.0)`
#[inline]
#[must_use]
pub const fn expit_linear(x: f64) -> f64 {
    0.2f64.mul_add(x, 0.5).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::{expit, expit_linear};
    use assert_float_eq::assert_float_absolute_eq;

    macro_rules! write_tests {
        ($name:ident, $func:expr) => {
            mod $name {
                use super::*;
                #[test]
                fn limit_points() {
                    let pos_inf_result = $func(f64::INFINITY);
                    let neg_inf_result = $func(-f64::INFINITY);

                    assert_float_absolute_eq!(pos_inf_result, 1.0);
                    assert_float_absolute_eq!(neg_inf_result, 0.0);
                }

                #[test]
                fn zero_point() {
                    let zero_point = $func(0.0);

                    assert_float_absolute_eq!(zero_point, 0.5);
                }
            }
        };
    }

    write_tests!(sigmoid, expit);
    write_tests!(linear_sigmoid, expit_linear);
}
