use itertools::izip;

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

#[cfg(test)]
mod tests {
    use super::ln_likelihood_naive;
    use super::ln_likelihood_wide;
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
