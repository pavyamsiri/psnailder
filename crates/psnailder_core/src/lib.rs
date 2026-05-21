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
