pub fn ln_likelihood_f64_for(data: &[f64], prediction: &[f64], mask: &[f64]) -> f64 {
    assert_eq!(data.len(), prediction.len());
    assert_eq!(data.len(), mask.len());

    let mut result = 0.0;
    for ((current_data, current_prediction), current_mask) in
        data.iter().zip(prediction.iter()).zip(mask.iter())
    {
        if *current_prediction == 0.0 {
            continue;
        }
        let num = (current_mask * (current_data - current_prediction)).powi(2);
        result += num / current_prediction;
    }

    -0.5 * result
}

pub fn ln_likelihood_f64_iter(data: &[f64], prediction: &[f64], mask: &[f64]) -> f64 {
    assert_eq!(data.len(), prediction.len());
    assert_eq!(data.len(), mask.len());

    -0.5 * data
        .iter()
        .zip(prediction.iter())
        .zip(mask.iter())
        .map(|((d, p), m)| {
            if *p > 0.0 {
                let num = (m * (d - p)).powi(2);
                num / p
            } else {
                0.0
            }
        })
        .sum::<f64>()
}
