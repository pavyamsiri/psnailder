use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use psnailder_core::likelihood::ln_likelihood_naive;
use psnailder_core::likelihood::ln_likelihood_wide;
use rand::RngExt as _;
use rand::SeedableRng as _;
use rand::rngs::SmallRng;

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
        .map(|&current_prediction| {
            if current_prediction == 0.0 {
                0.0
            } else {
                current_prediction * rng.random_range(0.8..1.2)
            }
        })
        .collect();

    let mask: Vec<f64> = (0..n).map(|_| rng.random_range(0.5..2.0)).collect();

    (data, prediction, mask)
}

fn bench_likelihood(crit: &mut Criterion) {
    let mut group = crit.benchmark_group("ln_likelihood");

    for size in [1_000, 10_000, 100_000] {
        let (data, prediction, mask) = make_data(size, 0.05, 42);

        group.bench_with_input(BenchmarkId::new("original", size), &size, |bench, _| {
            bench.iter(|| {
                ln_likelihood_naive(black_box(&data), black_box(&prediction), black_box(&mask))
            });
        });

        group.bench_with_input(BenchmarkId::new("wide", size), &size, |bench, _| {
            bench.iter(|| {
                ln_likelihood_wide(black_box(&data), black_box(&prediction), black_box(&mask))
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_likelihood);
criterion_main!(benches);
