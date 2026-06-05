use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use psnailder_math::expit;
use psnailder_math::expit_linear;
use rand::RngExt as _;
use rand::SeedableRng as _;
use rand::rngs::SmallRng;

fn make_data(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = SmallRng::seed_from_u64(seed);

    (0..n).map(|_| rng.random_range(-1e50..1e50)).collect()
}

fn expit_array(data: &[f64], func: impl Fn(f64) -> f64) -> f64 {
    data.iter().map(|val| func(*val)).sum::<f64>()
}

fn bench_expit(crit: &mut Criterion) {
    let mut group = crit.benchmark_group("expit");

    for size in [1_000, 10_000, 100_000] {
        let data = make_data(size, 42);

        group.bench_with_input(BenchmarkId::new("original", size), &size, |bench, _| {
            bench.iter(|| expit_array(black_box(&data), expit));
        });

        group.bench_with_input(BenchmarkId::new("linear", size), &size, |bench, _| {
            bench.iter(|| expit_array(black_box(&data), expit_linear));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_expit);
criterion_main!(benches);
