use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use psnailder_math::arctan2_vec;
use rand::RngExt as _;
use rand::SeedableRng as _;
use rand::rngs::SmallRng;

fn make_data(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = SmallRng::seed_from_u64(seed);

    (0..n).map(|_| rng.random_range(-1e50..1e50)).collect()
}

fn arctan2_vec_stdlib(xs: &[f64], ys: &[f64], out: &mut [f64]) {
    for ((x, y), oo) in xs.iter().zip(ys.iter()).zip(out.iter_mut()) {
        *oo = y.atan2(*x);
    }
}

fn bench_arctan2(crit: &mut Criterion) {
    let mut group = crit.benchmark_group("arctan2");

    for size in [1_000, 10_000, 100_000] {
        let x = make_data(size, 42);
        let y = make_data(size, 51);
        let mut out = vec![0.0; size];

        group.bench_with_input(BenchmarkId::new("stdlib", size), &size, |bench, _| {
            bench.iter(|| arctan2_vec_stdlib(black_box(&x), black_box(&y), black_box(&mut out)));
        });

        group.bench_with_input(BenchmarkId::new("fast", size), &size, |bench, _| {
            bench.iter(|| arctan2_vec(black_box(&x), black_box(&y), black_box(&mut out)));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_arctan2);
criterion_main!(benches);
