use argmin_testfunctions::rosenbrock;
use basin::CostFunction;
use core::{cmp, fmt};
use std::collections::BinaryHeap;

#[derive(Debug)]
pub struct OptimizationResult {
    pub params: Vec<f64>,
    pub cost: f64,
    pub nfev: u64,
}

struct OrderedPoint {
    cost: f64,
    point: Vec<f64>,
}

impl PartialEq for OrderedPoint {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost
    }
}

impl Eq for OrderedPoint {}

impl PartialOrd for OrderedPoint {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedPoint {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        // Max-heap by cost — so the *worst* kept point is always at the top
        self.cost
            .partial_cmp(&other.cost)
            .unwrap_or(cmp::Ordering::Equal)
    }
}

pub struct TikTak<const N: usize> {
    pub num_samples: usize,
    pub num_star: usize,
    pub min_weight: f64,
    pub max_weight: f64,

    pub points: Vec<Vec<f64>>,
}

impl<const N: usize> TikTak<N> {
    pub fn new(
        log_num_samples: u8,
        keep_ratio: f32,
        min_weight: f64,
        max_weight: f64,
    ) -> TikTak<N> {
        assert!(log_num_samples <= 16);
        assert!(log_num_samples > 0);
        assert!(keep_ratio > 0.0);
        assert!(keep_ratio <= 1.0);
        assert!(min_weight < max_weight);
        assert!(min_weight >= 0.0);
        assert!(max_weight <= 1.0);

        let ndim = N;

        let num_samples = 2 << log_num_samples;
        assert!(num_samples >= 1);

        let mut num_star = (keep_ratio * num_samples as f32).ceil() as usize;
        if num_star > num_samples {
            num_star = num_samples;
        }
        if num_star < 1 {
            num_star = 1;
        }

        let points = (0..num_samples)
            .map(|i| {
                let mut point = Vec::with_capacity(ndim);
                let num_batches = ndim / 4 + 1;
                for dimension_set in 0..num_batches {
                    point.extend(
                        sobol_burley::sample_4d(i as u32, dimension_set as u32, 0)
                            .into_iter()
                            .map(|v| v as f64),
                    );
                }
                point.truncate(ndim);
                point
            })
            .collect();

        Self {
            num_samples,
            num_star,
            min_weight,
            max_weight,
            points,
        }
    }
}

impl<const N: usize> TikTak<N> {
    pub fn minimize<C>(
        &self,
        cost_func: C,
        bounds: &[(f64, f64)],
    ) -> Result<OptimizationResult, C::Error>
    where
        C: CostFunction<Param = Vec<f64>, Output = f64>
            + Clone
            + fmt::Debug
            + Sync
            + Send
            + basin::BoxConstraints,
        C::Error: Send + fmt::Display,
    {
        let mut heap: BinaryHeap<OrderedPoint> = BinaryHeap::with_capacity(self.num_star + 1);

        use rayon::prelude::*;

        let evaluated_points: Vec<_> = self
            .points
            .par_iter()
            .map(|point| {
                let scaled_point: Vec<f64> = point
                    .iter()
                    .zip(bounds.iter())
                    .map(|(p, (lb, ub))| lb + p * (ub - lb))
                    .collect();

                let cost = cost_func.cost(&scaled_point)?;
                Ok::<_, C::Error>(OrderedPoint {
                    cost,
                    point: scaled_point,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        for ep in evaluated_points {
            heap.push(ep);
            if heap.len() > self.num_star {
                heap.pop();
            }
        }

        assert_eq!(heap.len(), self.num_star);

        let best_points = heap.into_sorted_vec();

        let global_best_param: Vec<f64> = best_points.first().unwrap().point.clone();

        let mut solutions = best_points
            .into_par_iter()
            .enumerate()
            .filter_map(|(idx, current_seed)| {
                let weight = (((idx + 1) as f64) / (self.num_star as f64))
                    .powf(0.5)
                    .clamp(self.min_weight, self.max_weight);

                let new_seed: Vec<f64> = current_seed
                    .point
                    .iter()
                    .zip(global_best_param.iter())
                    .map(|(current_param, best_param)| {
                        (1.0 - weight) * current_param + weight * best_param
                    })
                    .collect();

                match basin::Executor::new(
                    cost_func.clone(),
                    basin::NelderMead::standard().projected(),
                    basin::BasicSimplexState::new(new_seed),
                )
                .max_iter(200)
                .run()
                {
                    Ok(r) => {
                        let nfev = r.cost_evals();
                        let best_param = r.best_param();
                        let best_cost = r.best_cost();
                        return Some((best_cost, best_param.to_owned(), nfev));
                    }
                    Err(e) => eprintln!("restart {idx} failed: {e}"),
                }
                None
            })
            .collect::<Vec<(f64, Vec<f64>, u64)>>();

        solutions.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let nfev = solutions.iter().map(|(_, _, nfev)| nfev).sum::<u64>() + self.num_samples as u64;
        let (global_best_cost, global_best_param, _) = solutions.first().unwrap();

        Ok(OptimizationResult {
            params: global_best_param.to_owned(),
            cost: *global_best_cost,
            nfev,
        })
    }
}

#[derive(Debug, Clone)]
struct Rosenbrock {
    lb: Vec<f64>,
    ub: Vec<f64>,
}

impl basin::CostFunction for Rosenbrock {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = core::convert::Infallible;
    fn cost(&self, p: &Self::Param) -> Result<Self::Output, Self::Error> {
        Ok(rosenbrock(p))
    }
}

impl basin::BoxConstraints for Rosenbrock {
    fn lower(&self) -> &Self::Param {
        &self.lb
    }

    fn upper(&self) -> &Self::Param {
        &self.ub
    }
}

pub fn run() -> Result<(), core::convert::Infallible> {
    let tiktak = TikTak::<2>::new(10, 128.0f32.recip(), 0.1, 0.995);
    let res = tiktak.minimize(
        Rosenbrock {
            lb: vec![-5.0, -5.0],
            ub: vec![5.0, 5.0],
        },
        &[(-5.0, 5.0), (-5.0, 5.0)],
    )?;
    println!("{res:?}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::run;
    #[test]
    fn smoke() {
        assert!(run().is_ok());
    }
}
