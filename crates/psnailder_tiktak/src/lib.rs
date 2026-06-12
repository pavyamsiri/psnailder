extern crate alloc;

use alloc::collections::BinaryHeap;
use argmin_testfunctions::rosenbrock;
use basin::CostFunction;
use core::{cmp, convert, fmt};
use psnailder_core::usize_to_f64;

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
    /// Initialise a `TikTak` global optimizer.
    ///
    /// # Panics
    /// This function assumes limits for the given parameters:
    /// - `log_num_samples`: can not exceed `16` as that would require too much memory and needs to be greater than `0`.
    /// - `keep_ratio`: this is a percentage and so should be between 0 and 1.
    /// - `min_weight`: the minimum weight is `0.0`.
    /// - `max_weight`: the maximum weight is `1.0`.
    #[must_use]
    pub fn new(
        log_num_samples: u8,
        keep_ratio: f32,
        min_weight: f64,
        max_weight: f64,
    ) -> TikTak<N> {
        assert!(
            log_num_samples <= 16,
            "too much memory required for more than 2^16 samples."
        );
        assert!(
            log_num_samples > 0,
            "the log of the number of samples can't be less than or equal to 0."
        );
        assert!(
            keep_ratio > 0.0,
            "the number of points to keep must be greater than 0."
        );
        assert!(
            keep_ratio <= 1.0,
            "the number of points to keep must be less than the total number of samples."
        );
        assert!(
            min_weight < max_weight,
            "the minimum weight must be less than the maximum weight."
        );
        assert!(
            min_weight >= 0.0,
            "the minimum weight must be non-negative."
        );
        assert!(max_weight <= 1.0, "the maximum weight can not exceed 1.");

        let ndim = N;

        let num_samples = 2 << log_num_samples;
        assert!(
            num_samples >= 1,
            "the number of samples must at least be 1."
        );

        let num_samples_f64 = usize_to_f64!(num_samples, "`num_samples` can't exceed 2^16.");
        let num_star = (f64::from(keep_ratio) * num_samples_f64)
            .ceil()
            .clamp(1.0, num_samples_f64) as usize;

        let points = (0..num_samples)
            .map(|i| {
                let mut point = Vec::with_capacity(ndim);
                let num_batches = ndim / 4 + 1;
                for dimension_set in 0..num_batches {
                    point.extend(
                        sobol_burley::sample_4d(i as u32, dimension_set as u32, 0)
                            .into_iter()
                            .map(f64::from),
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
    /// Minimize a cost function using the `TikTak` global optimizer.
    ///
    /// # Errors
    /// This function fails if the cost function does not succeed in at least one evaluation.
    ///
    /// # Panics
    /// This function can panic if the number of kept points is not equal to the intended number of
    /// kept points.
    pub fn minimize<C>(
        &self,
        cost_func: &C,
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
        use rayon::prelude::*;
        let mut heap: BinaryHeap<OrderedPoint> = BinaryHeap::with_capacity(self.num_star + 1);

        let evaluated_points: Vec<_> = self
            .points
            .par_iter()
            .map(|point| {
                let scaled_point: Vec<f64> = point
                    .iter()
                    .zip(bounds.iter())
                    .map(|(point, (lb, ub))| lb + point * (ub - lb))
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

        assert_eq!(
            heap.len(),
            self.num_star,
            "the heap contains the top `num_star` points."
        );

        let best_points = heap.into_sorted_vec();

        let global_best_param: Vec<f64> = best_points
            .first()
            .expect("the number of kept points is at least 1.")
            .point
            .clone();

        let mut solutions = best_points
            .into_par_iter()
            .enumerate()
            .filter_map(|(idx, current_seed)| {
                let weight = ((usize_to_f64!(idx + 1, "the indices will never exceed 2^52."))
                    / (usize_to_f64!(self.num_star, "`num_star` can never exceed 2^52.")))
                .sqrt()
                .clamp(self.min_weight, self.max_weight);

                let new_seed: Vec<f64> = current_seed
                    .point
                    .iter()
                    .zip(global_best_param.iter())
                    .map(|(current_param, best_param)| {
                        current_param.mul_add(1.0 - weight, weight * best_param)
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
                    Ok(res) => {
                        let nfev = res.cost_evals();
                        let best_param = res.best_param();
                        let best_cost = res.best_cost();
                        return Some((best_cost, best_param.to_owned(), nfev));
                    }
                    Err(err) => eprintln!("restart {idx} failed: {err}"),
                }
                None
            })
            .collect::<Vec<(f64, Vec<f64>, u64)>>();

        solutions.sort_by(|left, right| {
            left.0
                .partial_cmp(&right.0)
                .expect("there should be no NaNs.")
        });

        let nfev = solutions.iter().map(|(_, _, nfev)| nfev).sum::<u64>()
            + u64::try_from(self.num_samples)
                .expect("the number of samples will never exceed 2^16.");
        let (global_best_cost, actual_global_best_param, _) = solutions
            .first()
            .expect("the number of solutions will at least be 1.");

        Ok(OptimizationResult {
            params: actual_global_best_param.to_owned(),
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
    type Error = convert::Infallible;
    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Self::Error> {
        Ok(rosenbrock(param))
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

/// Run a minimisation test using a 2D Rosenbrock function.
///
/// # Errors
/// Minimisation can fail.
#[must_use]
pub fn run() -> Result<(), convert::Infallible> {
    let tiktak = TikTak::<2>::new(10, 128.0f32.recip(), 0.1, 0.995);
    let res = tiktak.minimize(
        &Rosenbrock {
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
