use argmin::{
    core::{CostFunction, Error, Executor, Problem, State},
    solver::neldermead::NelderMead,
};
use argmin_testfunctions::rosenbrock;
use core::default;
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

pub struct TikTak {
    pub num_samples: usize,
    pub num_star: usize,
    pub min_weight: f64,
    pub max_weight: f64,
}

impl TikTak {
    pub const fn new(
        log_num_samples: u8,
        keep_ratio: f32,
        min_weight: f64,
        max_weight: f64,
    ) -> TikTak {
        assert!(log_num_samples <= 16);
        assert!(log_num_samples > 0);
        assert!(keep_ratio > 0.0);
        assert!(keep_ratio <= 1.0);
        assert!(min_weight < max_weight);
        assert!(min_weight >= 0.0);
        assert!(max_weight <= 1.0);

        let num_samples = 2 << log_num_samples;
        assert!(num_samples >= 1);

        let mut num_star = (keep_ratio * num_samples as f32).ceil() as usize;
        if num_star > num_samples {
            num_star = num_samples;
        }
        if num_star < 1 {
            num_star = 1;
        }

        Self {
            num_samples,
            num_star,
            min_weight,
            max_weight,
        }
    }
}

impl default::Default for TikTak {
    fn default() -> Self {
        Self::new(10, 128.0f32.recip(), 0.1, 0.995)
    }
}

impl TikTak {
    pub fn minimize(
        &self,
        cost_func: impl CostFunction<Param = Vec<f64>, Output = f64> + Clone + fmt::Debug,
        bounds: &[(f64, f64)],
    ) -> Result<OptimizationResult, Error> {
        const SIMPLEX_STEP: f64 = 0.05;
        const SD_TOLERANCE: f64 = 1e-6;
        let ndim = bounds.len();
        let seq = sobol::Sobol::<f64>::new(ndim, &sobol::params::JoeKuoD6::minimal());
        let mut heap: BinaryHeap<OrderedPoint> = BinaryHeap::with_capacity(self.num_star + 1);

        let mut nfev: u64 = self.num_samples as u64;

        for point in seq.take(self.num_samples) {
            let scaled_point: Vec<f64> = point
                .iter()
                .zip(bounds.iter())
                .map(|(p, (lb, ub))| lb + p * (ub - lb))
                .collect();

            let cost = cost_func.cost(&scaled_point)?;
            heap.push(OrderedPoint {
                cost,
                point: scaled_point,
            });
            // If we are full we evict the current worst
            if heap.len() > self.num_star {
                heap.pop();
            }
        }

        assert_eq!(heap.len(), self.num_star);

        let best_points = heap.into_sorted_vec();

        let mut global_best_cost = f64::INFINITY;
        let mut global_best_param: Option<Vec<f64>> = None;

        for (idx, current_seed) in best_points.into_iter().enumerate() {
            let weight = (((idx + 1) as f64) / (self.num_star as f64))
                .powf(0.5)
                .clamp(self.min_weight, self.max_weight);
            let new_seed = if let Some(global_best_param) = &global_best_param {
                current_seed
                    .point
                    .iter()
                    .zip(global_best_param.iter())
                    .map(|(current_param, best_param)| {
                        (1.0 - weight) * current_param + weight * best_param
                    })
                    .collect()
            } else {
                current_seed.point.clone()
            };

            let mut vertices = vec![];
            for d in 0..ndim {
                let mut v = new_seed.clone();
                v[d] += if v[d].abs() > 1e-8 {
                    SIMPLEX_STEP * v[d].abs()
                } else {
                    SIMPLEX_STEP
                };
                vertices.push(v);
            }
            vertices.push(new_seed);

            let solver = match NelderMead::new(vertices).with_sd_tolerance(SD_TOLERANCE) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("restart {idx}: failed to build solver – {e}");
                    continue;
                }
            };

            match Executor::new(cost_func.clone(), solver)
                .configure(|state| state.max_iters(200))
                .run()
            {
                Ok(r) => {
                    if let Some(best_param) = r.state().get_best_param() {
                        let best_cost = r.state().get_best_cost();
                        if best_cost < global_best_cost {
                            global_best_cost = best_cost;
                            global_best_param = Some(best_param.clone());
                        }
                    }
                    nfev += r.problem().counts.get("cost_count").unwrap();
                }
                Err(e) => eprintln!("restart {idx} failed: {e}"),
            }
        }

        let Some(global_best_param) = global_best_param else {
            panic!("no best param?");
        };
        Ok(OptimizationResult {
            params: global_best_param,
            cost: global_best_cost,
            nfev,
        })
    }
}

#[derive(Debug, Clone)]
struct Rosenbrock;

impl CostFunction for Rosenbrock {
    type Param = Vec<f64>;
    type Output = f64;
    fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        Ok(rosenbrock(p))
    }
}

pub fn run() -> Result<(), Error> {
    let tiktak = TikTak::new(10, 128.0f32.recip(), 0.1, 0.995);
    let res = tiktak.minimize(Rosenbrock, &[(-5.0, 5.0), (-5.0, 5.0)])?;
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
