use psnailder_core::PSpiralComponent;
use rand::RngExt as _;

/// An interface representing components of a mock data model.
pub trait Component {
    /// Evaluate the component's contribution to either the signal or background given `x` and `y`.
    fn evaluate(&self, x: f64, y: f64) -> f64;
}

/// Types of signal component.
pub enum SignalComponent {
    /// A log spiral component.
    LogSpiral(PSpiralComponent),
    /// A custom component.
    Custom(Box<dyn Component>),
}

/// Types of background component.
pub enum BackgroundComponent {
    /// A Gaussian background component.
    Gaussian(GaussianComponent),
    /// A custom component.
    Custom(Box<dyn Component>),
}

/// A Gaussian component.
pub struct GaussianComponent {
    /// The scale of the x coordinate.
    pub x_scale: f64,
    /// The scale of the y coordinate.
    pub y_scale: f64,
    /// The amplitude.
    pub amplitude: f64,
    /// The variance.
    pub variance: f64,
    /// The centre of the Gaussian in x.
    pub x_offset: f64,
    /// The centre of the Gaussian in y.
    pub y_offset: f64,
}

impl Component for GaussianComponent {
    fn evaluate(&self, x: f64, y: f64) -> f64 {
        let xs = (x - self.x_offset) / self.x_scale;
        let ys = (y - self.y_offset) / self.y_scale;
        let rxy2 = xs.mul_add(xs, ys * ys);
        self.amplitude * (-0.5 * rxy2 / self.variance).exp()
    }
}

impl Component for PSpiralComponent {
    fn evaluate(&self, x: f64, y: f64) -> f64 {
        self.perturbation_scalar(x, y)
    }
}

impl Component for SignalComponent {
    fn evaluate(&self, x: f64, y: f64) -> f64 {
        match self {
            SignalComponent::LogSpiral(comp) => comp.evaluate(x, y),
            SignalComponent::Custom(comp) => comp.evaluate(x, y),
        }
    }
}

impl Component for BackgroundComponent {
    fn evaluate(&self, x: f64, y: f64) -> f64 {
        match self {
            BackgroundComponent::Gaussian(comp) => comp.evaluate(x, y),
            BackgroundComponent::Custom(comp) => comp.evaluate(x, y),
        }
    }
}

/// The mock data in the form of a number count grid.
pub struct MockGridResult {
    /// The density values flattened out in row-major order.
    pub density: Vec<f64>,
    /// The background values flattened out in row-major order.
    pub background: Vec<f64>,
    /// The x values flattened out in row-major order.
    pub mesh_x: Vec<f64>,
    /// The y values flattened out in row-major order.
    pub mesh_y: Vec<f64>,
    /// The number of bins in x.
    pub num_x: usize,
    /// The number of bins in y.
    pub num_y: usize,
}

/// The mock data in the form of a collection of points.
pub struct MockParticlesResult {
    /// The x coordinates of the particles.
    pub particle_x: Vec<f64>,
    /// The y coordinates of the particles.
    pub particle_y: Vec<f64>,
    /// The mock data in grid form.
    pub grid: MockGridResult,
}

pub struct MockModel {
    signal: Vec<SignalComponent>,
    background: Vec<BackgroundComponent>,
}

impl MockModel {
    pub fn new(signal: Vec<SignalComponent>, background: Vec<BackgroundComponent>) -> Self {
        Self { signal, background }
    }

    #[inline]
    fn evaluate_grid_scalar(&self, x: f64, y: f64) -> (f64, f64) {
        let mut bb = 0.0;
        for background in self.background.iter() {
            bb += background.evaluate(x, y);
        }

        let mut ss = f64::NEG_INFINITY;
        for signal in self.signal.iter() {
            ss = ss.max(signal.evaluate(x, y));
        }
        let ss = if ss.is_finite() { ss } else { 1.0 };

        (ss, bb)
    }

    pub fn mock_grid_in_place(
        &self,
        xs: &[f64],
        ys: &[f64],
        out_density: &mut [f64],
        out_background: &mut [f64],
    ) {
        assert_eq!(xs.len(), ys.len());
        assert_eq!(xs.len(), out_density.len());
        assert_eq!(xs.len(), out_background.len());

        for (x, y, ood, oob) in itertools::izip!(
            xs.iter(),
            ys.iter(),
            out_density.iter_mut(),
            out_background.iter_mut()
        ) {
            let (ss, bb) = self.evaluate_grid_scalar(*x, *y);
            *oob = bb;
            *ood = bb * ss;
        }
        let norm = out_density.iter().sum::<f64>();
        out_density.iter_mut().for_each(|oo| *oo /= norm);
        out_background.iter_mut().for_each(|oo| *oo /= norm);
    }

    pub fn mock_grid(
        &self,
        x_edges: &[f64],
        y_edges: &[f64],
        num_particles: usize,
    ) -> MockGridResult {
        assert!(x_edges.len() >= 2);
        assert!(y_edges.len() >= 2);

        let num_particles = num_particles as f64;

        let num_x = x_edges.len() - 1;
        let num_y = y_edges.len() - 1;
        let num_cells = num_x * num_y;
        let mut density = Vec::with_capacity(num_cells);
        let mut background = Vec::with_capacity(num_cells);
        let mut mesh_x = Vec::with_capacity(num_cells);
        let mut mesh_y = Vec::with_capacity(num_cells);

        for y_idx in 0..num_y {
            let y_cen = 0.5 * (y_edges[y_idx] + y_edges[y_idx + 1]);
            for x_idx in 0..num_x {
                let x_cen = 0.5 * (x_edges[x_idx] + x_edges[x_idx + 1]);

                let (ss, bb) = self.evaluate_grid_scalar(x_cen, y_cen);

                density.push(ss * bb);
                background.push(bb);
                mesh_x.push(x_cen);
                mesh_y.push(y_cen);
            }
        }

        let norm = density.iter().sum::<f64>();
        density
            .iter_mut()
            .for_each(|oo| *oo = num_particles * *oo / norm);
        background
            .iter_mut()
            .for_each(|oo| *oo = num_particles * *oo / norm);

        MockGridResult {
            density,
            background,
            mesh_x,
            mesh_y,
            num_x,
            num_y,
        }
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "not sure of the API at the moment."
    )]
    pub fn mock_particles_in_place(
        &self,
        x_edges: &[f64],
        y_edges: &[f64],
        num_samples: usize,
        out_x: &mut [f64],
        out_y: &mut [f64],
        scratch: &mut [f64],
        mut rng: impl rand::Rng,
    ) {
        assert!(x_edges.len() >= 2);
        assert!(y_edges.len() >= 2);
        assert_eq!(out_x.len(), num_samples);
        assert_eq!(out_y.len(), num_samples);

        let num_x = x_edges.len() - 1;
        let num_y = y_edges.len() - 1;
        let num_cells = num_x * num_y;

        // [density (num_cells), background (num_cells), cdf (num_cells)]
        assert!(scratch.len() >= num_cells * 3);

        let (density, rest) = scratch.split_at_mut(num_cells);
        let (background, cdf) = rest.split_at_mut(num_cells);

        assert_eq!(density.len(), num_cells);
        assert_eq!(background.len(), num_cells);
        assert_eq!(cdf.len(), num_cells);

        for y_idx in 0..num_y {
            let y_cen = 0.5 * (y_edges[y_idx] + y_edges[y_idx + 1]);
            for x_idx in 0..num_x {
                let flat_idx = y_idx * num_x + x_idx;
                let x_cen = 0.5 * (x_edges[x_idx] + x_edges[x_idx + 1]);

                let (ss, bb) = self.evaluate_grid_scalar(x_cen, y_cen);

                density[flat_idx] = ss * bb;
                background[flat_idx] = bb;
            }
        }

        let norm = density.iter().sum::<f64>();
        density.iter_mut().for_each(|val| *val /= norm);
        background.iter_mut().for_each(|val| *val /= norm);

        let mut cumsum = 0.0;
        for (c, &d) in cdf.iter_mut().zip(density.iter()) {
            cumsum += d;
            *c = cumsum;
        }

        // Normalise CDF
        let cdf_norm = cdf.last().copied().expect("should not be zero length.");
        cdf.iter_mut().for_each(|val| *val /= cdf_norm);

        let dx_half = 0.5 * mean_diff(x_edges);
        let dy_half = 0.5 * mean_diff(y_edges);

        for k in 0..num_samples {
            let u: f64 = rng.random();
            let flat_idx = cdf.partition_point(|&c| c < u).min(num_cells - 1);

            let y_idx = flat_idx / num_x;
            let x_idx = flat_idx % num_x;

            let x_cen = 0.5 * (x_edges[x_idx] + x_edges[x_idx + 1]);
            let y_cen = 0.5 * (y_edges[y_idx] + y_edges[y_idx + 1]);

            let jx: f64 = rng.random_range(-dx_half..=dx_half);
            let jy: f64 = rng.random_range(-dy_half..=dy_half);

            out_x[k] = x_cen + jx;
            out_y[k] = y_cen + jy;
        }
    }

    pub fn mock_particles(
        &self,
        x_edges: &[f64],
        y_edges: &[f64],
        num_samples: usize,
        mut rng: impl rand::Rng,
    ) -> MockParticlesResult {
        assert!(x_edges.len() >= 2);
        assert!(y_edges.len() >= 2);

        let mut particle_x = Vec::with_capacity(num_samples);
        let mut particle_y = Vec::with_capacity(num_samples);

        let num_x = x_edges.len() - 1;
        let num_y = y_edges.len() - 1;
        let num_cells = num_x * num_y;

        let mut density = Vec::with_capacity(num_cells);
        let mut background = Vec::with_capacity(num_cells);
        let mut mesh_x = Vec::with_capacity(num_cells);
        let mut mesh_y = Vec::with_capacity(num_cells);
        let mut cdf = Vec::with_capacity(num_cells);

        for y_idx in 0..num_y {
            let y_cen = 0.5 * (y_edges[y_idx] + y_edges[y_idx + 1]);
            for x_idx in 0..num_x {
                let x_cen = 0.5 * (x_edges[x_idx] + x_edges[x_idx + 1]);

                let (ss, bb) = self.evaluate_grid_scalar(x_cen, y_cen);

                density.push(ss * bb);
                background.push(bb);
                mesh_x.push(x_cen);
                mesh_y.push(y_cen);
            }
        }

        let norm = density.iter().sum::<f64>();
        density.iter_mut().for_each(|val| *val /= norm);
        background.iter_mut().for_each(|val| *val /= norm);

        let mut cumsum = 0.0;
        for d in density.iter() {
            cumsum += d;
            cdf.push(cumsum);
        }

        // Normalise CDF
        let cdf_norm = cdf.last().copied().expect("should not be zero length.");
        cdf.iter_mut().for_each(|val| *val /= cdf_norm);

        let dx_half = 0.5 * mean_diff(x_edges);
        let dy_half = 0.5 * mean_diff(y_edges);

        for _ in 0..num_samples {
            let u: f64 = rng.random();
            let flat_idx = cdf.partition_point(|&c| c < u).min(num_cells - 1);

            let y_idx = flat_idx / num_x;
            let x_idx = flat_idx % num_x;

            let x_cen = 0.5 * (x_edges[x_idx] + x_edges[x_idx + 1]);
            let y_cen = 0.5 * (y_edges[y_idx] + y_edges[y_idx + 1]);

            let jx: f64 = rng.random_range(-dx_half..=dx_half);
            let jy: f64 = rng.random_range(-dy_half..=dy_half);

            particle_x.push(x_cen + jx);
            particle_y.push(y_cen + jy);
        }

        MockParticlesResult {
            particle_x,
            particle_y,
            grid: MockGridResult {
                density,
                background,
                mesh_x,
                mesh_y,
                num_x,
                num_y,
            },
        }
    }
}

#[inline]
fn mean_diff(edges: &[f64]) -> f64 {
    (edges[edges.len() - 1] - edges[0]) / (edges.len() - 1) as f64
}
