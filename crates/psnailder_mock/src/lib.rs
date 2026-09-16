use psnailder_core::PSpiralComponent;
use psnailder_core::usize_to_f64;
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

/// A model of the phase spiral signal used to mock data.
pub struct MockModel {
    /// The signal components of the phase spiral i.e. the log spiral arms.
    signal: Vec<SignalComponent>,
    /// The background components.
    background: Vec<BackgroundComponent>,
}

// public
impl MockModel {
    /// Create a new model.
    #[must_use]
    pub const fn new(signal: Vec<SignalComponent>, background: Vec<BackgroundComponent>) -> Self {
        Self { signal, background }
    }

    /// Evaluate the density at the given place.
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

    /// Generate mock data in a grid at the given locations by writing to the given arrays in place.
    ///
    /// # Panics
    /// This function assumes the given arrays all have the same length and so will panic if any of the
    /// arrays differ in length.
    pub fn mock_grid_in_place(
        &self,
        xs: &[f64],
        ys: &[f64],
        out_density: &mut [f64],
        out_background: &mut [f64],
    ) {
        assert_eq!(xs.len(), ys.len(), "`xs` and `ys` must be the same length");
        assert_eq!(
            xs.len(),
            out_density.len(),
            "`xs` and `out_density` must be the same length."
        );
        assert_eq!(
            xs.len(),
            out_background.len(),
            "`xs` and `out_background` must be the same length."
        );

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

        for oo in out_density.iter_mut() {
            *oo /= norm;
        }
        for oo in out_background.iter_mut() {
            *oo /= norm;
        }
    }

    /// Generate mock data in a grid defined by the bin edges.
    ///
    /// # Panics
    /// This function assumes that the given bin edges are at least of length 2 as it
    /// would not form valid bin edges otherwise.
    #[must_use]
    pub fn mock_grid(
        &self,
        x_edges: &[f64],
        y_edges: &[f64],
        num_particles: u32,
    ) -> MockGridResult {
        assert!(x_edges.len() >= 2, "`x_edges` must be at least length 2.");
        assert!(y_edges.len() >= 2, "`y_edges` must be at least length 2.");

        let num_particles = f64::from(num_particles);

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
        for oo in density.iter_mut() {
            *oo = num_particles * *oo / norm;
        }
        for oo in background.iter_mut() {
            *oo = num_particles * *oo / norm;
        }

        MockGridResult {
            density,
            background,
            mesh_x,
            mesh_y,
            num_x,
            num_y,
        }
    }

    /// Generate mock particles given the bin edges to determine the grid to evaluate.
    ///
    /// # Panics
    /// This function assumes that the given bin edges are at least length 2 otherwise they are
    /// do not define valid bin edges.
    #[must_use]
    pub fn mock_particles(
        &self,
        x_edges: &[f64],
        y_edges: &[f64],
        num_samples: usize,
        rng: impl rand::Rng,
    ) -> MockParticlesResult {
        assert!(x_edges.len() >= 2, "`x_edges` must be at least length 2.");
        assert!(y_edges.len() >= 2, "`y_edges` must be at least length 2.");

        let num_x = x_edges.len() - 1;
        let num_y = y_edges.len() - 1;
        let num_cells = num_x * num_y;

        let mut density = vec![0.0; num_cells];
        let mut background = vec![0.0; num_cells];
        let mut cdf = vec![0.0; num_cells];
        let mut mesh_x = vec![0.0; num_cells];
        let mut mesh_y = vec![0.0; num_cells];
        let mut particle_x = vec![0.0; num_samples];
        let mut particle_y = vec![0.0; num_samples];
        self.mock_particles_in_place(
            x_edges,
            y_edges,
            MockParticlesInPlaceBuffer {
                x: &mut particle_x,
                y: &mut particle_y,
                density: &mut density,
                background: &mut background,
                mesh_x: &mut mesh_x,
                mesh_y: &mut mesh_y,
                cdf: &mut cdf,
            },
            rng,
        );

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

/// An intermediate struct to store the buffers necessary to
/// generate mock particle data.
struct MockParticlesInPlaceBuffer<'out> {
    /// The particles' x position buffer; must be `num_samples` in size.
    x: &'out mut [f64],
    /// The particles' y position buffer; must be `num_samples` in size.
    y: &'out mut [f64],
    /// The density grid; must be `num_cells` in size.
    density: &'out mut [f64],
    /// The background grid; must be `num_cells` in size.
    background: &'out mut [f64],
    /// The x mesh; must be `num_cells` in size.
    mesh_x: &'out mut [f64],
    /// The y mesh; must be `num_cells` in size.
    mesh_y: &'out mut [f64],
    /// The cumulative distribution function; must be `num_cells` in size.
    cdf: &'out mut [f64],
}
// private
impl MockModel {
    fn mock_particles_in_place(
        &self,
        x_edges: &[f64],
        y_edges: &[f64],
        buffers: MockParticlesInPlaceBuffer<'_>,
        mut rng: impl rand::Rng,
    ) {
        assert!(x_edges.len() >= 2, "`x_edges` must be at least length 2.");
        assert!(y_edges.len() >= 2, "`y_edges` must be at least length 2.");

        let out_x = buffers.x;
        let out_y = buffers.y;
        let out_density = buffers.density;
        let out_background = buffers.background;
        let out_mesh_x = buffers.mesh_x;
        let out_mesh_y = buffers.mesh_y;
        let cdf = buffers.cdf;

        assert_eq!(
            out_x.len(),
            out_y.len(),
            "`out_x` and `out_y` must be the same length."
        );
        let num_samples = out_x.len();

        let num_x = x_edges.len() - 1;
        let num_y = y_edges.len() - 1;
        let num_cells = num_x * num_y;

        assert_eq!(
            out_density.len(),
            num_cells,
            "`density` must be `num_cells` in length."
        );
        assert_eq!(
            out_background.len(),
            num_cells,
            "`background` must be `num_cells` in length."
        );
        assert_eq!(cdf.len(), num_cells, "`cdf` must be `num_cells` in length.");
        assert_eq!(
            out_mesh_x.len(),
            num_cells,
            "`out_mesh_x` must be `num_cells` in length."
        );
        assert_eq!(
            out_mesh_y.len(),
            num_cells,
            "`out_mesh_y` must be `num_cells` in length."
        );

        for y_idx in 0..num_y {
            let y_cen = 0.5 * (y_edges[y_idx] + y_edges[y_idx + 1]);
            for x_idx in 0..num_x {
                let flat_idx = y_idx * num_x + x_idx;
                let x_cen = 0.5 * (x_edges[x_idx] + x_edges[x_idx + 1]);

                let (ss, bb) = self.evaluate_grid_scalar(x_cen, y_cen);

                out_density[flat_idx] = ss * bb;
                out_background[flat_idx] = bb;
                out_mesh_x[flat_idx] = x_cen;
                out_mesh_y[flat_idx] = y_cen;
            }
        }

        let norm = out_density.iter().sum::<f64>();
        for val in out_density.iter_mut() {
            *val /= norm;
        }
        for val in out_background.iter_mut() {
            *val /= norm;
        }

        let mut cumsum = 0.0;
        for (current_cdf, &current_density) in cdf.iter_mut().zip(out_density.iter()) {
            cumsum += current_density;
            *current_cdf = cumsum;
        }

        // Normalise CDF
        let cdf_norm = cdf.last().copied().expect("should not be zero length.");
        for val in cdf.iter_mut() {
            *val /= cdf_norm;
        }

        let dx = 0.5 * mean_diff(x_edges);
        let dy = 0.5 * mean_diff(y_edges);

        for out_index in 0..num_samples {
            let random_value: f64 = rng.random();
            let flat_idx = cdf
                .partition_point(|&current_cdf| current_cdf < random_value)
                .min(num_cells - 1);

            let y_idx = flat_idx / num_x;
            let x_idx = flat_idx % num_x;

            let x_cen = 0.5 * (x_edges[x_idx] + x_edges[x_idx + 1]);
            let y_cen = 0.5 * (y_edges[y_idx] + y_edges[y_idx + 1]);

            let jx: f64 = rng.random_range(-dx..=dx);
            let jy: f64 = rng.random_range(-dy..=dy);

            out_x[out_index] = x_cen + jx;
            out_y[out_index] = y_cen + jy;
        }
    }
}

#[inline]
fn mean_diff(edges: &[f64]) -> f64 {
    assert!(
        edges.len() >= 2,
        "`edges` must be at least length 2 to be valid bin edges"
    );

    let left = edges.first().expect("`edges` is at least length 2.");
    let right = edges.last().expect("`edges` is at least length 2.");

    assert!(
        left < right,
        "the leftmost edge must be smaller than the rightmost edge."
    );

    let num_bins = usize_to_f64!(
        edges.len() - 1,
        "the number of bins will always fit in an f64."
    );
    (right - left) / num_bins
}
