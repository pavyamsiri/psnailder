use psnailder_core::PSpiralComponent;
use psnailder_fit::{PSpiralFitter, PSpiralFitterND};
use psnailder_mock::{BackgroundComponent, GaussianComponent, MockModel, SignalComponent};

fn linspace(start: f64, end: f64, n: usize) -> Vec<f64> {
    let step = (end - start) / (n - 1) as f64;
    (0..n).map(|i| start + i as f64 * step).collect()
}

fn main() {
    let model = MockModel::new(
        vec![SignalComponent::LogSpiral(PSpiralComponent {
            alpha: 0.5,
            b: 0.05,
            c: 0.002,
            theta0: 0.0,
            scale_factor: 40.0,
            rho: 0.09,
            winding: -1,
            flattening_strength: 0.1,
        })],
        vec![BackgroundComponent::Gaussian(GaussianComponent {
            x_scale: 1.0,
            y_scale: 40.0,
            amplitude: 1.0,
            variance: 0.125,
            x_offset: 0.0,
            y_offset: 0.0,
        })],
    );
    let num_x_bins = 100;
    let num_y_bins = 100;
    let x_edges = linspace(-1.2, 1.2, num_x_bins + 1);
    let y_edges = linspace(-60.0, 60.0, num_y_bins + 1);
    let mock_result = model.mock_grid(&x_edges, &y_edges, 1_000_000);

    let tiktak1d = psnailder_tiktak::TikTak::<6>::new(10, 128.0f32.recip(), 0.1, 0.995);
    let tiktak2d = psnailder_tiktak::TikTak::<12>::new(10, 128.0f32.recip(), 0.1, 0.995);
    let fitter = PSpiralFitter {
        fitter1d: PSpiralFitterND {
            tiktak: tiktak1d,
            alpha_bounds: (0.0, 1.0),
            b_bounds: (0.005, 0.1),
            c_bounds: (0.0, 0.004),
            theta0_bounds: (-core::f64::consts::PI, core::f64::consts::PI),
            scale_factor_bounds: (30.0, 70.0),
            rho_bounds: (0.0, 0.18),
        },
        fitter2d: PSpiralFitterND {
            tiktak: tiktak2d,
            alpha_bounds: (0.0, 1.0),
            b_bounds: (0.005, 0.1),
            c_bounds: (0.0, 0.004),
            theta0_bounds: (-core::f64::consts::PI, core::f64::consts::PI),
            scale_factor_bounds: (30.0, 70.0),
            rho_bounds: (0.0, 0.18),
        },
    };

    let mask: Vec<f64> = mock_result
        .mesh_x
        .iter()
        .zip(mock_result.mesh_y.iter())
        .map(|(x, y)| -psnailder_core::expit(x * x + (y * y) / (40.0 * 40.0) - 1.0) + 1.0)
        .collect();

    let (density, background, filtered_mask, mesh_x, mesh_y) = {
        let mut d = vec![];
        let mut b = vec![];
        let mut m = vec![];
        let mut x = vec![];
        let mut y = vec![];
        for ((((&di, &bi), &mi), &xi), &yi) in mock_result
            .density
            .iter()
            .zip(&mock_result.background)
            .zip(&mask)
            .zip(&mock_result.mesh_x)
            .zip(&mock_result.mesh_y)
            .filter(|((((_, _), mi), _), _)| **mi >= 0.5)
        {
            d.push(di);
            b.push(bi);
            m.push(mi);
            x.push(xi);
            y.push(yi);
        }
        (d, b, m, x, y)
    };

    println!("density length = {}", mock_result.density.len());
    println!("filtered density length = {}", density.len());

    fitter.fit_spiral_with_background(&density, &background, &filtered_mask, &mesh_x, &mesh_y);
}
