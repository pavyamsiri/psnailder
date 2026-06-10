use psnailder_core::PSpiralComponent;
use psnailder_core::usize_to_f64;
use psnailder_mock::{
    BackgroundComponent, GaussianComponent, MockGridResult, MockModel, SignalComponent,
};

fn linspace(start: f64, end: f64, n: usize) -> Vec<f64> {
    let nf = usize_to_f64!(n, "`n` must fit inside `f64`.");
    let step = (end - start) / (nf - 1.0);
    (0..n)
        .map(|i| {
            let i_f = usize_to_f64!(i, "`i` should fit in a `u32`.");
            i_f.mul_add(step, start)
        })
        .collect()
}

fn print_grid(grid: &MockGridResult) {
    let fmt = |values: &[f64]| {
        values
            .iter()
            .map(|x| format!("{x:.6}"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    println!("import numpy as np");
    println!("import matplotlib.pyplot as plt");
    println!("num_x, num_y = {}, {}", grid.num_x, grid.num_y);
    println!(
        "density    = np.array([{}]).reshape(num_y, num_x)",
        fmt(&grid.density)
    );
    println!(
        "background = np.array([{}]).reshape(num_y, num_x)",
        fmt(&grid.background)
    );
    println!(
        "mesh_x     = np.array([{}]).reshape(num_y, num_x)",
        fmt(&grid.mesh_x)
    );
    println!(
        "mesh_y     = np.array([{}]).reshape(num_y, num_x)",
        fmt(&grid.mesh_y)
    );
    println!("fig, axes = plt.subplots(1, 2)");
    println!("axes[0].pcolormesh(mesh_x, mesh_y, density)");
    println!("axes[0].set_title('density')");
    println!("axes[1].pcolormesh(mesh_x, mesh_y, background)");
    println!("axes[1].set_title('background')");
    println!("plt.show()");
}

fn main() {
    let model = MockModel::new(
        vec![SignalComponent::LogSpiral(PSpiralComponent {
            alpha: 0.5,
            b_winding: 0.05,
            c_winding: 0.002,
            theta0: 0.0,
            scale_factor: 40.0,
            rho: 0.09,
            winding: 1,
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
    let x_bins = 1000;
    let y_bins = 1000;
    let x_edges = linspace(-1.2, 1.2, x_bins + 1);
    let y_edges = linspace(-60.0, 60.0, y_bins + 1);
    let mock_result = model.mock_grid(&x_edges, &y_edges, 1_000_000);

    print_grid(&mock_result);
}
