import numpy as np
import math
from psnailder.component import PSpiralComponent
from psnailder.model import PSpiralModel
from psnailder.likelihood_utils import ln_likelihood


def analytic_grad_single(parameters, z_mesh, vz_mesh, background, data, mask, winding=1, flattening_strength=0.1):
    # parameters: [alpha, b, c, theta0, scale_factor, rho]
    alpha, b, c, theta0, scale, rho = [float(x) for x in parameters]
    z_flat = z_mesh.ravel()
    vz_flat = vz_mesh.ravel()
    n = z_flat.size

    scaled_z = z_flat * scale
    scaled_vz = vz_flat / scale
    r = np.hypot(z_flat, scaled_vz)
    theta = np.arctan2(vz_flat, scaled_z)

    # phase and derivatives
    if c != 0.0:
        half = 0.5 * b / c
        A = half * half + r / c
        sqrtA = np.sqrt(A)
        phase = -half + sqrtA
        dhalf_db = 0.5 / c
        dA_db = 2.0 * half * dhalf_db
        dphase_db = -dhalf_db + 0.5 * dA_db / sqrtA
        dhalf_dc = -0.5 * b / (c * c)
        dA_dc = 2.0 * half * dhalf_dc - r / (c * c)
        dphase_dc = -dhalf_dc + 0.5 * dA_dc / sqrtA
        dphase_dr = 0.5 / sqrtA * (1.0 / c)
    else:
        phase = r / b
        dphase_db = -r / (b * b)
        dphase_dc = 0.0
        dphase_dr = 1.0 / b

    ex = np.exp(-(r - rho) / flattening_strength)
    F = 1.0 / (1.0 + ex)
    dF_dr = (ex / (flattening_strength * (1.0 + ex) * (1.0 + ex)))
    dF_drho = -dF_dr

    U = winding * theta - phase - theta0
    cosU = np.cos(U)
    sinU = np.sin(U)

    pert = 1.0 + alpha * F * cosU
    y = background.ravel() * pert
    rvec = data.ravel() - y
    maskf = mask.ravel()

    with np.errstate(divide='ignore', invalid='ignore'):
        dlnL_dy = 0.5 * (maskf ** 2) * rvec * (2.0 * y + rvec) / (y * y)
        dlnL_dy = np.where(np.isfinite(dlnL_dy), dlnL_dy, 0.0)

    grad = np.zeros(6, dtype=np.float64)
    # dalpha
    dalpha = F * cosU
    grad[0] = -np.sum(dlnL_dy * background.ravel() * dalpha)
    # db
    ddb = alpha * F * sinU * dphase_db
    grad[1] = -np.sum(dlnL_dy * background.ravel() * ddb)
    # dc
    ddc = alpha * F * sinU * dphase_dc
    grad[2] = -np.sum(dlnL_dy * background.ravel() * ddc)
    # dtheta0
    dtheta0 = alpha * F * sinU * 1.0
    grad[3] = -np.sum(dlnL_dy * background.ravel() * dtheta0)
    # dscale
    with np.errstate(divide='ignore', invalid='ignore'):
        dr_dscale = - (vz_flat * vz_flat) / (r * (scale ** 3))
        denom_theta = (z_flat * scale) ** 2 + vz_flat ** 2
        dtheta_dscale = - (vz_flat * z_flat) / denom_theta
        dtheta_dscale = np.where(denom_theta == 0.0, 0.0, dtheta_dscale)
        dr_dscale = np.where(r == 0.0, 0.0, dr_dscale)
    dphase_dscale = dphase_dr * dr_dscale
    dF_dscale = dF_dr * dr_dscale
    dscale = alpha * (dF_dscale * cosU + F * (-sinU) * (winding * dtheta_dscale - dphase_dscale))
    grad[4] = -np.sum(dlnL_dy * background.ravel() * dscale)
    # drho
    drho = alpha * cosU * dF_drho
    grad[5] = -np.sum(dlnL_dy * background.ravel() * drho)

    return grad


def test_analytic_jacobian_single_component():
    # Small grid
    z = np.linspace(-0.5, 0.5, 30)
    vz = np.linspace(-20.0, 20.0, 30)
    Z, VZ = np.meshgrid(z, vz)
    background = np.ones_like(Z)
    mask = np.ones_like(Z)

    true = PSpiralComponent(alpha=0.4, b=0.04, c=0.001, theta0=0.5, scale_factor=30.0, rho=0.08, winding=1)
    data = (true.perturbation(Z, VZ) * background)

    # pick test parameters near true
    params = np.array([0.42, 0.041, 0.0011, 0.52, 30.1, 0.081], dtype=np.float64)

    def objective(p):
        comp = PSpiralComponent.from_array(p, winding=1)
        model = PSpiralModel((comp,), Z, VZ, background)
        return -ln_likelihood(data, model.prediction(), mask)

    # numeric gradient (central difference)
    eps = 1e-6
    numgrad = np.zeros_like(params)
    for i in range(len(params)):
        p1 = params.copy()
        p2 = params.copy()
        p1[i] -= eps
        p2[i] += eps
        f1 = objective(p1)
        f2 = objective(p2)
        numgrad[i] = (f2 - f1) / (2 * eps)

    angrad = analytic_grad_single(params, Z, VZ, background, data, mask, winding=1)

    # compare
    np.testing.assert_allclose(angrad, numgrad, rtol=1e-3, atol=1e-5)
