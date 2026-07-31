"""
Schroedinger's Penalty -- Maximum-Entropy variant.

This script replaces the Fisher-information / ground-state (Schroedinger
equation) derivation of the target density with a maximum-entropy (Gibbs)
derivation, and is otherwise structured to parallel fit_and_visualize_pdf.py
as closely as possible so the two can be compared directly.

Instead of solving for u(x,z) as the ground state of a Schroedinger
equation (equivalently: the density that MINIMIZES a combination of the 
FISHER INFORMATION and ENERGY), we take xi(x,z) to be the density that
MAXIMIZES SHANNON ENTROPY and the expected potential <V>. That density 
has the classic Gibbs/Boltzmann form:

    xi(x,z) = exp(-V(x,z)) / Z,   Z = integral of exp(-V) over the goal rectangle

with the same effective potential as the Schroedinger version:

    V(x,z) = a*U_phys(x,z) - bx*(x-x0)^2 - bz*(z-z0)^2
    U_phys(x,z) = (g/2) * [ sqrt(d^2+x^2+z^2) + z ]

`a` is fixed at 1 (as in the main analysis for a fair comparison); 
(bx, bz, x0, z0) are fit jointly by maximum likelihood, using the same 
starting point and bounds as the Schroedinger version.

Note on boundary behaviour -- an intrinsic difference between the two
derivations, not a bug: the Schroedinger version imposes Dirichlet
boundary conditions (u=0, hence zero density) at the posts and crossbar,
since a wavefunction confined to a hard-walled box must vanish there. The
maximum-entropy version has no such constraint built in -- exp(-V) is
simply evaluated and normalized over the bounded goal rectangle, so the
density at the posts/crossbar need not be zero, and can even be the
*maximum* of the density if bx or bz is large enough to push mass toward
the edges. Whether this matters for the fit is an empirical question,
which is exactly why this script exists as a point of comparison.

Practical consequence of the above (found by testing this script, not
theoretical): because -bx*(x-x0)^2 grows without bound away from x0, once
bx is more than a few units it can push essentially all of exp(-V)'s mass
out to the domain edges, starving the interior (where the actual data
sit) down to the interpolator's numerical floor. Past that point every
larger bx looks equally bad to the optimizer -- the negative log-likelihood
surface goes flat rather than guiding it back down -- so bx/bz are bounded
more tightly here than in the Schroedinger version, and the starting guess
is deliberately modest. If you widen B_BOUNDS, check that the reported
Hessian eigenvalues are still sensible (not the artifact of a flat plateau)
rather than trusting the fit blindly.

Requires: data_xz.npy (see 02_build_data_xz.py)
"""

import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import RegularGridInterpolator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1. Physical constants and grid
# ----------------------------------------------------------------------
g = 9.81         # m/s^2
d = 11.0         # penalty-spot-to-goal-line distance, m
a_half = 3.66    # goal half-width, m
b_height = 2.44  # crossbar height, m
A_FIXED = 1.00   # physical-cost weight, fixed (matches the Schroedinger version)

nx, nz = 39, 26
x_full = np.linspace(-a_half, a_half, nx)
z_full = np.linspace(0, b_height, nz)
dx = x_full[1] - x_full[0]
dz = z_full[1] - z_full[0]
cellsize = dx * dz

# No interior/boundary split needed here -- there's no PDE and no Dirichlet
# condition to carve out, so we evaluate directly on the full grid.
Xg, Zg = np.meshgrid(x_full, z_full, indexing='ij')
U_phys = (g / 2.0) * (np.sqrt(d**2 + Xg**2 + Zg**2) + Zg)


def build_V(a, bx, bz, x0, z0):
    return a * U_phys - bx * (Xg - x0)**2 - bz * (Zg - z0)**2


def solve_maxent_density(V):
    """Maximum-entropy density subject to <V>: the Gibbs/Boltzmann form
    xi ~ exp(-V), normalized over the goal-rectangle grid."""
    logp = -V
    logp -= logp.max()   # numerical stability before exponentiating
    p = np.exp(logp)
    p /= p.sum() * cellsize
    return p


def nll_from_grid(grid, data_xz):
    interp = RegularGridInterpolator((x_full, z_full), grid,
                                      bounds_error=False, fill_value=1e-9)
    dens = np.clip(interp(data_xz), 1e-9, None)
    return -np.sum(np.log(dens))


# ----------------------------------------------------------------------
# 2. Joint fit: (bx, bz, x0, z0), with a fixed at A_FIXED
# ----------------------------------------------------------------------
def nll_joint(params, data_xz):
    bx, bz, x0, z0 = params
    V = build_V(A_FIXED, bx, bz, x0, z0)
    grid = solve_maxent_density(V)
    return nll_from_grid(grid, data_xz)


# Kept much tighter than the Schroedinger version's (0, 200): under the
# Gibbs/exp(-V) map, bx/bz above roughly single digits already drive the
# density to the domain edges and flatten the likelihood surface (see the
# docstring note above). Widen only if you've checked the fit doesn't land
# on that plateau.
B_BOUNDS = (0.0, 15.0)
X0_BOUNDS = (-a_half, a_half)
Z0_BOUNDS = (0.0, b_height)


def fit_and_report(data_xz, x0_start=(0.5, 0.5, 0.0, 1.05)):
    bounds = [B_BOUNDS, B_BOUNDS, X0_BOUNDS, Z0_BOUNDS]

    res = minimize(nll_joint, x0_start, args=(data_xz,), method='Nelder-Mead',
                    bounds=bounds,
                    options={'xatol': 1e-4, 'fatol': 1e-2, 'maxiter': 600, 'maxfev': 800})
    res = minimize(nll_joint, res.x, args=(data_xz,), method='Nelder-Mead',
                    bounds=bounds,
                    options={'xatol': 1e-4, 'fatol': 1e-2, 'maxiter': 300, 'maxfev': 400})
    th, f0 = res.x, res.fun

    # full (not per-parameter) finite-difference Hessian, for honest SEs
    h = np.array([0.06, 3.0, 0.02, 0.02])
    H = np.zeros((4, 4))
    for i in range(4):
        ei = np.zeros(4); ei[i] = h[i]
        H[i, i] = (nll_joint(th + ei, data_xz) - 2 * f0 + nll_joint(th - ei, data_xz)) / h[i]**2
    for i in range(4):
        for j in range(i + 1, 4):
            ei = np.zeros(4); ei[i] = h[i]
            ej = np.zeros(4); ej[j] = h[j]
            H[i, j] = H[j, i] = (
                nll_joint(th + ei + ej, data_xz) - nll_joint(th + ei - ej, data_xz)
                - nll_joint(th - ei + ej, data_xz) + nll_joint(th - ei - ej, data_xz)
            ) / (4 * h[i] * h[j])

    eigvals = np.linalg.eigvalsh(H)
    cov = np.linalg.inv(H)
    se = np.sqrt(np.abs(np.diag(cov)))

    bx, bz, x0, z0 = th
    print(f"bx = {bx:.3f} +/- {se[0]:.3f}")
    print(f"bz = {bz:.2f} +/- {se[1]:.2f}")
    print(f"x0 = {x0:.4f} +/- {se[2]:.4f} m")
    print(f"z0 = {z0:.4f} +/- {se[3]:.4f} m")
    print(f"-log L = {f0:.2f}")
    print(f"Hessian eigenvalues: {np.round(eigvals, 3)}  (near-positive => genuine local min)")
    return th, f0, se


# ----------------------------------------------------------------------
# 3. Visualization: same three-panel layout as the Schroedinger version,
#    written to separate files so both can be generated side by side
# ----------------------------------------------------------------------
def make_figure(data_xz, th,
                 out_density='fig_density_entropy.pdf',
                 out_marg_x='fig_marginal_x_entropy.pdf',
                 out_marg_z='fig_marginal_z_entropy.pdf'):
    bx, bz, x0, z0 = th
    V = build_V(A_FIXED, bx, bz, x0, z0)
    grid = solve_maxent_density(V)

    marg_x = grid.sum(axis=1) * dz
    marg_x /= marg_x.sum() * dx
    marg_z = grid.sum(axis=0) * dx
    marg_z /= marg_z.sum() * dz

    # (a) fitted density
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    im = ax.imshow(grid.T, origin='lower', extent=[-a_half, a_half, 0, b_height],
                    aspect='auto', cmap='viridis')
    ax.scatter(data_xz[:, 0], data_xz[:, 1], s=4, c='white', alpha=0.3, edgecolors='none')
    ax.set_xlabel('x, lateral offset (m)'); ax.set_ylabel('z, height (m)')
    ax.set_title(f'Fitted density (max-entropy); n={len(data_xz)} penalties')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(out_density)
    plt.close(fig)
    print(f'saved {out_density}')

    # (b) lateral marginal
    fig, ax = plt.subplots(figsize=(5, 4.6))
    ax.hist(data_xz[:, 0], bins=26, density=True, alpha=0.4, color='gray', label='data')
    ax.plot(x_full, marg_x, color='tab:red', lw=2.4, label='fitted model')
    ax.set_xlabel('x (m)'); ax.set_ylabel('probability density')
    ax.set_title('Lateral marginal (max-entropy)'); ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_marg_x)
    plt.close(fig)
    print(f'saved {out_marg_x}')

    # (c) height marginal
    fig, ax = plt.subplots(figsize=(5, 4.6))
    ax.hist(data_xz[:, 1], bins=20, density=True, alpha=0.4, color='gray', label='data')
    ax.plot(z_full, marg_z, color='tab:red', lw=2.4, label='fitted model')
    ax.set_xlabel('z (m)'); ax.set_ylabel('probability density')
    ax.set_title('Height marginal (max-entropy)'); ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_marg_z)
    plt.close(fig)
    print(f'saved {out_marg_z}')


# ----------------------------------------------------------------------
# 4. Run
# ----------------------------------------------------------------------
if __name__ == '__main__':
    data_xz = np.load('data_xz.npy')   # (N,2) array: on-target (x,z) in metres
    print(f"n = {len(data_xz)}")

    theta, nll, se = fit_and_report(data_xz)
    make_figure(data_xz, theta)
