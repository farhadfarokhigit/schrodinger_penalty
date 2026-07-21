"""
Compare cost-function specifications by negative log-likelihood.

Common setup: goal rectangle Omega = [-3.66,3.66] x [0,2.44] (m),
U_phys(x,z) = (g/2)[sqrt(d^2+x^2+z^2)+z], V(x,z) = a*U_phys - bx*(x-x0)^2 - bz*(z-z0)^2.

IMPORTANT: a, bx, bz are all non-negative by construction (a is an energy
weight; bx=lambda*(1-wx)^2 and bz=lambda*(1-wz)^2 are lambda>0 times a
square). Every fit below enforces a>=0, bx>=0, bz>=0 as bounds, rather
than leaving the optimiser free to exploit sign flips (e.g. a<0 would
make U_phys anti-confining directly, mimicking bx/bz without testing
what a row actually claims to test).

Three rows:
  1. Full model, a FIXED at 1 -- (bx,bz,x0,z0) then fit jointly.
  2. Physics only -- bx=bz=0 (the keeper-modelling terms turned off
     entirely), only a is fit.
  3. Keeper only -- a=0 (the physical-cost term turned off entirely),
     (bx,bz,x0,z0) fit jointly.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import eigh
from scipy.interpolate import RegularGridInterpolator
import pandas as pd

# ---------------- shared grid / physics ----------------
g, d = 9.81, 11.0
a_half, b_height = 3.66, 2.44
nx, nz = 39, 26
x_full = np.linspace(-a_half, a_half, nx)
z_full = np.linspace(0, b_height, nz)
xi_ = x_full[1:-1]; zi_ = z_full[:-1]
dx = x_full[1]-x_full[0]; dz = z_full[1]-z_full[0]
Nx, Nz = len(xi_), len(zi_)
cellsize = dx*dz
Xg, Zg = np.meshgrid(xi_, zi_, indexing='ij')
U_phys = (g/2.0)*(np.sqrt(d**2+Xg**2+Zg**2)+Zg)

Lx = np.zeros((Nx,Nx))
for i in range(Nx):
    Lx[i,i] = -2.0/dx**2
    if i>0: Lx[i,i-1]=1.0/dx**2
    if i<Nx-1: Lx[i,i+1]=1.0/dx**2
Lz = np.zeros((Nz,Nz))
for j in range(Nz):
    Lz[j,j] = -2.0/dz**2
    if j==0: Lz[j,j+1]=2.0/dz**2
    else:
        Lz[j,j-1]=1.0/dz**2
        if j<Nz-1: Lz[j,j+1]=1.0/dz**2
Lap = np.kron(Lx, np.eye(Nz)) + np.kron(np.eye(Nx), Lz)


def full_grid(psi2):
    full = np.zeros((nx, nz))
    full[1:-1, :-1] = psi2
    return full


def build_V(a, bx, bz, x0, z0):
    return a*U_phys - bx*(Xg-x0)**2 - bz*(Zg-z0)**2


def solve_fisher_density(a, bx, bz, x0, z0):
    """Ground state of -Lap + V/4 (the stationary Schroedinger equation)."""
    V = build_V(a, bx, bz, x0, z0)
    H = -Lap + np.diag(0.25*V.flatten())
    vals, vecs = eigh(H, subset_by_index=[0, 0])
    psi2 = (vecs[:, 0]**2).reshape(Nx, Nz)
    psi2 /= psi2.sum()*cellsize
    return full_grid(psi2)


def nll_from_grid(grid, data_xz):
    interp = RegularGridInterpolator((x_full, z_full), grid,
                                      bounds_error=False, fill_value=1e-9)
    return -np.sum(np.log(np.clip(interp(data_xz), 1e-9, None)))


# ---------------- fitting helpers, all with physically-required bounds ----------------
A_BOUNDS = (0.0, 60.0)
B_BOUNDS = (0.0, 200.0)
X0_BOUNDS = (-a_half, a_half)
Z0_BOUNDS = (0.0, b_height)


def fit_full_model_a_fixed(data_xz, a_fixed=1.0, x0start=(2.06, 64.5, 0.0, 1.05)):
    """Row 1: a fixed; (bx,bz,x0,z0) jointly fit."""
    def nll(p):
        bx, bz, x0, z0 = p
        return nll_from_grid(solve_fisher_density(a_fixed, bx, bz, x0, z0), data_xz)
    bounds = [B_BOUNDS, B_BOUNDS, X0_BOUNDS, Z0_BOUNDS]
    res = minimize(nll, x0start, method='Nelder-Mead', bounds=bounds,
                    options={'xatol': 1e-4, 'fatol': 1e-2, 'maxiter': 600, 'maxfev': 800})
    return res.x, res.fun, a_fixed


def fit_physics_only(data_xz, x0start=(1.0,)):
    """Row 2: bx=bz=0 (keeper modelling off); only a is fit."""
    def nll(p):
        a, = p
        return nll_from_grid(solve_fisher_density(a, 0.0, 0.0, 0.0, 0.0), data_xz)
    res = minimize(nll, x0start, method='Nelder-Mead', bounds=[A_BOUNDS],
                    options={'xatol': 1e-4, 'fatol': 1e-2, 'maxiter': 300})
    return res.x, res.fun


def fit_keeper_only(data_xz, x0start=(2.56, 64.5, 0.0, 1.05)):
    """Row 3: a=0 (physical cost off); (bx,bz,x0,z0) jointly fit."""
    def nll(p):
        bx, bz, x0, z0 = p
        return nll_from_grid(solve_fisher_density(0.0, bx, bz, x0, z0), data_xz)
    bounds = [B_BOUNDS, B_BOUNDS, X0_BOUNDS, Z0_BOUNDS]
    res = minimize(nll, x0start, method='Nelder-Mead', bounds=bounds,
                    options={'xatol': 1e-3, 'fatol': 1e-2, 'maxiter': 600, 'maxfev': 800})
    return res.x, res.fun


# ---------------- run all three rows and assemble the table ----------------
def build_table(data_xz):
    rows = []

    th, nll, a_fixed = fit_full_model_a_fixed(data_xz)
    rows.append(dict(row='1. Full model, a FIXED at 1', a=a_fixed,
                      bx=th[0], bz=th[1], x0=th[2], z0=th[3], k=4, negLogL=nll))

    th, nll = fit_physics_only(data_xz)
    rows.append(dict(row='2. Physics only (keeper off)', a=th[0], bx=0.0, bz=0.0,
                      x0=np.nan, z0=np.nan, k=1, negLogL=nll))

    th, nll = fit_keeper_only(data_xz)
    rows.append(dict(row='3. Keeper only (physics off)', a=0.0, bx=th[0], bz=th[1],
                      x0=th[2], z0=th[3], k=4, negLogL=nll))

    df = pd.DataFrame(rows, columns=['row', 'a', 'bx', 'bz', 'x0', 'z0', 'k', 'negLogL'])
    return df


if __name__ == '__main__':
    data_xz = np.load('data_xz.npy')   # (N,2) on-target penalty (x,z) in metres
    print(f"n = {len(data_xz)}")
    table = build_table(data_xz)
    pd.set_option('display.float_format', lambda v: f'{v:.3f}')
    print(table.to_string(index=False))
