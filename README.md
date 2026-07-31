# Schrödinger's Penalty

Code accompanying the paper modeling the spatial distribution of football
(soccer) penalty-kick placement as the ground state of a Schrödinger-like
equation on the goal rectangle. The kicker's target density is treated as
`\xi(x, z) = u(x, z)^2`, where `u` minimizes an effective potential that trades
off a physical "cost" term (harder-to-save shots require more precisely
placed, faster kicks) against a "keeper" term (a soft Gaussian-like penalty
for aiming too close to where the keeper is expected to cover).

Data: real penalty kicks from [StatsBomb's open data](https://github.com/statsbomb/open-data).

## Pipeline

The code runs in two phases: **data preparation** (network-dependent, run
once) and **modeling** (offline, run as needed).

### 1. Data preparation

| Script | Purpose |
|---|---|
| `01_fetch_penalties.py` | Downloads every competition/season StatsBomb has released, walks every match's event log, and extracts all penalty-kick shot events (in-game spot-kicks and shootouts). Saves raw records to `penalties_raw.json`. Requires internet access to `github.com` / `raw.githubusercontent.com`. Processes ~3,900 matches; takes several minutes. Deliberately rate-limited (small thread pool, retries with backoff) to avoid hammering GitHub's servers. |
| `02_build_data_xz.py` | Converts `penalties_raw.json` into the `(N, 2)` array of **on-target** `(x, z)` shot locations, in metres, used by every fit downstream. Converts StatsBomb's yard-based pitch coordinates to metres, and filters out shots that missed the goal frame entirely (execution errors, not chosen targets). Saves `data_xz.npy`. No network needed. |

Run these once, in order:

```bash
python 01_fetch_penalties.py     # -> penalties_raw.json
python 02_build_data_xz.py       # -> data_xz.npy
```

### 2. Modeling

| Script | Purpose |
|---|---|
| `fit_and_visualize_pdf.py` | Main fit. Solves for the ground state of `-∇²u + (1/4)V u = (1/4)κu` on the goal rectangle `[-3.66, 3.66] × [0, 2.44]` m via finite differences (Dirichlet at the posts/crossbar, Neumann at the ground), fits `(bx, bz, x0, z0)` jointly by maximum likelihood with `a` held fixed, reports parameter estimates with finite-difference-Hessian standard errors, and saves three PDF figures: fitted density (`fig_density.pdf`), lateral marginal (`fig_marginal_x.pdf`), and height marginal (`fig_marginal_z.pdf`). |
| `fit_and_visualize_pdf_entropy.py` | Maximum-entropy variant of the main fit (see [below](#maximum-entropy-comparison) for details). Same `V(x,z)`, same fixed `a=1`, same `(bx, bz, x0, z0)` fit by maximum likelihood — but the target density is derived by minimizing a combination of negative Shannon entropy and `⟨V⟩` instead of Fisher information and `⟨V⟩`, giving the closed-form Gibbs density `ξ(x,z) ∝ exp(-V(x,z))` in place of the Schrödinger ground state. Saves `fig_density_entropy.pdf`, `fig_marginal_x_entropy.pdf`, `fig_marginal_z_entropy.pdf`. |
| `compare_terms.py` | Model-comparison table. Fits three nested specifications by negative log-likelihood: (1) the full model with `a` fixed and `(bx, bz, x0, z0)` free, (2) physics-only (`bx = bz = 0`, only `a` free), (3) keeper-only (`a = 0`, `(bx, bz, x0, z0)` free). All fits enforce `a, bx, bz ≥ 0` as bounds, since these are meant to be non-negative by construction. |

All three take `data_xz.npy` as input:

```bash
python fit_and_visualize_pdf.py
python fit_and_visualize_pdf_entropy.py
python compare_terms.py
```

### Maximum-entropy comparison

`fit_and_visualize_pdf_entropy.py` is a deliberate structural twin of
`fit_and_visualize_pdf.py`, built to isolate the effect of one modeling
choice: how the target density `ξ(x,z)` is derived from the potential
`V(x,z) = a·U_phys(x,z) − bx(x−x0)² − bz(z−z0)²`.

- **Schrödinger version:** `ξ = u²`, where `u` minimizes a combination of
  Fisher information and `⟨V⟩` — the ground state of a Schrödinger-like
  equation, solved by finite differences with Dirichlet boundary
  conditions at the posts/crossbar (`u = 0` there) and a Neumann condition
  at the ground.
- **Entropy version:** `ξ` minimizes a combination of negative Shannon
  entropy and `⟨V⟩` instead of Fisher information and `⟨V⟩`. This has a
  closed-form solution, the Gibbs/Boltzmann density `ξ(x,z) ∝ exp(-V(x,z))`,
  normalized directly over the goal-rectangle grid — no PDE solve, and no
  Dirichlet condition, so the density need not vanish at the posts/crossbar.

Both are fit with `a` fixed at 1 and `(bx, bz, x0, z0)` free, so the two
`-log L` values are directly comparable (same free-parameter count, same
fixed `a`). On the project's real penalty data, the Schrödinger version
fits noticeably better than the entropy version (`-log L ≈ 3686` vs.
`4234.01`), suggesting the two-lobed placement pattern is captured more
naturally by penalizing spread via Fisher information than via entropy.

**Why `a` is fixed at 1 in both models.** `a` plays the role of an inverse
temperature in the entropy version's Gibbs form (`exp(-V/T)` with `T=1`
is exactly `exp(-V)` when `a` absorbs the `1/T` factor), so fixing `a=1`
is equivalent to fixing that temperature. We tested letting `a` float
jointly with `(bx, bz, x0, z0)` in both models and found it is not
identifiable in either:
- In the **entropy** model, freeing `a` drives `bz` to exactly 0, at which
  point `z0` has no effect on the likelihood and is left completely
  unconstrained (different starting points converge to wildly different
  `z0` at identical `-log L`).
- In the **Schrödinger** model, freeing `a` does not converge at all — the
  optimizer drifts along an unbounded ridge, with `a`, `bx`, and `bz`
  climbing together while the likelihood barely improves (well under 1%
  over the fixed-`a=1` fit).

Fixing `a=1` in both models keeps the parameter count matched, keeps the
nested-model comparisons in `compare_terms.py` interpretable, and costs
essentially nothing in fit quality.

**Numerical note.** Because `-bx(x−x0)²` and `-bz(z−z0)²` are unbounded
below away from `(x0, z0)`, and the entropy version has no boundary
condition to contain that growth, `bx`/`bz` above roughly single digits
can push nearly all of `exp(-V)`'s probability mass to the domain edges,
flattening the likelihood surface. `B_BOUNDS` is therefore set tighter in
`fit_and_visualize_pdf_entropy.py` (`(0, 15)`) than in the Schrödinger
version's `compare_terms.py` (`(0, 200)`).

## Requirements

```
numpy
scipy
pandas
matplotlib
```

No API keys are required; `01_fetch_penalties.py` uses StatsBomb's public
open-data repository directly.

## Repository structure

```
.
├── 01_fetch_penalties.py       # Stage 1: scrape StatsBomb open data (network)
├── 02_build_data_xz.py         # Stage 2: convert to on-target (x,z) array
├── fit_and_visualize_pdf.py    # main joint fit + figures (Schrödinger / Fisher-information)
├── fit_and_visualize_pdf_entropy.py  # maximum-entropy variant of the main fit
├── compare_terms.py            # nested-model comparison table
├── penalties_raw.json          # generated by stage 1
├── data_xz.npy                 # generated by stage 2
├── fig_density.pdf             # generated by fit_and_visualize_pdf.py
├── fig_marginal_x.pdf          # generated by fit_and_visualize_pdf.py
├── fig_marginal_z.pdf          # generated by fit_and_visualize_pdf.py
├── fig_density_entropy.pdf     # generated by fit_and_visualize_pdf_entropy.py
├── fig_marginal_x_entropy.pdf  # generated by fit_and_visualize_pdf_entropy.py
└── fig_marginal_z_entropy.pdf  # generated by fit_and_visualize_pdf_entropy.py
```

## Citation

If you use this code, please cite the accompanying paper. *(Add full
citation/BibTeX here once available.)*
