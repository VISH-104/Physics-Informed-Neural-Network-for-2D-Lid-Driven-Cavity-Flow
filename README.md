# Physics-Informed Neural Network for 2D Lid-Driven Cavity Flow

A PyTorch implementation of a Physics-Informed Neural Network (PINN) for steady, incompressible 2D lid-driven cavity flow at Re = 1000. The network enforces the incompressible Navier-Stokes equations as a soft constraint at interior collocation points, while satisfying boundary conditions exactly through an output transformation.

---

## Problem Formulation

### Governing equations

Non-dimensional steady incompressible Navier-Stokes on the unit square `[0,1]^2`:

* Continuity:   `du/dx + dv/dy = 0`
* x-momentum:   `u*du/dx + v*du/dy + dp/dx - (1/Re)*(d2u/dx2 + d2u/dy2) = 0`
* y-momentum:   `u*dv/dx + v*dv/dy + dp/dy - (1/Re)*(d2v/dx2 + d2v/dy2) = 0`

### Boundary conditions — smooth regularized lid

| Boundary | u | v |
|---|---|---|
| Top lid (`y=1`) | `u_lid(x) = 16x^2(1-x)^2` | 0 |
| Bottom (`y=0`) | 0 | 0 |
| Left (`x=0`) | 0 | 0 |
| Right (`x=1`) | 0 | 0 |

The smooth lid profile `u_lid = 16x^2(1-x)^2` is a regularized alternative to the classical discontinuous lid (`u=1`, used in most CFD benchmarks). It removes the velocity singularity at the top corners while preserving the primary recirculating vortex structure. Peak velocity is 1 at `x=0.5`. This regularization is standard practice in PINN-based cavity flow studies to avoid singularity-driven training failure.

> **Note on comparison with Botella & Peyret (1998):** Botella & Peyret (1998) provide a high-accuracy spectral reference solution for the *classical* lid-driven cavity with `u=1` uniformly on the top lid. Because the present implementation uses a different boundary condition, any comparison with Botella & Peyret data is **qualitative structural comparison only** — the two solve different boundary value problems. No claim of quantitative benchmark validation against Botella & Peyret is made.

---

## Network Architecture

```
Input: (x, y) ∈ [0,1]^2

Hidden layer 0:  Linear(2  -> 64),  sin activation
Hidden layer 1:  Linear(64 -> 64),  tanh activation
  ...
Hidden layer 8:  Linear(64 -> 64),  tanh activation
Output layer:    Linear(64 -> 3),   no activation  -> [u_raw, v_raw, p_raw]

Total hidden layers: 9 (1 sin + 8 tanh)
Neurons per layer:   64
Trainable parameters: ~33,600
Initialization: Xavier normal
```

The sine activation in the first layer provides spectral-like expressiveness for capturing smooth shear-dominated flow features. Tanh in subsequent layers gives smooth, bounded representations. This mixed-activation architecture is motivated by Wang et al. (2021) who demonstrated improved gradient flow in deep PINNs.

### Output transformation — exact BC enforcement

Following Lagaris et al. (1998), boundary conditions are enforced exactly through an output transformation rather than as a soft loss term:

`B(x,y) = x(1-x)y(1-y)` (vanishes on all four walls)

`u(x,y) = B * NN_u + y^4 * u_lid(x)`
`v(x,y) = B * NN_v`
`p(x,y) = NN_p - NN_p(0,0)` (pressure gauge: `p(0,0)=0`)

This eliminates the boundary-condition loss term entirely. The optimizer minimizes only the PDE residual.

---

## Training

### Collocation points

128 x 128 = **16,384** interior points on a Chebyshev-Gauss-Lobatto grid, which clusters points near the walls for better boundary-layer resolution.

### Reynolds-number curriculum

Cold-starting at Re = 1000 often produces a trivial (near-Stokes) solution. Curriculum training stages the Reynolds number:

| Stage | Re | Adam iterations | LR schedule |
|---|---|---|---|
| 1 | 100 | 3,000 | Cosine anneal 1e-3 -> 1e-5 |
| 2 | 400 | 3,000 | Cosine anneal 1e-3 -> 1e-5 |
| 3 | 700 | 3,000 | Cosine anneal 1e-3 -> 1e-5 |
| 4 | 1000 | 15,000 | Cosine anneal 1e-3 -> 1e-5 |

After Adam: L-BFGS refinement at Re = 1000 (max 3,000 iterations, strong Wolfe line search).

### Hardware Requirement

Estimated training time is ~90-120 minutes on an NVIDIA RTX 3050 Ti Laptop GPU. Using a more powerful GPU like an NVIDIA T4 or A100 is recommended for faster convergence.

---

## Output

After training, the script generates:

| File | Description |
|---|---|
| `pinn_ldc_re1000.pt` | Model checkpoint (PyTorch state dict) |
| `pinn_ldc_results.npz` | NumPy archive: full field arrays and centreline data |
| `figures/loss_history.png` | Training loss curves (total, continuity, x/y-momentum) |
| `figures/fields_re1000.png` | Six-panel flow field: u, v, p, speed, vorticity, streamlines |
| `figures/streamlines_re1000.png` | Dense streamlines coloured by speed magnitude |
| `figures/vorticity_re1000.png` | Vorticity contours |
| `figures/centerlines_re1000.png` | Centreline profiles with qualitative structural comparison |

---

## Reproducing the Results

### Requirements

```
torch >= 2.0
numpy >= 1.24
matplotlib >= 3.7
```

Install:

```bash
pip install -r requirements.txt
```

### Run full training

```bash
python pinn_ldc.py
```

### Smoke test (GPU/CPU, ~10 seconds)

Verifies imports, forward pass, autograd graph, vorticity computation, BC enforcement, and a short training run:

```bash
python smoke_test.py
```

---

## Current Status and Future Validation Work

**What this project has demonstrated:**
- Implementation of a 9x64 PINN for 2D lid-driven cavity flow.
- Exact enforcement of regularized boundary conditions via output transformations.
- Computation of PDE residuals using PyTorch automatic differentiation.
- Formulation of an end-to-end curriculum training pipeline using Adam + L-BFGS.

**Limitations and Future Work:**
- The current implementation solves a smooth regularized lid boundary condition, making the solution physically distinct from classical uniform-lid benchmarks.
- Future work involves modifying the formulation to solve the classical $u=1$ lid problem (e.g., using corner singularity subtraction or enriched collocation) in order to claim strict quantitative validation against reference solutions like Botella & Peyret (1998).

---

## References

| Reference | Role |
|---|---|
| Raissi, Perdikaris & Karniadakis (2019). *J. Comput. Phys.* 378, 686-707 | PINN framework; Adam -> L-BFGS two-stage training |
| Lagaris, Likas & Fotiadis (1998). *IEEE Trans. Neural Netw.* 9(5), 987-1000 | Output transform for exact BC enforcement |
| Wang, Teng & Perdikaris (2021). *SIAM J. Sci. Comput.* 43(5), A3055-A3081 | Gradient pathologies in PINNs; motivation for mixed activations |
| Haghighat & Juanes (2021). *Neurocomputing* 457, 166-182 | Adam -> L-BFGS for PINNs |
| Botella & Peyret (1998). *Comput. Fluids* 27(4), 421-433 | High-accuracy spectral reference (for qualitative comparison) |
