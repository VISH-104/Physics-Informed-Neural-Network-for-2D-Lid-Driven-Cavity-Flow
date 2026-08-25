"""
=============================================================================
PINN for 2D Lid-Driven Cavity Flow (Smooth Regularized Lid)
Re = 1000  |  Domain [0,1]^2
=============================================================================

Problem formulation
-------------------
Steady, incompressible 2D Navier-Stokes in non-dimensional form on [0,1]^2:

    Continuity:   du/dx + dv/dy = 0
    x-momentum:   u*du/dx + v*du/dy + dp/dx - (1/Re)*(d2u/dx2 + d2u/dy2) = 0
    y-momentum:   u*dv/dx + v*dv/dy + dp/dy - (1/Re)*(d2v/dx2 + d2v/dy2) = 0

Boundary conditions (smooth regularized lid)
--------------------------------------------
    Top lid   (y=1):  u = u_lid(x) = 16*x^2*(1-x)^2,  v = 0
    Left wall  (x=0): u = 0,  v = 0
    Right wall (x=1): u = 0,  v = 0
    Bottom     (y=0): u = 0,  v = 0

The smooth lid u_lid = 16*x^2*(1-x)^2 is a regularized alternative to the
classical discontinuous lid (u=1). It removes the corner velocity singularity
while preserving the primary recirculating vortex structure. Peak velocity is 1
at x=0.5. This formulation is standard practice in PINN-based cavity flow
studies to avoid singularity-driven training failure.

Architecture
------------
    Layer 0:    Linear(2->H),  sin activation
    Layers 1-8: Linear(H->H), tanh activation
    Output:     Linear(H->3),  no activation  -> raw [u, v, p]
    H = 64  |  total hidden layers = 9
    Xavier normal initialization throughout.

Output transform (exact BC enforcement)
---------------------------------------
    B(x,y) = x*(1-x)*y*(1-y)              [vanishes on all 4 walls]
    u(x,y) = B*NN_u + y^4*u_lid(x)        [u=u_lid at y=1; u=0 elsewhere]
    v(x,y) = B*NN_v                       [v=0 on all walls]
    p(x,y) = NN_p - NN_p(0,0)             [gauge: p(0,0)=0]

Collocation: 128x128 = 16384 Chebyshev-Gauss-Lobatto interior points.

Training:
    Curriculum Re in [100, 400, 700, 1000] -- Adam (cosine LR)
    Followed by L-BFGS refinement at Re=1000.

References
----------
    Lagaris et al. (1998)     IEEE TNNLS
    Raissi et al. (2019)      JCP
    McFall & Mahan (2009)     IEEE TNNLS
    Wang et al. (2021)        SIAM J. Sci. Comput. (motivated mixed activations)
    Haghighat & Juanes (2021) Neurocomputing (optimizer sequence)
=============================================================================
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
import os

# ── Reproducibility ──────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ── Device ───────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Directories ──────────────────────────────────────────────
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)


# ============================================================
# Hyperparameters
# ============================================================

# Domain & Physics
RE_TARGET   = 1000.0           # target Reynolds number
N_GRID      = 128              # collocation grid per side -> 16,384 pts

# Architecture
N_HIDDEN    = 64               # neurons per hidden layer
N_LAYERS    = 9                # number of hidden layers (1 sin + 8 tanh)

# Loss weights
W_PDE       = 1.0
W_CONT      = 1.0

# Curriculum
RE_SCHEDULE         = [100.0, 400.0, 700.0, 1000.0]
ADAM_ITERS_WARMUP   = 3000
ADAM_ITERS_FINAL    = 15000
LBFGS_ITERS         = 3000
LR_ADAM             = 1e-3
PRINT_EVERY         = 500

SAVE_PATH = "pinn_ldc_re1000.pt"
RESULT_PATH = "pinn_ldc_results.npz"


# ============================================================
# Network architecture
# ============================================================

class ModifiedMLP(nn.Module):
    """
    9-hidden-layer MLP.
    Layer 0:    sin activation
    Layers 1-8: tanh activation
    Output:     no activation -> [u_raw, v_raw, p_raw]
    Xavier normal initialization.
    """
    def __init__(self, n_hidden: int = 64, n_layers: int = 9):
        super().__init__()
        assert n_layers >= 2, "Need at least 2 hidden layers"
        
        layers = [nn.Linear(2, n_hidden)]
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(n_hidden, n_hidden))
        self.hidden = nn.ModuleList(layers)
        self.out    = nn.Linear(n_hidden, 3)

        for layer in self.hidden:
            nn.init.xavier_normal_(layer.weight, gain=1.0)
            nn.init.zeros_(layer.bias)
        nn.init.xavier_normal_(self.out.weight, gain=1.0)
        nn.init.zeros_(self.out.bias)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        h = torch.sin(self.hidden[0](xy))
        for layer in self.hidden[1:]:
            h = torch.tanh(layer(h))
        return self.out(h)


class PINNModel(nn.Module):
    """
    PINN with exact boundary condition enforcement via output transform.
    Smooth lid BC: u_lid(x) = 16*x^2*(1-x)^2
    """
    def __init__(self, n_hidden: int = 64, n_layers: int = 9):
        super().__init__()
        self.net = ModifiedMLP(n_hidden, n_layers)

    def _lid_velocity(self, x: torch.Tensor) -> torch.Tensor:
        return 16.0 * x**2 * (1.0 - x)**2

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        xy   = torch.cat([x, y], dim=1)
        raw  = self.net(xy)
        u_r  = raw[:, 0:1]
        v_r  = raw[:, 1:2]
        p_r  = raw[:, 2:3]

        # Boundary-vanishing mask
        B = x * (1.0 - x) * y * (1.0 - y)

        # Exact BC enforcement
        u = B * u_r  +  y**4 * self._lid_velocity(x)
        v = B * v_r

        # Pressure gauge: subtract value at (0,0)
        xy_00 = torch.zeros(1, 2, device=x.device)
        p0    = self.net(xy_00)[:, 2:3]
        p     = p_r - p0

        return u, v, p


# ============================================================
# Collocation point sampling
# ============================================================

def make_interior_points(n_side: int, device: torch.device):
    """Chebyshev-Gauss-Lobatto grid on [0,1]^2."""
    k  = torch.arange(n_side, dtype=torch.float64, device=device)
    xi = 0.5 * (1.0 - torch.cos(np.pi * k / (n_side - 1)))
    X, Y = torch.meshgrid(xi, xi, indexing="xy")
    x = X.reshape(-1, 1).float().requires_grad_(True)
    y = Y.reshape(-1, 1).float().requires_grad_(True)
    return x, y


# ============================================================
# PDE residuals
# ============================================================

def compute_gradients(f: torch.Tensor, xs: torch.Tensor):
    """Compute df/dx via autograd."""
    return torch.autograd.grad(
        f, xs,
        grad_outputs=torch.ones_like(f),
        create_graph=True,
        retain_graph=True,
    )[0]

def pde_residuals(model: PINNModel, x: torch.Tensor, y: torch.Tensor, Re: float):
    u, v, p = model(x, y)

    u_x = compute_gradients(u, x)
    u_y = compute_gradients(u, y)
    v_x = compute_gradients(v, x)
    v_y = compute_gradients(v, y)
    p_x = compute_gradients(p, x)
    p_y = compute_gradients(p, y)

    u_xx = compute_gradients(u_x, x)
    u_yy = compute_gradients(u_y, y)
    v_xx = compute_gradients(v_x, x)
    v_yy = compute_gradients(v_y, y)

    nu = 1.0 / Re
    R_cont  = u_x + v_y
    R_mom_u = u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
    R_mom_v = u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)

    return R_cont, R_mom_u, R_mom_v

def total_loss(model: PINNModel, x: torch.Tensor, y: torch.Tensor, Re: float):
    R_cont, R_mom_u, R_mom_v = pde_residuals(model, x, y, Re)
    loss_cont  = (R_cont  ** 2).mean()
    loss_mom_u = (R_mom_u ** 2).mean()
    loss_mom_v = (R_mom_v ** 2).mean()
    loss = W_CONT * loss_cont + W_PDE * loss_mom_u + W_PDE * loss_mom_v
    return loss, loss_cont, loss_mom_u, loss_mom_v


# ============================================================
# Training
# ============================================================

def train(model: PINNModel, save_path: str = SAVE_PATH):
    model.to(device)
    x_int, y_int = make_interior_points(N_GRID, device)

    history = []
    t_start = time.time()

    for stage_idx, Re in enumerate(RE_SCHEDULE):
        is_final = (stage_idx == len(RE_SCHEDULE) - 1)
        n_iters  = ADAM_ITERS_FINAL if is_final else ADAM_ITERS_WARMUP

        print(f"\n{'='*60}")
        print(f"  Stage {stage_idx+1}/{len(RE_SCHEDULE)}:  Re = {Re:.0f}  ({n_iters} Adam iters)")
        print(f"{'='*60}")

        optimizer = torch.optim.Adam(model.parameters(), lr=LR_ADAM)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_iters, eta_min=1e-5)

        for it in range(1, n_iters + 1):
            optimizer.zero_grad()
            loss, lc, lu, lv = total_loss(model, x_int, y_int, Re)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            if it % PRINT_EVERY == 0 or it == 1:
                elapsed = time.time() - t_start
                lr_now  = scheduler.get_last_lr()[0]
                print(f"  [Re={Re:6.0f}  Adam {it:6d}/{n_iters}]  loss={loss.item():.3e}  "
                      f"cont={lc.item():.3e}  mom_u={lu.item():.3e}  mom_v={lv.item():.3e}  ({elapsed:.0f}s)")
                history.append((Re, it, loss.item(), lc.item(), lu.item(), lv.item()))

        torch.save({"model": model.state_dict(), "stage": stage_idx, "Re": Re, "history": history}, save_path)
        print(f"  -> Checkpoint saved: {save_path}")

    print(f"\n{'='*60}")
    print(f"  L-BFGS refinement at Re = {RE_SCHEDULE[-1]:.0f}  (max {LBFGS_ITERS} iters)")
    print(f"{'='*60}")

    lbfgs = torch.optim.LBFGS(
        model.parameters(), lr=1.0, max_iter=LBFGS_ITERS, max_eval=LBFGS_ITERS * 2,
        history_size=100, line_search_fn="strong_wolfe", tolerance_grad=1e-9, tolerance_change=1e-12
    )
    lbfgs_counter = [0]

    def closure():
        lbfgs.zero_grad()
        loss, lc, lu, lv = total_loss(model, x_int, y_int, RE_SCHEDULE[-1])
        loss.backward()
        lbfgs_counter[0] += 1
        if lbfgs_counter[0] % 100 == 0:
            print(f"  [L-BFGS {lbfgs_counter[0]:4d}]  loss={loss.item():.3e}  "
                  f"cont={lc.item():.3e}  mom_u={lu.item():.3e}  mom_v={lv.item():.3e}")
            history.append((RE_SCHEDULE[-1], lbfgs_counter[0], loss.item(), lc.item(), lu.item(), lv.item()))
        return loss

    lbfgs.step(closure)

    torch.save({"model": model.state_dict(), "stage": "lbfgs_final", "Re": RE_SCHEDULE[-1], "history": history}, save_path)
    print(f"\nFinal model saved: {save_path}")
    print(f"Total training time: {(time.time() - t_start)/60:.1f} min")
    return history


# ============================================================
# Field evaluation
# ============================================================

def evaluate_field(model: PINNModel, n: int = N_GRID):
    """Evaluate fields. Note: requires_grad=True for vorticity computation."""
    model.eval()
    xs = torch.linspace(0, 1, n, device=device)
    ys = torch.linspace(0, 1, n, device=device)
    Xg, Yg = torch.meshgrid(xs, ys, indexing="xy")
    
    x_g = Xg.reshape(-1, 1).requires_grad_(True)
    y_g = Yg.reshape(-1, 1).requires_grad_(True)

    u, v, p = model(x_g, y_g)

    # Vorticity
    v_x = torch.autograd.grad(v, x_g, grad_outputs=torch.ones_like(v), create_graph=False, retain_graph=True)[0]
    u_y = torch.autograd.grad(u, y_g, grad_outputs=torch.ones_like(u), create_graph=False, retain_graph=False)[0]
    w = (v_x - u_y)

    with torch.no_grad():
        U = u.reshape(n, n).cpu().numpy()
        V = v.reshape(n, n).cpu().numpy()
        P = p.reshape(n, n).cpu().numpy()
        W = w.reshape(n, n).cpu().numpy()
        X = Xg.cpu().numpy()
        Y = Yg.cpu().numpy()

    model.train()
    return X, Y, U, V, P, W


def check_boundary_conditions(model: PINNModel, n_check: int = 128):
    """Verify output-transform BCs are satisfied to machine precision."""
    model.eval()
    t = torch.linspace(0, 1, n_check, device=device).view(-1, 1)
    o = torch.ones_like(t); z = torch.zeros_like(t)
    with torch.no_grad():
        u_lid_ref  = 16.0 * t**2 * (1.0 - t)**2
        u_lid_p, v_lid, _ = model(t, o)
        u_bot, v_bot, _   = model(t, z)
        u_left, v_left, _ = model(z, t)
        u_right,v_right,_ = model(o, t)
    print("\n" + "="*52)
    print("  BC Verification (max |error|)")
    print("="*52)
    print(f"  Lid   |u - u_lid| max : {(u_lid_p - u_lid_ref).abs().max().item():.2e}")
    print(f"  Lid   |v|         max : {v_lid.abs().max().item():.2e}")
    print(f"  Bottom|u|         max : {u_bot.abs().max().item():.2e}")
    print(f"  Bottom|v|         max : {v_bot.abs().max().item():.2e}")
    print(f"  Left  |u|         max : {u_left.abs().max().item():.2e}")
    print(f"  Left  |v|         max : {v_left.abs().max().item():.2e}")
    print(f"  Right |u|         max : {u_right.abs().max().item():.2e}")
    print(f"  Right |v|         max : {v_right.abs().max().item():.2e}")
    print("="*52)
    model.train()


# ============================================================
# Plotting
# ============================================================

def plot_loss_history(history):
    arr  = np.array(history)
    iters = np.arange(len(arr))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.semilogy(iters, arr[:,2], 'k-',  lw=1.5, label='Total loss')
    ax.semilogy(iters, arr[:,3], 'b--', lw=1.2, label='Continuity')
    ax.semilogy(iters, arr[:,4], 'r:',  lw=1.2, label='Momentum u')
    ax.semilogy(iters, arr[:,5], 'g:',  lw=1.2, label='Momentum v')
    ax.set_xlabel('Log iteration index'); ax.set_ylabel('MSE loss')
    ax.set_title('PINN Training History - LDC Re=1000')
    ax.legend(); ax.grid(True, which='both', ls=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "loss_history.png"), dpi=150, bbox_inches='tight')
    plt.close()


def plot_fields(model: PINNModel):
    X, Y, U, V, P, W = evaluate_field(model, n=N_GRID)
    speed = np.sqrt(U**2 + V**2)
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)
    def _cb(ax, cf, label):
        cb = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(label, fontsize=10)
    fields = [
        (U,     'RdBu_r',   'u-velocity',        'u'),
        (V,     'RdBu_r',   'v-velocity',        'v'),
        (P,     'viridis',  'Pressure',          'p'),
        (speed, 'magma',    'Speed |U|',         '|U|'),
        (W,     'RdBu_r',   'Vorticity w',       'w'),
    ]
    axs = [fig.add_subplot(gs[i//3, i%3]) for i in range(5)]
    for ax, (F, cmap, title, label) in zip(axs, fields):
        lv  = np.linspace(F.min(), F.max(), 60)
        cf  = ax.contourf(X, Y, F, levels=lv, cmap=cmap)
        ax.contour(X, Y, F, levels=10, colors='k', linewidths=0.4, alpha=0.5)
        _cb(ax, cf, label)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('x/L'); ax.set_ylabel('y/L')
        ax.set_aspect('equal')
    ax6 = fig.add_subplot(gs[1, 2])
    cf6 = ax6.contourf(X, Y, speed, levels=60, cmap='viridis')
    ax6.streamplot(X[0, :], Y[:, 0], U, V, color='white', density=2.5, linewidth=0.8, arrowsize=0.8, arrowstyle='->')
    _cb(ax6, cf6, '|U|')
    ax6.set_title('Streamlines', fontsize=11, fontweight='bold')
    ax6.set_xlabel('x/L'); ax6.set_ylabel('y/L')
    ax6.set_aspect('equal')
    plt.suptitle(f'PINN Lid-Driven Cavity (Smooth Lid) - Re = {RE_TARGET:.0f}', fontsize=14, fontweight='bold', y=1.01)
    plt.savefig(os.path.join(FIG_DIR, "fields_re1000.png"), dpi=150, bbox_inches='tight')
    plt.close()


def plot_streamlines_reference_style(model: PINNModel):
    X, Y, U, V, P, W = evaluate_field(model, n=N_GRID)
    speed = np.sqrt(U**2 + V**2)
    fig, ax = plt.subplots(figsize=(7, 7))
    cf = ax.contourf(X, Y, speed, levels=100, cmap='rainbow', alpha=0.85)
    fig.colorbar(cf, ax=ax, label='Speed |U|', fraction=0.046, pad=0.04)
    ax.streamplot(X[0, :], Y[:, 0], U, V, color=speed, cmap='Blues_r', density=4.0, linewidth=1.0, arrowsize=0.6, arrowstyle='->')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel('x/L', fontsize=12); ax.set_ylabel('y/L', fontsize=12)
    ax.set_title(f'LDC Streamlines (Smooth Lid) - Re = {RE_TARGET:.0f}', fontsize=13, fontweight='bold')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "streamlines_re1000.png"), dpi=200, bbox_inches='tight')
    plt.close()


def plot_vorticity_contours(model: PINNModel):
    X, Y, U, V, P, W = evaluate_field(model, n=N_GRID)
    fig, ax = plt.subplots(figsize=(7, 7))
    pos_levels = np.array([0.5, 1.0, 2.0, 3.0, 5.0])
    neg_levels = np.array([-0.5, -1.0, -2.0, -3.0, -5.0, -10.0, -20.0, -30.0, -50.0])
    cf = ax.contourf(X, Y, W, levels=80, cmap='RdBu_r')
    fig.colorbar(cf, ax=ax, label='Vorticity w', fraction=0.046, pad=0.04)
    cs_neg = ax.contour(X, Y, W, levels=neg_levels, colors='black', linewidths=0.8, linestyles='solid')
    cs_pos = ax.contour(X, Y, W, levels=pos_levels, colors='black', linewidths=0.8, linestyles='dashed')
    ax.clabel(cs_neg, fmt='%.1f', fontsize=7)
    ax.clabel(cs_pos, fmt='%.1f', fontsize=7)
    ax.set_xlabel('x/L', fontsize=12); ax.set_ylabel('y/L', fontsize=12)
    ax.set_title(f'Vorticity Contours (Smooth Lid) - Re = {RE_TARGET:.0f}', fontsize=13, fontweight='bold')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "vorticity_re1000.png"), dpi=200, bbox_inches='tight')
    plt.close()


# ============================================================
# Save results
# ============================================================

def save_results(model: PINNModel):
    X, Y, U, V, P, W = evaluate_field(model, n=N_GRID)
    np.savez(RESULT_PATH, X=X, Y=Y, U=U, V=V, P=P, W=W)
    print(f"Results saved to {RESULT_PATH}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  PINN - 2D Lid-Driven Cavity (Smooth Lid)  Re=1000  [0,1]^2")
    print("  Architecture: ModifiedMLP  9x64  sin+tanh")
    print("  Output transform: exact BC enforcement")
    print("  Curriculum: Re in [100, 400, 700, 1000]")
    print("="*60 + "\n")

    print(f"Running on: {device}")

    model = PINNModel(n_hidden=N_HIDDEN, n_layers=N_LAYERS).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    check_boundary_conditions(model)

    history = train(model, save_path=SAVE_PATH)

    print("\nGenerating plots...")
    plot_loss_history(history)
    plot_fields(model)
    plot_streamlines_reference_style(model)
    plot_vorticity_contours(model)

    save_results(model)
    print("\nDone! All outputs saved.")
