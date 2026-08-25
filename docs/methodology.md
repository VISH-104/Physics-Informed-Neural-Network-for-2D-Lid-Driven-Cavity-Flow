# Methodology: Physics-Informed Neural Network for Lid-Driven Cavity Flow

This document details the mathematical formulation and numerical methods used in the implementation of the PINN for the regularized lid-driven cavity problem.

## 1. Governing Equations

The problem solves the steady, incompressible 2D Navier-Stokes equations in non-dimensional form on the unit square domain $\Omega = [0,1] \times [0,1]$.

**Continuity Equation:**\
$$\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0$$

**x-Momentum Equation:**\
$$u\frac{\partial u}{\partial x} + v\frac{\partial u}{\partial y} + \frac{\partial p}{\partial x} - \frac{1}{\mathrm{Re}}\left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right) = 0$$

**y-Momentum Equation:**\
$$u\frac{\partial v}{\partial x} + v\frac{\partial v}{\partial y} + \frac{\partial p}{\partial y} - \frac{1}{\mathrm{Re}}\left(\frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2}\right) = 0$$

The system is parameterized by the Reynolds number, $\mathrm{Re}$.

## 2. Boundary Conditions
The boundary conditions enforce no-slip walls on the left, right, and bottom, and a driven lid on the top. 

To avoid the mathematical singularity at the top corners $(0,1)$ and $(1,1)$ present in the classical problem (where $u$ abruptly changes from $1$ to $0$), a **smooth regularized lid velocity profile** is used:

$$ u_{\text{lid}}(x) = 16 x^2 (1-x)^2 $$

This guarantees that $u(0,1) = 0$ and $u(1,1) = 0$, while the peak velocity at $x=0.5$ remains exactly $1.0$.
*   Top wall ($y=1$): $u = u_{\text{lid}}(x)$, $v = 0$
*   Bottom wall ($y=0$): $u = 0$, $v = 0$
*   Left wall ($x=0$): $u = 0$, $v = 0$
*   Right wall ($x=1$): $u = 0$, $v = 0$

## 3. Network Architecture

A Multi-Layer Perceptron (MLP) acts as the function approximator for the solution $[u, v, p]$. The architecture is motivated by findings that deep PINNs can suffer from gradient pathologies, and utilizing mixed activation functions can help address spectral bias and improve gradient flow.

* **Input Layer:** 2 neurons for $(x, y)$.
* **Hidden Layer 0:** 64 neurons with **Sine** activation (sin).
* **Hidden Layers 1 to 8:** 64 neurons with **Hyperbolic Tangent** activation (tanh(x)).
* **Output Layer:** 3 neurons for raw predictions $[u_{\text{raw}}, v_{\text{raw}}, p_{\text{raw}}]$.
* **Initialization:** Xavier Normal initialization.

## 4. Output Transformation (Exact Boundary Enforcement)

Instead of adding boundary condition violations to the loss function as soft constraints, a trial function is constructed that inherently satisfies the boundary conditions through an output transformation based on the hard-constraint approach of Lagaris et al. (1998).

First, define a distance function $B(x,y)$ that is zero on all boundaries:

$$B(x,y) = x(1-x)y(1-y)$$

The final physical predictions $[u, v, p]$ are constructed as follows:

$$u(x,y) = B(x,y) \cdot u_{\text{raw}}(x,y) + y^4 \cdot u_{\text{lid}}(x)$$

$$v(x,y) = B(x,y) \cdot v_{\text{raw}}(x,y)$$

The term $y^4 \cdot u_{\text{lid}}(x)$ satisfies the top-lid condition when $y=1$ and decays to $0$ as $y \rightarrow 0$. Because $B(x,y)$ is zero everywhere on the boundary, $u$ and $v$ satisfy the prescribed velocity boundary conditions irrespective of the network's raw output.

## 5. Pressure Gauge

In incompressible flow without pressure boundary conditions, pressure is unique only up to an additive constant. A soft gauge constraint such as $p(0,0)=0$ is often added to the loss function. Here, the gauge is enforced structurally by evaluating the raw pressure at the origin and subtracting it from the predicted field:

$$p(x,y) = p_{\text{raw}}(x,y) - p_{\text{raw}}(0,0)$$

## 6. Automatic Differentiation and PDE Residuals

The partial derivatives required for the Navier-Stokes residuals are computed using PyTorch's automatic differentiation engine (`torch.autograd.grad`). Computing the Laplacian requires differentiating the first derivatives, so the computational graph of the first derivative is retained using `create_graph=True`.

## 7. Loss Function

The total loss is the Mean Squared Error (MSE) of the PDE residuals evaluated at a set of collocation points $(x_i, y_i)$:

$$\mathcal{L} = \frac{1}{N}\sum_{i=1}^{N}\left(R_{\text{cont}}^2 + R_{\text{mom},u}^2 + R_{\text{mom},v}^2\right)$$

where $N=16{,}384$ interior points are distributed using a Chebyshev-Gauss-Lobatto (CGL) grid, which clusters points near the boundaries to better capture boundary-layer gradients.

## 8. Curriculum Training
Directly training at $\text{Re} = 1000$ often leads the optimizer into poor local minima (e.g., trivial Stokes flow solutions). A curriculum strategy is used, sequentially increasing the Reynolds number: $\text{Re} \in \{100, 400, 700, 1000\}$.

## 9. Optimization
The training utilizes a two-stage optimization strategy:
1.  **Adam Optimizer:** Used for rapid exploration. 3,000 iterations for warmup stages, and 15,000 iterations for the final stage. The learning rate uses a Cosine Annealing schedule from $10^{-3}$ to $10^{-5}$. Gradients are clipped to a maximum norm of $1.0$ to ensure stability.
2.  **L-BFGS Optimizer:** A quasi-Newton method used for final refinement at $\text{Re} = 1000$ to polish the solution, limited to 3,000 iterations with Strong Wolfe line search.

## 10. Limitations
Because the boundary condition is formulated using a regularized lid profile, this numerical experiment does not replicate the exact physical conditions of the classic $u=1$ benchmark problem (e.g., Botella & Peyret 1998 or Ghia et al. 1982). Any centerline profile comparisons provided by this codebase should be treated as qualitative visual overlays demonstrating overall flow features rather than strict quantitative validations.
