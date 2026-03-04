# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.19.7",
#     "numpy>=1.24.0",
#     "matplotlib>=3.7.0",
#     "scikit-learn>=1.3.0",
#     "scipy>=1.10.0",
# ]
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App()

with app.setup:
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Loss Landscape Visualization

    _Based on a [blog post](https://kindxiaoming.github.io/blog/2026/loss-visualization-1/) by [Ziming Liu](https://kindxiaoming.github.io/)._

    Explore the **local-plane** method for visualizing 2D loss landscapes of neural
    networks, based on
    [Ziming Liu's blog post](https://kindxiaoming.github.io/blog/2026/loss-visualization-1/).

    **The idea:** Standard PCA projections of weight trajectories capture global
    structure but miss local geometry. Instead, we build local 2D coordinate systems
    at each trajectory point using **velocity** (tangent) and **acceleration**
    (curvature) directions, then blend them with Gaussian weighting to produce a
    smooth landscape.

    Use the controls below to pick a target function, network architecture, and
    visualization parameters, then watch the optimizer roll through the loss landscape.
    """)
    return


@app.cell(hide_code=True)
def _():
    config_form = mo.ui.form(
        mo.md("""
        **Target function & network**

        - {target_fn}
        - {hidden_width}
        - {n_hidden}
        - {lr}
        - {n_steps}
        - {seed}

        **Landscape parameters**

        - {k}
        - {sigma}
        - {grid_res}
        - {loss_cap}
        """).batch(
            target_fn=mo.ui.dropdown(
                options={"f(x) = x": "x", "f(x) = x²": "x2", "f(x) = x³": "x3", "f(x) = sin(x)": "sin"},
                value="f(x) = x³",
                label="Target function",
            ),
            hidden_width=mo.ui.slider(start=5, stop=50, step=5, value=10, label="Hidden layer width"),
            n_hidden=mo.ui.slider(start=1, stop=3, step=1, value=1, label="Number of hidden layers"),
            lr=mo.ui.slider(start=-4, stop=-1, step=0.5, value=-2, label=r"$\log$(learning rate)"),
            n_steps=mo.ui.slider(start=100, stop=1000, step=50, value=300, label="Training steps"),
            seed=mo.ui.number(start=0, stop=9999, value=42, label="Random seed"),
            k=mo.ui.slider(start=1, stop=30, step=1, value=10, label="k (lookback for velocity)"),
            sigma=mo.ui.slider(start=0.01, stop=1.0, step=0.01, value=0.1, label="σ (Gaussian smoothing)"),
            grid_res=mo.ui.slider(start=20, stop=80, step=5, value=40, label="Grid resolution"),
            loss_cap=mo.ui.slider(start=0.5, stop=5.0, step=0.25, value=1.5, label="Loss cap"),
        ),
        bordered=False,
        submit_button_label="Train & Visualize",
    )
    config_form
    return (config_form,)


@app.cell(hide_code=True)
def _(config_form):
    mo.stop(config_form.value is None, mo.md("_Configure parameters above and click **Train & Visualize** to start._"))

    _cfg = config_form.value
    _seed = int(_cfg["seed"])
    rng = np.random.default_rng(_seed)

    # Data
    _N = 1000
    inputs = rng.uniform(-1, 1, (_N, 1))
    _fn_map = {
        "x": lambda x: x,
        "x2": lambda x: x ** 2,
        "x3": lambda x: x ** 3,
        "sin": lambda x: np.sin(3 * x),
    }
    target_fn_key = _cfg["target_fn"]
    targets = _fn_map[target_fn_key](inputs)

    # Model
    _w = int(_cfg["hidden_width"])
    _n = int(_cfg["n_hidden"])
    widths = [1] + [_w] * _n + [1]
    params = init_mlp(widths, rng)
    _lr = 10 ** _cfg["lr"]

    n_steps = int(_cfg["n_steps"])
    trajectory, training_losses = collect_trajectory(params, widths, inputs, targets, n_steps, _lr)

    # Landscape parameters
    k_val = int(_cfg["k"])
    sigma_val = float(_cfg["sigma"])
    grid_res_val = int(_cfg["grid_res"])
    loss_cap_val = float(_cfg["loss_cap"])

    mo.md(
        f"**Trained** {n_steps} steps | arch `{widths}` | "
        f"lr={_lr:.4f} | final loss={training_losses[-1]:.6f} | "
        f"{len(trajectory[0])} params"
    )
    return (
        grid_res_val,
        inputs,
        k_val,
        loss_cap_val,
        n_steps,
        sigma_val,
        target_fn_key,
        targets,
        training_losses,
        trajectory,
        widths,
    )


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Training loss curve
    """)
    return


@app.cell(hide_code=True)
def _(target_fn_key, training_losses):
    _fn_labels = {"x": "x", "x2": "x²", "x3": "x³", "sin": "sin(3x)"}

    fig_loss, ax_loss = plt.subplots(figsize=(8, 3.5))
    ax_loss.plot(training_losses, "b-", lw=0.8)
    ax_loss.set_yscale("log")
    ax_loss.set_xlabel("Step")
    ax_loss.set_ylabel("Loss (log)")
    ax_loss.set_title(f"Training loss: f(x) = {_fn_labels[target_fn_key]}")
    fig_loss.tight_layout()
    fig_loss
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Loss landscape
    """)
    return


@app.cell(hide_code=True)
def _(
    grid_res_val,
    inputs,
    k_val,
    loss_cap_val,
    sigma_val,
    targets,
    trajectory,
    widths,
):
    with mo.status.spinner("Computing loss landscape..."):
        pc1_1d, pc2_1d, loss_grid, traj_pc, plane_indices = combined_landscape(
            trajectory, widths, inputs, targets,
            k=k_val,
            sigma=sigma_val,
            stride=5,
            grid_res=grid_res_val,
            margin=0.2,
            loss_cap=loss_cap_val,
        )
    mo.md(
        f"Landscape computed: **{loss_grid.shape[0]}×{loss_grid.shape[1]}** grid, "
        f"**{len(plane_indices)}** local planes"
    )
    return loss_grid, pc1_1d, pc2_1d, traj_pc


@app.cell(hide_code=True)
def _(loss_grid, pc1_1d, pc2_1d, target_fn_key, traj_pc):
    _fn_labels = {"x": "x", "x2": "x²", "x3": "x³", "sin": "sin(3x)"}

    fig_heat, ax_heat = plt.subplots(figsize=(8, 6))
    _im = ax_heat.imshow(
        loss_grid,
        extent=[pc1_1d[0], pc1_1d[-1], pc2_1d[0], pc2_1d[-1]],
        origin="lower",
        aspect="auto",
        cmap="viridis",
    )
    plt.colorbar(_im, ax=ax_heat, label="Blended loss")
    ax_heat.plot(traj_pc[:, 0], traj_pc[:, 1], "w-", alpha=0.5, lw=0.8)
    ax_heat.plot(traj_pc[0, 0], traj_pc[0, 1], "wo", markersize=8, label="start")
    ax_heat.plot(traj_pc[-1, 0], traj_pc[-1, 1], "r*", markersize=12, label="end")
    ax_heat.legend(loc="upper right")
    ax_heat.set_xlabel("PC1")
    ax_heat.set_ylabel("PC2")
    ax_heat.set_title(f"Loss landscape: f(x) = {_fn_labels[target_fn_key]}")
    fig_heat.tight_layout()
    fig_heat
    return


@app.cell(hide_code=True)
def _(loss_grid, pc1_1d, pc2_1d, target_fn_key, traj_pc):
    _fn_labels = {"x": "x", "x2": "x²", "x3": "x³", "sin": "sin(3x)"}

    fig_3d = plt.figure(figsize=(10, 7))
    ax_3d = fig_3d.add_subplot(111, projection="3d")
    _PC1_m, _PC2_m = np.meshgrid(pc1_1d, pc2_1d)
    ax_3d.plot_surface(_PC1_m, _PC2_m, loss_grid, cmap="viridis", alpha=0.85, edgecolor="none")
    ax_3d.plot(
        traj_pc[:, 0], traj_pc[:, 1],
        np.nanmin(loss_grid) * np.ones(len(traj_pc)),
        "r-", lw=1.5, alpha=0.9,
    )
    ax_3d.set_xlabel("PC1")
    ax_3d.set_ylabel("PC2")
    ax_3d.set_zlabel("Loss")
    ax_3d.set_title(f"Loss surface: f(x) = {_fn_labels[target_fn_key]}")
    ax_3d.view_init(elev=25, azim=135)
    fig_3d.tight_layout()
    fig_3d
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## 5. Trajectory explorer

    Drag the slider to move through training and see the optimizer's position on the landscape.
    """)
    return


@app.cell(hide_code=True)
def _(n_steps):
    step_slider = mo.ui.slider(start=0, stop=n_steps, step=1, value=0, label="Training step", full_width=True)
    step_slider
    return (step_slider,)


@app.cell(hide_code=True)
def _(loss_grid, pc1_1d, pc2_1d, step_slider, training_losses, traj_pc):
    from scipy.interpolate import RegularGridInterpolator

    _t = int(step_slider.value)
    _PC1_m, _PC2_m = np.meshgrid(pc1_1d, pc2_1d)

    # Interpolate loss at trajectory points for the 3D ball
    _fill = np.nanmean(loss_grid)
    _loss_safe = np.where(np.isfinite(loss_grid), loss_grid, _fill)
    _interp = RegularGridInterpolator((pc2_1d, pc1_1d), _loss_safe, bounds_error=False, fill_value=_fill)
    _loss_range = np.nanmax(loss_grid) - np.nanmin(loss_grid)
    _z_offset = 0.05 * _loss_range if _loss_range > 0 else 0.01

    fig_frame = plt.figure(figsize=(10, 7))
    ax_frame = fig_frame.add_subplot(111, projection="3d")
    ax_frame.plot_surface(_PC1_m, _PC2_m, loss_grid, cmap="viridis", alpha=0.8, edgecolor="none")

    # Trail up to current step (on surface)
    _trail_x = traj_pc[:_t + 1, 0]
    _trail_y = traj_pc[:_t + 1, 1]
    _trail_z = np.array([float(_interp([_trail_y[i], _trail_x[i]])[0]) + _z_offset for i in range(len(_trail_x))])
    ax_frame.plot(_trail_x, _trail_y, _trail_z, "r-", lw=2, alpha=0.9, zorder=5)

    # Current position (ball on surface)
    _bx = traj_pc[_t, 0]
    _by = traj_pc[_t, 1]
    _bz = float(_interp([_by, _bx])[0]) + _z_offset
    ax_frame.plot([_bx], [_by], [_bz], "o", color="red", markersize=9, markeredgecolor="white", markeredgewidth=1.25, zorder=10)

    ax_frame.set_xlabel("PC1")
    ax_frame.set_ylabel("PC2")
    ax_frame.set_zlabel("Loss")
    ax_frame.set_title(f"Step {_t}: loss = {training_losses[_t]:.6f}")
    ax_frame.view_init(elev=25, azim=135)
    fig_frame.tight_layout()
    fig_frame
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## How it all works

    **Problem.** Neural network loss landscapes live in thousands of dimensions.
    Projecting via global PCA onto 2 directions gives a single fixed plane that
    misses local curvature information.

    **Solution: local planes.** At each trajectory point $\mathbf{w}_t$, define a
    local 2D coordinate system:

    - **$\mathbf{e}_1$** = normalized velocity $\mathbf{v}_t = \mathbf{w}_t - \mathbf{w}_{t-k}$
    - **$\mathbf{e}_2$** = normalized acceleration component orthogonal to $\mathbf{e}_1$:
      $\mathbf{a}_t = \mathbf{v}_t - \mathbf{v}_{t-1}$, then Gram-Schmidt

    **Blending.** For each query point in the global PCA grid, evaluate the loss on
    every local plane and combine with Gaussian weights:

    $$L(\text{query}) = \frac{\sum_i g_i \cdot L_i}{\sum_i g_i}, \quad g_i = \exp\!\left(-\frac{d_i^2}{2\sigma^2}\right)$$

    where $d_i$ is the distance from the query to the $i$-th trajectory point in
    PC space.

    **Parameters:**
    - **$k$**: lookback steps for velocity/acceleration (larger = smoother directions)
    - **$\sigma$**: Gaussian scale (larger = more blending between planes)
    - **Loss cap**: clips extreme values for better color contrast

    Based on [Ziming Liu's blog](https://kindxiaoming.github.io/blog/2026/loss-visualization-1/).
    """)
    return


@app.function
def silu(x):
    """SiLU (Swish) activation."""
    return x / (1.0 + np.exp(-x))


@app.function
def silu_deriv(x):
    """Derivative of SiLU."""
    sig = 1.0 / (1.0 + np.exp(-x))
    return sig + x * sig * (1.0 - sig)


@app.function
def init_mlp(widths, rng):
    """Initialize MLP weights with Xavier uniform initialization.

    Returns a list of (W, b) tuples, one per layer.
    """
    params = []
    for i in range(len(widths) - 1):
        fan_in, fan_out = widths[i], widths[i + 1]
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        W = rng.uniform(-limit, limit, (fan_in, fan_out)).astype(np.float64)
        b = np.zeros(fan_out, dtype=np.float64)
        params.append((W, b))
    return params


@app.function
def mlp_forward(params, x):
    """Forward pass. Returns output and list of pre-activation values for backprop."""
    pre_acts = []
    h = x
    for i, (W, b) in enumerate(params):
        z = h @ W + b
        pre_acts.append((h, z))
        if i < len(params) - 1:
            h = silu(z)
        else:
            h = z  # linear output
    return h, pre_acts


@app.function
def mse_loss(params, x, y):
    """Compute MSE loss."""
    pred, _ = mlp_forward(params, x)
    return np.mean((pred - y) ** 2)


@app.function
def mlp_backward(params, x, y):
    """Backprop through the MLP. Returns gradients for each (W, b)."""
    pred, pre_acts = mlp_forward(params, x)
    N = x.shape[0]
    # d(MSE)/d(pred) = 2*(pred - y)/N
    delta = 2.0 * (pred - y) / N

    grads = []
    for i in range(len(params) - 1, -1, -1):
        h_in, z = pre_acts[i]
        dW = h_in.T @ delta
        db = delta.sum(axis=0)
        grads.append((dW, db))
        if i > 0:
            delta = (delta @ params[i][0].T) * silu_deriv(pre_acts[i - 1][1])
    grads.reverse()
    return grads


@app.function
def flatten_params(params):
    """Flatten all (W, b) pairs into a single 1D array."""
    return np.concatenate([np.concatenate([W.ravel(), b.ravel()]) for W, b in params])


@app.function
def unflatten_params(flat, widths):
    """Restore (W, b) list from a flat array."""
    params = []
    offset = 0
    for i in range(len(widths) - 1):
        fan_in, fan_out = widths[i], widths[i + 1]
        n_w = fan_in * fan_out
        W = flat[offset : offset + n_w].reshape(fan_in, fan_out)
        offset += n_w
        b = flat[offset : offset + fan_out]
        offset += fan_out
        params.append((W.copy(), b.copy()))
    return params


@app.function
def adam_step(grads, m_state, v_state, t, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
    """One step of Adam optimizer. Updates m_state, v_state in-place, returns updated params delta."""
    updates = []
    for i, (dW, db) in enumerate(grads):
        for j, g in enumerate([dW, db]):
            idx = i * 2 + j
            m_state[idx] = beta1 * m_state[idx] + (1 - beta1) * g
            v_state[idx] = beta2 * v_state[idx] + (1 - beta2) * g ** 2
            m_hat = m_state[idx] / (1 - beta1 ** t)
            v_hat = v_state[idx] / (1 - beta2 ** t)
            updates.append(-lr * m_hat / (np.sqrt(v_hat) + eps))
    return updates


@app.function
def collect_trajectory(params, widths, x, y, n_steps, lr):
    """Train with Adam and collect weight snapshots + losses."""
    # Initialize Adam state
    m_state = [np.zeros_like(p) for W, b in params for p in [W, b]]
    v_state = [np.zeros_like(p) for W, b in params for p in [W, b]]

    trajectory = [flatten_params(params).copy()]
    losses = [mse_loss(params, x, y)]

    for step in range(1, n_steps + 1):
        grads = mlp_backward(params, x, y)
        updates = adam_step(grads, m_state, v_state, step, lr=lr)

        idx = 0
        for i in range(len(params)):
            W, b = params[i]
            W += updates[idx]
            idx += 1
            b += updates[idx]
            idx += 1

        trajectory.append(flatten_params(params).copy())
        losses.append(mse_loss(params, x, y))

    return trajectory, losses


@app.function
def local_plane_axes(w_t, w_tk, w_t2k):
    """Compute orthonormal (velocity, acceleration) basis at w_t."""
    v = w_t - w_tk
    v_prev = w_tk - w_t2k
    a = v - v_prev
    e1 = v / (np.linalg.norm(v) + 1e-12)
    a_perp = a - np.dot(a, e1) * e1
    e2 = a_perp / (np.linalg.norm(a_perp) + 1e-12)
    return w_t, e1, e2


@app.function
def combined_landscape(
    trajectory, widths, x, y, k, sigma, stride, grid_res, margin, loss_cap
):
    """Build 2D loss landscape from blended local planes."""
    W = np.stack(trajectory, axis=0)
    n_total = W.shape[0]
    w_mean = W.mean(axis=0)

    pca = PCA(n_components=2)
    pca.fit(W)
    PC1 = pca.components_[0]
    PC2 = pca.components_[1]
    traj_pc = pca.transform(W)

    pc1_min, pc1_max = traj_pc[:, 0].min(), traj_pc[:, 0].max()
    pc2_min, pc2_max = traj_pc[:, 1].min(), traj_pc[:, 1].max()
    pad1 = (pc1_max - pc1_min) * margin
    pad2 = (pc2_max - pc2_min) * margin
    pc1_1d = np.linspace(pc1_min - pad1, pc1_max + pad1, grid_res)
    pc2_1d = np.linspace(pc2_min - pad2, pc2_max + pad2, grid_res)
    pc1_grid, pc2_grid = np.meshgrid(pc1_1d, pc2_1d)

    plane_indices = [t for t in range(2 * k, n_total) if (t - 2 * k) % stride == 0]
    if not plane_indices:
        plane_indices = list(range(2 * k, n_total))

    local_planes = []
    for t in plane_indices:
        center, e1, e2 = local_plane_axes(W[t], W[t - k], W[t - 2 * k])
        local_planes.append((center, e1, e2, traj_pc[t]))

    loss_grid = np.full((grid_res, grid_res), np.nan)
    for i in range(grid_res):
        for j in range(grid_res):
            pc1, pc2 = pc1_grid[i, j], pc2_grid[i, j]
            w_query = w_mean + pc1 * PC1 + pc2 * PC2
            num, den = 0.0, 0.0
            for center, e1, e2, traj_pt in local_planes:
                d_sq = (pc1 - traj_pt[0]) ** 2 + (pc2 - traj_pt[1]) ** 2
                g = np.exp(-d_sq / (2 * sigma ** 2))
                offset = w_query - center
                alpha = np.dot(offset, e1)
                beta = np.dot(offset, e2)
                w_local = center + alpha * e1 + beta * e2
                p = unflatten_params(w_local, widths)
                L_i = mse_loss(p, x, y)
                if loss_cap is not None:
                    L_i = min(L_i, loss_cap)
                num += g * L_i
                den += g
            if den > 0:
                loss_grid[i, j] = num / den

    return pc1_1d, pc2_1d, loss_grid, traj_pc, plane_indices


if __name__ == "__main__":
    app.run()
