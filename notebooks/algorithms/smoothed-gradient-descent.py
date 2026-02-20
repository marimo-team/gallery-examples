# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo>=0.19.7",
#     "matplotlib==3.10.8",
#     "numpy==2.3.5",
#     "scipy==1.17.0",
# ]
# ///

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.integrate import quad
    from scipy.stats import norm


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Problem of Hard Functions

    Sometimes we want to optimise functions that are hard to optimise. Here are two examples:

    $$f(x) = \text{sinc}(x) \quad \text{and} \quad f(x) = \lfloor 10 \cdot \text{sinc}(x) + 4 \sin(x) \rfloor$$
    """)
    return


@app.cell
def _():
    # Shared x values for all plots
    x_vals = np.linspace(-15, 15, 1000)
    s_vals = np.linspace(0.01, 6.0, 40)
    return s_vals, x_vals


@app.cell
def _():
    def sinc_func(x):
        """Standard sinc function: sin(x)/x with sinc(0)=1"""
        return np.sinc(x / np.pi)

    def hard_func(x):
        """Non-differentiable function with many local optima"""
        return np.floor(10 * np.sinc(x / np.pi) + 4 * np.sin(x))

    return hard_func, sinc_func


@app.cell
def _(hard_func, sinc_func, x_vals):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(x_vals, sinc_func(x_vals), color='#1f77b4')
    ax1.set_xlabel('x')
    ax1.set_title(r'$f(x) = \mathrm{sinc}(x)$')
    ax1.set_ylim(-0.3, 1.1)

    ax2.plot(x_vals, hard_func(x_vals), color='#1f77b4')
    ax2.set_xlabel('x')
    ax2.set_title(r'$f(x) = \lfloor 10 \cdot \mathrm{sinc}(x) + 4 \sin(x) \rfloor$')

    plt.tight_layout()
    fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Both functions have multiple peaks, making them hard to optimise via gradient descent. The right function is also non-differentiable.

    **Idea:** Add a smoothing parameter $s$ to turn the 1D problem into a 2D problem:

    $$g(x, s) = \int_{-\infty}^{\infty} f(t) \cdot \mathcal{N}(t; \mu=x, \sigma=s) \, dt$$

    When $s = 0$: $g(x, 0) = f(x)$. When $s \gg 0$: $g(x, s)$ becomes a smooth average.
    """)
    return


@app.cell
def _():
    def smoothed_value(f, x, sigma, n_points=500):
        """Compute Gaussian-smoothed value of f at x with smoothing sigma."""
        if sigma < 0.01:
            return f(x)
        # Integrate over ±4 sigma
        t = np.linspace(x - 4*sigma, x + 4*sigma, n_points)
        weights = norm.pdf(t, loc=x, scale=sigma)
        values = f(t)
        return np.trapezoid(values * weights, t)

    def compute_landscape(f, x_arr, s_arr):
        """Compute the smoothed landscape g(x, s) over a grid."""
        Z = np.zeros((len(s_arr), len(x_arr)))
        for i, s in enumerate(s_arr):
            for j, x in enumerate(x_arr):
                Z[i, j] = smoothed_value(f, x, s)
        return Z

    return compute_landscape, smoothed_value


@app.cell
def _(compute_landscape, hard_func, s_vals, sinc_func, x_vals):
    # Compute landscapes (this may take a moment)
    Z_sinc = compute_landscape(sinc_func, x_vals, s_vals)
    Z_hard = compute_landscape(hard_func, x_vals, s_vals)
    return Z_hard, Z_sinc


@app.cell
def _(Z_hard, Z_sinc, s_vals, x_vals):
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(10, 4))

    X, S = np.meshgrid(x_vals, s_vals)

    cf1 = ax3.contourf(X, S, Z_sinc, levels=20, cmap='viridis')
    ax3.set_xlabel('x')
    ax3.set_ylabel('smoothing')
    ax3.set_title(r'$g(x, s)$ for sinc')
    plt.colorbar(cf1, ax=ax3)

    cf2 = ax4.contourf(X, S, Z_hard, levels=20, cmap='viridis')
    ax4.set_xlabel('x')
    ax4.set_ylabel('smoothing')
    ax4.set_title(r'$g(x, s)$ for floor function')
    plt.colorbar(cf2, ax=ax4)

    plt.tight_layout()
    fig2
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Intuition

    How does the smoothing work? We convolve $f(x)$ with a Gaussian:

    $$g(x, \sigma) = \int_{-\infty}^{\infty} f(t) \cdot \mathcal{N}(t; x, \sigma) \, dt$$

    This integral computes a weighted average of $f$, where points closer to $x$ get higher weight.

    **Use the sliders below to explore how different $(x, \sigma)$ values produce different pixel values.**
    """)
    return


@app.cell
def _():
    x_slider = mo.ui.slider(start=-14.0, stop=14.0, step=0.1, value=0.0, label="x")
    sigma_slider = mo.ui.slider(start=0.1, stop=3.0, step=0.1, value=1.0, label="σ")
    mo.hstack([x_slider, sigma_slider], justify="start")
    return sigma_slider, x_slider


@app.cell(hide_code=True)
def _(
    Z_hard,
    hard_func,
    s_vals,
    sigma_slider,
    smoothed_value,
    x_slider,
    x_vals,
):
    x_pos = x_slider.value
    sigma_val = sigma_slider.value

    fig_int, (ax_conv, ax_heat) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Function, Gaussian, product, filled area
    ax_conv.plot(x_vals, hard_func(x_vals), 'C0-', linewidth=1.5, label=r'$f(x)$')

    gaussian = norm.pdf(x_vals, loc=x_pos, scale=sigma_val)
    scale = 5
    ax_conv.plot(x_vals, gaussian * scale, 'C1-', linewidth=1.5,
                 label=rf'$\mathcal{{N}}(\mu={x_pos:.1f}, \sigma={sigma_val:.1f})$')

    product = hard_func(x_vals) * gaussian
    ax_conv.plot(x_vals, product * scale, 'C2-', linewidth=2, label=r'$f(x) \cdot \mathcal{N}$')
    ax_conv.fill_between(x_vals, 0, product * scale, alpha=0.3, color='C2')
    ax_conv.axvline(x=x_pos, color='red', linestyle='--', alpha=0.5)

    ax_conv.set_xlabel('x')
    ax_conv.set_ylim(-8, 12)
    ax_conv.set_xlim(-15, 15)
    ax_conv.legend(loc='upper right', fontsize=9)
    ax_conv.set_title('Gaussian convolution')

    # Right: Heatmap with pixel
    _X_grid, _S_grid = np.meshgrid(x_vals, s_vals)
    ax_heat.contourf(_X_grid, _S_grid, Z_hard, levels=20, cmap='viridis')
    ax_heat.scatter([x_pos], [sigma_val], s=200, c='white', marker='o',
                    edgecolors='red', linewidths=3, zorder=5)

    pixel_val = smoothed_value(hard_func, x_pos, sigma_val)
    ax_heat.set_xlabel('x')
    ax_heat.set_ylabel('smoothing (σ)')
    ax_heat.set_title(f'Pixel value = {pixel_val:.2f}')
    plt.colorbar(ax_heat.collections[0], ax=ax_heat)

    plt.tight_layout()

    mo.vstack([
        fig_int,
        mo.md(f"The green shaded area on the left (integral) equals the pixel value **{pixel_val:.2f}** on the right.")
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The key insight: instead of optimising $f(x)$ directly, we optimise in the $(x, s)$ space.
    Starting with high smoothing, the landscape is smooth and gradients point toward the global optimum.
    As we reduce $s$ toward 0, we converge to the true optimum of $f(x)$.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Gradient Field Visualization

    The arrows show the gradient direction at each point. Notice how at high smoothing (top),
    gradients consistently point toward the global maximum at $x=0$. At low smoothing (bottom),
    the gradients become chaotic with many local optima.
    """)
    return


@app.cell
def _(Z_sinc, s_vals, sinc_func, smoothed_value, x_vals):
    # Compute gradient field for visualization
    # Use a coarser grid for the arrows, derived from x_vals and s_vals
    x_arrow = np.linspace(x_vals.min() + 1, x_vals.max() - 1, 20)
    s_arrow = np.linspace(s_vals.min() + 0.1, s_vals.max() - 0.1, 15)

    U = np.zeros((len(s_arrow), len(x_arrow)))  # gradient in x direction
    V = np.zeros((len(s_arrow), len(x_arrow)))  # gradient in s direction
    eps = 1e-4

    for i, s in enumerate(s_arrow):
        for j, x in enumerate(x_arrow):
            # Gradient in x
            g_x_plus = smoothed_value(sinc_func, x + eps, s)
            g_x_minus = smoothed_value(sinc_func, x - eps, s)
            U[i, j] = (g_x_plus - g_x_minus) / (2 * eps)

            # Gradient in s - this shows how the landscape wants to move in s direction
            g_s_plus = smoothed_value(sinc_func, x, s + eps)
            g_s_minus = smoothed_value(sinc_func, x, max(0.01, s - eps))
            V[i, j] = (g_s_plus - g_s_minus) / (2 * eps)

    fig_quiver, ax_quiver = plt.subplots(figsize=(10, 6))

    _X_grid, _S_grid = np.meshgrid(x_vals, s_vals)
    ax_quiver.contourf(_X_grid, _S_grid, Z_sinc, levels=20, cmap='viridis', alpha=0.8)

    _X_arrow, _S_arrow = np.meshgrid(x_arrow, s_arrow)
    # Normalize for display - show direction only
    magnitude = np.sqrt(U**2 + V**2 + 0.001)
    ax_quiver.quiver(_X_arrow, _S_arrow, U/magnitude, V/magnitude, color='white', alpha=0.9, scale=30)

    ax_quiver.set_xlabel('x')
    ax_quiver.set_ylabel('smoothing (σ)')
    ax_quiver.set_title('Gradient field: arrows show steepest ascent direction in (x, σ) space')
    plt.colorbar(ax_quiver.collections[0], ax=ax_quiver)

    fig_quiver
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Gradient Descent in Smoothed Space

    Key insight:
    - When $s \approx 0$: $g(x, s) \approx f(x)$ (original function)
    - When $s \gg 0$: $g(x, s)$ is smooth, gradients point toward global optimum region
    - Starting at high $s$ and descending toward $s \approx 0$ helps escape local optima
    """)
    return


@app.cell(hide_code=True)
def _(smoothed_value):
    def gradient_descent_smoothed(f, x0, s0, lr_x=0.5, lr_s=0.1, steps=100, eps=1e-4, s_bias=-0.01):
        """
        Gradient descent on the smoothed landscape g(x, s).
        Follows gradients in BOTH x and s directions.

        The key insight: s is a searchable dimension, not just an annealing schedule.
        We maximize g in x direction, and follow gradient in s direction with a small
        bias toward lower s (to eventually converge to the true optimum).
        """
        trajectory = [(x0, s0)]
        x, s = x0, s0

        for step in range(steps):
            # Numerical gradient in x (for maximization)
            g_x_plus = smoothed_value(f, x + eps, s)
            g_x_minus = smoothed_value(f, x - eps, s)
            grad_x = (g_x_plus - g_x_minus) / (2 * eps)

            # Numerical gradient in s - this is the key!
            # Follow the gradient in s direction too
            g_s_plus = smoothed_value(f, x, s + eps)
            g_s_minus = smoothed_value(f, x, max(0.01, s - eps))
            grad_s = (g_s_plus - g_s_minus) / (2 * eps)

            # Update x (gradient ascent to find maximum)
            x = x + lr_x * grad_x

            # Update s: follow gradient + small bias toward lower s
            # The bias ensures we eventually converge to s≈0
            s = s + lr_s * grad_s + s_bias
            s = max(0.01, min(3.0, s))  # clamp to valid range

            trajectory.append((x, s))

        return np.array(trajectory)

    return (gradient_descent_smoothed,)


@app.function(hide_code=True)
def es_on_f(f, mu0, sigma0, alpha_mu=0.3, alpha_sigma=0.05, n_samples=100, steps=100, seed=42):
    """
    ES on f(x) - Original blogpost style.

    σ is the search distribution width, not a dimension being optimized.
    Samples x ~ N(μ, σ), evaluates raw f(x), with normalization.
    """
    np.random.seed(seed)
    trajectory = [(mu0, sigma0)]
    mu, sigma = mu0, sigma0

    for _ in range(steps):
        samples = np.random.normal(mu, sigma, n_samples)
        f_vals = f(samples)

        # Normalize rewards (baseline subtraction + scaling)
        f_norm = (f_vals - np.mean(f_vals)) / (np.std(f_vals) + 1e-8)

        d_mu = alpha_mu * np.mean(f_norm * (samples - mu)) / sigma
        d_sigma = alpha_sigma * np.mean(f_norm * ((samples - mu)**2 / sigma**2 - 1)) - 0.01

        mu = mu + d_mu
        sigma = np.clip(sigma + d_sigma, 0.1, 6.0)
        trajectory.append((mu, sigma))

    return np.array(trajectory)


@app.cell(hide_code=True)
def _(smoothed_value):
    def es_on_g(f, x0, s0, eps_x=0.5, eps_s=0.2, alpha=0.3, n_samples=50, steps=100, seed=42):
        """
        ES on g(x,σ) - True 2D optimization.

        Samples perturbations in both x and σ, evaluates smoothed g(x,σ).
        Actually navigates the 2D landscape.
        """
        np.random.seed(seed)
        trajectory = [(x0, s0)]
        x, s = x0, s0

        for _ in range(steps):
            dx = np.random.normal(0, eps_x, n_samples)
            ds = np.random.normal(0, eps_s, n_samples)

            x_samples = x + dx
            s_samples = np.clip(s + ds, 0.1, 6.0)

            # Evaluate the SMOOTHED function g(x, σ)
            g_vals = np.array([smoothed_value(f, xi, si) for xi, si in zip(x_samples, s_samples)])
            g_norm = (g_vals - np.mean(g_vals)) / (np.std(g_vals) + 1e-8)

            d_x = alpha * np.mean(g_norm * dx) / eps_x
            d_s = alpha * np.mean(g_norm * ds) / eps_s - 0.005

            x = x + d_x
            s = np.clip(s + d_s, 0.1, 6.0)
            trajectory.append((x, s))

        return np.array(trajectory)

    return (es_on_g,)


@app.cell
def _():
    # Configuration: ONE place to set all trajectory parameters
    traj_x0_slider = mo.ui.slider(start=-12.0, stop=12.0, step=0.5, value=-8.0, label="Start x₀")
    traj_s0_slider = mo.ui.slider(start=0.5, stop=5.0, step=0.1, value=3.0, label="Start σ₀")
    traj_steps_slider = mo.ui.slider(start=50, stop=500, step=50, value=200, label="Steps")
    traj_func_dropdown = mo.ui.dropdown(options=["sinc", "floor"], value="sinc", label="Function")
    mo.hstack([traj_func_dropdown, traj_x0_slider, traj_s0_slider, traj_steps_slider], justify="start")
    return (
        traj_func_dropdown,
        traj_s0_slider,
        traj_steps_slider,
        traj_x0_slider,
    )


@app.cell(hide_code=True)
def _(
    Z_hard,
    Z_sinc,
    es_on_g,
    gradient_descent_smoothed,
    hard_func,
    s_vals,
    sinc_func,
    traj_func_dropdown,
    traj_s0_slider,
    traj_steps_slider,
    traj_x0_slider,
    x_vals,
):
    # Get config values
    _x0 = traj_x0_slider.value
    _s0 = traj_s0_slider.value
    _steps = traj_steps_slider.value
    _func_name = traj_func_dropdown.value
    _func = sinc_func if _func_name == "sinc" else hard_func
    _Z = Z_sinc if _func_name == "sinc" else Z_hard

    # Compute all 3 trajectories from same starting point
    _traj_grad = gradient_descent_smoothed(_func, x0=_x0, s0=_s0, steps=_steps)
    _traj_es_f = es_on_f(_func, mu0=_x0, sigma0=_s0, steps=_steps, seed=42)
    _traj_es_g = es_on_g(_func, x0=_x0, s0=_s0, steps=_steps, seed=42)

    # Single chart comparing all 3 methods
    fig_traj, ax_traj = plt.subplots(figsize=(10, 6))

    _X2, _S2 = np.meshgrid(x_vals, s_vals)
    ax_traj.contourf(_X2, _S2, _Z, levels=20, cmap='viridis')

    # Red for gradient, green for ES on f(x), blue for ES on g(x,σ)
    ax_traj.plot(_traj_grad[:, 0], _traj_grad[:, 1], 'r.-', linewidth=2, markersize=3, label='Gradient on g')
    ax_traj.plot(_traj_es_f[:, 0], _traj_es_f[:, 1], 'g.-', linewidth=2, markersize=3, label='ES on f(x)')
    ax_traj.plot(_traj_es_g[:, 0], _traj_es_g[:, 1], 'b.-', linewidth=2, markersize=3, label='ES on g(x,σ)')
    ax_traj.scatter([_x0], [_s0], s=150, c='yellow', marker='*', zorder=10, edgecolors='black', label='Start')

    ax_traj.set_xlabel('x')
    ax_traj.set_ylabel('smoothing (σ)')
    ax_traj.set_title(f'{_func_name} function: start (x={_x0}, σ={_s0}), {_steps} steps')
    ax_traj.legend(loc='upper right')
    plt.colorbar(ax_traj.collections[0], ax=ax_traj)

    plt.tight_layout()
    fig_traj
    return


if __name__ == "__main__":
    app.run()
