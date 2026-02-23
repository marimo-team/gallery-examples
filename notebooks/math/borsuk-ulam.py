# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "drawdata",
#     "numpy==2.4.2",
#     "matplotlib",
#     "polars==1.38.1",
#     "scipy==1.17.0",
# ]
# ///

import marimo

__generated_with = "0.20.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Borsuk-Ulam Theorem (1D)

    **Theorem:** For any continuous function $f$ on a circle, there exist two
    antipodal points with equal values: $f(\theta) = f(\theta + \pi)$.

    **How this works:**
    1. Draw a curve below — think of it as sketching a function.
    2. The x-axis becomes angle $\theta \in [0, 2\pi]$, the y-axis becomes $f(\theta)$.
    3. We interpolate your points into a continuous function and find where
       $g(\theta) = f(\theta) - f(\theta + \pi) = 0$.

    The theorem guarantees that zero **always** exists. Try to draw one where it doesn't!
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from drawdata import ScatterWidget
    from scipy.ndimage import gaussian_filter1d

    return ScatterWidget, gaussian_filter1d, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Draw your function

    Draw points below. The x-coordinate becomes the angle, y becomes the value.
    """)
    return


@app.cell
def _(ScatterWidget, mo):
    widget = mo.ui.anywidget(ScatterWidget(n_classes=1, height=400, width=600, ))
    widget
    return (widget,)


@app.cell
def _(mo):
    sigma_slider = mo.ui.slider(1, 30, value=5, label="Smoothing (σ)")
    sigma_slider
    return (sigma_slider,)


@app.cell
def _(gaussian_filter1d, mo, np, plt, sigma_slider, widget):
    df = widget.widget.data_as_polars
    mo.stop(len(df) < 3, mo.md("⬆️ **Draw at least 3 points to get started.**"))

    # Extract and sort by x
    x_raw = df["x"].to_numpy().astype("float64")
    y_raw = df["y"].to_numpy().astype("float64")
    order = np.argsort(x_raw)
    x_pts, y_pts = x_raw[order], y_raw[order]

    # Normalize x to [0, 2pi], y to [-1, 1]
    x_min, x_max = x_pts.min(), x_pts.max()
    y_min, y_max = y_pts.min(), y_pts.max()
    mo.stop(x_max - x_min < 1e-9, mo.md("**Spread your points out horizontally.**"))

    theta_pts = (x_pts - x_min) / (x_max - x_min) * 2 * np.pi
    if y_max - y_min > 1e-9:
        f_pts = (y_pts - y_min) / (y_max - y_min) * 2 - 1
    else:
        f_pts = np.zeros_like(y_pts)

    # Interpolate onto a fine grid, then apply Gaussian smoothing with wrap
    theta = np.linspace(0, 2 * np.pi, 500)
    f_raw = np.interp(theta, theta_pts, f_pts)
    f_raw[-1] = f_raw[0]  # close the loop
    f_vals = gaussian_filter1d(f_raw, sigma=sigma_slider.value, mode='wrap')

    # Compute g(theta) = f(theta) - f(theta + pi)
    theta_shifted = (theta + np.pi) % (2 * np.pi)
    f_shifted = np.interp(theta_shifted, theta, f_vals)
    g_vals = f_vals - f_shifted

    # Find zero crossings of g
    sign_changes = np.where(np.diff(np.sign(g_vals)))[0]

    # Refine to get the best zero crossing
    if len(sign_changes) > 0:
        best_idx = sign_changes[np.argmin(np.abs(g_vals[sign_changes]))]
        t0, t1 = theta[best_idx], theta[best_idx + 1]
        g0, g1 = g_vals[best_idx], g_vals[best_idx + 1]
        if abs(g1 - g0) > 1e-12:
            t_star = t0 - g0 * (t1 - t0) / (g1 - g0)
        else:
            t_star = (t0 + t1) / 2
        f_star = float(np.interp(t_star, theta, f_vals))
        t_anti = (t_star + np.pi) % (2 * np.pi)
        f_anti = float(np.interp(t_anti, theta, f_vals))
        found = True
    else:
        found = False

    # --- PLOTS ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Plot 1: f(theta) with antipodal pair
    ax1 = axes[0]
    ax1.plot(theta, f_vals, 'b-', linewidth=2, label=r'$f(\theta)$')
    ax1.plot(theta, f_shifted, 'r--', linewidth=1.5, alpha=0.7, label=r'$f(\theta + \pi)$')
    if found:
        ax1.plot(t_star, f_star, 'go', markersize=12, zorder=5, label=f'θ* = {t_star:.2f}')
        ax1.plot(t_anti, f_anti, 'g^', markersize=12, zorder=5, label=f'θ* + π = {t_anti:.2f}')
        ax1.axhline(y=f_star, color='green', linestyle=':', alpha=0.5)
    ax1.set_xlabel(r'$\theta$')
    ax1.set_ylabel(r'$f(\theta)$')
    ax1.set_title('Your function on the circle')
    ax1.legend(fontsize=9)
    ax1.set_xlim(0, 2 * np.pi)
    ax1.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax1.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])

    # Plot 2: g(theta) = f(theta) - f(theta + pi)
    ax2 = axes[1]
    ax2.plot(theta, g_vals, 'purple', linewidth=2)
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax2.fill_between(theta, g_vals, 0, alpha=0.15, color='purple')
    if found:
        ax2.plot(t_star, 0, 'go', markersize=12, zorder=5, label='Zero crossing!')
    ax2.set_xlabel(r'$\theta$')
    ax2.set_ylabel(r'$g(\theta)$')
    ax2.set_title(r'$g(\theta) = f(\theta) - f(\theta + \pi)$')
    ax2.legend(fontsize=9)
    ax2.set_xlim(0, 2 * np.pi)
    ax2.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax2.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])

    # Plot 3: Polar view
    ax3 = axes[2]
    ax3 = fig.add_subplot(1, 3, 3, projection='polar')
    axes[2].set_visible(False)  # hide the rectangular one
    r_vals = f_vals - f_vals.min() + 0.5  # shift so all radii positive
    ax3.plot(theta, r_vals, 'b-', linewidth=2)
    if found:
        r_star = f_star - f_vals.min() + 0.5
        r_anti = f_anti - f_vals.min() + 0.5
        ax3.plot(t_star, r_star, 'go', markersize=12, zorder=5)
        ax3.plot(t_anti, r_anti, 'g^', markersize=12, zorder=5)
        ax3.plot([t_star, t_anti], [r_star, r_anti], 'g--', linewidth=2, alpha=0.7)
    ax3.set_title('Polar view', pad=20)

    plt.tight_layout()
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Why does this always work?

    Define $g(\theta) = f(\theta) - f(\theta + \pi)$. Then:

    $$g(0) = f(0) - f(\pi)$$
    $$g(\pi) = f(\pi) - f(2\pi) = f(\pi) - f(0) = -g(0)$$

    So $g(0)$ and $g(\pi)$ have **opposite signs** (unless one is already zero).
    By the **Intermediate Value Theorem**, $g$ must cross zero somewhere in between.

    That zero is your antipodal pair: $f(\theta^*) = f(\theta^* + \pi)$. ∎

    This is the 1D case of the Borsuk-Ulam theorem. The full theorem says:
    for any continuous map $f: S^n \to \mathbb{R}^n$, there exists a point $x$ where
    $f(x) = f(-x)$. The 2D version implies that right now, there are two
    diametrically opposite points on Earth with the **same temperature and pressure**.
    """)
    return


if __name__ == "__main__":
    app.run()