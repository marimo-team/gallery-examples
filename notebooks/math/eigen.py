# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.20.1",
#     "matplotlib==3.10.8",
#     "numpy==2.4.2",
# ]
# ///

import marimo

__generated_with = "0.20.1"
app = marimo.App()

with app.setup:
    import numpy as np
    import marimo as mo
    import matplotlib.pyplot as plt


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Spectral Decomposition

    A matrix $M$ can be decomposed as:

    $$M = V D V^T$$

    where $V$ is the matrix of eigenvectors and $D = \text{diag}(\lambda_1, \lambda_2)$
    is the diagonal matrix of eigenvalues.

    To get a sense of **what this means**, we can image a matrix transforming a unit circle into an ellipse. To get use started, we have provided an initial transform matrix $M$ below.

    **Drag the values of M to adjust.**
    """)
    return


@app.cell
def _():
    _A = np.random.rand(2, 2)
    initial_matrix = _A @ _A.T
    matrix_ui = mo.ui.matrix(
        initial_matrix,
        step=0.05,
        precision=3,
        symmetric=True,
        label=r"$M$",
    )
    matrix_ui
    return (matrix_ui,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    This notebook will illustrate the transform, and will demonstrate that the axes of the transformed circle are aligned with the eigenvectors and scaled by the eigenvalues.
    We can grab the "eigenvalues" and "eigenvectors" of the matrix easily with `numpy`.
    """)
    return


@app.cell
def _(matrix_ui):
    M = np.array(matrix_ui.value)
    vals, vecs = np.linalg.eigh(M)
    eigen = mo.ui.matrix(
        np.diag(vals),
        precision=3,
        disabled=~np.identity(2, dtype=bool),
        label=r"$D$",
        step=0.1,
    )
    return M, eigen, vals, vecs


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Drag the matrix values below to see the decomposition and transform update live.**
    Adjusting the eigenvalues in $D$ produces $M^\star = V D^\star V^T$.
    """)
    return


@app.cell
def _(eigen, vecs):
    mo.hstack(
        [
            mo.ui.matrix(vecs, precision=3, disabled=True, label=r"$V$"),
            mo.md(r"$$\times$$"),
            eigen,
            mo.md(r"$$\times$$"),
            mo.ui.matrix(vecs.T, precision=3, disabled=True, label=r"$V^T$"),
            mo.md(r"$$=$$"),
            mo.ui.matrix(
                (vecs @ eigen.value) @ vecs.T,
                precision=3,
                disabled=True,
                label=r"$M^\star$",
            ),
        ],
        justify="start",
    )
    return


@app.cell(hide_code=True)
def _():
    use_m_star = mo.ui.switch(label=r"Use $M^\star$", value=False)
    mo.hstack([
        mo.md(
            "See how $M$ transforms the unit circle, then toggle to $M^\\star$"
            " to see the effect of scaling the eigenvalues."
        ),
        use_m_star,
    ], justify="start", gap=1)
    return (use_m_star,)


@app.cell(hide_code=True)
def _(M, eigen, use_m_star, vals, vecs):
    if use_m_star.value:
        _D = np.array(eigen.value)
        _plot_vals = np.diag(_D)
        _plot_matrix = (vecs @ _D) @ vecs.T
        _label = r"$M^\star$"
    else:
        _plot_vals = vals
        _plot_matrix = M
        _label = r"$M$"

    # Regular grid colored by distance from (-1, -1)
    _g = np.linspace(-2, 2, 25)
    _gx, _gy = np.meshgrid(_g, _g)
    grid_x = _gx.ravel()
    grid_y = _gy.ravel()
    _dist = np.hypot(grid_x + 1, grid_y + 1)
    _grid_pts = np.stack([grid_x, grid_y])
    _grid_transformed = _plot_matrix @ _grid_pts
    grid_tx = _grid_transformed[0]
    grid_ty = _grid_transformed[1]

    _theta = np.linspace(0, 2 * np.pi, 200)
    _circle = np.array([np.cos(_theta), np.sin(_theta)])
    _ellipse = _plot_matrix @ _circle

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    # Grid points colored by distance from (-1,-1)
    ax.scatter(grid_x, grid_y, c=_dist, cmap="viridis", s=15, alpha=0.6, zorder=1)

    # Unit circle
    ax.plot(
        _circle[0], _circle[1], "--", color="gray", alpha=0.5, label="Unit circle"
    )

    # Transformed ellipse
    ax.plot(
        _ellipse[0],
        _ellipse[1],
        "-",
        color="steelblue",
        linewidth=2,
        label="Transformed",
    )

    # Eigenvectors scaled by eigenvalues
    _colors = ["#e74c3c", "#2ecc71"]
    for i in range(2):
        _ev = vecs[:, i] * _plot_vals[i]
        ax.annotate(
            "",
            xy=_ev,
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color=_colors[i], lw=2.5),
        )
        ax.text(
            _ev[0] * 1.15,
            _ev[1] * 1.15,
            rf"$\lambda_{i + 1}={_plot_vals[i]:.2f}$",
            fontsize=10,
            color=_colors[i],
            ha="center",
        )

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.axvline(0, color="k", linewidth=0.5)
    ax.legend(loc="upper left")
    ax.set_title(f"Transform of unit circle by {_label}")

    plot = mo.ui.matplotlib(ax)
    plot
    return grid_tx, grid_ty, grid_x, grid_y, plot


@app.cell(hide_code=True)
def _(grid_tx, grid_ty, grid_x, grid_y, plot):
    _sel = plot.value
    if _sel:
        _mask = _sel.get_mask(grid_x, grid_y)
        _sx = grid_x[_mask]
        _sy = grid_y[_mask]
        _dx = grid_tx[_mask] - _sx
        _dy = grid_ty[_mask] - _sy
        _dist = np.hypot(_sx + 1, _sy + 1)

        _fig, _ax = plt.subplots(1, 1, figsize=(6, 6))
        _ax.quiver(
            _sx,
            _sy,
            _dx,
            _dy,
            _dist,
            cmap="viridis",
            angles="xy",
            scale_units="xy",
            scale=1,
            alpha=0.8,
        )
        _ax.scatter(_sx, _sy, c=_dist, cmap="viridis", s=12, zorder=2)
        _ax.set_aspect("equal")
        _ax.grid(True, alpha=0.3)
        _ax.axhline(0, color="k", linewidth=0.5)
        _ax.axvline(0, color="k", linewidth=0.5)
        _ax.set_title(f"Translation vectors ({_mask.sum()} selected points)")
        _out = _ax
    else:
        _out = mo.md(
            "*Select a region on the plot above to see translation vectors.*"
        )
    _out
    return


@app.cell(hide_code=True)
def _(eigen):
    _D = np.array(eigen.value)
    _ev = np.diag(_D)
    _det = _ev[0] * _ev[1]
    _circle_area = np.pi
    _ellipse_area = np.pi * abs(_det)

    _orient = "preserved" if _det > 0 else "reversed"

    mo.md(
        rf"""
    ### Aside: Area Scaling

    The product of eigenvalues ($\det(M^\star)$) tells us how $M^\star$ scales area:

    $$\lambda_1 \cdot \lambda_2 = {_ev[0]:.3f} \times {_ev[1]:.3f} = {_det:.3f}$$

    $$\begin{{array}}{{l|r}}
    & \text{{Area}} \\ \hline
    \text{{Unit circle}} & \pi \approx {_circle_area:.3f} \\
    \text{{Ellipse}} & \pi \,|\lambda_1 \lambda_2| \approx {_ellipse_area:.3f} \\
    \text{{Scale factor}} & {abs(_det):.3f}\times
    \end{{array}}$$

    Orientation is **{_orient}** (sign of $\det(M^\star)$).
    """
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
