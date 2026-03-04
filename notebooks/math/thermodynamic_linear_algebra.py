# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.20.4",
#     "matplotlib==3.10.8",
#     "numpy==2.4.2",
# ]
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    return mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Thermodynamic Linear Algebra: estimating the inverse of a matrix

    _By [Simone Conradi](https://profconradi.com/)._

    This notebook demonstrates a key result from *Thermodynamic Linear Algebra* (Aifer et al., 2023): **the inverse of a symmetric positive-definite matrix can be estimated from the time-averaged covariance of the overdamped Langevin dynamics in the corresponding quadratic potential**.

    ## The core idea

    A system with $d$ degrees of freedom evolving in the quadratic potential $U(\mathbf{x}) = \frac{1}{2}\mathbf{x}^\top A\mathbf{x}$ (with $A$ symmetric positive-definite) has, at thermal equilibrium with inverse temperature $\beta = 1/k_BT$, a Boltzmann distribution $f(\mathbf{x})\propto\exp(-\beta U(\mathbf{x}))$ that is a zero-mean multivariate Gaussian: $\mathbf{x}\sim\mathcal{N}[\mathbf{0},\,\beta^{-1}A^{-1}]$. The equilibrium covariance directly encodes the inverse: $A^{-1}=\beta\,\langle\mathbf{x}\mathbf{x}^\top\rangle_{\mathrm{time}}$.

    Aifer et al. propose to realize this potential on thermodynamic hardware (e.g. coupled RLC circuits, where $A$ is encoded in resistances and capacitances). On such hardware, simply letting the system equilibrate and measuring its fluctuations would yield $A^{-1}$. This notebook implements the same idea purely numerically.

    ## Overdamped Langevin dynamics and ergodicity

    I simulate the **overdamped Langevin equation** $d\mathbf{x} = -\frac{1}{\gamma}A\mathbf{x}\,dt + \sqrt{\frac{2}{\beta\gamma}}\,d\mathbf{W}_t$, where $\gamma>0$ is the damping constant and $\mathbf{W}_t$ a standard Wiener process. The deterministic drift pushes $\mathbf{x}$ toward the potential minimum; the stochastic term models thermal noise. This defines a vector Ornstein-Uhlenbeck process with relaxation time $\tau_r=\gamma/\|A\|$ and unique stationary covariance $\Sigma_s=\beta^{-1}A^{-1}$.

    By **ergodicity**, the time-averaged covariance along a single trajectory converges to the ensemble average: $\beta \langle\mathbf{x}\mathbf{x}^\top\rangle_{\mathrm{time}}\xrightarrow{\tau\to\infty}A^{-1}$.

    ## What this notebook does

    Set the elements of a $2\times 2$ SPD matrix $A$. The simulation integrates the Langevin equation via Euler-Maruyama discretization, discards a burn-in transient, and computes $\beta \langle\mathbf{x}\mathbf{x}^\top\rangle_{\mathrm{time}}$ as an estimator of $A^{-1}$. Observe how convergence depends on $\beta$ (inverse temperature), $\gamma$ (damping) and the number of steps.
    """)
    return


@app.cell
def _(mo, np):
    matrix_ui = mo.ui.matrix(
        np.array([[4,2],[2,9]], dtype=np.int32),
        step=1,
        precision=0,
        symmetric=True,
        label=r"$A$",
    )
    return (matrix_ui,)


@app.cell(hide_code=True)
def _(mo):
    n_iter_slider = mo.ui.slider(10_000, 100_000, 10_000, value=10_000, label="Number of steps")
    beta_slider = mo.ui.slider(0.01, 2, 0.01, value=0.1, label=r"$\beta$")
    gamma_slider = mo.ui.slider(1, 50, 0.5, value=10., label=r"$\gamma$")
    dt_slider = mo.ui.slider(0.001, 0.5, 0.001, value=0.01, label=r"$\delta t$")
    return beta_slider, dt_slider, gamma_slider, n_iter_slider


@app.cell(hide_code=True)
def _(beta_slider, dt_slider, gamma_slider, matrix_ui, mo, n_iter_slider):
    mo.hstack(
        [
            matrix_ui,
            mo.vstack(
                [mo.md("**Simulation**"), n_iter_slider, dt_slider],
                align="stretch",
            ),
            mo.vstack(
                [mo.md("**Physics**"), beta_slider, gamma_slider],
                align="stretch",
            ),
        ],
        justify="center",
        align="center",
        gap=4,
    )
    return


@app.cell(hide_code=True)
def _(matrix_ui, mo, np):
    A = np.array(matrix_ui.value)
    is_positive_definite = np.all(np.linalg.eigvals(A)>0)

    mo.stop(
        not is_positive_definite,
        mo.md(
            "**Matrix is not positive definite!**"
            " Please adjust the matrix elements."
        ).style({"color": "red", "font-size": "1.1em"}),
    )
    return (A,)


@app.cell(hide_code=True)
def _(A, beta_slider, dt_slider, gamma_slider, n_iter_slider, np):
    D = 2
    gamma = gamma_slider.value
    beta = beta_slider.value
    dt = dt_slider.value

    n_iter = n_iter_slider.value
    x0 = np.array([15.0, 17.0])

    x = np.zeros((n_iter, D), dtype=np.float64)
    x[0] = x0
    for _i in range(1, n_iter):
        x[_i] = (
            x[_i - 1]
            - 1 / gamma * A @ x[_i - 1] * dt
            + np.random.normal(0, np.sqrt(2 / (gamma * beta) * dt), size=(2,))
        )
    return beta, n_iter, x


@app.cell(hide_code=True)
def _():
    def quadratic_potential_to_latex(A):
        a, b = A[0, 0], A[0, 1]
        c, d = A[1, 0], A[1, 1]
        return (
            r"$U(\mathbf{x})=\frac{1}{2}\mathbf{x}^\top A \mathbf{x}"
            rf" \quad A = \genfrac{{(}}{{)}}{{0}}{{}}{{{a} \quad {b}}}{{{c} \quad {d}}}$"
        )

    def langevin_equation():
        return r"$d\mathbf{x} = -\frac{1}{\gamma}A\mathbf{x}dt + \mathcal{N}\left[0, \frac{2}{\beta\gamma}I\,dt\right]$"

    def time_averages_latex(A_inv, B):
        a, b = round(A_inv[0, 0], 2), round(A_inv[0, 1], 2)
        c, d = round(A_inv[1, 0], 2), round(A_inv[1, 1], 2)
        a1, b1 = round(B[0, 0], 4), round(B[0, 1], 4)
        c1, d1 = round(B[1, 0], 4), round(B[1, 1], 4)
        return (
            rf"$A^{{-1}}=\genfrac{{(}}{{)}}{{0}}{{}}{{{a} \quad {b}}}{{{c} \quad {d}}}"
            rf" \quad \beta \langle\mathbf{{x}}\mathbf{{x}}^\top\rangle_{{\mathrm{{time}}}}"
            rf"\approx \genfrac{{(}}{{)}}{{0}}{{}}{{{a1:+.4f} \quad {b1:+.4f}}}{{{c1:+.4f} \quad {d1:+.4f}}}$"
        )

    return langevin_equation, quadratic_potential_to_latex, time_averages_latex


@app.cell(hide_code=True)
def _(
    A,
    beta,
    langevin_equation,
    n_iter,
    np,
    plt,
    quadratic_potential_to_latex,
    time_averages_latex,
    x,
):
    n_grid = 100
    skip = 5000
    xmin, xmax, ymin, ymax = -17, 17, -10, 24
    xmesh, ymesh = np.meshgrid(
        np.linspace(xmin, xmax, n_grid), np.linspace(ymin, ymax, n_grid)
    )
    grid = np.vstack((xmesh.ravel(), ymesh.ravel())).T
    U = np.einsum("ik,kl,il->i", grid, A, grid).reshape((n_grid, n_grid))

    n_max = n_iter - 1
    B = np.einsum("ti,tj->ij", x[skip:n_max], x[skip:n_max]) / (n_max - skip) * beta

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.set_facecolor("#f4f0e8")
    ax.set_facecolor("#f4f0e8")
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)

    ax.contour(xmesh, ymesh, U, colors="#383b3e", levels=40, linewidths=0.2, zorder=-10)
    ax.plot(x[:n_max, 0], x[:n_max, 1], color="blue", linewidth=.1, alpha=0.7, zorder=0)
    ax.scatter(x[n_max, 0], x[n_max, 1], c="red", s=10, linewidths=0.0, alpha=1.0, zorder=10)

    ax.set_aspect("equal", "box")
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])

    props1 = dict(boxstyle="round", facecolor="wheat", alpha=0.5)

    ax.text(
        -15,
        13,
        quadratic_potential_to_latex(A),
        size=12,
        verticalalignment="top",
        color = "#383b3e",
        bbox=props1,
    )
    ax.text(
        1,
        16,
        langevin_equation(),
        size=12,
        verticalalignment="top",
        color = "blue",
        bbox=props1,
    )

    props2 = dict(boxstyle="roundtooth", facecolor="green", alpha=0.7)
    ax.text(
        -15.6,
        23,
        time_averages_latex(np.linalg.inv(A), B),
        size=15,
        verticalalignment="top",
        bbox=props2,
    )

    ax.set_xlabel(r"$x_1$", fontsize=14)
    ax.set_ylabel(r"$x_2$", fontsize=14)
    ax.set_title(
        "Thermodynamic Linear Algebra: Estimating the Inverse of a Matrix",
        fontsize=11,
        fontweight="bold",
    )
    ax.text(7, -9.4, "Simone Conradi, 2026", fontsize=10)

    plt.gca()
    return


if __name__ == "__main__":
    app.run()
