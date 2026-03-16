# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "matplotlib",
#     "numpy",
#     "scipy",
#     "wigglystuff",
# ]
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    from wigglystuff.chart_puck import ChartPuck
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.integrate import solve_ivp

    return ChartPuck, np, plt, solve_ivp


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Interactive ODE Solver

    Drag the red initial-condition point to explore solutions of

    $$\frac{dy}{dt} = \frac{1}{2}y^2 - 5t^2 + 1$$

    The solver integrates forward (blue) and backward (orange) in time
    from the current initial condition.
    """)
    return


@app.cell
def _(ChartPuck, np, ode_rhs, plt, solve_ivp):
    X_MIN, X_MAX = -3, 3
    Y_MIN, Y_MAX = -10, 10


    def draw_ode_soln(ax: plt.Axes, widget: ChartPuck) -> None:
        x, y = widget.x[0], widget.y[0]

        # --- Direction field (quiver plot) ---
        _t_grid: np.ndarray = np.linspace(X_MIN, X_MAX, 25)
        _y_grid: np.ndarray = np.linspace(Y_MIN, Y_MAX, 25)
        _T, _Y = np.meshgrid(_t_grid, _y_grid)
        _dT: np.ndarray = np.ones_like(_T)  # dt/dt = 1
        _dY: np.ndarray = ode_rhs(_T.flatten(), _Y.flatten()).reshape(
            _T.shape
        )  # dy/dt from the ODE

        # Normalize arrows so they all have the same length
        _speed: np.ndarray = np.sqrt(_dT**2 + _dY**2)
        _dT_norm: np.ndarray = _dT / _speed
        _dY_norm: np.ndarray = _dY / _speed

        ax.quiver(
            _T,
            _Y,
            _dT_norm,
            _dY_norm,
            _speed,
            cmap="coolwarm",
            alpha=0.45,
            scale=30,
            width=0.003,
            headwidth=3.5,
            headlength=4,
        )

        # --- Compute forward ODE solution ---
        t_forward_span: tuple[float, float] = (x, float(X_MAX))
        sol_forward = solve_ivp(
            ode_rhs, t_forward_span, [y], dense_output=True, max_step=0.05
        )
        t_fwd: np.ndarray = np.linspace(x, float(X_MAX), 300)
        y_fwd: np.ndarray = sol_forward.sol(t_fwd)[0]

        # --- Compute backward ODE solution ---
        t_backward_span: tuple[float, float] = (x, float(X_MIN))
        sol_backward = solve_ivp(
            ode_rhs, t_backward_span, [y], dense_output=True, max_step=0.05
        )
        t_bwd: np.ndarray = np.linspace(x, float(X_MIN), 300)
        y_bwd: np.ndarray = sol_backward.sol(t_bwd)[0]

        # --- Plot forward and backward solutions ---
        ax.plot(
            t_fwd, y_fwd, color="tab:blue", linewidth=2.5, label="Forward", zorder=3
        )
        ax.plot(
            t_bwd,
            y_bwd,
            color="tab:orange",
            linewidth=2.5,
            label="Backward",
            zorder=3,
        )

        ax.set_title(f"Position: ({x:.2f}, {y:.2f})")
        ax.set_xlim(X_MIN, X_MAX)
        ax.set_ylim(Y_MIN, Y_MAX)
        ax.set_xlabel("t")
        ax.set_ylabel("y")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.figure.tight_layout()

    return (draw_ode_soln,)


@app.cell
def _(np):
    def ode_rhs(t: float, y: np.ndarray) -> np.ndarray:
        """dy/dt = 0.5 y^2 - 5t^2 + 1."""
        return np.array([0.5 * y**2 - 5 * t**2 + 1])

    return (ode_rhs,)


@app.cell
def _(ChartPuck, draw_ode_soln):
    dynamic_puck = ChartPuck.from_callback(
        draw_fn=draw_ode_soln,
        x_bounds=(-3, 3),
        y_bounds=(-5, 5),
        figsize=(10, 6),
        x=0,
        y=0,
        puck_color="red",
        puck_radius=5,
        throttle=10,
    )
    return (dynamic_puck,)


@app.cell
def _(dynamic_puck, mo):
    dynamic_widget = mo.ui.anywidget(dynamic_puck)
    return (dynamic_widget,)


@app.cell
def _(dynamic_widget):
    dynamic_widget
    return


if __name__ == "__main__":
    app.run()
