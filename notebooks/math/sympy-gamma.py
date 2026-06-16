# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#     "marimo>=0.20.2",
#     "sympy==1.13.3",
#     "numpy==2.3.5",
#     "matplotlib==3.10.0",
#     "mohtml==0.1.11",
# ]
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    import numpy as np
    import sympy as sp
    import matplotlib.pyplot as plt
    from sympy.parsing.sympy_parser import (
        parse_expr,
        standard_transformations,
        implicit_multiplication_application,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 🧮 SymPy Gamma NB

    A tiny [SymPy Gamma](https://gamma.sympy.org/) in a notebook: type a single
    expression and get back a stack of **cards** that analyse it — simplified and
    factored forms, roots, derivative, integral, series expansion, and a plot.

    Everything below recomputes live as you edit the box.

    **Syntax tips:**

    You can use `x` as the variable, `**` for powers (`x**2`), `/` for
    fractions, and function names like `sin`, `exp`, `sqrt`. Implicit
    multiplication works too, so `2x` means `2*x`.
    """)
    return


@app.cell(hide_code=True)
def _():
    expr_input = mo.ui.text(
        value="(x-1)(x+1)(x**2-2)",
        full_width=True,
        label="Expression",
    )
    return (expr_input,)


@app.cell(hide_code=True)
def _(expr_input):
    _transforms = standard_transformations + (implicit_multiplication_application,)

    try:
        expr = parse_expr(expr_input.value, transformations=_transforms)
    except Exception as e:
        mo.stop(
            True,
            mo.callout(
                mo.md(f"**Could not parse that expression.**\n\n`{e}`"),
                kind="danger",
            ),
        )

    # Pick the working variable: the single free symbol if there is exactly one,
    # otherwise fall back to `x`.
    _free = sorted(expr.free_symbols, key=lambda s: s.name)
    var = _free[0] if len(_free) == 1 else sp.Symbol("x")

    # Show the parsed expression as LaTeX directly beneath the input box.
    return expr, var


@app.cell(hide_code=True)
def _(expr, var):
    def _plot():
        f = sp.lambdify(var, expr, "numpy")

        # Bracket the real roots so the interesting region is on screen.
        try:
            real_roots = [
                float(r)
                for r in sp.solve(sp.Eq(expr, 0), var)
                if r.is_real
            ]
        except Exception:
            real_roots = []

        if real_roots:
            lo, hi = min(real_roots), max(real_roots)
            pad = max(1.0, (hi - lo) * 0.5)
            x_lo, x_hi = lo - pad, hi + pad
        else:
            x_lo, x_hi = -10.0, 10.0

        xs = np.linspace(x_lo, x_hi, 800)
        ys = np.asarray(f(xs), dtype=float)
        # Broadcast scalars (e.g. constant expressions) onto the x-grid.
        ys = np.broadcast_to(ys, xs.shape)

        fig, ax = plt.subplots(figsize=(6, 3.5), facecolor="white")
        ax.set_facecolor("white")
        ax.plot(xs, ys, color="#1f77b4", lw=2)
        ax.axhline(0, color="0.6", lw=1)
        if real_roots:
            ax.scatter(real_roots, [0] * len(real_roots), color="#d62728", zorder=5)
        ax.set_xlabel(var.name)
        ax.set_ylabel(f"f({var.name})")
        ax.set_title(f"Plot of  ${sp.latex(expr)}$")
        ax.grid(True, alpha=0.3)
        # Force a light theme so the plot stays readable regardless of marimo's
        # dark/light app theme (marimo otherwise restyles figures to match it).
        ax.tick_params(colors="black")
        for _spine in ax.spines.values():
            _spine.set_color("black")
        ax.xaxis.label.set_color("black")
        ax.yaxis.label.set_color("black")
        ax.title.set_color("black")
        # Return the figure (not a bare Axes): wrapping an Axes in a layout
        # container sends marimo's HTML formatter into infinite recursion.
        return fig

    try:
        plot_output = _plot()
    except Exception as e:  # noqa: BLE001
        plot_output = mo.md(f"*Plot unavailable — {e}*")
    return (plot_output,)


@app.cell
def _(expr_input, plot_output, table):
    from mohtml import br,p 

    mo.hstack([
        mo.vstack([
            expr_input,
            mo.md(table),
        ]),
        mo.vstack([
            *([br()] * 2),
            plot_output,
        ])
    ])
    return


@app.cell(hide_code=True)
def _(expr, var):
    def _ltx(fn):
        try:
            return rf"$\,{sp.latex(fn())}\,$"
        except Exception:  # noqa: BLE001
            return "*n/a*"

    def _roots():
        try:
            sols = sp.solve(sp.Eq(expr, 0), var)
        except Exception:  # noqa: BLE001
            return "*n/a*"
        if not sols:
            return "—"
        return ", ".join(rf"$\,{var.name} = {sp.latex(s)}\,$" for s in sols)

    _rows = [
        ("Simplified", _ltx(lambda: sp.simplify(expr))),
        ("Factored", _ltx(lambda: sp.factor(expr))),
        ("Expanded", _ltx(lambda: sp.expand(expr))),
        ("Roots ( = 0)", _roots()),
        (f"Derivative (d/d{var.name})", _ltx(lambda: sp.diff(expr, var))),
        (f"Integral (d{var.name})", _ltx(lambda: sp.integrate(expr, var))),
        (f"Series ({var.name} → 0)", _ltx(lambda: expr.series(var, 0, 6))),
    ]
    table = "| Operation | Result |\n| :-- | :-- |\n" + "\n".join(
        f"| **{_name}** | {_val} |" for _name, _val in _rows
    )
    return (table,)


if __name__ == "__main__":
    app.run()
