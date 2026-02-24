# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.20.2",
#     "matplotlib",
#     "numpy",
# ]
# ///

import marimo

__generated_with = "0.20.2"
app = marimo.App()

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np


@app.cell(hide_code=True)
def _():
    mo.md("""
    # Interactive Matplotlib Selection Demo

    Interactive region selection for matplotlib plots using `mo.ui.matplotlib`.

    - Click and drag to draw a **box** selection
    - Hold **Shift** and drag for **lasso** selection
    - Click outside the selection to clear it
    """)
    return


@app.cell
def _():
    np.random.seed(42)
    x_data = np.random.randn(300)
    y_data = np.random.randn(300)
    return x_data, y_data


@app.cell
def _(x_data, y_data):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(x_data, y_data, alpha=0.6, color="#3b82f6")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, alpha=0.3)
    ax.set_title("Draw a selection to highlight points")
    ax = mo.ui.matplotlib(ax)
    return (ax,)


@app.cell
def _(ax):
    ax
    return


@app.cell
def _(ax, x_data, y_data):
    _mask = ax.value.get_mask(x_data, y_data)
    np.column_stack([x_data[_mask], y_data[_mask]])
    return


if __name__ == "__main__":
    app.run()
