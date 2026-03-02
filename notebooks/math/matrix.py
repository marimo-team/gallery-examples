# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.20.2",
#     "numpy==2.3.5",
# ]
# ///

import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np


@app.cell
def _():
    matrix = mo.ui.matrix(np.eye(3))
    matrix
    return (matrix,)


@app.cell
def _(matrix):
    np.asarray(matrix.value)
    return
