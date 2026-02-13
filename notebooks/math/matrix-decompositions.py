# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy==2.3.5",
#     "scipy==1.17.0",
#     "wigglystuff",
# ]
# ///

import marimo

__generated_with = "0.19.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from wigglystuff import Matrix
    return Matrix, mo, np


@app.cell
def _(Matrix, mo, slider_cols, slider_rows):
    matrix_widget = mo.ui.anywidget(Matrix(rows=slider_rows.value, cols=slider_cols.value, min_value=-10, max_value=10))
    return (matrix_widget,)


@app.cell
def _(mo):
    slider_rows = mo.ui.slider(1, 10, 1, value=3, label="rows")
    slider_cols = mo.ui.slider(1, 10, 1, value=3, label="cols")
    return slider_cols, slider_rows


@app.cell
def _(slider_cols, slider_rows):
    slider_rows, slider_cols
    return


@app.cell
def _(matrix_widget):
    matrix_widget
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## SVD: $A = U \Sigma V^\top$

    $U$ and $V$ are orthogonal matrices, $\Sigma$ is diagonal with singular values. Reveals the rank, range, and null space of a matrix.
    """)
    return


@app.cell
def _(Matrix, matrix_widget, mo, np):
    A = np.array(matrix_widget.matrix)
    U, S, Vt = np.linalg.svd(A)
    Sigma = np.diag(S)

    mo.hstack([
        Matrix(matrix=A.tolist(), static=True),
        mo.md("# $=$"),
        Matrix(matrix=U.tolist(), static=True),
        mo.md("# $\\times$"),
        Matrix(matrix=Sigma.tolist(), static=True),
        mo.md("# $\\times$"),
        Matrix(matrix=Vt.tolist(), static=True),
    ], justify="start", align="center")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## QR: $A = QR$

    $Q$ is orthogonal, $R$ is upper triangular. Used for solving least squares problems and computing eigenvalues.
    """)
    return


@app.cell
def _(Matrix, matrix_widget, mo, np):
    A_qr = np.array(matrix_widget.matrix)
    Q, R = np.linalg.qr(A_qr)

    mo.hstack([
        Matrix(matrix=A_qr.tolist(), static=True),
        mo.md("# $=$"),
        Matrix(matrix=Q.tolist(), static=True),
        mo.md("# $\\times$"),
        Matrix(matrix=R.tolist(), static=True, triangular="upper"),
    ], justify="start", align="center")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## LU: $A = PLU$

    $P$ is a permutation matrix, $L$ is lower triangular, $U$ is upper triangular. Efficient for solving linear systems via forward/backward substitution.
    """)
    return


@app.cell
def _(Matrix, matrix_widget, mo, np):
    from scipy.linalg import lu
    A_lu = np.array(matrix_widget.matrix)
    P, L, U_lu = lu(A_lu)

    mo.hstack([
        Matrix(matrix=A_lu.tolist(), static=True),
        mo.md("# $=$"),
        Matrix(matrix=P.tolist(), static=True),
        mo.md("# $\\times$"),
        Matrix(matrix=L.tolist(), static=True, triangular="lower"),
        mo.md("# $\\times$"),
        Matrix(matrix=U_lu.tolist(), static=True, triangular="upper"),
    ], justify="start", align="center")
    return


if __name__ == "__main__":
    app.run()
