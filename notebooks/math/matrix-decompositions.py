# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.20.2",
#     "numpy==2.3.5",
#     "scipy==1.17.0",
# ]
# ///

import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np
    from scipy.linalg import lu


@app.cell
def _(slider_cols, slider_rows):
    matrix_widget = mo.ui.matrix(
        np.zeros((slider_rows.value, slider_cols.value)).tolist(),
        min_value=-10,
        max_value=10,
    )
    return (matrix_widget,)


@app.cell
def _():
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
def _():
    mo.md(r"""
    ## SVD: $A = U \Sigma V^\top$

    $U$ and $V$ are orthogonal matrices, $\Sigma$ is diagonal with singular values. Reveals the rank, range, and null space of a matrix.
    """)
    return


@app.cell
def _(matrix_widget):
    A = np.array(matrix_widget.value)
    U, S, Vt = np.linalg.svd(A)
    Sigma = np.diag(S)

    mo.hstack(
        [
            mo.ui.matrix(A.tolist(), disabled=True),
            mo.md("# $=$"),
            mo.ui.matrix(U.tolist(), disabled=True),
            mo.md("# $\\times$"),
            mo.ui.matrix(Sigma.tolist(), disabled=True),
            mo.md("# $\\times$"),
            mo.ui.matrix(Vt.tolist(), disabled=True),
        ],
        justify="start",
        align="center",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## QR: $A = QR$

    $Q$ is orthogonal, $R$ is upper triangular. Used for solving least squares problems and computing eigenvalues.
    """)
    return


@app.cell
def _(matrix_widget):
    A_qr = np.array(matrix_widget.value)
    Q, R = np.linalg.qr(A_qr)

    mo.hstack(
        [
            mo.ui.matrix(A_qr.tolist(), disabled=True),
            mo.md("# $=$"),
            mo.ui.matrix(Q.tolist(), disabled=True),
            mo.md("# $\\times$"),
            mo.ui.matrix(R.tolist(), disabled=True),
        ],
        justify="start",
        align="center",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## LU: $A = PLU$

    $P$ is a permutation matrix, $L$ is lower triangular, $U$ is upper triangular. Efficient for solving linear systems via forward/backward substitution.
    """)
    return


@app.cell
def _(matrix_widget):
    A_lu = np.array(matrix_widget.value)
    P, L, U_lu = lu(A_lu)

    mo.hstack(
        [
            mo.ui.matrix(A_lu.tolist(), disabled=True),
            mo.md("# $=$"),
            mo.ui.matrix(P.tolist(), disabled=True),
            mo.md("# $\\times$"),
            mo.ui.matrix(L.tolist(), disabled=True),
            mo.md("# $\\times$"),
            mo.ui.matrix(U_lu.tolist(), disabled=True),
        ],
        justify="start",
        align="center",
    )
    return


if __name__ == "__main__":
    app.run()
