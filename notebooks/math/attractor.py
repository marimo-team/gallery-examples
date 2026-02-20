# /// script
# requires-python = "==3.10"
# dependencies = [
#     "jupyter-scatter",
#     "marimo",
#     "numpy",
#     "geoindex_rs>=0.2.1",
# ]
# requires-python = ">=3.11"
# ///

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import jscatter
    import numpy as np
    from math import inf


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Try panning and zooming on the scatter plot, then **click and hold** to make a selection with your mouse and get the points back in Python!
    """)
    return


@app.cell
def _():
    points = roessler_attractor(num=1_000_000)
    scatter = jscatter.Scatter(points[:, 0], points[:, 1], height=800)
    widget = mo.ui.anywidget(scatter.widget)
    widget
    return points, widget


@app.cell
def _(points, widget):
    points[widget.selection, :]
    return


@app.function(hide_code=True)
def roessler_attractor(num: int) -> np.array:
    points = []

    xn = 2.644838333129883
    yn = 4.060488700866699
    zn = 2.8982460498809814
    a = 0.2
    b = 0.2
    c = 5.7
    dt = 0.006

    minX = inf
    maxX = -inf
    minY = inf
    maxY = -inf
    for i in range(num):
        dx = -yn - zn
        dy = xn + a * yn
        dz = b + zn * (xn - c)

        xh = xn + 0.5 * dt * dx
        yh = yn + 0.5 * dt * dy
        zh = zn + 0.5 * dt * dz

        dx = -yh - zh
        dy = xh + a * yh
        dz = b + zh * (xh - c)

        xn1 = xn + dt * dx
        yn1 = yn + dt * dy
        zn1 = zn + dt * dz

        points.append([xn1, yn1])

        minX = min(xn1, minX)
        maxX = max(xn1, maxX)
        minY = min(yn1, minY)
        maxY = max(yn1, maxY)

        xn = xn1
        yn = yn1
        zn = zn1

    dX = maxX - minX
    dY = maxY - minY

    for i in range(num):
        points[i][0] -= minX
        points[i][0] /= dX / 2
        points[i][0] -= 1
        points[i][1] -= minY
        points[i][1] /= dY / 2
        points[i][1] -= 1

    return np.array(points)


if __name__ == "__main__":
    app.run()
