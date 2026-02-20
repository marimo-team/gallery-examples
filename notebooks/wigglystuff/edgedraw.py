# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "wigglystuff",
# ]
# ///

import marimo

__generated_with = "0.19.11"
app = marimo.App()

with app.setup:
    import marimo as mo
    from wigglystuff import EdgeDraw


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## EdgeDraw

    We created this widget to make it easy to dynamically draw a graph.
    """)
    return


@app.cell
def _():
    widget = mo.ui.anywidget(EdgeDraw(["a", "b", "c", "d"], directed=True))
    widget
    return (widget,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The widget has all sorts of useful attributes and properties that you can retreive. These update as you interact with the widget.
    """)
    return


@app.cell
def _(widget):
    widget.names
    return


@app.cell
def _(widget):
    widget.links
    return


@app.cell
def _(widget):
    widget.get_adjacency_matrix()
    return


@app.cell
def _(widget):
    widget.get_neighbors("c")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Cycle Detection

    The widget can detect cycles in the graph. You can specify whether to treat the graph as directed or undirected.
    """)
    return


@app.cell
def _(widget):
    widget.has_cycle(directed=False), widget.has_cycle(directed=True)
    return


if __name__ == "__main__":
    app.run()
