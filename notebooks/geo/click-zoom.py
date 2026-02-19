# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo",
#     "openlayers==0.1.6",
# ]
# ///

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import openlayers as ol

    return mo, ol


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This notebook shows how to use OpenLayers with marimo. Notice how when you interact with the map, the map state is sent back to Python.
    """)
    return


@app.cell
def _(ol):
    m = ol.MapWidget()
    m.add_click_interaction()
    return (m,)


@app.cell
def _(m, mo):
    widget = mo.ui.anywidget(m)
    return (widget,)


@app.cell
def _(widget):
    widget
    return


@app.cell
def _(widget):
    widget.value["view_state"]
    return


@app.cell
def _(widget):
    widget.value["clicked"]
    return


if __name__ == "__main__":
    app.run()
