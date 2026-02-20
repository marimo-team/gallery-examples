# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "wigglystuff",
# ]
# ///

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="columns", sql_output="polars")

with app.setup:
    import marimo as mo
    from wigglystuff import SortableList


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## `SortableList`

    This widget lets you maintain a list that you can sort around.
    """)
    return


@app.cell
def _():
    widget = mo.ui.anywidget(
        SortableList(
            ["a", "b", "c"],
            editable=True,
            addable=True,
            removable=True,
            label="My Sortable List"
        )

    )
    widget
    return (widget,)


@app.cell
def _(widget):
    widget.value.get("value")
    return


if __name__ == "__main__":
    app.run()
