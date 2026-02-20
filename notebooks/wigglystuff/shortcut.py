# /// script
# requires-python = ">=3.12"
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
    from wigglystuff import KeystrokeWidget


@app.cell
def _():
    widget = mo.ui.anywidget(KeystrokeWidget())
    widget
    return (widget,)


@app.cell
def _(widget):
    widget.value
    return


if __name__ == "__main__":
    app.run()
