# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "mohtml",
#     "wigglystuff",
# ]
# ///

import marimo

__generated_with = "0.17.8"
app = marimo.App(width="medium", sql_output="polars")

with app.setup:
    import marimo as mo
    from mohtml import div, img, tailwind_css
    from wigglystuff import WebcamCapture


@app.cell
def _():
    tailwind_css()
    return


@app.cell
def _():
    widget = mo.ui.anywidget(WebcamCapture(interval_ms=1000))
    return (widget,)


@app.cell
def _(widget):
    widget
    return


@app.cell
def _(widget):
    div(
        img(src=widget.image_base64),
        klass="bg-slate-100 border border-slate-200 rounded-2xl p-4",
    )
    return


@app.cell
def _(widget):
    widget.get_pil()
    return


if __name__ == "__main__":
    app.run()
