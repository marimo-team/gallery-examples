# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "mohtml",
#     "wigglystuff",
# ]
# ///

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium", sql_output="polars")

with app.setup:
    import marimo as mo
    from mohtml import div, img, tailwind_css
    from wigglystuff import Paint


@app.cell
def _():
    tailwind_css()
    return


@app.cell
def _():
    widget = mo.ui.anywidget(Paint(height=550))
    return (widget,)


@app.cell
def _(widget):
    widget
    return


@app.cell
def _(widget):
    div(img(src=widget.get_base64()), klass="bg-gray-200 p-4")
    return


@app.cell
def _(widget):
    widget.get_pil()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    You can also draw over existing images with this library, this can be useful when interacting with multimodal LLMs.
    """)
    return


@app.cell
def _():
    redraw_widget = mo.ui.anywidget(
        Paint(
            init_image="https://marimo.io/_next/image?url=%2Fimages%2Fblog%2F8%2Fthumbnail.png&w=1920&q=75"
        )
    )
    return (redraw_widget,)


@app.cell
def _(redraw_widget):
    redraw_widget
    return


@app.cell
def _(redraw_widget):
    redraw_widget.get_pil()
    return


if __name__ == "__main__":
    app.run()
