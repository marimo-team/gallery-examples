# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "wigglystuff",
# ]
# ///

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    from wigglystuff import CopyToClipboard


@app.cell
def _():
    default_snippet = "pip install wigglystuff"
    widget = mo.ui.anywidget(CopyToClipboard(text_to_copy=default_snippet))
    editor = mo.ui.text_area(label="Text to copy", value=default_snippet)
    return editor, widget


@app.cell
def _(widget):
    widget
    return


@app.cell
def _(editor):
    editor
    return


@app.cell
def _(editor, widget):
    widget.text_to_copy = editor.value
    return


@app.cell
def _(widget):
    preview = widget.text_to_copy
    truncated = preview if len(preview) < 80 else preview[:77] + "..."

    mo.callout("Click the button to copy the payload below:")
    mo.md(f"```text\n{truncated}\n```")
    return


if __name__ == "__main__":
    app.run()
