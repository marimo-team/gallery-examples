# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "anywidget>=0.9.0",
#     "marimo",
#     "traitlets>=5.0.0",
#     "wigglystuff==0.2.37",
# ]
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    from rpsls_widget.rpsls_widget import RpslsWidget

    return RpslsWidget, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Rock, Paper, Scissors ... and beyond?

    Rock-Paper-Scissors has **3** elements. Rock-Paper-Scissors-Lizard-Spock has **5**.
    Could we make a variant with **4**? Or **6**?

    For a game to be *fair*, every element must beat the same number of others.
    With $n$ elements, each must beat exactly $k = \frac{n - 1}{2}$ others.
    That's only a whole number when $n$ is **odd**.

    Use the slider to explore: odd values produce a perfectly balanced tournament,
    while even values show why perfect balance is impossible.
    """)
    return


@app.cell
def _(RpslsWidget, mo):
    widget = mo.ui.anywidget(RpslsWidget(n=3))
    return (widget,)


@app.cell
def _(widget):
    widget
    return


@app.cell
def _(mo):
    from wigglystuff import KeystrokeWidget

    keystroke = mo.ui.anywidget(KeystrokeWidget())
    keystroke
    return (keystroke,)


@app.cell
def _(keystroke):
    keystroke.last_key["key"]
    return


@app.cell
def _(get_highlight, keystroke, set_highlight, widget):
    names = "abcdefghijklmnopqrstuvwxyz".upper()

    if keystroke.last_key["key"] == "ArrowRight":
        widget.animate_node(1)
        set_highlight(-1)
    if keystroke.last_key["key"] == "ArrowLeft":
        widget.animate_node(-1)
        set_highlight(-1)
    if keystroke.last_key["key"] == "ArrowUp":
        set_highlight(lambda _: (_ + 1) % widget.n)
    if keystroke.last_key["key"] == "ArrowDown":
        set_highlight(lambda _: (_ - 1) % widget.n)
    if keystroke.last_key["key"] == "q":
        set_highlight(-1)
        widget.clear_highlight()

    if get_highlight() >= 0:
        widget.animate_highlight(names[get_highlight()])
    return


@app.cell
def _(get_highlight):
    get_highlight()
    return


@app.cell
def _(mo):
    get_highlight, set_highlight = mo.state(-1)
    return get_highlight, set_highlight


if __name__ == "__main__":
    app.run()
