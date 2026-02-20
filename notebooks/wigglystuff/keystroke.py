# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "wigglystuff",
# ]
# ///

import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    from wigglystuff import KeystrokeWidget


@app.cell
def _():
    listener = mo.ui.anywidget(KeystrokeWidget())
    return (listener,)


@app.cell
def _():
    mo.md(r"""
    ## Capture shortcuts from the browser

    Click the widget, then press any key combination (e.g. `Ctrl + Shift + P`).
    The latest shortcut is synced to Python through the `last_key` trait.
    """)
    return


@app.cell
def _(listener):
    listener
    return


@app.cell
def _(listener):
    info = listener.last_key or {}

    modifiers = [
        label
        for key, label in [
            ("ctrlKey", "Ctrl"),
            ("shiftKey", "Shift"),
            ("altKey", "Alt"),
            ("metaKey", "Meta"),
        ]
        if info.get(key)
    ]

    shortcut = " + ".join(modifiers + [info.get("key", "—")]).strip(" + ")
    rows = {
        "Shortcut": shortcut or "—",
        "Code": info.get("code", "—"),
        "Timestamp": info.get("timestamp", "—"),
    }

    mo.callout("Latest keyboard event from the browser:")
    mo.md("\n".join(f"- **{label}:** `{value}`" for label, value in rows.items()))
    return


if __name__ == "__main__":
    app.run()
