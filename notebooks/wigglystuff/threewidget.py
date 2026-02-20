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
    import random
    from wigglystuff import ThreeWidget


@app.cell
def _():
    random.seed(42)
    data = []
    for _ in range(900):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        hex_value = f"#{r:02x}{g:02x}{b:02x}"
        data.append(
            {
                "x": r / 255.0,
                "y": g / 255.0,
                "z": b / 255.0,
                "color": hex_value,
                "size": random.uniform(0.08, 0.2),
            }
        )

    three = ThreeWidget(
        data=data,
        width=640,
        height=420,
        # show_grid=True,
        # show_axes=True,
        # axis_labels=["R", "G", "B"],
    )
    widget = mo.ui.anywidget(three)
    return data, three, widget


@app.cell
def _(reset, shuffle):
    btn_reset = mo.ui.button(on_click=reset, label="reset")
    btn_shuffle = mo.ui.button(on_click=shuffle, label="make some noise")
    [btn_reset, btn_shuffle]
    return


@app.cell
def _(widget):
    widget
    return


@app.cell(hide_code=True)
def _(data, three):
    def shuffle(_):
        updates = []
        for _point in three.data:
            updates.append(
                {
                    "x": _point["x"] + (random.random()-0.5)*0.1,
                    "y": _point["y"] + (random.random()-0.5)*0.3,
                    "z": _point["z"] + (random.random()-0.5)*0.1,
                }
            )
        three.update_points(updates, animate=True, duration_ms=650)

    def reset(_):
        three.update_points(data, animate=True, duration_ms=650)

    return reset, shuffle


@app.cell
def _(widget):
    widget.start_rotate(speed=10.0)
    return


if __name__ == "__main__":
    app.run()
