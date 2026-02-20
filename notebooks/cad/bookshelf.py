# /// script
# requires-python = "==3.12"
# dependencies = [
#     "build123d==0.10.0",
#     "marimo>=0.19.7",
#     "marimo-cad==0.1.0",
# ]
# ///
"""Parametric Bookshelf - marimo-cad example."""

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import marimo_cad as cad
    from build123d import Box, Pos


@app.cell
def _():
    mo.md("""
    # Parametric Bookshelf

    Adjust sliders - camera position is preserved!
    """)
    return


@app.cell
def _():
    shelf_slider = mo.ui.slider(2, 8, value=4, label="Shelves")
    height_slider = mo.ui.slider(60, 200, value=120, label="Height (cm)")
    return height_slider, shelf_slider


@app.cell
def _():
    viewer = cad.Viewer()
    return (viewer,)


@app.cell
def _(height_slider, shelf_slider, viewer):
    # Fixed dimensions (cm)
    WIDTH = 80
    DEPTH = 30
    SIDE_T = 2
    SHELF_T = 2
    BACK_T = 1

    # Colors
    SIDE_COLOR = "#8B4513"
    SHELF_COLOR = "#DEB887"
    BACK_COLOR = "#A0522D"

    def build_bookshelf(shelf_count: int, height: int) -> list:
        """Build bookshelf parts from parameters."""
        parts = []
        inner_width = WIDTH - 2 * SIDE_T

        # Left side
        left = Pos(-WIDTH / 2 + SIDE_T / 2, 0, height / 2) * Box(SIDE_T, DEPTH, height)
        parts.append({"shape": left, "name": "Left Side", "color": SIDE_COLOR})

        # Right side
        right = Pos(WIDTH / 2 - SIDE_T / 2, 0, height / 2) * Box(SIDE_T, DEPTH, height)
        parts.append({"shape": right, "name": "Right Side", "color": SIDE_COLOR})

        # Back panel (at +Y so front faces camera)
        back = Pos(0, DEPTH / 2 - BACK_T / 2, height / 2) * Box(inner_width, BACK_T, height)
        parts.append({"shape": back, "name": "Back", "color": BACK_COLOR})

        # Top panel
        top = Pos(0, 0, height - SHELF_T / 2) * Box(inner_width, DEPTH, SHELF_T)
        parts.append({"shape": top, "name": "Top", "color": SIDE_COLOR})

        # Bottom shelf
        bottom = Pos(0, 0, SHELF_T / 2) * Box(inner_width, DEPTH, SHELF_T)
        parts.append({"shape": bottom, "name": "Bottom", "color": SHELF_COLOR})

        # Internal shelves
        if shelf_count > 2:
            spacing = (height - SHELF_T) / (shelf_count - 1)
            for i in range(1, shelf_count - 1):
                z = SHELF_T / 2 + i * spacing
                shelf = Pos(0, 0, z) * Box(inner_width, DEPTH, SHELF_T)
                parts.append({"shape": shelf, "name": f"Shelf {i}", "color": SHELF_COLOR})

        return parts

    parts = build_bookshelf(shelf_slider.value, height_slider.value)
    viewer.render(parts)

    mo.vstack([mo.hstack([shelf_slider, height_slider]), viewer])
    return


if __name__ == "__main__":
    app.run()
