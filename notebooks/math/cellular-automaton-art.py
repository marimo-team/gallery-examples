# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "matplotlib",
#     "numpy",
#     "openai==2.24.0",
#     "pydantic-ai-slim==1.63.0",
#     "scipy",
# ]
# ///

import marimo

__generated_with = "0.20.2"
app = marimo.App()

with app.setup:
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap
    from scipy.ndimage import convolve, gaussian_filter


@app.cell(hide_code=True)
def _():
    mo.md("""
    # Cellular Automaton Art

    This notebook simulates a cellular automaton using a voter model, and uses the
    result to make art with matplotlib. It was inspired by [an art piece](https://x.com/S_Conradi/status/2022222382136766904) made by [Simone Conradi](https://x.com/S_Conradi).
    """)
    return


@app.cell(hide_code=True)
def _():
    form = mo.ui.form(
        mo.ui.dictionary({
            "seed": mo.ui.slider(start=0, stop=200, value=42, show_value=True),
            "n_states": mo.ui.slider(start=4, stop=20, value=14, show_value=True),
            "grid_size": mo.ui.slider(start=100, stop=1000, value=400, step=50, show_value=True),
            "steps": mo.ui.slider(start=50, stop=1000, value=400, step=10, show_value=True),
            "block_size": mo.ui.slider(start=1, stop=32, value=1, show_value=True),
        }, label="parameters"),
        submit_button_label="Make art!",
        bordered=False
    )
    form
    return (form,)


@app.cell(hide_code=True)
def _(form):
    mo.stop(form.value is None, mo.md("Click **Make art!** to cook up a plot"))
    with mo.persistent_cache("art"):
        _v = form.value
        grid = voter_model(_v["grid_size"], _v["n_states"], _v["steps"], _v["seed"], _v["block_size"])
    return (grid,)


@app.cell
def _(grid, plot_grid):
    plot_grid(grid)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The rendering parameters below affect the look and feel of the plot, without changing the simulated grid.
    """)
    return


@app.cell(hide_code=True)
def _():
    smoothing = mo.ui.slider(start=0, stop=10, value=6, label="Smoothing", show_value=True)
    blur = mo.ui.slider(start=0.0, stop=8.0, value=0.0, step=0.5, label="Gaussian blur", show_value=True)
    alpha = mo.ui.slider(start=0.0, stop=1.0, value=0.65, step=0.05, label="Smooth alpha", show_value=True)
    palette = mo.ui.dropdown(
        options=["Blue-Cream-Red", "Viridis", "Twilight", "Ocean-Earth", "Neon"],
        value="Blue-Cream-Red",
        label="Palette",
    )
    mo.hstack([smoothing, blur, alpha, palette], wrap=True, justify="start")
    return alpha, blur, palette, smoothing


@app.cell
def _(alpha, blur, palette, smoothing):
    def plot_grid(grid):
        _n = int(grid.max()) + 1
        display_grid = smooth_grid(grid, smoothing.value, _n)
        display_grid = blur_grid(display_grid, blur.value, _n)

        cmap = build_cmap(palette.value, _n)
        fig, ax = plt.subplots(figsize=(10, 10), facecolor="white")
        ax.imshow(grid, cmap=cmap, interpolation="nearest")
        ax.imshow(display_grid, cmap=cmap, interpolation="nearest", alpha=alpha.value)
        ax.set_axis_off()
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        return fig

    return (plot_grid,)


@app.function
def voter_model(
    size: int, n: int, num_steps: int, rng_seed: int, block_size: int = 1
) -> np.ndarray:
    """Each cell copies a random Moore-neighbour's state."""
    rng = np.random.default_rng(rng_seed)
    if block_size > 1:
        small = rng.integers(
            0, n, size=(size // block_size + 1, size // block_size + 1)
        )
        grid = np.repeat(
            np.repeat(small, block_size, axis=0), block_size, axis=1
        )[:size, :size].copy()
    else:
        grid = rng.integers(0, n, size=(size, size))
    shifts = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]
    for i in mo.status.progress_bar(
        range(num_steps), subtitle="Simulating automaton ...", completion_subtitle="Done!"
    ):
        direction = rng.integers(0, 8, size=(size, size))
        new_grid = grid.copy()
        for d, (dr, dc) in enumerate(shifts):
            mask = direction == d
            new_grid[mask] = np.roll(np.roll(grid, -dr, axis=0), -dc, axis=1)[
                mask
            ]
        grid = new_grid
    return grid


@app.function(hide_code=True)
def build_cmap(name: str, n: int) -> ListedColormap:
    """Build a ListedColormap with *n* colours from the chosen palette."""
    palettes = {
        "Blue-Cream-Red": [
            "#3060ff",
            "#50a0ff",
            "#60d0ff",
            "#80eeff",
            "#c0ffff",
            "#fffff0",
            "#ffee60",
            "#ffb040",
            "#ff6060",
            "#ff4040",
            "#ee2020",
        ],
        "Ocean-Earth": [
            "#2080ff",
            "#00c8f0",
            "#00f0d0",
            "#40ffa0",
            "#a0ff60",
            "#ffff50",
            "#ffc020",
            "#ff7030",
            "#ff4060",
            "#f020a0",
        ],
        "Neon": [
            "#8040ff",
            "#c040ff",
            "#ff40ff",
            "#ff40a0",
            "#ff4060",
            "#ff8020",
            "#ffc000",
            "#e0ff00",
            "#40ff40",
            "#00ffc0",
        ],
    }
    if name in palettes:
        base = LinearSegmentedColormap.from_list(name, palettes[name], N=256)
        return ListedColormap([base(i / max(n - 1, 1)) for i in range(n)])
    # Fall back to matplotlib built-in colormaps
    builtin = {"Viridis": "viridis", "Twilight": "twilight"}
    base = plt.get_cmap(builtin.get(name, "viridis"))
    return ListedColormap([base(i / max(n - 1, 1)) for i in range(n)])


@app.function
def smooth_grid(grid: np.ndarray, radius: int, n_states: int) -> np.ndarray:
    """Mode-filter: replace each cell with the most common state in a (2r+1)x(2r+1) window."""
    if radius == 0:
        return grid
    kernel = np.ones((2 * radius + 1, 2 * radius + 1))
    counts = np.stack(
        [
            convolve((grid == s).astype(float), kernel, mode="wrap")
            for s in range(n_states)
        ]
    )
    return np.argmax(counts, axis=0).astype(grid.dtype)


@app.function
def blur_grid(grid: np.ndarray, sigma: float, n_states: int) -> np.ndarray:
    """Gaussian-blur each state channel then re-quantize to the dominant state."""
    if sigma == 0:
        return grid
    blurred = np.stack(
        [
            gaussian_filter(
                (grid == s).astype(float), sigma=sigma, mode="wrap"
            )
            for s in range(n_states)
        ]
    )
    return np.argmax(blurred, axis=0).astype(grid.dtype)


if __name__ == "__main__":
    app.run()
