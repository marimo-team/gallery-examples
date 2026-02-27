# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.20.2",
#     "matplotlib==3.10.8",
#     "numpy==2.4.2",
# ]
# ///

import marimo

__generated_with = "0.20.2"
app = marimo.App(
    css_file="/usr/local/_marimo/custom.css",
    auto_download=["html"],
)


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    return mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Bak-Sneppen Model — Interactive 3D Visualization

    ## What is Self-Organized Criticality?

    In nature, many complex systems, from earthquakes to forest fires to biological evolution,
    seem to spontaneously organize themselves into a **critical state**, poised at the edge between
    order and disorder. This phenomenon is called **Self-Organized Criticality (SOC)**, a concept
    introduced by Per Bak, Chao Tang, and Kurt Wiesenfeld in 1987 [1]. Unlike continuous phase transitions in physics, such as the ferromagnetic transition, which occurs only at a precise critical temperature, SOC systems reach their critical state without any fine-tuning of external parameters: they drive themselves there naturally.

    ## The Bak-Sneppen Model

    The **Bak-Sneppen model** (1993) [2] is one of the simplest and most elegant models of SOC. It was
    designed to capture a key feature of biological evolution: species don't evolve in isolation: when one species changes, its neighbors in the ecosystem are affected too.

    Here's how the model works:

    1. **Setup**: imagine a ring of species, each assigned a random **fitness** value between 0 and 1.
       Higher values mean the species is better adapted to its environment.
    2. **Evolution step**: at each time step, find the species with the **lowest fitness** — the
       weakest link in the ecosystem. Replace its fitness with a new random value. Crucially, also
       replace the fitness of its **two nearest neighbors**, simulating the ecological disruption
       caused by one species changing.
    3. **Repeat**: keep iterating this simple rule.

    ## What Emerges?

    After many steps, something remarkable happens: the system **self-organizes** into a critical
    state where almost all fitness values sit above a threshold of approximately **0.667**. Species
    below this threshold are quickly eliminated and replaced. This emergent threshold was not built
    into the rules, it arises spontaneously from the dynamics.

    This is analogous to **punctuated equilibrium** in evolutionary biology: long periods of
    stability interrupted by bursts of change (avalanches of co-evolutionary activity).

    ## How to Use This Notebook

    - Adjust the **number of species** and **steps per click** using the sliders on the left
    - Press **Simulate** to advance the evolution
    - Watch how the fitness landscape (the 3D stem plot) evolves over time
    - Use the **camera controls** to rotate and tilt the view
    - Notice how, after enough steps, the stems cluster near the top — that's the self-organized critical state!
    """)
    return


@app.cell(hide_code=True)
def _(
    azimuth_slider,
    controls,
    elevation_slider,
    get_state,
    mo,
    np,
    plt,
    total_steps_get,
):
    fitness = get_state()
    _n_creatures = len(fitness)
    _theta = np.linspace(0, 2 * np.pi, _n_creatures, endpoint=False)
    _x = np.cos(_theta)
    _y = np.sin(_theta)

    _bg_color = "#1d252b"
    _dr_color = "#fbfcfc"

    _fig = plt.figure(figsize=(7, 7))
    _fig.set_facecolor(_bg_color)
    _ax = _fig.add_subplot(projection="3d")
    _ax.set_facecolor(_bg_color)

    _markerline, _stemlines, _baseline = _ax.stem(_x, _y, fitness, linefmt=":w")
    _stemlines.set_linewidths(0.5)
    _markerline.set_markerfacecolor(_dr_color)
    _markerline.set_markeredgecolor("#07BEB8")
    _markerline.set_markeredgewidth(0)
    _markerline.set_markersize(4)
    _baseline.set_color("#FF8552")
    _baseline.set_linewidth(2)
    _ax.set_xlim([-1, 1])
    _ax.set_ylim([-1, 1])
    _ax.set_zlim([0, 1.1])
    _ax.set_aspect("equal")
    _ax.view_init(elevation_slider.value, azimuth_slider.value, 0)
    _ax.axis("off")

    _plot = mo.vstack([
        _fig,
        mo.md(f"**Total steps: {total_steps_get()}**"),
    ])
    mo.hstack([controls, _plot], widths=[1, 3], align="start", gap=2)
    return


@app.cell
def _(mo):
    n_creatures_slider = mo.ui.slider(
        50, 500, value=200, step=10, label="Number of species"
    )
    n_steps_slider = mo.ui.slider(
        1, 50, value=10, step=1, label="Number of steps"
    )
    get_n_steps, set_n_steps = mo.state(10)
    run_button = mo.ui.run_button(label="Simulate")
    elevation_slider = mo.ui.slider(
        10, 60, value=30, step=5, label="Camera elevation"
    )
    azimuth_slider = mo.ui.slider(
        0, 360, value=45, step=5, label="Camera azimuth"
    )
    sim_controls = mo.vstack([
        n_creatures_slider,
        n_steps_slider,
        run_button,
    ])
    camera_controls = mo.vstack([
        elevation_slider,
        azimuth_slider,
    ])
    controls = mo.vstack([
        mo.md("**Simulation**"),
        sim_controls,
        mo.md("**Camera**"),
        camera_controls,
    ])
    return (
        azimuth_slider,
        controls,
        elevation_slider,
        get_n_steps,
        n_creatures_slider,
        n_steps_slider,
        run_button,
        set_n_steps,
    )


@app.cell
def _(n_steps_slider, set_n_steps):
    set_n_steps(n_steps_slider.value)
    return


@app.cell
def _(mo, n_creatures_slider, np):
    get_state, set_state = mo.state(np.random.uniform(0.0, 1.0, n_creatures_slider.value))
    total_steps_get, total_steps_set = mo.state(0)
    return get_state, set_state, total_steps_get, total_steps_set


@app.cell
def _(np):
    def bak_sneppen(fitness, size):
        i = np.argmin(fitness)
        i_update = np.array([i - 1, i, i + 1]) % size
        fitness[i_update] = np.random.uniform(0.0, 1.0, 3)
        return fitness

    return (bak_sneppen,)


@app.cell
def _(
    bak_sneppen,
    get_n_steps,
    get_state,
    mo,
    run_button,
    set_state,
    total_steps_get,
    total_steps_set,
):
    mo.stop(not run_button.value)

    _fitness = get_state().copy()
    _n_steps = get_n_steps()
    for _ in range(_n_steps):
        _fitness = bak_sneppen(_fitness, len(_fitness))
    set_state(_fitness)
    total_steps_set(total_steps_get() + _n_steps)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    © 2026 Simone Conradi. This work is licensed under a
    [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    **References**

    1. P. Bak, C. Tang, K. Wiesenfeld, *Self-organized criticality: An explanation
       of 1/f noise*, Phys. Rev. Lett. **59**, 381 (1987).
       [DOI:10.1103/PhysRevLett.59.381](https://doi.org/10.1103/PhysRevLett.59.381)

    2. P. Bak, K. Sneppen, *Punctuated equilibrium and criticality in a simple
       model of evolution*, Phys. Rev. Lett. **71**, 4083 (1993).
       [DOI:10.1103/PhysRevLett.71.4083](https://doi.org/10.1103/PhysRevLett.71.4083)
    """)
    return


if __name__ == "__main__":
    app.run()
