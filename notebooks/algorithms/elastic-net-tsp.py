# /// script
# dependencies = [
#     "drawdata==0.5.1",
#     "marimo",
#     "matplotlib==3.11.0",
#     "numpy==2.5.0",
#     "wigglystuff==0.5.10",
# ]
# requires-python = ">=3.14"
# ///

import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from drawdata import ScatterWidget
    from wigglystuff import PlaySlider

    return PlaySlider, ScatterWidget, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The Elastic Net Method for the Travelling Salesman Problem

    Durbin & Willshaw, *Nature* 326 (1987): [An analogue approach to the
    travelling salesman problem using an elastic net
    method](https://ena-tsp.mathieularose.com/durbin1987.pdf).

    The travelling salesman problem (TSP) asks for the shortest closed tour
    that visits every city exactly once. Durbin & Willshaw's **elastic net**
    solves it with a physical metaphor: imagine a rubber band laid out as a
    small circle near the centroid of the cities. Every point on the band is
    pulled towards nearby cities, while the band's own elasticity keeps it
    smooth and short. As the "pull radius" is slowly shrunk, the band
    stretches out, unfurls, and settles into a tour that passes near every
    city.

    Below, draw a set of cities, run the algorithm, and then **play** through
    its iterations to watch the band unfurl into a tour.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The method

    Let $x_i$ be the position of city $i$, and $y_j$ the position of point
    $j$ on the elastic band (the "path"), with $M$ points forming a closed
    loop. **The $y_j$ are the free variables** — the $2M$ coordinates of the
    path points are what the algorithm adjusts. The cities $x_i$ are fixed
    data; they never move.

    Everything follows from a single energy function of the $y_j$'s:

    $$E(y_1, \dots, y_M) = -\alpha K \sum_i \ln \sum_j \phi(|x_i - y_j|, K) \;+\; \beta \sum_j |y_{j+1} - y_j|^2$$

    The first term is small when every city sits close to *some* path point
    (pulls the band towards cities); the second is small when the band is
    short and smooth (elasticity). Each iteration nudges every path point
    downhill on $E$:

    $$\Delta y_j = -K \frac{\partial E}{\partial y_j} = \alpha \sum_i w_{ij}(x_i - y_j) + \beta K (y_{j+1} - 2y_j + y_{j-1}) \qquad (1)$$

    The weight $w_{ij}$ — how much city $i$ pulls on point $j$ — falls
    straight out of differentiating the log-sum-exp term above; it isn't a
    separate design choice. It's a normalized, distance-based responsibility:

    $$w_{ij} = \frac{\phi(|x_i - y_j|, K)}{\sum_k \phi(|x_i - y_k|, K)}, \qquad \phi(d, K) = \exp\left(-\frac{d^2}{2K^2}\right) \qquad (2)$$

    | Symbol | Meaning |
    |---|---|
    | $x_i$ | position of city $i$ (fixed data) |
    | $y_j$ | position of path point $j$ (the free variable) |
    | $K$ | "radius of influence" of a city over the path |
    | $\alpha$ | strength of the pull towards cities |
    | $\beta$ | strength of the path's own elasticity |

    $K$ starts large ($K=0.2$ on the unit square) so every city pulls on
    every path point roughly equally, keeping the band a smooth, near-circular
    loop. Every $n=25$ iterations $K$ is reduced by 1%, so cities become
    progressively more selective about which point they pull on — this
    gradual sharpening is what unfurls the loop into a tour. This plays the
    same role as lowering the temperature in simulated annealing.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Draw some cities

    Click and drag on the canvas below to place a handful of "cities" (a
    single color is enough — color isn't meaningful here, only position).
    """)
    return


@app.cell(hide_code=True)
def _(ScatterWidget, mo):
    scatter_widget = mo.ui.anywidget(ScatterWidget())
    scatter_widget
    return (scatter_widget,)


@app.cell(hide_code=True)
def _(mo):
    alpha_slider = mo.ui.slider(start=0.05, stop=0.5, step=0.05, value=0.2, label="α (pull to cities)")
    beta_slider = mo.ui.slider(start=0.5, stop=4.0, step=0.1, value=2.0, label="β (path elasticity)")
    run_button = mo.ui.run_button(label="Run elastic net")
    mo.hstack([alpha_slider, beta_slider, run_button])
    return alpha_slider, beta_slider, run_button


@app.cell(hide_code=True)
def _(mo, np, scatter_widget):
    mo.stop(
        len(scatter_widget.widget.data) < 3,
        mo.md("*Draw at least 3 cities above to get started.*"),
    )

    _raw = np.array([[p["x"], p["y"]] for p in scatter_widget.widget.data])
    _mins = _raw.min(axis=0)
    _span = (_raw.max(axis=0) - _mins).max()
    cities = (_raw - _mins) / _span
    cities = cities - cities.mean(axis=0) + 0.5
    n_cities = len(cities)
    return cities, n_cities


@app.cell(hide_code=True)
def _(alpha_slider, beta_slider, cities, mo, np, run_button):
    mo.stop(not run_button.value, mo.md("*Click **Run elastic net** above to compute the tour.*"))


    def elastic_net_history(cities, m_ratio=2.5, alpha=0.2, beta=2.0, k_init=0.2, k_final=0.01, decay=0.99, n=25, settle_iters=500, seed=0):
        """Anneal an elastic net (eq. 1, 2) from a small ring towards a tour.

        Records a snapshot every time K is reduced, then keeps iterating at the
        final K for `settle_iters` more steps so the path fully relaxes onto the
        cities before a tour is read off.
        """
        rng = np.random.default_rng(seed)
        n_points = max(int(round(m_ratio * len(cities))), 5)
        centroid = cities.mean(axis=0)
        theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        radius = 0.1 * (1 + 0.1 * rng.standard_normal(n_points))
        y = centroid + radius[:, None] * np.stack([np.cos(theta), np.sin(theta)], axis=1)

        def step(y, K):
            diff = cities[:, None, :] - y[None, :, :]  # x_i - y_j, shape (N, M, 2)
            phi = np.exp(-np.sum(diff**2, axis=2) / (2 * K**2))
            w = phi / (phi.sum(axis=1, keepdims=True) + 1e-12)  # eq. 2
            pull = np.einsum("ij,ijk->jk", w, diff)  # sum_i w_ij (x_i - y_j)
            neighbor = np.roll(y, -1, axis=0) - 2 * y + np.roll(y, 1, axis=0)
            return y + alpha * pull + beta * K * neighbor  # eq. 1

        history = [y.copy()]
        k_history = [k_init]
        K = k_init
        while K > k_final:
            for _ in range(n):
                y = step(y, K)
            K *= decay
            history.append(y.copy())
            k_history.append(K)

        for i in range(settle_iters):
            y = step(y, k_final)
            if (i + 1) % n == 0:
                history.append(y.copy())
                k_history.append(k_final)
        return history, k_history


    history, k_history = elastic_net_history(cities, alpha=alpha_slider.value, beta=beta_slider.value)
    return history, k_history


@app.cell(hide_code=True)
def _(PlaySlider, history, mo):
    frame_slider = mo.ui.anywidget(
        PlaySlider(min_value=0, max_value=len(history) - 1, step=1, interval_ms=60, loop=False, width=500)
    )
    frame_slider
    return (frame_slider,)


@app.cell(hide_code=True)
def _(cities, frame_slider, history, k_history, np, plt):
    _frame = int(frame_slider.widget.value)
    _y = history[_frame]
    _path = np.vstack([_y, _y[0]])

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(_path[:, 0], _path[:, 1], "-", color="steelblue", linewidth=1.2)
    ax.scatter(cities[:, 0], cities[:, 1], marker="s", color="black", s=25, zorder=5)
    ax.set_title(f"K = {k_history[_frame]:.4f}   (checkpoint {_frame}/{len(history) - 1})")
    ax.set_aspect("equal")
    ax.axis("off")
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## From path to tour

    Once the band has settled, each city is closest to exactly one point on
    it. Reading off cities in the order their closest path point appears
    along the loop gives a discrete tour.
    """)
    return


@app.cell(hide_code=True)
def _(cities, history, mo, n_cities, np, plt):
    _final_y = history[-1]
    _dists = np.linalg.norm(cities[:, None, :] - _final_y[None, :, :], axis=2)
    _nearest_point = np.argmin(_dists, axis=1)
    tour_order = np.argsort(_nearest_point)
    tour = cities[tour_order]
    tour_length = np.sum(np.linalg.norm(np.roll(tour, -1, axis=0) - tour, axis=1))

    fig_tour, ax_tour = plt.subplots(figsize=(5, 5))
    _tour_closed = np.vstack([tour, tour[0]])
    ax_tour.plot(_tour_closed[:, 0], _tour_closed[:, 1], "-o", color="darkorange", markersize=4)
    ax_tour.set_title(f"Tour length = {tour_length:.3f}  ({n_cities} cities)")
    ax_tour.set_aspect("equal")
    ax_tour.axis("off")

    mo.vstack([fig_tour, mo.md(f"Tour visits cities in order: `{tour_order.tolist()}`")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Takeaways

    - The elastic net turns a discrete combinatorial problem (which order to
      visit cities) into a continuous one (where to place points on a loop),
      solved by gradient descent on an energy function.
    - $K$-annealing is the key trick: a soft, ambiguous assignment of cities
      to path points early on avoids bad local minima, and sharpening it
      gradually locks in a tour.
    - The method is cheap per iteration ($O(NM)$) and fully parallel across
      path points — Durbin & Willshaw were originally motivated by biological
      neural mappings, not optimization, which is why the algorithm looks so
      different from typical local-search TSP heuristics (2-opt, simulated
      annealing).
    """)
    return


if __name__ == "__main__":
    app.run()
