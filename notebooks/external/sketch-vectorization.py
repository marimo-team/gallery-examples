# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "sketchy-svg",
#     "marimo",
#     "yarl"
# ]
# ///

import marimo

__generated_with = "0.20.1"
app = marimo.App(width="full")


@app.cell(hide_code=True)
async def _():
    import tempfile
    import urllib.parse
    import urllib.request
    from importlib.util import find_spec
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    _DATA_BASE = "https://raw.githubusercontent.com/rambip/sketch-vectorization/1c5377984b0f3bd207aa636739ac94366708fff9/data"

    if find_spec("js"):
        # WASM (Pyodide): load data over HTTPS via yarl.URL
        from yarl import URL

        from sketchy_svg.onnxruntime_compat import patch_onnx

        await patch_onnx()
        DATA_DIR = URL(_DATA_BASE)
    else:
        # Non-WASM: download data files lazily into a local cache dir so
        # load_normalized / show_example receive a plain filesystem path.
        class _DataDir:
            def __init__(self, base_url, cache_dir):
                self._base = base_url
                self._cache = cache_dir

            def __truediv__(self, relative):
                local = self._cache / relative
                if not local.exists():
                    local.parent.mkdir(parents=True, exist_ok=True)
                    encoded = urllib.parse.quote(str(relative), safe="/")
                    urllib.request.urlretrieve(f"{self._base}/{encoded}", local)
                return local

        _cache_dir = Path(tempfile.gettempdir()) / "sketchy-svg-data-cache"
        _cache_dir.mkdir(exist_ok=True)
        DATA_DIR = _DataDir(_DATA_BASE, _cache_dir)

    from sketchy_svg.prepare import (
        BinarySketchPredictor,
        compute_thickness_map,
        load_normalized,
    )

    return (
        BinarySketchPredictor,
        DATA_DIR,
        compute_thickness_map,
        load_normalized,
        mo,
        np,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # From sketch to vector graphics

    *By [Antonin Peronnet (@rambip)](https://github.com/rambip) — adapted from [rambip/sketch-vectorization](https://github.com/rambip/sketch-vectorization).*

    You probably have already heard about SVG, or Scalable Vector Graphics. It is a file format that represents images as a collection of curves, instead of pixels. This allows to have very small files, and to scale them without losing quality.

    But SVG can be complicated and counter-intuitive to create. Some artists prefer staying on pen and paper because of the expressivity this provide.

    So naturally, people have tried to convert raster images (or photographs) of drawn sketches and convert them to SVG.

    This (this notebook and the library it uses) is our attempt to re-implement the paper [Fidelity vs. Simplicity: a Global Approach to Line Drawing Vectorization](https://www-sop.inria.fr/reves/Basilic/2016/FLB16/fidelity_simplicity.pdf).
    It started up as a School project (Telecom Paris), but I ended up spending a lot of time on it, to create a usable library.

    In this notebook, we will try to give a complete overview of the techniques used and the implementation choices we made.

    We strongly advise to read the paper a first time before looking at the notebook. Except for some details, it is well written and have nice illustrations.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Pipeline

    Let's start with the central idea: we want to take a photography of a sketch as input, and return a SVG made up of a few curves

    ![](https://raw.githubusercontent.com/rambip/sketch-vectorization/1c5377984b0f3bd207aa636739ac94366708fff9/images/illustration_butterfly.png)

    In a nushell, here is what we do:
    - we apply **preprocessing** to translate the image into a black and white mask
    - we create a **skeleton** and convert it to a graph (chains of pixels)
    - we use **hypergraph optimization** to find the optimal curves to approximate these chains of pixels
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here is the image we will use for demonstration purposes:
    """)
    return


@app.function
def show(ax, title, axis=False):
    ax.axis("equal")
    ax.axis(axis)
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    ax.set_title(title)
    return ax


@app.cell
def _(DATA_DIR, load_normalized, plt):
    img = load_normalized(DATA_DIR / "sketches/butterfly.png", size=256)
    plt.imshow(img, cmap="binary")
    plt.colorbar()
    show(plt.gca(), "original image", axis=True)
    return (img,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Preprocessing

    Before doing anything, we need a binary representation of our drawing: black and white pixels, not gray.

    The paper uses advanced trapped-ball and paint-filling techniques, but that would be a lot of work, to be honnest. And not that interesting !

    Because I'm lazy, I was wondering: what if we cloud cheat ? Going from someone's drawing to a logo-like black and white image sounds a lot like a [Style transfer](https://en.wikipedia.org/wiki/Neural_style_transfer):

    ![](https://s3.amazonaws.com/book.keras.io/img/ch8/style_transfer.png)

    We are in 2025 after all, since we don't have a huge brain and a lot of time, maybe we could use a GPU and a lot of data instead ?

    So, that's what I did !


    Basically, I followed a simple recipe:
    1. Get a good quality SVG dataset ([this one](https://huggingface.co/datasets/OmniSVG/MMSVG-Icon))
    2. Convert the SVG to a sketch-like black and white representation
    3. Add different kind of noises and texture to simulate drawing on paper
    4. Train a CNN to denoise the images.

    This was not very hard conceptually, if you're curious you can look at the notebooks "cnn.py" and "svg_dataset.py" in one of these places:
    - https://github.com/rambip/sketch-vectorization/tree/main/notebooks/ -> the code
    - https://github.com/rambip/sketch-vectorization/tree/main/notebooks/__marimo__ -> the exported notebooks
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    _details = mo.as_html(
        mo.md(r"""
    - the noise must really look like your test images. Otherwise, you're too much out of distribution and the model can't learn properly.
    - There is a tradeoff for the resolution: we want a model that can process large images, but it's more costly to train.
    - **Don't try a small model**. I thought that a model with an inner dimension of 8 and 4 layers was enough, but it was not the case. Don't fear the overfitting: if you train for long enough and you have diverse enough datapoints, your model will generalize even if it has a lot of parameters. If that seems counter-intuitive to you, go read about [Double Descent](https://en.wikipedia.org/wiki/Double_descent)
    - residual connections work really well. We ended up adding them at each layer.
    - The loss is key. We started with BCE, but the model had too much false negative, which broke the connectivity of the figure. We used our custom loss instead, $L = L_1 + \lambda L_2$ where $L_1$ penalized false negatives and $L_2$ penalized overall positives.

    We ended up with:
    - 3000 data points of 256x256
    - 8 layers of 3x3 convolutions, with a direct residual connection from the input to the output
    - an inner dimension of 32
    - SiLU activation function
    - 50 epochs, with a batch size of 50
    - running on T4 gpus for something like 20min
    """)
    )
    mo.md(f"""
    <details>
    <summary>
    In a nutshell, a few lessons we learned:
    </summary>
    {_details.text}
    </details>

    But now, let's try it !

    We converted it to ONNX format, so it can run without a gpu.
    """)
    return


@app.cell
async def _(BinarySketchPredictor, img, plt):
    classifier = BinarySketchPredictor(0.5, gaussian_blur_sigma=1)
    proba = await classifier.predict_proba(img)
    plt.imshow(proba, cmap="binary")
    plt.colorbar()
    show(plt.gca(), "Predicted probability for each pixel")
    return (classifier,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Since the model is not perfect, there are still a few artefacts in the image.
    To remove them, we use a simple gaussian filter before thresholding.
    """)
    return


@app.cell
async def _(classifier, img, plt):
    img_binary = await classifier.predict(img)
    plt.imshow(img_binary, cmap="binary")
    show(plt.gca(), title="Predicted pixels")
    return (img_binary,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next, we compute the thickness of the drawing using erosion. For each pixel $P$, we want to get the number of time we have to "erode" the image before $P$ disappear.

    > Note: erosion consists in taking the minimum value of all pixels in a 3x3 neighborhood.
    """)
    return


@app.cell
def _(compute_thickness_map, img_binary, plt):
    thickness_map = compute_thickness_map(img_binary)
    plt.imshow(thickness_map)
    plt.colorbar()
    show(plt.gca(), "Thickness of the drawing")
    return (thickness_map,)


@app.cell(hide_code=True)
def _(mo):
    _skeleton_illustration = mo.Html("""
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 450 150" width="400px">
      <defs>
        <style>
          .grid-bg { fill: #e0e0e0; }
          .pixel { fill: black; stroke: #333; stroke-width: 1; }
          .touch { fill: #888; }
        </style>
      </defs>

      <!-- Grid 1 -->
      <g transform="translate(10, 10)">
        <rect class="grid-bg" width="130" height="130" rx="5"/>
        <g transform="translate(25, 25)">
          <!-- Pixels -->
          <rect class="pixel" x="28" y="28" width="24" height="24"/>
          <rect class="pixel" x="54" y="28" width="24" height="24"/>
          <rect class="pixel" x="2" y="54" width="24" height="24"/>
          <!-- Touch points -->
          <circle class="touch" cx="53" cy="40" r="5"/>
          <circle class="touch" cx="27" cy="53" r="5"/>
        </g>
      </g>

      <!-- Grid 2 -->
      <g transform="translate(160, 10)">
        <rect class="grid-bg" width="130" height="130" rx="5"/>
        <g transform="translate(25, 25)">
          <!-- Pixels -->
          <rect class="pixel" x="54" y="2" width="24" height="24"/>
          <rect class="pixel" x="28" y="28" width="24" height="24"/>
          <rect class="pixel" x="2" y="54" width="24" height="24"/>
          <!-- Touch points -->
          <circle class="touch" cx="53" cy="27" r="5"/>
          <circle class="touch" cx="27" cy="53" r="5"/>
        </g>
      </g>

      <!-- Grid 3 -->
      <g transform="translate(310, 10)">
        <rect class="grid-bg" width="130" height="130" rx="5"/>
        <g transform="translate(25, 25)">
          <!-- Pixels -->
          <rect class="pixel" x="2" y="28" width="24" height="24"/>
          <rect class="pixel" x="28" y="28" width="24" height="24"/>
          <rect class="pixel" x="54" y="28" width="24" height="24"/>
          <!-- Touch points -->
          <circle class="touch" cx="27" cy="40" r="5"/>
          <circle class="touch" cx="53" cy="40" r="5"/>
        </g>
      </g>
    </svg>
    """)
    mo.md(rf"""
    ## Skeleton and topological graph

    Once we have a nice black and white image, we can create the skeleton very easily using iterative thining.

    > A **skeleton** is a new binary image where all pixels have only 2 neighbors, like this:

    > {_skeleton_illustration}

    Hopefuly, this is already implemented in `scikit-learn`.
    """)
    return


@app.cell
def _():
    from scipy import signal
    from skimage.morphology import skeletonize

    from sketchy_svg.bezier import fit_bezier, interpolate_bezier
    from sketchy_svg.topology import (
        extract_chains,
        refine_all_chains,
        remove_parasite_chains,
    )

    return (
        extract_chains,
        fit_bezier,
        interpolate_bezier,
        refine_all_chains,
        remove_parasite_chains,
        signal,
        skeletonize,
    )


@app.cell
def _(img_binary, plt, skeletonize):
    skeleton = skeletonize(img_binary, method="zhang")

    plt.imshow(skeleton, cmap="binary")
    show(plt.gca(), "Skeleton of the sketch")
    return (skeleton,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Once we have this graph, we traverse it in order to get the pixel chains.

    This is not explained explicitely in the original paper, but the trick is to look at the **number of neighbours** in the skeleton. Indeed, most nodes in the skeleton have only 2 neighbours. If it is not the case, we call it a "special" pixel or an "endpoint".
    """)
    return


@app.cell
def _(np, plt, signal, skeleton):
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    neighbours = signal.convolve2d(skeleton.astype(int), kernel, mode="same")
    neighbours = neighbours * skeleton

    def pos_where(mask):
        return np.where(mask)[1], np.where(mask)[0]

    plt.imshow(1 - skeleton, cmap="gray")
    plt.scatter(
        *pos_where(neighbours == 1),
        label="1 neighbour",
        s=5,
    )
    plt.scatter(
        *pos_where(neighbours == 3),
        label="3 neighbours",
        s=5,
    )
    plt.scatter(
        *pos_where(neighbours > 3),
        label=">3 neighbours",
        s=5,
    )
    plt.legend()
    show(plt.gca(), "skeleton with number of neighbours")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We use this information to perform a graph traversal, from arbitrary pixels to these "special" pixels.
    For the special case of loops, we break them into a chain that ends where it starts.

    We get a list of pixel "chains":

    > Note: we have to be careful about "clusters". If you zoom in on the previous image, you will see clusters of pixels with at least 3 neighbours. We merge them into one single mega-pixel when it's the case.
    """)
    return


@app.cell
def _(extract_chains, plt, skeleton):
    pixel_chains = extract_chains(skeleton)
    _ax = plt.gca()
    for chain in pixel_chains:
        # chain is shape (N, 2) where each row is [row, col] or [y, x]
        _ax.plot(chain[:, 1], chain[:, 0], "-o", markersize=2, linewidth=1)

    show(_ax, "Pixel chains")
    return (pixel_chains,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's remove the pixel chains that are too small:

    > Note: we only remove small pixel chains that don't affect the *connectivity*, like chains that have a dead end or multiple chains that start and end at the same place.
    """)
    return


@app.cell
def _(pixel_chains, plt, remove_parasite_chains):
    pixel_chains_clean = remove_parasite_chains(pixel_chains, min_length=10)
    ax_chain_clean = plt.gca()
    for _chain in pixel_chains_clean:
        # chain is shape (N, 2) where each row is [row, col] or [y, x]
        ax_chain_clean.plot(_chain[:, 1], _chain[:, 0], "-o", markersize=2, linewidth=1)

    show(ax_chain_clean, "Pixel chains (cheaned)")
    return ax_chain_clean, pixel_chains_clean


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now that he have a clean skeleton, the goal will be to approximate as well as possible using [Bezier curves](https://pomax.github.io/bezierinfo/).

    This was not explicit in the paper, but fitting a bezier on a chain of pixel is pretty cheap once you have the right representation.

    You want to minimize $\mathcal L(\omega) = \sum_t ||B(\omega)_t - C_t||^2$ where $\omega$ are the 8 parameters of our bezier curve, $B(\omega)$ is the list of positions of the bezier with parameters $\omega$, and $C$ is the list of positions in our pixel chains.

    For a fixed $t$, we know that $\omega \to B(\omega)_t$ is linear, and we can construct the transformation matrix pretty easily. This means that the problems boils down to a **linear least-square problem** (see [this wiki page](https://en.wikipedia.org/wiki/Linear_least_squares#Applications) for more information)
    """)
    return


@app.cell
def _(fit_bezier, interpolate_bezier, np):
    def show_fitted(ax, chain):
        # Fit Bézier curve
        instants = np.linspace(0, 1, len(chain))
        control_points = fit_bezier(chain, instants, degree=3)
        bezier_curve = interpolate_bezier(control_points, instants)

        # Plot the original pixels
        ax.plot(chain[:, 1], chain[:, 0], "o", markersize=3, alpha=0.3, color="gray")

        # Plot the Bézier curve
        ax.plot(
            bezier_curve[:, 1],
            bezier_curve[:, 0],
            "-",
            linewidth=1,
            alpha=1,
            color="red",
        )

    return (show_fitted,)


@app.cell
def _(pixel_chains_clean, plt, show_fitted):
    ax_fitted = plt.gca()
    for _chain in pixel_chains_clean:
        show_fitted(ax_fitted, _chain)
    show(ax_fitted, "Fitted Bézier Curves")
    return (ax_fitted,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This already looks pretty good, but we see that some curves fit better than other. This is because in our skeleton, some chains of pixels where very long and with complicated shapes.
    To solve this problem, we split our pixel chains into pieces when the fitting error is too high.

    > Note that we modified the original metric used for fitting. Instead of using $\sum_{i=1}^n \|p_i - q_i\|^2$ (total distance between the 2 pixel chains), we use :

    > $\displaystyle \frac{\sum_{i=1}^n \|p_i - q_i\|^2}{\ln(n)}$
    >
    > This ensures that long chains that are almost perfect will not be split, and so we get fewer refined parts.

    Here is what we get:
    """)
    return


@app.cell
def _(pixel_chains_clean, plt, refine_all_chains):
    pixel_chains_refined = refine_all_chains(pixel_chains_clean, tolerance=2)
    ax_chain_refined = plt.gca()
    for _chain in pixel_chains_refined:
        # chain is shape (N, 2) where each row is [row, col] or [y, x]
        ax_chain_refined.plot(
            _chain[:, 1], _chain[:, 0], "-o", markersize=2, linewidth=1
        )
    show(ax_chain_refined, "Pixel chains (refined)")
    return ax_chain_refined, pixel_chains_refined


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The pixel chains seem simpler this way. Let's check that the curves fit better this time:
    """)
    return


@app.cell
def _(
    ax_chain_clean,
    ax_chain_refined,
    ax_fitted,
    mo,
    pixel_chains_refined,
    plt,
    show_fitted,
):
    ax_fitted_refined = plt.gca()
    for _chain in pixel_chains_refined:
        show_fitted(ax_fitted_refined, _chain)

    mo.vstack(
        [
            mo.hstack(
                [
                    show(ax_chain_clean, "Pixel chains before refine"),
                    show(ax_fitted, "Fitted curves before refine"),
                ]
            ),
            mo.hstack(
                [
                    show(ax_chain_refined, "Pixel chains after refine"),
                    show(ax_fitted_refined, "Fitted curves after refine"),
                ]
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now, we have to implement what is probably the most difficult part of the paper: the global optimisation.

    Here is the intuition: if there are multiple chains of pixels that could be joined into a nice curve, we do it. But we have to be careful about which chains we join !


    We will manipulate sequences of edges, that we call **hyperedges**.
    This optimisation process is based on the idea of doing **perturbations** on these sequences.

    Since the paper leaves room for interpretation on how to implement, we implemented it in a slightly different (but similar) way as they describe.

    We first sample pairs of hyperedges $(U, V)$ such that $U$ end at a node that is anywhere inside $V$. We call such a configuation a "T" configuration.

    ![](https://raw.githubusercontent.com/rambip/sketch-vectorization/1c5377984b0f3bd207aa636739ac94366708fff9/images/schema_t.svg)

    Note that hyperedges are oriented. Once we sample one "T", we can apply 6 different transformations:

    - increase or decrease the degree of $V$
    - splitting V into 2 (if it has more than 3 nodes)
    - overlaping V onto U (here, $V$ will become $(y, x, c, d)$)
    - dissociating V from U (for that, the last edge of V must be in U)
    - merging V and U. This is only possible when the T is in the special configuration showed below:

    ![](https://raw.githubusercontent.com/rambip/sketch-vectorization/1c5377984b0f3bd207aa636739ac94366708fff9/images/schema_t2.svg)

    We add a special transformation: **reverse**. As the name suggests, we reverse the order of all nodes in the sequence.

    With the addition of this transformation, this is completely equivalent to the transformations proposed by the paper.


    We try to chose the same parameters as the paper each time we can, to be able to compare results.
    """)
    return


@app.cell
def _():
    from sketchy_svg.demo import OptimPlayBack
    from sketchy_svg.optim import SketchOptimizer, align_boundaries

    return OptimPlayBack, SketchOptimizer, align_boundaries


@app.cell
def _(SketchOptimizer, mo, pixel_chains_refined, thickness_map):
    lam = 0.6  # interpolation between fidelity and simplicity
    mu = 0.2  # penalty for high degree
    t_decrease = 0.9997  # rate of temperature decrease
    t_min = 0.001
    optim = SketchOptimizer(
        lam=lam,
        t_min=t_min,
        mu=mu,
        t_decrease=t_decrease,
        status_function=lambda x, title: mo.status.progress_bar(x, title=title),
    )
    curves = optim.fit_transform(pixel_chains_refined, thickness_map)
    return curves, optim


@app.cell
def _(optim, plt):
    plt.plot(range(len(optim.error_)), optim.error_)
    plt.ylabel("Energy")
    plt.xlabel("steps")
    plt.title("Evolution of energy during optimization")
    plt.show()
    return


@app.cell
def _(optim, plt):
    from collections import Counter

    from sketchy_svg.optim import Perturbation

    counts = Counter(x[0] for x in optim.history_)
    plt.figure(figsize=[12, 6])
    plt.bar(range(7), [counts[i] for i in range(7)])
    plt.xticks(range(7), [x.name for x in Perturbation], rotation=0, fontsize=8)
    plt.gca()
    return


@app.cell
def _(curves, interpolate_bezier, np, plt):
    instants = np.linspace(0, 1, 100)
    _ax = plt.gca()
    for i, _c in enumerate(curves):
        bezier_curve = interpolate_bezier(_c.control_points, instants)

        # Plot the Bézier curve
        _ax.plot(bezier_curve[:, 1], bezier_curve[:, 0], "-", linewidth=1, alpha=1)
        _ax.text(
            bezier_curve[:, 1].mean(), bezier_curve[:, 0].mean(), str(i), fontsize=10
        )
    show(_ax, "final curves")
    return (instants,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If you are viewing the interactive version, you can view the entire history of transformations done to the network of curves:
    """)
    return


@app.cell
def _(OptimPlayBack, mo, optim, pixel_chains_refined, thickness_map):
    f = mo.ui.anywidget(
        OptimPlayBack(pixel_chains_refined, thickness_map, optim.history_)
    )
    f
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If we look closely, we see that some lines do not join exactly at the same point. It is logical, since every curve fitting is independent.
    The way to fix this problem is specified in the paper, but not exactly.

    We chose to implement what seemed like a reasonnable solution.

    First for the point-edges junctions, we look again at all our "T" configurations (see above) and force the control point of the incoming curve to fall exactly on the other curve.

    For the point-point junctions, we compute the barycenter $B$ of all the points that are supposed to arrive at the same position, and we force the control point of each incoming edge to be $B$.
    """)
    return


@app.cell
def _(align_boundaries, curves, instants, interpolate_bezier, optim, plt):
    final_curves = align_boundaries(
        curves, optim.endpoint_mapping_, optim.interior_mapping_
    )
    _ax = plt.gca()
    for _i, _c in enumerate(final_curves):
        _bezier_curve = interpolate_bezier(_c.control_points, instants)

        # Plot the Bézier curve
        _ax.plot(_bezier_curve[:, 1], _bezier_curve[:, 0], "-", linewidth=1, alpha=1)
    show(_ax, "final curves (with junction)")
    return (final_curves,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## SVG export

    Here is the final result:
    """)
    return


@app.cell
def _():
    from marimo import Html

    from sketchy_svg.demo import set_options, show_example
    from sketchy_svg.export import export_svg

    return Html, export_svg, set_options, show_example


@app.cell
def _(Html, export_svg, final_curves, img):
    out = export_svg(final_curves, img.shape[0], img.shape[1])
    Html(out)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Gallery
    """)
    return


@app.cell
def _(mo, set_options):
    set_options(
        {
            "status_function": lambda x, title: mo.status.progress_bar(x, title=title),
        }
    )
    return


@app.cell
async def _(DATA_DIR, show_example):
    await show_example(DATA_DIR / "sketches/butterfly.png")
    return


@app.cell
async def _(DATA_DIR, show_example):
    await show_example(DATA_DIR / "original_paper/figure_2/input.png")
    return


@app.cell
async def _(DATA_DIR, show_example):
    await show_example(DATA_DIR / "original_paper/figure_1/input.png")
    return


@app.cell
async def _(DATA_DIR, show_example):
    await show_example(DATA_DIR / "sketches/triangle.png")
    return


@app.cell
async def _(DATA_DIR, show_example):
    await show_example(DATA_DIR / "sketches/dress.png")
    return


@app.cell
async def _(DATA_DIR, show_example):
    await show_example(DATA_DIR / "sketches/piano.png")
    return


@app.cell
async def _(DATA_DIR, show_example):
    await show_example(DATA_DIR / "sketches/house.png")
    return


@app.cell
async def _(DATA_DIR, show_example):
    await show_example(DATA_DIR / "sketches/cube_bend.png")
    return


@app.cell
async def _(DATA_DIR, show_example):
    await show_example(DATA_DIR / "sketches/smiley.png")
    return


@app.cell
async def _(DATA_DIR, show_example):
    await show_example(
        DATA_DIR / "CAD_dataset/Dataset_B/ESB_Sketches/90 degree elbows/001_1.png"
    )
    return


@app.cell
async def _(DATA_DIR, show_example):
    await show_example(
        DATA_DIR / "CAD_dataset/Dataset_B/ESB_Sketches/U shaped parts/005_1.png"
    )
    return


@app.cell
async def _(DATA_DIR, show_example):
    await show_example(DATA_DIR / "original_paper/figure_10/archi.png")
    return


@app.cell
async def _(DATA_DIR, show_example):
    await show_example(DATA_DIR / "original_paper/figure_14/bag/input.png")
    return


if __name__ == "__main__":
    app.run()
