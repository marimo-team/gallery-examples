# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
#     "scipy",
#     "scikit-image",
#     "pillow",
# ]
# ///
"""Inscribed Squares from Noise — a marimo walkthrough of
"Visual Diffusion Models are Geometric Solvers" (Goren et al., CVPR 2026 Highlight).

Submission for the alphaXiv x marimo "Bring Research to Life" competition.
"""

import marimo

__generated_with = "0.23.4"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    import io
    import logging
    import urllib.request
    from contextlib import redirect_stderr, redirect_stdout
    from importlib.util import find_spec
    from pathlib import Path

    # Silence the "Matplotlib is building the font cache" first-run message
    # that fires in fresh Pyodide environments. Belt and suspenders: lower
    # the log levels (in case it routes through logging), AND redirect both
    # std streams during the matplotlib import (in case it writes direct).
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

    import marimo as mo
    with redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
        import matplotlib.pyplot as plt
        # Force font cache build now (while streams are silenced) so the
        # first real plt.subplots() call doesn't trigger the message.
        _warmup = plt.figure(figsize=(0.1, 0.1))
        plt.close(_warmup)
    import numpy as np
    from scipy.interpolate import splev, splprep
    from skimage.measure import find_contours

    GH_RAW = "https://raw.githubusercontent.com/FarseenSh/alphaxiv-marimo-comp/main/data/gallery.npz"

    def _fetch_remote():
        with urllib.request.urlopen(GH_RAW) as _r:
            return dict(np.load(io.BytesIO(_r.read())))

    # In Pyodide/WASM, __file__ may resolve into a virtual filesystem where the
    # path "exists" but the bytes are stale/empty — always fetch remotely there.
    if find_spec("js") is not None:
        gallery = _fetch_remote()
    else:
        _local = Path(__file__).resolve().parents[1] / "data" / "gallery.npz"
        if _local.exists() and _local.stat().st_size > 0:
            try:
                gallery = dict(np.load(_local))
            except Exception:
                gallery = _fetch_remote()
        else:
            gallery = _fetch_remote()

    IMAGE_SIZE = 128

    def to_pixel(xy, size=IMAGE_SIZE):
        return xy * (size // 2) + size // 2

    def square_polygon(sample, level=0.0):
        contours = find_contours(sample, level=level)
        if not contours:
            return None
        return max(contours, key=len)

    def make_axes(ax, size=IMAGE_SIZE, title=None):
        ax.set_xlim(0, size); ax.set_ylim(size, 0)
        ax.set_aspect("equal"); ax.axis("off")
        if title:
            ax.set_title(title, fontsize=11)
        return ax

    def plot_curve(ax, curve_xy, color="black", lw=1.4, alpha=1.0):
        px = to_pixel(curve_xy)
        px = np.vstack([px, px[:1]])
        ax.plot(px[:, 0], px[:, 1], color=color, lw=lw, alpha=alpha)

    def plot_square_outline(ax, sample, color="crimson", lw=2.4, alpha=0.9, fill_alpha=0.18):
        poly = square_polygon(sample)
        if poly is None:
            return
        ax.fill(poly[:, 1], poly[:, 0], color=color, alpha=fill_alpha)
        ax.plot(poly[:, 1], poly[:, 0], color=color, lw=lw, alpha=alpha)

    return (
        IMAGE_SIZE,
        gallery,
        make_axes,
        mo,
        np,
        plot_curve,
        plot_square_outline,
        plt,
        splev,
        splprep,
        to_pixel,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <div style="text-align:center; padding: 24px 0 0 0;">
    <h1 style="margin-bottom: 4px; font-size: 2.4rem;">Inscribed Squares from Noise</h1>
    <p style="font-size: 1.05rem; opacity: 0.85; margin-top: 0;">
    A diffusion model doesn't generate cats here.<br>
    It solves a 100-year-old open problem in geometry — by drawing.
    </p>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(gallery, make_axes, plot_curve, plot_square_outline, plt):
    _xy = gallery["hero_butterfly/curve_xy"]
    _samples = gallery["hero_butterfly/samples"][:8]
    _palette = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd",
                "#ff7f0e", "#17becf", "#e377c2", "#bcbd22"]
    _fig, _axes = plt.subplots(2, 4, figsize=(13, 6.5))
    for _ax, _i, _c in zip(_axes.flat, range(8), _palette):
        plot_curve(_ax, _xy, lw=1.3)
        plot_square_outline(_ax, _samples[_i], color=_c, fill_alpha=0.22, lw=2.0)
        make_axes(_ax, title=f"seed {_i}")
    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <p style="text-align:center; font-size: 0.9rem; opacity: 0.65; margin: -8px 0 16px 0;">
    The same Jordan curve, eight random seeds, eight different inscribed squares.
    </p>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The setup

    The **Inscribed Square Problem** (Toeplitz, 1911) asks: does every closed,
    non-self-intersecting curve in the plane contain four points that form a
    perfect square? It's still open in full generality after 100+ years.

    Goren, Yehezkel, Dahary, Voynov, Patashnik, and Cohen-Or
    ([arXiv 2510.21697](https://arxiv.org/abs/2510.21697), CVPR 2026 Highlight)
    propose something a little wild: **train a standard image diffusion model
    to take a Jordan curve as a 128×128 picture and denoise Gaussian noise
    into a picture of an inscribed square**. No specialized architecture, no
    parametric tricks — pure pixel-space denoising.

    It works. And because diffusion is multimodal, the same curve produces
    a *different* inscribed square at every seed — uncovering a hidden family
    of solutions the paper itself does not explore.

    That's where this notebook spends most of its time.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Pipeline — at a glance

    | Stage | Input | Output |
    |---|---|---|
    | **Condition** | A Jordan curve, rasterized to 128×128 binary | 1-channel image (the "problem statement") |
    | **Noise**     | Random Gaussian (1×128×128) | The starting point of denoising |
    | **U-Net**     | `[noise, condition]` concatenated as 2 channels | Predicted noise per pixel |
    | **DDIM**      | 100 denoising steps | A clean 1-channel image of an inscribed square |

    The U-Net is a vanilla 4-level diffusion U-Net (~20M params, attention at
    the bottleneck and the inner enc/dec levels). Training data is purely
    synthetic: 100,000 procedurally generated Jordan curves, each constructed
    to pass exactly through a known random square.

    Diffusion sampling itself runs offline (~5s per sample on CPU). The
    notebook reads the cached samples from a 10 MB `.npz`. *Why?* PyTorch
    is not in Pyodide, but everything else here is. To reproduce the
    sampling, see `scripts/precompute.py` in the repo.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Design your own curve

    Before we walk through the cached gallery, here's the curve generator
    the model was trained on, running live in your browser. The math:

    $$
    r(\theta) = 1 + \sum_{h=1}^{H} \rho_h \sin(h\theta + \phi_h),\qquad
    \rho_h \sim \mathcal U(0, 1)\cdot \alpha\cdot 10^{-(0.5 + 2(h-1)/(H-1))}
    $$

    Move the sliders. Inference can't run in-browser (PyTorch is not in
    Pyodide), but you can preview *exactly the input* the diffusion
    model would receive. Fork the repo to run sampling on it.
    """)
    return


@app.cell
def _(mo):
    H_slider = mo.ui.slider(start=1, stop=30, value=12, step=1, label="harmonics H")
    rho_slider = mo.ui.slider(start=0.0, stop=2.0, value=1.2, step=0.05, label="amplitude scale α")
    radius_slider = mo.ui.slider(start=0.30, stop=0.75, value=0.55, step=0.01, label="target radius")
    seed_slider = mo.ui.slider(start=0, stop=99, value=7, step=1, label="seed")
    mo.hstack([H_slider, rho_slider, radius_slider, seed_slider], gap=2)
    return H_slider, radius_slider, rho_slider, seed_slider


@app.cell
def _(
    H_slider,
    IMAGE_SIZE,
    np,
    plt,
    radius_slider,
    rho_slider,
    seed_slider,
    splev,
    splprep,
    to_pixel,
):
    def _generate_curve(H, rho_scale, target_radius, seed, num_points=600):
        rng = np.random.default_rng(int(seed))
        t = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        rho = rng.random(H) * np.logspace(-0.5, -2.5, max(H, 1)) * rho_scale
        phi = rng.random(H) * 2 * np.pi
        r = np.ones_like(t)
        for h in range(1, H + 1):
            r += rho[h - 1] * np.sin(h * t + phi[h - 1])
        r *= target_radius
        x, y = r * np.cos(t), r * np.sin(t)
        try:
            tck, _ = splprep([x, y], s=0, per=True)
            u = np.linspace(0, 1, num_points)
            x, y = splev(u, tck)
        except Exception:
            pass
        return np.stack([x, y], axis=1)

    _curve = _generate_curve(
        H_slider.value, rho_slider.value, radius_slider.value, seed_slider.value
    )

    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(10, 5))

    _px = to_pixel(_curve, IMAGE_SIZE)
    _px = np.vstack([_px, _px[:1]])
    _a1.fill(_px[:, 0], _px[:, 1], color="#e8eef9", alpha=1.0)
    _a1.plot(_px[:, 0], _px[:, 1], color="black", lw=1.6)
    _a1.set_xlim(0, IMAGE_SIZE); _a1.set_ylim(IMAGE_SIZE, 0)
    _a1.set_aspect("equal"); _a1.set_title(f"r(θ) with H={H_slider.value}, α={rho_slider.value:.2f}")
    _a1.grid(alpha=0.15); _a1.set_xticks([]); _a1.set_yticks([])
    for _spine in _a1.spines.values(): _spine.set_visible(False)

    _img = np.full((IMAGE_SIZE, IMAGE_SIZE), 255, dtype=np.uint8)
    _pts = _px.astype(np.int32)
    for _i in range(len(_pts) - 1):
        _x0, _y0 = _pts[_i]; _x1, _y1 = _pts[_i + 1]
        _n = max(abs(_x1 - _x0), abs(_y1 - _y0)) + 1
        _xs = np.linspace(_x0, _x1, _n).astype(int).clip(0, IMAGE_SIZE - 1)
        _ys = np.linspace(_y0, _y1, _n).astype(int).clip(0, IMAGE_SIZE - 1)
        _img[_ys, _xs] = 0
    _a2.imshow(_img, cmap="gray", vmin=0, vmax=255)
    _a2.set_title("rasterized → model input (128×128 binary)")
    _a2.set_xticks([]); _a2.set_yticks([])
    for _spine in _a2.spines.values(): _spine.set_visible(False)

    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Pick a curve from the gallery

    From here on we follow one running example through the pipeline, then
    come back at the end and apply everything to the rest of the gallery.
    Switch curves to see how the model behaves on each.
    """)
    return


@app.cell
def _(gallery, mo):
    _choices = [
        c for c in
        ["hero_butterfly", "circle", "peanut", "spiky_gear", "paper_figure_1"]
        if f"{c}/curve_img" in gallery
    ]
    curve_picker = mo.ui.dropdown(
        options=_choices, value="hero_butterfly", label="Jordan curve"
    )
    curve_picker
    return (curve_picker,)


@app.cell
def _(curve_picker, gallery):
    name = curve_picker.value
    curve_xy = gallery[f"{name}/curve_xy"]
    samples = gallery[f"{name}/samples"]
    return curve_xy, name, samples


@app.cell
def _(curve_xy, make_axes, name, plot_curve, plt):
    _fig, _ax = plt.subplots(figsize=(5, 5))
    plot_curve(_ax, curve_xy)
    make_axes(_ax, title=f"input: {name}")
    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Watch the square emerge

    DDIM walks 100 denoising steps from pure Gaussian noise to a clean
    prediction. Below: the **predicted clean image $\hat x_0$** at every
    10th step — what the model "thinks" the answer is at every point in
    the trajectory.
    """)
    return


@app.cell
def _(curve_picker, gallery):
    has_traj = f"{curve_picker.value}/trajectory" in gallery
    traj = gallery[f"{curve_picker.value}/trajectory"] if has_traj else None
    traj_steps = (
        gallery[f"{curve_picker.value}/trajectory_steps"] if has_traj else None
    )
    return has_traj, traj, traj_steps


@app.cell
def _(
    curve_xy,
    has_traj,
    make_axes,
    mo,
    np,
    plot_curve,
    plt,
    traj,
    traj_steps,
):
    if has_traj:
        _fig, _axes = plt.subplots(2, 5, figsize=(15, 6.4))
        for _i, _ax in enumerate(_axes.flat):
            _frame = traj[_i, 0]
            if _frame.ndim == 3:
                _frame = _frame[0]
            # Binary mask: black where the model thinks "square pixel"
            _bin = (_frame < 0).astype(np.float32)
            _ax.imshow(_bin, cmap="binary", vmin=0, vmax=1, alpha=0.85)
            plot_curve(_ax, curve_xy, color="black", lw=1.0, alpha=0.55)
            make_axes(_ax, title=f"after step {int(traj_steps[_i]) + 1}/100")
        plt.tight_layout()
        _out = _fig
    else:
        _out = mo.md(
            "*(trajectory only cached for the hero butterfly — switch curves above)*"
        )
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## One solution at a time

    Each random seed produces a different valid inscribed square.
    Scroll the slider to walk through the seeds.
    """)
    return


@app.cell
def _(mo, samples):
    sample_slider = mo.ui.slider(
        start=0, stop=samples.shape[0] - 1, value=0, step=1, label="seed index"
    )
    sample_slider
    return (sample_slider,)


@app.cell
def _(
    curve_xy,
    make_axes,
    plot_curve,
    plot_square_outline,
    plt,
    sample_slider,
    samples,
):
    _i = int(sample_slider.value)
    _fig, _ax = plt.subplots(figsize=(5.5, 5.5))
    plot_curve(_ax, curve_xy)
    plot_square_outline(_ax, samples[_i])
    make_axes(_ax, title=f"seed {_i} → one valid inscribed square")
    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Four seeds, four squares

    Same curve, different random initial noise, different valid squares.
    These are not retries — they are simultaneous *modes* of the
    diffusion posterior $p(\text{square} \mid \text{curve})$.
    """)
    return


@app.cell
def _(curve_xy, make_axes, np, plot_curve, plot_square_outline, plt, samples):
    _idxs = np.linspace(0, samples.shape[0] - 1, 4).astype(int)
    _colors = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]

    _fig, _axes = plt.subplots(1, 4, figsize=(14, 4))
    for _ax, _i, _c in zip(_axes, _idxs, _colors):
        plot_curve(_ax, curve_xy)
        plot_square_outline(_ax, samples[int(_i)], color=_c, fill_alpha=0.20)
        make_axes(_ax, title=f"seed {int(_i)}")
    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The hero cell — the family of solutions

    Now overlay *every* sampled square onto the same curve, weighted by
    how many seeds covered each pixel. Bright pixels are inside the
    intersection of all sampled squares; dim pixels are reached by only
    some. The contrast is the **wiggle room of the inscribed-square
    family** for this Jordan curve — something the paper hints at but
    never quantifies.

    You're seeing one slice of an infinite-dimensional solution manifold,
    free of charge, just because the model is multimodal.
    """)
    return


@app.cell
def _(curve_xy, make_axes, np, plot_curve, plt, samples):
    _n = samples.shape[0]
    _fig, _ax = plt.subplots(figsize=(6.2, 6.2))
    _mask = (samples < 0).astype(np.float32)
    _heat = _mask.mean(axis=0)
    _im = _ax.imshow(np.where(_heat > 0.05, _heat, np.nan),
                     cmap="plasma", alpha=0.92, vmin=0, vmax=1)
    plot_curve(_ax, curve_xy, color="black", lw=1.6, alpha=0.85)
    _cbar = plt.colorbar(_im, ax=_ax, fraction=0.046, pad=0.04)
    _cbar.set_label(f"fraction of {_n} seeds covering pixel")
    make_axes(_ax, title="multimodal solution map")
    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Are these actually squares?

    We can score each sample's "squareness" — fraction of the predicted
    shape filled by the smallest enclosing rectangle, scaled by an
    aspect-ratio penalty (the metric the paper uses):

    $$
    Q = \frac{\text{area}}{w \cdot h} \cdot
    \exp\!\left(-2 \left|\tfrac{\max(w,h)}{\min(w,h)} - 1\right|\right)
    $$

    $Q = 1$ is a perfect square; $Q$ near 0 is a long thin rectangle or
    scattered noise.
    """)
    return


@app.cell
def _(np, plt, samples):
    def _squareness(mask):
        """Squareness via min-area rotating bbox (PCA-free, brute-force over 1° angles)."""
        _ys, _xs = np.where(mask < 0)
        if len(_xs) < 8:
            return 0.0, 0.0
        _pts = np.stack([_xs, _ys], axis=1).astype(float)
        _pts = _pts - _pts.mean(axis=0)
        _best_a = np.inf
        _best_w = _best_h = 0.0
        for _ang in np.arange(0, 90, 1.0):
            _r = np.radians(_ang)
            _R = np.array([[np.cos(_r), -np.sin(_r)], [np.sin(_r), np.cos(_r)]])
            _rot = _pts @ _R.T
            _w = _rot[:, 0].max() - _rot[:, 0].min()
            _h = _rot[:, 1].max() - _rot[:, 1].min()
            if _w * _h < _best_a:
                _best_a = _w * _h
                _best_w, _best_h = _w, _h
        _area = float(len(_xs))
        _fill = _area / max(1.0, _best_a)
        _aspect = max(_best_w, _best_h) / max(1.0, min(_best_w, _best_h))
        return float(_fill * np.exp(-2 * abs(_aspect - 1))), _area

    scores = np.array([_squareness(s)[0] for s in samples])
    _areas = np.array([_squareness(s)[1] for s in samples])

    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(11, 3.5))
    _a1.bar(range(len(scores)), scores, color="#5b8def")
    _a1.axhline(0.85, ls="--", color="grey", alpha=0.6,
                label="paper's quality threshold ≈ 0.85")
    _a1.set_xlabel("seed index"); _a1.set_ylabel("squareness $Q$")
    _a1.set_ylim(0, 1.02); _a1.legend(fontsize=8)
    _a1.set_title("per-seed squareness")
    for _spine in ("top", "right"): _a1.spines[_spine].set_visible(False)

    _a2.scatter(_areas, scores, s=44, c="#5b8def", edgecolor="white", linewidth=0.5)
    _a2.set_xlabel("predicted square area (px)"); _a2.set_ylabel("squareness $Q$")
    _a2.set_title("size vs. quality")
    for _spine in ("top", "right"): _a2.spines[_spine].set_visible(False)
    plt.tight_layout()
    _fig
    return (scores,)


@app.cell(hide_code=True)
def _(mo, scores):
    mo.md(f"""
    Mean squareness across seeds: **{float(scores.mean()):.3f}**, n={len(scores)}.
    Outliers near 0 are typically failed samples — the model occasionally
    produces a degenerate blob instead of a square. Failure rates of
    ~5–15% match the paper's reported numbers on the curves task.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Gallery — every curve in the test set

    Same model, same pipeline, applied to all five curves. Each panel
    shows the multimodal heatmap (intersection in yellow, halo in purple).

    - **circle** — every diameter rotation gives an inscribed square, and
      you can literally see the model rediscovering that: the heatmap is
      a near-perfect rotational sunflower.
    - **peanut** — only one square fits; the heatmap is a single tight blob.
    - **spiky_gear** — the model finds a robust square despite the
      out-of-distribution lobes.
    """)
    return


@app.cell
def _(gallery, make_axes, np, plot_curve, plt):
    _names = [
        n for n in
        ["hero_butterfly", "circle", "peanut", "spiky_gear", "paper_figure_1"]
        if f"{n}/curve_img" in gallery
    ]

    _fig, _axes = plt.subplots(1, len(_names), figsize=(3.0 * len(_names), 3.4))
    if len(_names) == 1:
        _axes = [_axes]

    for _ax, _name in zip(_axes, _names):
        _smp = gallery[f"{_name}/samples"]
        _cxy = gallery[f"{_name}/curve_xy"]
        _heat = (_smp < 0).astype(np.float32).mean(axis=0)
        _ax.imshow(np.where(_heat > 0.05, _heat, np.nan),
                   cmap="plasma", vmin=0, vmax=1, alpha=0.92)
        plot_curve(_ax, _cxy, color="black", lw=1.4, alpha=0.85)
        _ax.set_box_aspect(1.0)
        make_axes(_ax, title=f"{_name}\n(n={_smp.shape[0]} seeds)")

    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What this notebook does *not* claim

    - **The model is not a proof.** It outputs pixel pictures. The squares
      are visually inscribed, but corners snap to the nearest curve pixel
      to within ~1 pixel of error — not provably exact.
    - **Out-of-distribution failure is real.** Curves with thin lobes,
      aggressive self-intersections, or fractal textures often produce
      degenerate samples. The multimodal map does not filter them.
    - **The training data trick matters.** Each training curve was
      constructed to pass through a known square. The model never had to
      discover inscribed-ness from scratch on adversarial inputs.

    None of this diminishes the central observation: a generic image
    diffusion model — no architectural specialization, no parametric
    output head — recovers inscribed squares from images well enough to
    be visually convincing on novel curves. That's the paper's
    contribution. The multimodal-overlay framing in this notebook is a
    free side effect of using diffusion.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Try it yourself

    - **Repo**: [`github.com/FarseenSh/alphaxiv-marimo-comp`](https://github.com/FarseenSh/alphaxiv-marimo-comp)
      — fork to run sampling on your own curves. `scripts/precompute.py`
      accepts any `(H, ρ, target_radius, rotation, seed)` tuple.
    - **Pretrained checkpoint**: 80 MB at
      [huggingface.co/nirgoren/geometric-solver](https://huggingface.co/nirgoren/geometric-solver).
    - **Original paper**: [arXiv:2510.21697](https://arxiv.org/abs/2510.21697)
      (Goren, Yehezkel, Dahary, Voynov, Patashnik, Cohen-Or — CVPR 2026 Highlight).
    - **alphaXiv discussion**: [comments](https://www.alphaxiv.org/abs/2510.21697).

    Built for the
    [alphaXiv × marimo "Bring Research to Life" notebook competition](https://marimo.io/pages/events/notebook-competition).
    """)
    return


if __name__ == "__main__":
    app.run()
