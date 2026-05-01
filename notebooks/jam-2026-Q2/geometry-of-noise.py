# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy==2.4.4",
#     "matplotlib==3.10.8",
#     "plotly==6.7.0",
#     "pywidget==0.1.0",
# ]
# ///
"""
The Geometry of Noise — why diffusion models don't need noise conditioning.

Interactive companion notebook for Sahraee-Ardakan, Delbracio & Milanfar
(2026). All analytics are closed-form over a discrete 2D-circles dataset
lifted to R^D via a random orthogonal projection. No neural networks,
no training. Pure NumPy + pywidget, runs entirely in Pyodide.
"""

import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import io
    import json
    from functools import lru_cache

    import numpy as np
    import plotly.graph_objects as go
    import traitlets
    from pywidget import PyWidget

    return PyWidget, io, json, lru_cache, np, traitlets


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The Geometry of Noise
    Paper: [Sahraee-Ardakan, Delbracio & Milanfar (2026)](https://www.alphaxiv.org/abs/2602.18428v1)
    · Notebook by Konstantin Taletskiy

    ---

    ## TL;DR

    Every diffusion model you have used passes the current noise level
    $t$ to its U-Net. This paper says you can **drop $t$ entirely** —
    turn `unet(x_t, t)` into `unet(x_t)`, a *blind* sampler — and it
    still works. The reason is geometric: in high-dimensional spaces,
    the image itself reveals its own noise level (concentration of
    measure). It also matters *what* the U-Net is trained to predict:
    clean image or velocity → stable; noise → fragile. The paper names
    the mechanism **posterior collapse on $t$** and proves a hard
    boundary at codimension $D - d > 2$, below which even the stable
    parameterizations break.

    This notebook builds the geometric intuition step by step and lets
    you watch that boundary live by dragging $D$.

    ---

    *Everything below is a closed-form analytical reconstruction — no
    neural networks, no training. Interactive elements use
    [pywidget](https://github.com/ktaletsk/pywidget): pure-Python widgets
    running in Pyodide, natively compatible with marimo.*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    from pathlib import Path as _Path

    # Prefer the PNG on disk; fall back to the base64-encoded text file if
    # the image asset isn't shipped alongside the notebook (e.g., in molab).
    _src = None
    try:
        _dir = _Path(__file__).parent / "figures"
        _png = _dir / "cifar10_comparison.png"
        _txt = _dir / "cifar10_comparison_b64.txt"
        if _png.exists():
            _src = _png
        elif _txt.exists():
            _src = f"data:image/png;base64,{_txt.read_text().strip()}"
    except (NameError, OSError):
        _src = None

    if _src is not None:
        _figure = mo.image(src=_src, width="100%", rounded=True)
    else:
        _figure = mo.md("*(CIFAR-10 figure not found)*")

    mo.vstack([
        _figure,
        mo.md(
            "*On real images, the difference is stark: noise prediction without "
            "knowing the noise level produces mush (left), while velocity "
            "prediction produces clean samples (right) — both using the exact "
            "same blind architecture. Our notebook explains why, analytically, "
            "on a toy dataset you can fully explore.*"
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A five-minute refresher on diffusion models

    If the phrase "diffusion model" already means something concrete to
    you, skip this section. Otherwise, here's the three-step story:

    - **Forward process.** Take a clean sample. Add a tiny dab of
      Gaussian noise. Add another. And another. After enough steps the
      sample is indistinguishable from pure Gaussian noise. The
      **noise level** $t$ labels how many steps in we are, from $t=0$
      (clean) to $t=1$ (pure noise).
    - **Training.** Ask a neural network to undo one step of noising.
      Given the noisy sample $\mathbf{x}_t$ and the noise level $t$,
      predict something that reveals the clean sample — either the
      clean data $\mathbf{x}$, or the noise $\boldsymbol{\epsilon}$ that
      was added, or the velocity $\boldsymbol{\epsilon} - \mathbf{x}$.
    - **Sampling.** Start from pure noise, run the trained "undo" step
      many times over, arrive at a fresh sample.

    ### The toy: concentric circles

    Everything in this notebook happens on a toy 2D dataset which is
    **200 points on two concentric circles** — so we can *see* each of these three steps with
    our own eyes. Let's do a lap.
    """)
    return


@app.cell(hide_code=True)
def _(X2_data, io):
    # Static view of the toy dataset before the reader starts moving D.
    # 200 points on two concentric circles in R^2 — the entire substrate
    # of every figure in the notebook, lifted into D-dim by a random
    # orthogonal embedding when D > 2.
    import re as _re
    import matplotlib as _mpl

    _mpl.use("agg")
    import matplotlib.pyplot as _plt

    _fig, _ax = _plt.subplots(figsize=(3.8, 3.8))
    _ax.scatter(
        X2_data[:100, 0],
        X2_data[:100, 1],
        s=18,
        c="#1f77b4",
        alpha=0.85,
        linewidths=0,
        label="inner ($r = 0.6$)",
    )
    _ax.scatter(
        X2_data[100:, 0],
        X2_data[100:, 1],
        s=18,
        c="#e67e22",
        alpha=0.85,
        linewidths=0,
        label="outer ($r = 1.2$)",
    )
    _ax.set_aspect("equal")
    _ax.set_xlim(-1.6, 1.6)
    _ax.set_ylim(-1.6, 1.6)
    _ax.grid(alpha=0.2)
    _ax.legend(loc="upper right", fontsize=8, frameon=False)
    _ax.set_title(
        r"Toy dataset: 200 points on two concentric circles in $\mathbb{R}^2$",
        fontsize=10,
        pad=8,
    )
    for _s in _ax.spines.values():
        _s.set_alpha(0.3)

    _fig.tight_layout()
    _buf = io.StringIO()
    _fig.savefig(_buf, format="svg", bbox_inches="tight")
    _plt.close(_fig)
    _svg = _buf.getvalue()
    _svg = _re.sub(r'(<svg[^>]*?)\s+width="[^"]*"', r"\1", _svg, count=1)
    _svg = _re.sub(r'(<svg[^>]*?)\s+height="[^"]*"', r"\1", _svg, count=1)
    _svg = _re.sub(
        r"<svg\b",
        '<svg style="width:100%;max-width:400px;height:auto;display:block;margin:0 auto;"',
        _svg,
        count=1,
    )
    from marimo import Html as _Html

    _Html(
        f'<div style="max-width:400px;margin:0 auto;padding:6px 0 14px;">{_svg}</div>'
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Watching the forward process

    As you scrub $t$ below, you're watching the forward process the way
    **every modern diffusion model actually sees it during training**:
    for each training step, a single noise vector
    $\boldsymbol{\epsilon} \sim \mathcal{N}(0, I)$ is drawn per data
    point and $\mathbf{u}_t = (1-t)\mathbf{x} + t\boldsymbol{\epsilon}$
    is computed in one shot. No sequential noising between timesteps.
    The network only ever sees marginal pairs $(\mathbf{u}_t, t)$.

    Each data point moves along its own straight line in 2D, from its
    starting position at $t=0$ to its personal noise target at $t=1$.
    The orange dot tracks one of them so you can follow its trajectory.
    """)
    return


@app.cell(hide_code=True)
def _(PyWidget, traitlets):
    class ForwardDiffusionWidget(PyWidget):
        config_json = traitlets.Unicode("{}").tag(sync=True)
        _py_packages = ["numpy"]

        def render(self, el, model):
            from pyodide.ffi import create_proxy
            from js import window
            import json
            import numpy as np

            config = json.loads(model.get("config_json"))
            if not config:
                el.innerHTML = (
                    '<div style="padding:24px;color:#888;">Loading…</div>'
                )
                return

            X2 = np.array(config["X2"])
            eps = np.array(config["eps"])
            k = int(config.get("highlight_index", 75))
            N = X2.shape[0]
            vb = 2.6
            W = 520

            hx, hy = float(X2[k, 0]), float(X2[k, 1])
            ex, ey = float(eps[k, 0]), float(eps[k, 1])

            # Straight-line trajectory for the highlighted point
            traj_line = (
                f'<line x1="{hx:.4f}" y1="{-hy:.4f}" '
                f'x2="{ex:.4f}" y2="{-ey:.4f}" '
                f'stroke="#e67e22" stroke-width="0.018" '
                f'stroke-dasharray="0.05,0.035" opacity="0.55"/>'
            )
            traj_endpoint = (
                f'<circle cx="{ex:.4f}" cy="{-ey:.4f}" r="0.035" '
                f'fill="none" stroke="#e67e22" stroke-width="0.02" '
                f'opacity="0.7"/>'
            )

            regular_dots = "".join(
                f'<circle class="fd-dot" data-i="{i}" '
                f'cx="{X2[i, 0]:.4f}" cy="{-X2[i, 1]:.4f}" '
                f'r="0.04" fill="#1f77b4" opacity="0.75"/>'
                for i in range(N)
                if i != k
            )
            highlight_dot = (
                f'<circle class="fd-dot" data-i="{k}" '
                f'cx="{hx:.4f}" cy="{-hy:.4f}" '
                f'r="0.08" fill="#e67e22" stroke="white" stroke-width="0.02" '
                f'opacity="0.98"/>'
            )

            el.innerHTML = f"""
            <div style="max-width:{W}px;margin:0 auto;">
                <svg viewBox="{-vb} {-vb} {2 * vb} {2 * vb}"
                     width="100%" style="aspect-ratio:1/1; max-width:{W}px;
                     border:1px solid #ddd; border-radius:6px; background:#fafafa;
                     display:block;"
                     id="fd-scatter">
                    {traj_line}
                    {traj_endpoint}
                    {regular_dots}
                    {highlight_dot}
                </svg>
                <div style="display:flex; align-items:center; gap:10px;
                            margin:10px 4px 0;">
                    <button id="fd-play"
                            style="padding:4px 12px; border-radius:4px;
                            border:1px solid #1f77b4; background:#fff;
                            color:#1f77b4; cursor:pointer; font-size:12px;
                            min-width:70px;">▶ play</button>
                    <input id="fd-slider" type="range" min="0" max="1000"
                           step="1" value="0" style="flex:1;"/>
                    <span id="fd-readout" style="font-family:monospace;
                          font-size:12px; min-width:70px;">t = 0.00</span>
                </div>
                <div style="text-align:center; font-size:11px; color:#666;
                     margin-top:4px; max-width:460px; margin-left:auto;
                     margin-right:auto;">
                    u<sub>t</sub> = (1−t) · x + t · ε.  ε is drawn once per
                    point with independent components per axis; the orange
                    dot tracks one point along its straight-line path.
                </div>
            </div>
            """

            svg_el = el.querySelector("#fd-scatter")
            slider = el.querySelector("#fd-slider")
            button = el.querySelector("#fd-play")
            readout = el.querySelector("#fd-readout")
            dot_nodes = svg_el.querySelectorAll(".fd-dot")

            dot_by_index = {}
            for j in range(dot_nodes.length):
                node = dot_nodes.item(j)
                idx = int(node.getAttribute("data-i"))
                dot_by_index[idx] = node

            state = {"playing": False, "interval_id": None, "direction": 1}

            def update_positions(t):
                a_t = 1.0 - t
                b_t = t
                for i in range(N):
                    ux = a_t * float(X2[i, 0]) + b_t * float(eps[i, 0])
                    uy = a_t * float(X2[i, 1]) + b_t * float(eps[i, 1])
                    d = dot_by_index[i]
                    d.setAttribute("cx", f"{ux:.4f}")
                    d.setAttribute("cy", f"{-uy:.4f}")
                readout.textContent = f"t = {t:.2f}"

            def on_slider_input(event):
                update_positions(int(slider.value) / 1000.0)

            def tick():
                v = int(slider.value) + state["direction"] * 8
                if v >= 1000:
                    v = 1000
                    state["direction"] = -1
                elif v <= 0:
                    v = 0
                    state["direction"] = 1
                slider.value = str(v)
                update_positions(v / 1000.0)

            def on_play_click(event):
                if state["playing"]:
                    window.clearInterval(state["interval_id"])
                    state["playing"] = False
                    button.textContent = "▶ play"
                else:
                    p = create_proxy(tick)
                    state["interval_id"] = window.setInterval(p, 40)
                    state["playing"] = True
                    button.textContent = "⏸ pause"
                    el._py_fd_tick_proxy = p

            _slider_proxy = create_proxy(on_slider_input)
            slider.addEventListener("input", _slider_proxy)
            el._py_fd_slider_proxy = _slider_proxy

            _button_proxy = create_proxy(on_play_click)
            button.addEventListener("click", _button_proxy)
            el._py_fd_button_proxy = _button_proxy

        def update(self, el, model):
            self.render(el, model)

    return (ForwardDiffusionWidget,)


@app.cell(hide_code=True)
def _(ForwardDiffusionWidget, X2_data, json, mo, np):
    # -------- Act 0.1: forward diffusion (interactive) ------------
    # Draw one fixed eps per data point. Scrubbing t = 0 → 1 then gives
    # each point a deterministic straight-line trajectory in 2D from x to
    # eps. This matches what the network actually sees during training: a
    # single (u_t, t) pair per example, sampled from the marginal.
    _rng_fd = np.random.default_rng(0)
    _eps_fixed = _rng_fd.standard_normal(X2_data.shape)

    _fd_config = {
        "X2": X2_data.tolist(),
        "eps": _eps_fixed.tolist(),
        "highlight_index": 75,
    }
    _fd_w = ForwardDiffusionWidget(config_json=json.dumps(_fd_config))
    forward_widget = mo.ui.anywidget(_fd_w)
    forward_widget
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Training, in three lines of pseudocode

    ```python
    t       = random.uniform(0, 1)           # pick a noise level
    x_t     = (1 - t) * x + t * eps          # noise the clean sample (eps ~ N(0, I))
    loss    = mse(unet(x_t, t), target)      # predict & regress
    ```

    The network gets both the noisy sample $\mathbf{x}_t$ *and* the
    noise level $t$, and learns to predict some `target` that pins
    down the clean $\mathbf{x}$. Simple. Works. This is the pattern
    every diffusion model in the wild uses.

    **What happens if we drop the `t`?** That is the question this
    paper answers.
    """)
    return


@app.cell(hide_code=True)
def _(PyWidget, traitlets):
    class ReverseDiffusionWidget(PyWidget):
        config_json = traitlets.Unicode("{}").tag(sync=True)
        _py_packages = ["numpy"]

        def render(self, el, model):
            from pyodide.ffi import create_proxy
            from js import window
            import json
            import numpy as np

            config = json.loads(model.get("config_json"))
            if not config:
                el.innerHTML = (
                    '<div style="padding:24px;color:#888;">Loading…</div>'
                )
                return

            traj = np.array(config["trajectories"])  # (n_frames, N, 2)
            ts = config["ts"]  # length n_frames
            X2 = np.array(config["X2"])
            n_frames = traj.shape[0]
            N = traj.shape[1]

            vb = 2.6
            W = 520

            target_svg = "".join(
                f'<circle cx="{X2[i, 0]:.4f}" cy="{-X2[i, 1]:.4f}" '
                f'r="0.025" fill="black" opacity="0.22"/>'
                for i in range(X2.shape[0])
            )

            dots_svg = "".join(
                f'<circle class="rd-dot" data-i="{i}" '
                f'cx="{traj[0, i, 0]:.4f}" cy="{-traj[0, i, 1]:.4f}" '
                f'r="0.045" fill="#1f77b4" opacity="0.85"/>'
                for i in range(N)
            )

            el.innerHTML = f"""
            <div style="max-width:{W}px;margin:0 auto;">
                <svg viewBox="{-vb} {-vb} {2 * vb} {2 * vb}"
                     width="100%" style="aspect-ratio:1/1; max-width:{W}px;
                     border:1px solid #ddd; border-radius:6px; background:#fafafa;
                     display:block;"
                     id="rd-scatter">
                    {target_svg}
                    {dots_svg}
                </svg>
                <div style="display:flex; align-items:center; gap:10px;
                            margin:10px 4px 0;">
                    <button id="rd-play"
                            style="padding:4px 12px; border-radius:4px;
                            border:1px solid #1f77b4; background:#fff;
                            color:#1f77b4; cursor:pointer; font-size:12px;
                            min-width:70px;">▶ play</button>
                    <input id="rd-slider" type="range" min="0" max="{n_frames - 1}"
                           step="1" value="0" style="flex:1;"/>
                    <span id="rd-readout" style="font-family:monospace;
                          font-size:12px; min-width:70px;">t = {ts[0]:.2f}</span>
                </div>
                <div style="text-align:center; font-size:11px; color:#666;
                     margin-top:4px;">
                    FM-conditional sampler in D = 16, projected to 2D.
                    Faint black dots are the target dataset.
                </div>
            </div>
            """

            svg_el = el.querySelector("#rd-scatter")
            slider = el.querySelector("#rd-slider")
            button = el.querySelector("#rd-play")
            readout = el.querySelector("#rd-readout")
            dots = svg_el.querySelectorAll(".rd-dot")

            state = {"playing": False, "interval_id": None}

            def update_frame(step):
                for i in range(N):
                    d = dots.item(i)
                    d.setAttribute("cx", f"{float(traj[step, i, 0]):.4f}")
                    d.setAttribute("cy", f"{-float(traj[step, i, 1]):.4f}")
                readout.textContent = f"t = {ts[step]:.2f}"

            def on_slider_input(event):
                update_frame(int(slider.value))

            def tick():
                step = int(slider.value) + 1
                if step >= n_frames:
                    step = 0
                slider.value = str(step)
                update_frame(step)

            def on_play_click(event):
                if state["playing"]:
                    window.clearInterval(state["interval_id"])
                    state["playing"] = False
                    button.textContent = "▶ play"
                else:
                    p = create_proxy(tick)
                    state["interval_id"] = window.setInterval(p, 55)
                    state["playing"] = True
                    button.textContent = "⏸ pause"
                    el._py_rd_tick_proxy = p

            _slider_proxy = create_proxy(on_slider_input)
            slider.addEventListener("input", _slider_proxy)
            el._py_rd_slider_proxy = _slider_proxy

            _button_proxy = create_proxy(on_play_click)
            button.addEventListener("click", _button_proxy)
            el._py_rd_button_proxy = _button_proxy

        def update(self, el, model):
            self.render(el, model)

    return (ReverseDiffusionWidget,)


@app.cell(hide_code=True)
def _(
    PARAMS,
    ReverseDiffusionWidget,
    X2_data,
    conditional_field,
    json,
    lift,
    mo,
    np,
    random_lift,
):
    # -------- Act 0.2: reverse diffusion (interactive) ------------
    # Precompute a full reverse-sampling trajectory on the circles at
    # D = 16, capturing one snapshot per Euler step. Then hand the frames
    # to a pywidget; the reader scrubs or auto-plays — no kernel round-trip
    # per frame, rendering is entirely in-browser via Pyodide.
    _D_demo = 16
    _P_rd = random_lift(_D_demo, seed=42)
    _X_rd = lift(X2_data, _P_rd)
    _cd_rd = PARAMS["FM"]

    _n_samples = 200
    _n_steps = 80
    _t_start, _t_end = 0.99, 0.005
    _ts_full = np.exp(np.linspace(np.log(_t_start), np.log(_t_end), _n_steps + 1))

    _rng_rd = np.random.default_rng(1)
    _u = _rng_rd.standard_normal((_n_samples, _D_demo)) * _t_start

    # Capture every 2nd step so the timeline stays smooth without bloating
    # the payload. For FM-linear (mu=0, nu=1) the Euler step is u += f*dt.
    _frames_2d = [(_u @ _P_rd).tolist()]
    _ts_captured = [float(_ts_full[0])]
    for _i in range(_n_steps):
        _t_i = _ts_full[_i]
        _dt = _ts_full[_i + 1] - _ts_full[_i]
        _f = conditional_field(_u, _X_rd, _t_i, _cd_rd)
        _u = _u + _f * _dt
        if (_i + 1) % 2 == 0 or _i == _n_steps - 1:
            _frames_2d.append((_u @ _P_rd).tolist())
            _ts_captured.append(float(_ts_full[_i + 1]))

    _rd_config = {
        "trajectories": _frames_2d,
        "ts": _ts_captured,
        "X2": X2_data.tolist(),
    }
    _rd_w = ReverseDiffusionWidget(config_json=json.dumps(_rd_config))
    reverse_widget = mo.ui.anywidget(_rd_w)
    reverse_widget
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    That's diffusion in one lap: noise goes in, data comes out.
    Everything else in this notebook is about what happens when you
    remove the `t` from the middle step.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What this paper is actually asking

    You just saw the training pseudocode. The line that matters for
    this paper is the one that calls the network:

    ```python
    predicted = unet(x_t, t)               # ← t is "noise conditioning"
    ```

    **Stable Diffusion does this.** When you generate an image with SD,
    the model runs ~50 denoising steps from $t \approx 1$ (pure noise)
    down to $t = 0$ (clean image). At every step, the current $t$ is
    fed into the U-Net via a learned sinusoidal timestep embedding.
    The same conditioning pattern shows up across the field — sometimes
    inside a U-Net (**DALL-E 2**, **Imagen**), sometimes inside a
    transformer with a rectified-flow objective (**Flux**) — but the
    network is *always* told what $t$ is. Without that argument, it
    would not know whether to make **big corrections** at $t = 0.95$
    (still pure noise) or **polish details** at $t = 0.05$ (almost a
    finished picture).

    **This paper asks: what if we just drop the `t`?**

    ```python
    predicted = unet(x_t)                  # no t — "blind" sampler
    ```

    Naively, this should fail catastrophically. It turns out —
    experimentally (Sun et al., ICML 2025) — that for some choices of
    what the U-Net predicts, it works *anyway*. This paper explains why:
    **in high-dimensional image spaces, the noise itself leaves a
    statistical fingerprint** that the U-Net can read. The geometry of
    high-$D$ noise makes the training trick the net would need to
    "notice" $t$ almost inevitable.

    Below we prove this on a toy dataset you can explore by hand — two
    concentric circles — with no U-Net, no training. Every field is
    computed in closed form from Bayes' rule.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Why this sounds impossible

    Here's the argument *against* dropping `t`: at $t = 0.95$ the image
    $\mathbf{x}_t$ is mostly noise and the U-Net needs to make **large**
    corrections. At $t = 0.05$ it's almost clean, so corrections should
    be **tiny**. How can a single bounded neural network do both if
    it's not told which situation it's in?

    It can't — unless the image itself secretly tells the network which
    $t$ it came from. That is exactly what happens in high dimensions,
    and building the intuition for *why* is the main goal of this
    notebook.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What "dimension" means here

    We just claimed that "in high dimensions, the image itself tells the
    network which $t$ it came from." That word **dimension** is doing a
    lot of work in that sentence — let's be precise about which dimension
    we mean before we go further.

    The dimension that matters for this paper is the **ambient data
    space dimension** $D$ — the number of raw numbers in a single
    sample. For a $32 \times 32$ RGB CIFAR-10 image, $D = 3072$. For a
    $256 \times 256$ ImageNet image, $D = 196{,}608$. Not the model
    width, not the dataset size, not any latent dimension — just the
    coordinate count of the thing being denoised.

    For the rest of the notebook, **$D$ is something you control.** The
    slider below sweeps $D$ from $2$ (two-number "samples") up to $128$.
    Every figure that depends on $D$ — apple peel, shell histogram,
    posterior, 4-panel sampler — recomputes live as you drag. It's the
    single most important control in the notebook.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Reintroducing the toy in $\mathbb{R}^D$.**  You already saw our
    200 points on two concentric circles in $\mathbb{R}^2$. From here
    on, we're going to **embed those same circles into $\mathbb{R}^D$**:
    pick a random 2D plane through the origin of $\mathbb{R}^D$, rotate
    the 2D data into it, and leave the remaining $D - 2$ coordinates as
    zeros. Distances and angles are preserved exactly — we're not
    making the problem harder, we're putting the same picture into a
    room with more empty dimensions.

    Intrinsic dimension of the data: $d = 1$ (circles are 1D curves).
    Codimension: $D - d = D - 1$ — the amount of empty room around
    the data. The paper proves a hard regime boundary at $D - d > 2$ —
    below it, even the "stable" parameterizations fail.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    D_slider = mo.ui.slider(
        steps=[2, 4, 8, 16, 32, 64, 128],
        value=8,
        label="ambient dimension $D$",
        show_value=True,
        full_width=True,
    )
    mo.callout(
        mo.vstack(
            [
                mo.md(
                    "🎚️ **The notebook's main control — drag $D$ here.**  "
                    "Everything below recomputes live: the apple peel, the "
                    "shell histogram, the posterior $p(t \\mid \\mathbf{u})$, "
                    "and the 4-panel sampler experiment. "
                    "**Come back to this slider any time** as you read — "
                    "the notebook is built to be played with at multiple "
                    "values of $D$, not consumed once at $D = 8$."
                ),
                D_slider,
            ]
        ),
        kind="info",
    )
    return (D_slider,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### A surprising fact about high dimensions

    Imagine a $D$-dimensional apple — a unit ball in $\mathbb{R}^D$. Slice off
    just the outer **1% of the radius**, a paper-thin skin. How much of
    the apple did you take?

    - In ordinary 3D: about **3%**. Most of the apple is still on the table.
    - In $D = 100$: **63%**. You took most of it.
    - In $D = 1000$: **99.996%**. You took *almost everything* — the
      "core" effectively doesn't exist.

    This isn't a trick or a paradox. It's elementary geometry, and it's
    the reason every other claim in this notebook works. Drag the local
    slider to see the volume shift as you peel; drag the global $D$
    slider above to watch the picture get more violent with dimension.
    """)
    return


@app.cell(hide_code=True)
def _(PyWidget, traitlets):
    class ApplePeelWidget(PyWidget):
        config_json = traitlets.Unicode("{}").tag(sync=True)
        _py_packages = ["numpy"]

        def render(self, el, model):
            from pyodide.ffi import create_proxy
            from js import DOMPoint
            import json
            import numpy as np

            config = json.loads(model.get("config_json"))
            if not config:
                el.innerHTML = (
                    '<div style="padding:24px;color:#888;">Loading…</div>'
                )
                return

            D = int(config.get("D", 8))

            AP_W, AP_H = 280, 360
            CX = AP_W / 2
            L = (CX**2 + AP_H**2) ** 0.5  # corner-to-apex distance
            R_OUT_FRAC = 0.85  # outer arc radius as fraction of L
            R_OUT = R_OUT_FRAC * L  # apple's outer radius

            # Curve panel
            CV_W, CV_H = 380, 360
            PAD_L, PAD_B, PAD_T, PAD_R = 38, 30, 12, 8
            plot_w = CV_W - PAD_L - PAD_R
            plot_h = CV_H - PAD_B - PAD_T

            s_arr = np.linspace(0.0, 1.0, 200)
            f_arr = 1.0 - (1.0 - s_arr) ** D
            pts = []
            for s_v, f_v in zip(s_arr, f_arr):
                x = PAD_L + s_v * plot_w
                y = PAD_T + plot_h - f_v * plot_h
                pts.append(f"{x:.2f},{y:.2f}")
            curve_pts = " ".join(pts)
            fill_pts = (
                f"{PAD_L:.0f},{PAD_T + plot_h:.0f} "
                + curve_pts
                + f" {PAD_L + plot_w:.0f},{PAD_T + plot_h:.0f}"
            )

            x_ticks = ""
            for sv in [0.0, 0.25, 0.5, 0.75, 1.0]:
                tx = PAD_L + sv * plot_w
                ty = PAD_T + plot_h
                x_ticks += (
                    f'<line x1="{tx:.0f}" y1="{ty}" x2="{tx:.0f}" y2="{ty + 4}" '
                    f'stroke="#999" stroke-width="0.6"/>'
                )
                x_ticks += (
                    f'<text x="{tx:.0f}" y="{ty + 15}" text-anchor="middle" '
                    f'font-size="10" fill="#666">{int(sv * 100)}%</text>'
                )
            y_ticks = ""
            for fv in [0.0, 0.25, 0.5, 0.75, 1.0]:
                ty = PAD_T + plot_h - fv * plot_h
                y_ticks += (
                    f'<line x1="{PAD_L - 4}" y1="{ty:.0f}" x2="{PAD_L}" y2="{ty:.0f}" '
                    f'stroke="#999" stroke-width="0.6"/>'
                )
                y_ticks += (
                    f'<text x="{PAD_L - 7}" y="{ty + 3:.0f}" text-anchor="end" '
                    f'font-size="10" fill="#666">{int(fv * 100)}%</text>'
                )

            s0 = 0.10

            # Cone wedge (cream): full V from corners to V apex

            # Outer arc wall intersections (constant — arc radius is fixed)
            t_out = R_OUT / L  # parameter from V apex along wall
            x_out_l = CX * (1.0 - t_out)
            y_out_w = AP_H * (1.0 - t_out)
            x_out_r = AP_W - x_out_l

            cone_d = (
                f"M {x_out_l:.2f},{y_out_w:.2f} "
                f"A {R_OUT:.2f} {R_OUT:.2f} 0 0 1 {x_out_r:.2f},{y_out_w:.2f} "
                f"L {CX:.2f},{AP_H:.2f} Z"
            )

            def peel_path(s):
                R_in = R_OUT * (1.0 - s)
                t_in = R_in / L
                x_in_l = CX * (1.0 - t_in)
                y_in_w = AP_H * (1.0 - t_in)
                x_in_r = AP_W - x_in_l
                d = (
                    f"M {x_out_l:.2f},{y_out_w:.2f} "
                    f"A {R_OUT:.2f} {R_OUT:.2f} 0 0 1 {x_out_r:.2f},{y_out_w:.2f} "
                    f"L {x_in_r:.2f},{y_in_w:.2f} "
                    f"A {R_in:.2f} {R_in:.2f} 0 0 0 {x_in_l:.2f},{y_in_w:.2f} "
                    f"L {x_out_l:.2f},{y_out_w:.2f} Z"
                )
                handle_y = AP_H - R_in
                return d, handle_y

            peel_d0, handle_y0 = peel_path(s0)
            f0 = 1.0 - (1.0 - s0) ** D

            el.innerHTML = f"""
            <div style="max-width:720px; margin:0 auto;">
              <div style="display:flex; gap:18px; align-items:flex-start;
                          justify-content:center; flex-wrap:wrap;">

                <div style="text-align:center;">
                  <div style="font-size:11px; color:#555; margin-bottom:4px;
                              font-weight:600;">peel the apple!</div>
                  <svg viewBox="0 0 {AP_W} {AP_H}" width="{AP_W}" height="{AP_H}"
                       id="ap-apple-svg"
                       style="border:1px solid #ddd; border-radius:6px;
                              background:#ffffff; cursor:ns-resize;
                              user-select:none; touch-action:none; display:block;">
                    <path id="ap-cone" d="{cone_d}" fill="#fdf6e3" stroke="none"/>
                    <g id="ap-seeds" pointer-events="none">
                      <ellipse cx="140" cy="326" rx="5" ry="8.5" fill="#2b1d12" opacity="0.9" transform="rotate(0 140 326)"/>
                      <ellipse cx="128" cy="308" rx="4.5" ry="8" fill="#2b1d12" opacity="0.9" transform="rotate(-22 128 308)"/>
                      <ellipse cx="152" cy="308" rx="4.5" ry="8" fill="#2b1d12" opacity="0.9" transform="rotate(22 152 308)"/>
                      <ellipse cx="133" cy="290" rx="4" ry="7.5" fill="#2b1d12" opacity="0.9" transform="rotate(-12 133 290)"/>
                      <ellipse cx="147" cy="290" rx="4" ry="7.5" fill="#2b1d12" opacity="0.9" transform="rotate(12 147 290)"/>
                    </g>
                                    <path id="ap-peel" d="{peel_d0}" fill="#c0392b" stroke="none"/>
                    <circle id="ap-handle" cx="{CX}" cy="{handle_y0:.2f}" r="6"
                            fill="#fff" stroke="#c0392b" stroke-width="2.5"/>
                  </svg>
                </div>

                <div>
                  <div style="font-size:11px; color:#555; margin-bottom:4px;
                              font-weight:600; text-align:center;">
                    volume removed vs skin thickness, at D = <span id="ap-Dlabel">{D}</span>
                  </div>
                  <svg viewBox="0 0 {CV_W} {CV_H}" width="{CV_W}" height="{CV_H}"
                       style="border:1px solid #ddd; border-radius:6px; background:#fafafa;">
                    <line x1="{PAD_L}" y1="{PAD_T}" x2="{PAD_L}" y2="{PAD_T + plot_h}"
                          stroke="#ccc" stroke-width="0.6"/>
                    <line x1="{PAD_L}" y1="{PAD_T + plot_h}" x2="{PAD_L + plot_w}" y2="{PAD_T + plot_h}"
                          stroke="#ccc" stroke-width="0.6"/>
                    {x_ticks}
                    {y_ticks}
                    <polygon id="ap-fill" points="{fill_pts}" fill="#c0392b" opacity="0.18"/>
                    <polyline id="ap-curve" points="{curve_pts}" fill="none"
                              stroke="#c0392b" stroke-width="2"/>
                    <circle id="ap-dot" cx="{PAD_L + s0 * plot_w:.2f}"
                            cy="{PAD_T + plot_h - f0 * plot_h:.2f}"
                            r="5" fill="#c0392b" stroke="white" stroke-width="1.5"/>
                    <text x="{PAD_L + plot_w / 2:.0f}" y="{CV_H - 6}" text-anchor="middle"
                          font-size="11" fill="#555">
                      skin thickness  s  (% of radius)
                    </text>
                    <text x="12" y="{PAD_T + plot_h / 2:.0f}" text-anchor="middle"
                          font-size="11" fill="#555"
                          transform="rotate(-90, 12, {PAD_T + plot_h / 2:.0f})">
                      volume removed
                    </text>
                  </svg>
                </div>
              </div>

              <div style="text-align:center; margin:12px auto 0;">
                <span id="ap-readout" style="font-family:monospace; font-size:13px;
                      color:#333;">s = {s0 * 100:.1f}%  →  removes {f0 * 100:.1f}%</span>
              </div>

              <div style="text-align:center; font-size:11px; color:#666;
                          margin-top:8px; max-width:520px;
                          margin-left:auto; margin-right:auto;">
                Drag the inner arc up/down to set the peel thickness.
                Try 1% peel and watch the curve as you raise D with the
                global slider above.
              </div>
            </div>
            """

            readout = el.querySelector("#ap-readout")
            peel_path_el = el.querySelector("#ap-peel")
            handle = el.querySelector("#ap-handle")
            dot = el.querySelector("#ap-dot")
            apple_svg = el.querySelector("#ap-apple-svg")

            S_MAX = 0.95
            # handle_y = AP_H - R_OUT*(1-s)  →  s = (handle_y - (AP_H-R_OUT)) / R_OUT
            Y_OFFSET = AP_H - R_OUT

            def update(s):
                s = max(0.0, min(S_MAX, s))
                f = 1.0 - (1.0 - s) ** D
                d, handle_y = peel_path(s)
                peel_path_el.setAttribute("d", d)
                handle.setAttribute("cy", f"{handle_y:.2f}")
                dot.setAttribute("cx", f"{PAD_L + s * plot_w:.2f}")
                dot.setAttribute("cy", f"{PAD_T + plot_h - f * plot_h:.2f}")
                if f >= 0.9999:
                    f_str = f"{f * 100:.4f}%"
                elif f >= 0.999:
                    f_str = f"{f * 100:.3f}%"
                elif f >= 0.99:
                    f_str = f"{f * 100:.2f}%"
                else:
                    f_str = f"{f * 100:.1f}%"
                readout.textContent = f"s = {s * 100:.1f}%  →  removes {f_str}"

            state = {"dragging": False}

            def svg_y_from_event(event):
                ctm = apple_svg.getScreenCTM()
                if ctm is None:
                    return None
                pt = DOMPoint.new(float(event.clientX), float(event.clientY))
                return float(pt.matrixTransform(ctm.inverse()).y)

            def y_to_s(y):
                return (y - Y_OFFSET) / R_OUT

            def on_pointer_down(event):
                event.preventDefault()
                state["dragging"] = True
                try:
                    apple_svg.setPointerCapture(event.pointerId)
                except Exception:
                    pass
                y = svg_y_from_event(event)
                if y is not None:
                    update(y_to_s(y))

            def on_pointer_move(event):
                if not state["dragging"]:
                    return
                y = svg_y_from_event(event)
                if y is not None:
                    update(y_to_s(y))

            def on_pointer_up(event):
                state["dragging"] = False
                try:
                    apple_svg.releasePointerCapture(event.pointerId)
                except Exception:
                    pass

            _down = create_proxy(on_pointer_down)
            apple_svg.addEventListener("pointerdown", _down)
            _move = create_proxy(on_pointer_move)
            apple_svg.addEventListener("pointermove", _move)
            _up = create_proxy(on_pointer_up)
            apple_svg.addEventListener("pointerup", _up)
            apple_svg.addEventListener("pointercancel", _up)
            el._py_ap_down = _down
            el._py_ap_move = _move
            el._py_ap_up = _up

        def update(self, el, model):
            self.render(el, model)

    return (ApplePeelWidget,)


@app.cell(hide_code=True)
def _(ApplePeelWidget, D_slider, json, mo):
    _apple_config = {"D": int(D_slider.value)}
    _apple_w = ApplePeelWidget(config_json=json.dumps(_apple_config))
    apple_widget = mo.ui.anywidget(_apple_w)
    apple_widget
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Why?** The volume of a $D$-ball of radius $r$ scales as $r^D$
    (up to a constant). So the fraction of volume removed by a peel of
    thickness $s$ (as a fraction of the radius) is
    $$
    \frac{V_{\text{peel}}}{V_{\text{total}}} = 1 - (1 - s)^D.
    $$
    For any fixed $s > 0$, $(1-s)^D \to 0$ as $D \to \infty$. Almost
    all the volume of a high-dimensional ball lives in its outermost peel.

    **What this means for our noise vector.** Add Gaussian noise at level
    $t$ to a $D$-dimensional point: $\mathbf{u} = t\,\boldsymbol{\epsilon}$
    with $\boldsymbol{\epsilon} \sim \mathcal{N}(0, I_D)$. The squared
    length is
    $$
    \|\mathbf{u}\|^2 = t^2 \sum_{i=1}^D \epsilon_i^2,
    $$
    with mean $D t^2$ and standard deviation only $\sqrt{2D}\,t^2$ — a
    relative spread of $1/\sqrt{D}$. So $\|\mathbf{u}\| \approx t\sqrt{D}$
    to a vanishing relative error. The noise doesn't fill the ball; it
    lives on a **shell** at radius $t\sqrt{D}$, by exactly the same
    "all the volume is in the peel" mechanism.
    """)
    return


@app.cell(hide_code=True)
def _(D_slider, io, np):
    import re as _re

    import matplotlib as _mpl
    _mpl.use("agg")
    import matplotlib.pyplot as _plt

    _D = int(D_slider.value)
    _rng = np.random.default_rng(0)
    _n = 5000
    _eps = _rng.standard_normal((_n, _D))
    _eps_norms = np.linalg.norm(_eps, axis=1)

    _ts = [0.1, 0.3, 0.5, 0.7, 0.9]
    _colors = _plt.cm.viridis(np.linspace(0.12, 0.85, len(_ts)))

    _fig, _ax = _plt.subplots(figsize=(9.0, 3.9))

    _all_r = np.concatenate([t * _eps_norms for t in _ts])
    _max_r = max(_all_r.max() * 1.05, 0.1)
    _bins = np.linspace(0.0, _max_r, 70)

    for _t, _color in zip(_ts, _colors):
        _r = _t * _eps_norms
        _ax.hist(
            _r, bins=_bins, color=_color, alpha=0.55,
            edgecolor="white", linewidth=0.3, label=f"$t = {_t}$",
        )
        _mean_r = _t * np.sqrt(_D)
        _ax.axvline(
            _mean_r, color=_color, linestyle="--", lw=1.2, alpha=0.9,
        )

    _ax.set_xlabel(
        r"$\|\mathbf{u}\|$  where  $\mathbf{u} = t \cdot \boldsymbol{\epsilon}$,  "
        r"$\boldsymbol{\epsilon} \sim \mathcal{N}(0, I_D)$",
        fontsize=10,
    )
    _ax.set_ylabel("count")
    _ax.set_title(
        f"Where does a noisy sample land, at $D = {_D}$?  "
        r"(dashed lines = theoretical radius $t\sqrt{D}$)",
        fontsize=11, pad=6,
    )
    _ax.legend(loc="upper right", fontsize=9, ncol=1)
    _ax.grid(alpha=0.2)
    _ax.set_yticks([])

    _fig.tight_layout()
    _buf = io.StringIO()
    _fig.savefig(_buf, format="svg", bbox_inches="tight")
    _plt.close(_fig)
    _svg = _buf.getvalue()
    _svg = _re.sub(r'(<svg[^>]*?)\s+width="[^"]*"', r"\1", _svg, count=1)
    _svg = _re.sub(r'(<svg[^>]*?)\s+height="[^"]*"', r"\1", _svg, count=1)
    _svg = _re.sub(
        r"<svg\b",
        '<svg style="width:100%;max-width:820px;height:auto;display:block;"',
        _svg, count=1,
    )
    from marimo import Html as _Html

    _Html(f'<div style="max-width:820px;margin:0 auto;padding:4px;">{_svg}</div>')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The a-ha — reading $t$ from $\|\mathbf{u}\|$

    At **low $D$**, shells at different noise levels overlap heavily.
    If someone hands you a noisy sample $\mathbf{u}$, you cannot tell
    which $t$ it came from — the blind network genuinely has no way to
    know.

    At **high $D$**, the shells sit at well-separated radii
    $t\sqrt{D}$, each one narrower than the gap between neighbors. A
    single measurement of $\|\mathbf{u}\|$ is enough to read off $t$
    with high precision.

    **That is the "trick."** A blind diffusion model doesn't infer $t$
    from some clever learned representation. It gets $t$ for free, from
    the *geometry of high-dimensional Gaussian noise*. The real-world
    success is a consequence of the fact that real images live at
    $D \sim 10^3$–$10^5$, where the shells are enormously
    well-separated.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### From shells to the posterior $p(t \mid \mathbf{u})$

    The probabilistic statement of "you can read $t$ from $\mathbf{u}$"
    is that the **posterior distribution over the noise level**,
    $p(t \mid \mathbf{u})$, concentrates on a single value when the
    shells separate. The paper calls this **posterior collapse** and it
    is the formal core of the mechanism.

    Below is a live demo. **Click anywhere** on the scatter to place a
    probe point; the right panel shows $p(t\mid\mathbf{u})$ at that
    location. Re-click the same spot after changing $D$ and watch the
    posterior sharpen from broad and uncertain to a narrow spike.
    """)
    return


@app.cell(hide_code=True)
def _(PyWidget, traitlets):
    class PosteriorCollapseWidget(PyWidget):
        config_json = traitlets.Unicode("{}").tag(sync=True)
        _py_packages = ["numpy"]

        def render(self, el, model):
            from pyodide.ffi import create_proxy
            import json
            import numpy as np

            config = json.loads(model.get("config_json"))
            if not config:
                el.innerHTML = '<div style="padding:24px;color:#888;">Loading…</div>'
                return

            X_D = np.array(config["X_D"])
            P = np.array(config["P"])
            X2 = np.array(config["X2"])
            T_grid = np.array(config["T_grid"])
            D = config["D"]
            N = X_D.shape[0]

            def a(t):
                return 1.0 - t

            def b(t):
                return t

            def compute_posterior(px, py):
                u2 = np.array([px, py])
                u_D = P @ u2
                log_p = np.zeros(len(T_grid))
                for i, t_val in enumerate(T_grid):
                    at, bt = a(t_val), b(t_val)
                    diff = u_D[None, :] - at * X_D
                    sq = (diff * diff).sum(axis=1)
                    log_g = (
                        -sq / (2.0 * bt * bt + 1e-30)
                        - 0.5 * D * np.log(2.0 * np.pi * bt * bt + 1e-30)
                    )
                    m = log_g.max()
                    log_p[i] = m + np.log(np.exp(log_g - m).sum()) - np.log(N)
                log_p -= log_p.max()
                p = np.exp(log_p)
                return p / (p.sum() + 1e-30)

            def posterior_to_svg_path(posterior, w, h, pad_l, pad_b, pad_t):
                t_min, t_max = float(T_grid[0]), float(T_grid[-1])
                p_max = float(posterior.max()) * 1.25 if posterior.max() > 0 else 1.0
                plot_w = w - pad_l - 10
                plot_h = h - pad_b - pad_t
                pts = []
                for t_val, p_val in zip(T_grid, posterior):
                    x = pad_l + (float(t_val) - t_min) / (t_max - t_min + 1e-30) * plot_w
                    y = pad_t + plot_h - (float(p_val) / p_max) * plot_h
                    pts.append(f"{x:.1f},{y:.1f}")
                line_pts = " ".join(pts)
                x0, x1, base_y = pad_l, pad_l + plot_w, pad_t + plot_h
                fill_pts = f"{x0:.0f},{base_y:.0f} " + line_pts + f" {x1:.0f},{base_y:.0f}"
                return line_pts, fill_pts

            vb = 2.5
            LW, LH = 280, 280
            RW, RH = 340, 280
            PAD_L, PAD_B, PAD_T = 44, 32, 22

            data_dots = "".join(
                f'<circle cx="{x:.4f}" cy="{-y:.4f}" r="0.025" fill="black" opacity="0.6"/>'
                for x, y in X2
            )
            probe_x0, probe_y0 = 0.9, 0.0
            left_svg = f"""
            <svg viewBox="{-vb} {-vb} {2*vb} {2*vb}"
                 width="{LW}" height="{LH}"
                 style="cursor:crosshair; border:1px solid #ddd; border-radius:6px; background:#fafafa;"
                 id="scatter-svg">
                <circle cx="0" cy="0" r="0.6" fill="none" stroke="#aaa" stroke-width="0.015" stroke-dasharray="0.05,0.05"/>
                <circle cx="0" cy="0" r="1.2" fill="none" stroke="#aaa" stroke-width="0.015" stroke-dasharray="0.05,0.05"/>
                {data_dots}
                <circle id="probe-marker" cx="{probe_x0}" cy="{-probe_y0}" r="0.09"
                        fill="#e74c3c" stroke="white" stroke-width="0.025" opacity="0.9"/>
            </svg>
            """

            post0 = compute_posterior(probe_x0, probe_y0)
            line_pts0, fill_pts0 = posterior_to_svg_path(post0, RW, RH, PAD_L, PAD_B, PAD_T)
            plot_w = RW - PAD_L - 10
            plot_h = RH - PAD_B - PAD_T
            x_ticks_svg = ""
            for tv in [0.0, 0.25, 0.5, 0.75, 1.0]:
                tx = PAD_L + (tv - float(T_grid[0])) / (float(T_grid[-1]) - float(T_grid[0]) + 1e-30) * plot_w
                ty = PAD_T + plot_h
                x_ticks_svg += f'<line x1="{tx:.0f}" y1="{ty}" x2="{tx:.0f}" y2="{ty+4}" stroke="#999" stroke-width="0.5"/>'
                x_ticks_svg += f'<text x="{tx:.0f}" y="{ty+14}" text-anchor="middle" font-size="10" fill="#666">{tv:.2g}</text>'

            right_svg = f"""
            <svg viewBox="0 0 {RW} {RH}" width="{RW}" height="{RH}"
                 style="border:1px solid #ddd; border-radius:6px; background:#fafafa;"
                 id="posterior-svg">
                <line x1="{PAD_L}" y1="{PAD_T}" x2="{PAD_L}" y2="{PAD_T+plot_h}" stroke="#ccc" stroke-width="0.5"/>
                <line x1="{PAD_L}" y1="{PAD_T+plot_h}" x2="{PAD_L+plot_w}" y2="{PAD_T+plot_h}" stroke="#ccc" stroke-width="0.5"/>
                {x_ticks_svg}
                <text x="{PAD_L + plot_w//2}" y="{RH - 4}" text-anchor="middle" font-size="11" fill="#555">noise level t</text>
                <text x="12" y="{PAD_T + plot_h//2}" text-anchor="middle" font-size="11" fill="#555"
                      transform="rotate(-90, 12, {PAD_T + plot_h//2})">p(t | u)</text>
                <polygon id="post-fill" points="{fill_pts0}" fill="#1f77b4" opacity="0.2"/>
                <polyline id="post-line" points="{line_pts0}" fill="none" stroke="#1f77b4" stroke-width="1.8"/>
            </svg>
            """

            el.innerHTML = f"""
            <div style="display:flex; gap:20px; align-items:flex-start; flex-wrap:wrap;">
                <div>
                    <div style="font-size:12px; font-weight:600; color:#555; margin-bottom:6px;">
                        click to place probe
                    </div>
                    {left_svg}
                    <div id="probe-readout" style="font-size:11px; color:#888; margin-top:4px;">
                        probe: ({probe_x0:.2f}, {probe_y0:.2f})
                    </div>
                </div>
                <div>
                    <div style="font-size:12px; font-weight:600; color:#555; margin-bottom:6px;">
                        posterior p(t | u) at probe
                    </div>
                    {right_svg}
                </div>
            </div>
            """

            svg_el = el.querySelector("#scatter-svg")

            def on_click(event):
                pt = svg_el.createSVGPoint()
                pt.x = float(event.clientX)
                pt.y = float(event.clientY)
                ctm = svg_el.getScreenCTM()
                if ctm is None:
                    return
                svg_pt = pt.matrixTransform(ctm.inverse())
                px, py = float(svg_pt.x), -float(svg_pt.y)
                px = max(-vb, min(vb, px))
                py = max(-vb, min(vb, py))

                marker = svg_el.querySelector("#probe-marker")
                marker.setAttribute("cx", str(round(px, 4)))
                marker.setAttribute("cy", str(round(-py, 4)))

                readout = el.querySelector("#probe-readout")
                readout.textContent = f"probe: ({px:.2f}, {py:.2f})"

                post = compute_posterior(px, py)
                new_line, new_fill = posterior_to_svg_path(
                    post, RW, RH, PAD_L, PAD_B, PAD_T
                )
                el.querySelector("#post-fill").setAttribute("points", new_fill)
                el.querySelector("#post-line").setAttribute("points", new_line)

            # Store the proxy on the DOM element so Pyodide's GC doesn't
            # collect the Python closure while the DOM listener is still
            # attached (`self` isn't bound in the extracted render body).
            _click_proxy = create_proxy(on_click)
            svg_el.addEventListener("click", _click_proxy)
            svg_el._py_click_proxy = _click_proxy

        def update(self, el, model):
            self.render(el, model)

    return (PosteriorCollapseWidget,)


@app.cell(hide_code=True)
def _(
    D_slider,
    PosteriorCollapseWidget,
    X2_data,
    json,
    lift,
    mo,
    np,
    random_lift,
):
    _D = int(D_slider.value)
    _P = random_lift(_D, seed=42)
    _X_D = lift(X2_data, _P)
    _config = {
        "X_D": _X_D.tolist(),
        "P": _P.tolist(),
        "X2": X2_data.tolist(),
        "T_grid": np.linspace(0.005, 0.99, 80).tolist(),
        "D": _D,
    }
    _w = PosteriorCollapseWidget(config_json=json.dumps(_config))
    posterior_widget = mo.ui.anywidget(_w)
    posterior_widget
    return


@app.cell(hide_code=True)
def _(D_slider, mo):
    _D = int(D_slider.value)

    if _D <= 2:
        _explanation = (
            f"At $D = {_D}$, the posterior still has a mode near $t \\approx 0$ "
            "when the probe is close to the data — the geometry provides "
            "*some* signal. But notice the **long heavy tail** stretching "
            "toward $t = 1$. At this dimension the shells overlap too much "
            "for the posterior to commit."
        )
    elif _D <= 16:
        _explanation = (
            f"At $D = {_D}$, the posterior is visibly **sharper** than at "
            "$D = 2$ — the tail has receded. The shells are starting to "
            "separate."
        )
    elif _D <= 48:
        _explanation = (
            f"At $D = {_D}$, the posterior is **concentrated into a narrow "
            "spike**. Shells are well-separated. The sampler effectively "
            "'knows' $t$ from the geometry alone."
        )
    else:
        _explanation = (
            f"At $D = {_D}$, the posterior is effectively a **Dirac delta**. "
            "The high-$D$ regime has fully kicked in."
        )

    mo.vstack([
        mo.md(_explanation),
        mo.callout(
            mo.md(
                "This posterior is recomputed **in-browser** on every click "
                "using [pywidget](https://github.com/ktaletsk/pywidget) — "
                "pure Python running in Pyodide, zero kernel round-trip."
            ),
            kind="neutral",
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Does the shell picture actually rescue sampling?

    Posterior collapse says a blind U-Net *can* infer $t$ from the
    geometry. But that's only part of the story. When we actually run
    the reverse-time sampler, does it produce correct samples? And does
    it matter what the U-Net was trained to predict?

    Let's run the experiment.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What does the network predict?

    The default training convention — and what you'll see in almost any
    tutorial — trains the U-Net to predict the **noise** `eps` that was
    added. This is the DDPM convention. But you could equivalently train
    it to predict:

    | Target | Predicts | Method |
    |:---|:---|:---|
    | `eps` | the noise that was added | **DDPM** (noise prediction) |
    | `x` | the clean image directly | **EDM** (signal prediction) |
    | `v = eps - x` | the "velocity" from clean to noise | **Flow Matching** (velocity prediction) |

    When the U-Net *sees* $t$, all three are equivalent: given any two
    of $\{x_t, x, \text{eps}\}$ you can compute the third. When the
    U-Net is **blind**, each target has to implicitly *average over the
    posterior on $t$*. The three averages behave very differently. Two
    of them are stable; one of them is fragile. The widget below shows
    all four side by side:

    1. **FM conditional** — velocity prediction *with* $t$. Baseline.
    2. **FM blind** — same target, no $t$. Paper says: should work.
    3. **EDM blind** — signal target, no $t$. Paper predicts stable —
       but the authors never actually test it. This is our extension.
    4. **DDPM blind** — noise target, no $t$. Paper's stress test.
    """)
    return


@app.cell(hide_code=True)
def _(PyWidget, traitlets):
    class GeometryOfNoiseWidget(PyWidget):
        payload_json = traitlets.Unicode("{}").tag(sync=True)
        _py_packages = ["numpy", "matplotlib"]

        def render(self, el, model):
            import io
            import json
            import re

            import matplotlib
            matplotlib.use("agg")
            import matplotlib.pyplot as plt

            payload = json.loads(model.get("payload_json"))
            D = payload.get("D", 0)
            circles = payload.get("circles", [])
            panels = payload.get("panels", [])

            if not panels:
                el.innerHTML = (
                    '<div style="padding:24px;color:#888;">Computing…</div>'
                )
                return

            n_panels = len(panels)
            fig, axes = plt.subplots(
                1, n_panels, figsize=(3.2 * n_panels, 3.8),
            )
            if n_panels == 1:
                axes = [axes]

            cx = [p[0] for p in circles]
            cy = [p[1] for p in circles]

            ROLE_COLORS = {
                "baseline":  "#555555",
                "stable":    "#2e7d32",
                "unstable":  "#c62828",
            }

            for ax, panel in zip(axes, panels):
                sx = [p[0] for p in panel["samples"]]
                sy = [p[1] for p in panel["samples"]]
                role = panel.get("role", "baseline")
                sub = panel.get("sub", "")
                color = ROLE_COLORS.get(role, "#555555")
                ax.scatter(
                    sx, sy, s=14, c="#1f77b4", alpha=0.55,
                    linewidths=0, zorder=2,
                )
                ax.scatter(
                    cx, cy, s=5, c="black", alpha=0.85,
                    linewidths=0, zorder=3,
                )
                ax.set_title(
                    panel["title"], fontsize=11,
                    color=color, fontweight="bold", pad=8,
                )
                ax.set_xlim(-2.2, 2.2)
                ax.set_ylim(-2.2, 2.2)
                ax.set_aspect("equal")
                ax.set_xticks([])
                ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_edgecolor(color)
                    s.set_linewidth(1.2)
                    s.set_alpha(0.55)

            fig.suptitle(
                f"$D = {D}$   ·   closed-form analytics, no NN training",
                y=1.02, fontsize=12,
            )
            fig.tight_layout()

            buf = io.StringIO()
            fig.savefig(buf, format="svg", bbox_inches="tight")
            plt.close(fig)
            svg = buf.getvalue()

            svg = re.sub(r'(<svg[^>]*?)\s+width="[^"]*"', r"\1", svg, count=1)
            svg = re.sub(r'(<svg[^>]*?)\s+height="[^"]*"', r"\1", svg, count=1)
            svg = re.sub(
                r"<svg\b",
                '<svg style="width:100%;max-width:100%;height:auto;display:block;"',
                svg, count=1,
            )

            # Inline legend: one label per panel, flex-aligned with the
            # equal-width subplots in the SVG below.
            role_labels = []
            for panel in panels:
                role = panel.get("role", "baseline")
                sub = panel.get("sub", "")
                color = ROLE_COLORS.get(role, "#555555")
                bg = {"baseline": "#f5f5f5", "stable": "#e8f5e9", "unstable": "#ffebee"}.get(role, "#f5f5f5")
                role_labels.append(
                    f'<div style="flex:1; text-align:center; font-size:11px; '
                    f'padding:3px 4px; color:{color}; background:{bg}; '
                    f'border-radius:4px; margin:0 2px;">{sub}</div>'
                )
            legend_html = (
                '<div style="display:flex; max-width:1100px; margin:0 auto 2px; gap:4px;">'
                + "".join(role_labels)
                + '</div>'
            )

            el.innerHTML = (
                '<div style="width:100%;max-width:1100px;margin:0 auto;padding:4px;">'
                + legend_html
                + svg
                + '</div>'
            )

        def update(self, el, model):
            self.render(el, model)

    return (GeometryOfNoiseWidget,)


@app.cell(hide_code=True)
def _(
    D_slider,
    GeometryOfNoiseWidget,
    X2_data,
    compute_panels_for_D,
    json,
    mo,
):
    _D = int(D_slider.value)
    _panels = compute_panels_for_D(_D)
    _payload = {
        "D": _D,
        "circles": X2_data.tolist(),
        "panels": _panels,
    }
    _w = GeometryOfNoiseWidget(payload_json=json.dumps(_payload))
    widget = mo.ui.anywidget(_w)
    widget
    return


@app.cell(hide_code=True)
def _(D_slider, mo):
    _D = int(D_slider.value)
    _codim = _D - 1
    _regime = (
        "**Regime failure** — codimension $D-d = "
        + str(_codim)
        + " < 2$. Theory predicts its own boundary."
        if _codim < 2
        else (
            "**Regime I** — high-$D$ concentration dominates. Even DDPM blind converges."
            if _D >= 64
            else "**Middle regime** — parameterization matters. Compare green vs red."
        )
    )
    mo.md(
        f"$D = {_D}$, codimension $D - d = {_codim}$.  {_regime}"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    **What you just saw.** Four samplers drew from the same distribution
    using the same closed-form reverse-time ODE. The only differences:
    what the network predicts and whether it gets $t$. Three regimes:

    - **$D = 2$** — shells overlap completely. Even the green panels
      fail; **EDM blind** collapses all samples to the centroid.
    - **$D = 8$–$32$** — shells start separating. Green panels match
      the conditional baseline. **DDPM blind** (red) is visibly noisier.
    - **$D \geq 64$** — shells are disjoint. Even DDPM blind cleans up.

    The shell picture explains most of what you see — but **why does
    DDPM blind stay scattered in the middle regime**, when its cousins
    don't? That's where the three parameterizations stop being
    equivalent.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(
            r"""
            **Physicist sidebar — the Brownian walker's rule of thumb.**
            The codimension threshold $D - d > 2$ that the paper needs for
            near-manifold concentration is *the same threshold* that separates
            recurrent from transient Brownian motion: a Brownian walker in
            $\mathbb{R}^k$ almost surely returns to the origin for $k \leq 2$
            and escapes to infinity for $k > 2$. What the paper calls
            "Regime II" is rediscovering this classical stochastic-analysis
            fact in a modern diffusion-model setting. The posterior at
            $(u_1, u_2) = (1.2, 0)$ is Inverse-Gamma with shape
            $(D - d)/2 - 1$; it concentrates iff the shape is positive, i.e.
            iff $D - d > 2$.
            """
        ),
        kind="info",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## That's the geometry of noise

    You watched the codimension boundary $D - d > 2$ surface in three
    places:

    - **Apple peel** — almost all the volume of a $D$-ball lives in its
      outermost shell.
    - **Shell histogram** — Gaussian noise concentrates at radius
      $t\sqrt{D}$, with shells separating cleanly as $D$ grows.
    - **4-panel sampler** — even DDPM-blind, the most fragile
      parameterization, recovers once $D - d > 2$.

    The headline of the paper is one line: *the U-Net does not need
    $t$, because the noise itself encodes $t$ in high enough $D$.*
    Everything you saw here is the geometry of why that
    line is true.

    Suggestion: if the slider stayed at $D = 8$ the whole time, scroll back and
    try $D = 2$, $D = 16$, $D = 64$ — every figure recomputes.
    """)
    return


@app.cell(hide_code=True)
def _(np):
    # FM-linear schedule: a(t)=1-t, b(t)=t.  t=0 data, t=1 noise.
    def a(t):
        return 1.0 - t

    def b(t):
        return t

    def adot(t):
        return -1.0

    def bdot(t):
        return 1.0

    # Parameterizations: network target = c * x + d * eps
    PARAMS = {
        "FM":   (-1.0, 1.0),  # velocity
        "EDM":  ( 1.0, 0.0),  # signal
        "DDPM": ( 0.0, 1.0),  # noise
    }

    def make_circles(n=200, r1=0.6, r2=1.2):
        n_per = n // 2
        th = np.linspace(0.0, 2.0 * np.pi, n_per, endpoint=False)
        inner = np.stack([r1 * np.cos(th), r1 * np.sin(th)], axis=1)
        outer = np.stack(
            [r2 * np.cos(th + np.pi / n_per), r2 * np.sin(th + np.pi / n_per)],
            axis=1,
        )
        return np.concatenate([inner, outer], axis=0)

    def random_lift(D, seed=42):
        rng = np.random.default_rng(seed)
        M = rng.standard_normal((D, 2))
        Q, _ = np.linalg.qr(M)
        return Q

    def lift(X2, P):
        return X2 @ P.T

    X2_data = make_circles(n=200)
    return PARAMS, X2_data, a, adot, b, bdot, lift, random_lift


@app.cell(hide_code=True)
def _(a, adot, b, bdot, np):
    def softmax_weights(u, X, t):
        at, bt = a(t), b(t)
        diff = u[:, None, :] - at * X[None, :, :]
        sq = np.einsum("bnd,bnd->bn", diff, diff)
        log_w = -sq / (2.0 * bt * bt + 1e-30)
        log_w -= log_w.max(axis=-1, keepdims=True)
        w = np.exp(log_w)
        return w / w.sum(axis=-1, keepdims=True)

    def denoiser(u, X, t):
        return softmax_weights(u, X, t) @ X

    def conditional_field(u, X, t, cd):
        c, d = cd
        at, bt = a(t), b(t)
        x_star = denoiser(u, X, t)
        eps_star = (u - at * x_star) / (bt + 1e-30)
        return c * x_star + d * eps_star

    def log_p_u_given_t(u, X, t):
        at, bt = a(t), b(t)
        D_dim = X.shape[1]
        diff = u[:, None, :] - at * X[None, :, :]
        sq = np.einsum("bnd,bnd->bn", diff, diff)
        log_g = (
            -sq / (2.0 * bt * bt + 1e-30)
            - 0.5 * D_dim * np.log(2.0 * np.pi * bt * bt + 1e-30)
        )
        m = log_g.max(axis=-1, keepdims=True)
        return (
            m.squeeze(-1)
            + np.log(np.exp(log_g - m).sum(axis=-1))
            - np.log(X.shape[0])
        )

    def posterior_t(u, X, T_grid):
        log_p = np.stack([log_p_u_given_t(u, X, ti) for ti in T_grid], axis=1)
        log_p -= log_p.max(axis=-1, keepdims=True)
        p = np.exp(log_p)
        return p / p.sum(axis=-1, keepdims=True)

    def autonomous_field(u, X, T_grid, cd):
        """Vectorized f*(u) = sum_t p(t|u) f_t*(u)."""
        c, d = cd
        aT = np.asarray(a(T_grid))
        bT = np.asarray(b(T_grid))
        D_dim = X.shape[1]
        N = X.shape[0]

        u_sq = (u * u).sum(axis=-1)
        X_sq = (X * X).sum(axis=-1)
        aT_sq_Xsq = (aT * aT)[:, None] * X_sq[None, :]
        cross = np.einsum("bd,nd->bn", u, X)
        cross_T = aT[None, :, None] * cross[:, None, :]
        sq_dists = u_sq[:, None, None] + aT_sq_Xsq[None, :, :] - 2.0 * cross_T

        inv_2bsq = 1.0 / (2.0 * bT * bT + 1e-30)
        log_k = -sq_dists * inv_2bsq[None, :, None]
        log_k_max = log_k.max(axis=-1, keepdims=True)
        w_unnorm = np.exp(log_k - log_k_max)
        w_row_sum = w_unnorm.sum(axis=-1, keepdims=True)
        weights = w_unnorm / w_row_sum

        log_row_sum = log_k_max.squeeze(-1) + np.log(w_row_sum.squeeze(-1))
        log_norm = -0.5 * D_dim * np.log(2.0 * np.pi * bT * bT + 1e-30)
        log_p_u_t = log_row_sum - np.log(N) + log_norm[None, :]

        log_p_u_t -= log_p_u_t.max(axis=-1, keepdims=True)
        post = np.exp(log_p_u_t)
        post = post / post.sum(axis=-1, keepdims=True)

        x_star = np.einsum("btn,nd->btd", weights, X)
        eps_star = (
            u[:, None, :] - aT[None, :, None] * x_star
        ) / (bT[None, :, None] + 1e-30)
        f_t = c * x_star + d * eps_star
        return np.einsum("bt,btd->bd", post, f_t)

    def mu_nu(t, cd):
        c, d = cd
        at, bt, ad, bd = a(t), b(t), adot(t), bdot(t)
        det = at * d - bt * c
        mu = (ad * d - bd * c) / det
        nu = (bd * at - ad * bt) / det
        return mu, nu

    def sample(
        X, cd, *,
        n_samples=120, n_steps=120,
        blind=True, seed=0,
        t_start=0.99, t_end=0.005,
    ):
        rng = np.random.default_rng(seed)
        D_dim = X.shape[1]
        u = rng.standard_normal((n_samples, D_dim)) * b(t_start)
        T_grid = np.linspace(t_end, t_start, 48)
        ts = np.exp(np.linspace(np.log(t_start), np.log(t_end), n_steps + 1))
        for i in range(n_steps):
            t = ts[i]
            dt = ts[i + 1] - ts[i]
            mu_t, nu_t = mu_nu(t, cd)
            if blind:
                f = autonomous_field(u, X, T_grid, cd)
            else:
                f = conditional_field(u, X, t, cd)
            u = u + (mu_t * u + nu_t * f) * dt
        return u

    return conditional_field, log_p_u_given_t, sample


@app.cell(hide_code=True)
def _(PARAMS, X2_data, lift, lru_cache, np, random_lift, sample):
    @lru_cache(maxsize=32)
    def compute_panels_for_D(D: int):
        P = random_lift(D, seed=42)
        X = lift(X2_data, P)
        spec = [
            ("FM, conditional", "FM",   False,
             "baseline", "sees t · always works"),
            ("FM, blind",       "FM",   True,
             "stable",   "paper's headline claim"),
            ("EDM, blind",      "EDM",  True,
             "stable",   "paper predicts · never tests"),
            ("DDPM, blind",     "DDPM", True,
             "unstable", "paper's stress test"),
        ]
        out = []
        for title, key, blind, role, sub in spec:
            cd = PARAMS[key]
            s = sample(X, cd, n_samples=200, n_steps=120, blind=blind, seed=1)
            s2 = np.asarray(s @ P)
            out.append({
                "title": title,
                "sub": sub,
                "role": role,
                "samples": s2.tolist(),
            })
        return out

    return (compute_panels_for_D,)


@app.cell(hide_code=True)
def _(X2_data, lift, log_p_u_given_t, lru_cache, np, random_lift):
    @lru_cache(maxsize=32)
    def compute_emarg_for_D(D: int):
        """Compute E_marg on a 2D grid in the data subspace. Cached per D."""
        P = random_lift(D, seed=42)
        X = lift(X2_data, P)
        res = 50
        xx = np.linspace(-2.0, 2.0, res)
        gx, gy = np.meshgrid(xx, xx)
        grid_2d = np.stack([gx.ravel(), gy.ravel()], axis=1)
        grid_D = grid_2d @ P.T
        T_grid = np.linspace(0.005, 0.99, 64)
        log_p_all = np.stack(
            [log_p_u_given_t(grid_D, X, t) for t in T_grid], axis=1
        )
        dt = T_grid[1] - T_grid[0]
        log_terms = log_p_all + np.log(dt)
        m = log_terms.max(axis=1, keepdims=True)
        log_p_u = m.squeeze() + np.log(np.exp(log_terms - m).sum(axis=1))
        return gx, gy, (-log_p_u).reshape(res, res)

    return


if __name__ == "__main__":
    app.run()
