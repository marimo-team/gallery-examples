# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "colorcet==3.2.1",
#     "matplotlib==3.10.8",
#     "numpy==2.4.4",
#     "torch==2.10.0",
#     "wigglystuff==0.5.9",
# ]
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(
    width="medium",
    css_file="/usr/local/_marimo/custom.css",
    auto_download=["html"],
)


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import colorcet as cc  # noqa: F401 — registers the cet_* colormaps with matplotlib
    import torch
    import matplotlib.pyplot as plt
    from wigglystuff import CurveEditor

    return CurveEditor, mo, np, plt, torch


@app.cell
def _(mo):
    mo.md(r"""
    # Julia Sets via *Lagrangian Descriptors*

    A **Lagrangian Descriptor (LD)** is a tool from dynamical-systems theory
    that reveals the hidden geometry of phase space by accumulating a positive
    quantity along trajectories. Invariant structures (stable/unstable
    manifolds) emerge as **singularities** of the resulting scalar field.

    Lifting the orbit $z_0 \mapsto f(z_0) \mapsto \dots$ onto the sphere as
    $w^{(n)}=(w_1^{(n)},w_2^{(n)},w_3^{(n)})\in S^2$, the *Discrete Lagrangian
    Descriptor* of the initial point $z_0$ is the **component-wise** sum

    $$\mathcal{M}_p(z_0) \;=\; \sum_{n=0}^{N-1}\;\sum_{i=1}^{3}\,
    \bigl|\,w_i^{(n+1)}-w_i^{(n)}\,\bigr|^{\,p},
    \qquad 0<p<1 .$$

    Applying it to the quadratic map $f(z)=z^2+c$ yields the **Julia sets**:
    the Julia set $J(f)$ is the locus of sensitive dependence on initial
    conditions, and the field $\mathcal{M}_p$ develops its **singular features**
    (loss of differentiability) precisely along $J(f)$.
    By visualizing the **gradient** of $\mathcal{M}_p$, the Julia set appears
    as a web of luminous filaments.

    > **Key trick.** Orbits of $z^2+c$ can escape to $\infty$.
    > To keep the accumulated increments finite even near infinity, we *lift*
    > the dynamics onto the **Riemann sphere** $S^2\subset\mathbb{R}^3$ via the
    > inverse stereographic projection, and sum the per-coordinate increments
    > $|\Delta w_i|^{\,p}$ there (the standard discrete-LD form).

    *Method: S. Conradi — Discrete Lagrangian Descriptors for Julia sets
    (arXiv:2001.08937).*
    """)
    return


@app.cell
def _(torch):
    # GPU-accelerated core (PyTorch / CUDA, float64 for parity with NumPy).
    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _DTYPE = torch.float64


    def stereographic_inverse(x, y):
        """Plane -> Riemann sphere S^2 (inverse stereographic projection)."""
        z2 = x * x + y * y
        return 2 * x / (z2 + 1), 2 * y / (z2 + 1), (z2 - 1) / (z2 + 1)


    def quadratic_map_on_sphere(w1, w2, w3, a, b):
        """One step of f(z) = z^2 + (a + i b) lifted onto the sphere S^2."""
        w3m1 = (1 - w3) ** 2
        safe = torch.isclose(w3m1, torch.zeros_like(w3m1))
        one = torch.ones_like(w1)
        r12 = torch.where(safe, one, w1 * w2 / w3m1)
        r1 = torch.where(safe, one, w1 * w1 / w3m1)
        r2 = torch.where(safe, one, w2 * w2 / w3m1)
        q1 = (b + 2 * r12) ** 2
        q2 = a + r1 - r2
        den = 1 + q1 + q2 * q2
        return 2 * q2 / den, (2 * b + 4 * r12) / den, (q1 + q2 * q2 - 1) / den


    def discrete_lagrangian_descriptor(w1, w2, w3, a, b, n_iter=150, p=0.05):
        """Accumulate per-coordinate |Δ|^p increments of the orbit on S^2 (runs on GPU)."""
        acc = torch.zeros_like(w1)
        for _ in range(n_iter):
            n1, n2, n3 = quadratic_map_on_sphere(w1, w2, w3, a, b)
            acc = acc + torch.abs(n1 - w1) ** p + torch.abs(n2 - w2) ** p + torch.abs(n3 - w3) ** p
            w1, w2, w3 = n1, n2, n3
        return acc


    def edge_gradient(field):
        """Centered finite-difference magnitude: spikes on the singular (Julia) set."""
        gy = torch.roll(field, 1, 0) - torch.roll(field, -1, 0)
        gx = torch.roll(field, 1, 1) - torch.roll(field, -1, 1)
        return torch.sqrt(gx * gx + gy * gy)


    def sphere_grid(res, span=1.4):
        """Lifted (w1, w2, w3) coords of a res x res grid over [-span, span]^2."""
        lin = torch.linspace(-span, span, res, device=_DEVICE, dtype=_DTYPE)
        gx, gy = torch.meshgrid(lin, lin, indexing="xy")
        return stereographic_inverse(gx, gy)


    def julia_dld(res, a, b, n_iter, p, span=1.4):
        """DLD field and its edge-gradient as NumPy arrays, computed on the GPU."""
        dld = discrete_lagrangian_descriptor(*sphere_grid(res, span), float(a), float(b), n_iter, p)
        return dld.cpu().numpy(), edge_gradient(dld).cpu().numpy()

    return (
        discrete_lagrangian_descriptor,
        edge_gradient,
        julia_dld,
        sphere_grid,
    )


@app.cell
def _(mo):
    SIZE = 1.6
    controls = (
        mo.md(
            r"""
            ### Parameters

            Complex parameter $c = a + b\,i$

            {re}{im}

            {niter}

            {pexp}

            {res} {view} {cmap}
            """
        )
        .batch(
            re=mo.ui.number(-2.0, 2.0, value=-0.123, step=0.001, label="Re(c) = a"),
            im=mo.ui.number(-2.0, 2.0, value=0.745, step=0.001, label="Im(c) = b"),
            niter=mo.ui.slider(20, 400, value=150, step=10, label="Iterations N", show_value=True),
            pexp=mo.ui.slider(0.02, 1.0, value=0.05, step=0.01, label="Exponent p", show_value=True),
            res=mo.ui.dropdown(
                {"400 (fast)": 400, "600": 600, "800": 800, "1000 (slow)": 1000},
                value="600", label="Resolution",
            ),
            view=mo.ui.dropdown(
                ["Gradient (Julia set)", "Descriptor field"],
                value="Gradient (Julia set)", label="View",
            ),
            cmap=mo.ui.dropdown(
                ["magma", "inferno", "cividis", "twilight_shifted", "gnuplot2", "bone", "cet_CET_L20", "cet_CET_L19", "YlGnBu"],
                value="magma", label="Colormap",
            ),
        )
        .form(submit_button_label="Render", show_clear_button=False, bordered=True)
    )
    return SIZE, controls


@app.cell
def _(SIZE, controls, julia_dld):
    def _():
        defaults = {
            "re": -0.123, "im": 0.745, "niter": 150, "pexp": 0.05,
            "res": 600, "view": "Gradient (Julia set)", "cmap": "magma",
        }
        params = controls.value or defaults
        dld, grad = julia_dld(
            params["res"], params["re"], params["im"],
            params["niter"], params["pexp"], span=SIZE,
        )
        return params, dld, grad


    P, dld, grad = _()
    return P, dld, grad


@app.cell
def _(P, SIZE, dld, grad, plt):
    def _():
        if P["view"].startswith("Gradient"):
            img = grad ** 0.3
            vmin, vmax = 0.0, 1.5
        else:
            img = dld ** 0.1
            vmin = vmax = None

        fig, ax = plt.subplots(figsize=(7, 7))
        fig.set_facecolor("#f4f0e8")
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
        ax.imshow(
            img, cmap=P["cmap"], vmin=vmin, vmax=vmax,
            extent=[-SIZE, SIZE, -SIZE, SIZE], interpolation="lanczos", origin="lower",
        )
        ax.set_axis_off()
        ax.set_title(
            f"$f(z)=z^2 + ({P['re']:.3f}{'+' if P['im']>=0 else '-'}{abs(P['im']):.3f}i)$",
            fontsize=13, pad=8,
        )
        return fig


    julia_plot = _()
    return (julia_plot,)


@app.cell
def _(controls, julia_plot, mo):
    mo.hstack([controls, julia_plot], widths=[1, 1.6], align="center", gap=2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Animating $c$ in the parameter plane

    Draw a **path of points** in the plane $c=a+b\,i$: the editor sweeps along
    the curve (press **play**) and the Julia set on the right is **recomputed**
    at the current value of $c$. The seed points trace a loop near the boundary
    of the Mandelbrot set, where the Julia geometry changes most dramatically.
    Try different interpolators (Catmull-Rom, cardinal, natural) and enable
    **closed**/**loop** for a periodic animation.
    """)
    return


@app.cell
def _(CurveEditor, mo):
    # Seed pucks: a loop near the Mandelbrot boundary (ordered by angle).
    c_path_widget = CurveEditor(
        points=[
            {"x": 0.285, "y": 0.450},
            {"x": 0.000, "y": 0.660},
            {"x": -0.123, "y": 0.745},
            {"x": -0.400, "y": 0.600},
            {"x": -0.800, "y": 0.160},
        ],
        curve="catmull_rom",
        alpha=0.5,
        closed=True,
        loop=True,
        x_bounds=(-1.6, 1.0),
        y_bounds=(-1.3, 1.3),
        width=400,
        height=400,
        duration_ms=18000,
        sync_throttle_ms=90,
        show_axes=True,
        n_samples=200,
    )
    c_path = mo.ui.anywidget(c_path_widget)

    live_res = mo.ui.dropdown(
        {"250 (smooth)": 250, "350": 350, "500 (detail)": 500},
        value="350", label="Live resolution",
    )
    live_niter = mo.ui.slider(40, 250, value=100, step=10, label="Iterations N", show_value=True)
    live_p = mo.ui.slider(0.02, 1.0, value=0.05, step=0.01, label="Exponent p", show_value=True)
    live_vmax = mo.ui.slider(0.3, 3.0, value=1.5, step=0.1, label="vmax (gradient)", show_value=True)
    anim_view = mo.ui.dropdown(
        ["Gradient (Julia set)", "Descriptor field"],
        value="Gradient (Julia set)", label="View",
    )
    anim_cmap = mo.ui.dropdown(
        ["magma", "inferno", "cividis", "twilight_shifted", "gnuplot2", "bone", "YlGnBu", "cet_linear_bmw_5_95_c89", "cet_CET_CBL2", "cet_CET_L8"],
        value="magma", label="Colormap",
    )
    return (
        anim_cmap,
        anim_view,
        c_path,
        c_path_widget,
        live_niter,
        live_p,
        live_res,
        live_vmax,
    )


@app.cell
def _(
    SIZE,
    anim_cmap,
    anim_view,
    c_path,
    c_path_widget,
    julia_dld,
    live_niter,
    live_p,
    live_res,
    live_vmax,
    mo,
    plt,
):
    def _():
        a = float(c_path.x)
        b = float(c_path.y)
        dld, grad = julia_dld(live_res.value, a, b, live_niter.value, live_p.value, span=SIZE)

        if anim_view.value.startswith("Gradient"):
            img = grad ** 0.3
            vmin, vmax = 0.0, live_vmax.value
        else:
            img = dld ** 0.1
            vmin = vmax = None

        sig = "#383b3e"
        fig, ax = plt.subplots(figsize=(6, 6))
        fig.patch.set_facecolor("#f4f0e8")
        fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
        ax.imshow(
            img, cmap=anim_cmap.value, vmin=vmin, vmax=vmax,
            extent=[-SIZE, SIZE, -SIZE, SIZE], interpolation="lanczos", origin="lower",
        )
        ax.set_axis_off()
        ax.text(
            0.03, 0.97, f"$c = {a:.3f}{'+' if b >= 0 else '-'}{abs(b):.3f}\\,i$",
            transform=ax.transAxes, ha="left", va="top",
            color=sig, fontsize=13,
            bbox=dict(boxstyle="round,pad=0.3", fc="#f4f0e8", ec="none", alpha=0.6),
        )

        # Inset: the (a, b) path with a playhead at current c.
        samples = c_path_widget.samples
        if len(samples) >= 2:
            sx = [s["x"] for s in samples]
            sy = [s["y"] for s in samples]
            inset = ax.inset_axes([0.72, 0.72, 0.26, 0.26])
            inset.set_facecolor("none")
            inset.plot(sx, sy, color=sig, lw=1.0, alpha=0.55)
            inset.plot([a], [b], "o", color="#d63a2f", ms=5)
            inset.set_xlim(-1.6, 1.0)
            inset.set_ylim(-1.3, 1.3)
            inset.set_aspect("equal")
            inset.set_xticks([])
            inset.set_yticks([])
            inset.set_xlabel(r"$a$", fontsize=9, color=sig, labelpad=1)
            inset.set_ylabel(r"$b$", fontsize=9, color=sig, labelpad=1, rotation=0)
            for spine in inset.spines.values():
                spine.set_edgecolor(sig)
                spine.set_alpha(0.4)
                spine.set_linewidth(0.6)

        render_controls = mo.vstack([
            mo.md("**rendering**"),
            live_res, live_niter, live_p, live_vmax, anim_view, anim_cmap,
        ])
        left = mo.vstack([mo.md("**path in $(a, b)$**"), c_path, render_controls])

        return mo.hstack([left, fig], justify="space-around", align="start", widths=[0.5, 0.5])


    _()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Export high-resolution animation

    Resample the current path, compute the *Lagrangian Descriptor* for every
    frame at the settings below, and encode an **MP4**. It uses the **cmap**,
    **view** and **vmax** from the live panel; **frames**, **resolution**,
    **iterations**, **p**, **dpi** and **fps** are independent.
    """)
    return


@app.cell
def _(mo):
    export_frames = mo.ui.slider(60, 600, value=240, step=10, label="frames", show_value=True)
    export_res = mo.ui.slider(300, 1200, value=600, step=50, label="resolution", show_value=True)
    export_niter = mo.ui.slider(50, 800, value=150, step=10, label="iterations N", show_value=True)
    export_p = mo.ui.slider(0.02, 1.0, value=0.05, step=0.01, label="exponent p", show_value=True)
    export_dpi = mo.ui.slider(72, 300, value=150, step=6, label="dpi", show_value=True)
    export_fps = mo.ui.slider(15, 60, value=30, step=1, label="fps", show_value=True)
    export_filename = mo.ui.text(value="julia_dld_animation.mp4", label="output file", full_width=True)
    export_button = mo.ui.run_button(label="Render & save")

    mo.hstack([
        mo.vstack([export_frames, export_res, export_niter, export_p]),
        mo.vstack([export_dpi, export_fps, export_filename, export_button]),
    ], justify="start", align="start", gap=2)
    return (
        export_button,
        export_dpi,
        export_filename,
        export_fps,
        export_frames,
        export_niter,
        export_p,
        export_res,
    )


@app.cell
def _(
    SIZE,
    anim_cmap,
    anim_view,
    c_path_widget,
    discrete_lagrangian_descriptor,
    edge_gradient,
    export_button,
    export_dpi,
    export_filename,
    export_fps,
    export_frames,
    export_niter,
    export_p,
    export_res,
    live_vmax,
    mo,
    np,
    plt,
    sphere_grid,
):
    def _():
        import matplotlib.animation as manim
        from pathlib import Path

        if not export_button.value:
            return mo.md("_(press **Render & save** to encode an MP4.)_")

        samples = c_path_widget.samples
        if len(samples) < 2:
            return mo.md("**The path has no samples — add at least 2 points.**")

        xs = np.array([s["x"] for s in samples], dtype=float)
        ys = np.array([s["y"] for s in samples], dtype=float)
        t_src = np.linspace(0.0, 1.0, len(xs))
        t_dst = np.linspace(0.0, 1.0, export_frames.value)
        a_path = np.interp(t_dst, t_src, xs)
        b_path = np.interp(t_dst, t_src, ys)

        res = export_res.value
        n_iter = export_niter.value
        p = export_p.value
        cmap = anim_cmap.value
        use_grad = anim_view.value.startswith("Gradient")
        vmax = live_vmax.value
        sig = "w"

        # Lift the grid onto the sphere once; reuse across every frame on the GPU.
        w1, w2, w3 = sphere_grid(res, span=SIZE)

        fig, ax = plt.subplots(figsize=(6, 6), dpi=export_dpi.value)
        fig.patch.set_facecolor("#f4f0e8")
        fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
        ax.set_axis_off()
        im = ax.imshow(
            np.zeros((res, res)),
            cmap=cmap, vmin=0.0, vmax=(vmax if use_grad else None),
            extent=[-SIZE, SIZE, -SIZE, SIZE], interpolation="lanczos", origin="lower",
        )
        ax.text(
            0.97, 0.02, "Simone Conradi 2026", transform=ax.transAxes,
            ha="right", va="bottom", color=sig, fontsize=10, style="italic", alpha=0.75,
        )
        ax.set_title("$f(z)=z^2 + c$", fontsize=15)

        inset = ax.inset_axes([0.72, 0.72, 0.26, 0.26])
        inset.set_facecolor("none")
        inset.plot(a_path, b_path, color=sig, lw=1.0, alpha=0.55)
        dot, = inset.plot([a_path[0]], [b_path[0]], "o", color="#d63a2f", ms=4)
        inset.set_xlim(-1.6, 1.0)
        inset.set_ylim(-1.3, 1.3)
        inset.set_aspect("equal")
        inset.set_xticks([-1.6, 1.0])
        inset.set_yticks([-1.3, 1.3])
        inset.set_xlabel(r"$\Re c$", fontsize=9, color=sig, labelpad=-4)
        inset.set_ylabel(r"$\Im c$", fontsize=9, color=sig, labelpad=-4, rotation=0)
        inset.tick_params(colors=sig, labelsize=6, which='both')
        for spine in inset.spines.values():
            spine.set_edgecolor(sig)
            spine.set_alpha(0.4)
            spine.set_linewidth(0.6)

        out = Path(export_filename.value).expanduser().resolve()
        writer = manim.FFMpegWriter(fps=export_fps.value, bitrate=50000, codec="libx264")
        n_frames = export_frames.value
        with mo.status.progress_bar(total=n_frames, title="Encoding MP4") as bar:
            with writer.saving(fig, str(out), dpi=export_dpi.value):
                for i in range(n_frames):
                    a, b = float(a_path[i]), float(b_path[i])
                    dld = discrete_lagrangian_descriptor(w1, w2, w3, a, b, n_iter, p)
                    field = edge_gradient(dld) if use_grad else dld
                    img = (field.cpu().numpy()) ** (0.3 if use_grad else 0.1)
                    im.set_data(img)
                    if not use_grad:
                        im.set_clim(img.min(), img.max())
                    dot.set_data([a], [b])
                    writer.grab_frame()
                    bar.update()
        plt.close(fig)

        mb = out.stat().st_size / (1024 * 1024)
        return mo.md(
            f"**Saved** `{out}` — {n_frames} frames @ {export_fps.value} fps, "
            f"{export_res.value}px, {export_dpi.value} dpi, {mb:.1f} MB"
        )


    _()
    return


if __name__ == "__main__":
    app.run()
