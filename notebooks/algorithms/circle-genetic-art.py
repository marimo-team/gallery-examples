# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy==2.3.5",
#     "pillow==12.3.0",
#     "matplotlib==3.10.8",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import io
    import base64
    import time
    from PIL import Image, ImageDraw
    import matplotlib.pyplot as plt

    return Image, ImageDraw, base64, io, mo, np, plt, time


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Circle Genetic Art

    Reconstruct an image out of **translucent shapes** with a **hill climber**,
    and always compare two strategies side by side:

    - **All at once** — fit every shape together.
    - **Waves** — fit a batch, freeze it, then fit the next batch on top.

    Each has its **own iteration budget** (equal steps ≠ equal time — a wave step
    is much cheaper since it renders onto a cached base), so compare on the
    **convergence vs time** chart. Each shape is `(x, y, size, R, G, B)` —
    **circles or squares** (toggle) — seeded with the target's color at its center;
    each step nudges a few shapes and keeps the change only if the pixel MSE drops.
    Both panels update live. Defaults to the **Mona Lisa** — or upload your own.
    Changing a setting only recomputes the strategy it affects.
    """)
    return


@app.cell
def _(mo):
    upload = mo.ui.file(kind="button", filetypes=[".png", ".jpg", ".jpeg"], label="Upload image")
    opacity = mo.ui.slider(0.05, 1.0, value=0.5, step=0.05, label="Shape opacity", show_value=True)
    circles = mo.ui.slider(50, 250, value=150, step=10, label="Circles", show_value=True)
    iterations = mo.ui.slider(1000, 25000, value=6000, step=1000, label="All-at-once iterations", show_value=True)
    waves = mo.ui.slider(2, 20, value=2, step=1, label="Number of waves", show_value=True)
    wave_iterations = mo.ui.slider(1000, 25000, value=6000, step=1000, label="Waves iterations (total)", show_value=True)
    squares = mo.ui.switch(value=False, label="Use squares instead of circles")
    run = mo.ui.run_button(label="Evolve")
    return (
        circles,
        iterations,
        opacity,
        run,
        squares,
        upload,
        wave_iterations,
        waves,
    )


@app.cell(hide_code=True)
def _(
    circles,
    iterations,
    mo,
    opacity,
    run,
    squares,
    upload,
    wave_iterations,
    waves,
):
    mo.vstack([upload, opacity, circles, iterations, waves, wave_iterations, squares, run])
    return


@app.cell(hide_code=True)
def _():
    # Default target: a small base64-embedded Mona Lisa so the demo needs no upload.
    MONA_B64 = (
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYn"
    "KSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgo"
    "KCgoKCgoKCj/wAARCACzAHgDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUF"
    "BAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVW"
    "V1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi"
    "4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAEC"
    "AxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVm"
    "Z2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq"
    "8vP09fb3+Pn6/9oADAMBAAIRAxEAPwDHksoj/Bn/AAqMWsJU5jGemcVreXtjTcMEjrR5QcL8vH6147qS3uZ8iMlLSMg5RQMdOKY9"
    "pbHO2NQMVtfZkHGOfWmSWhIGQMD0o531DkTMZbaIA/IOQPwqRbRNwYD5hg5rS+ykEcU9rfCHjn2pe0fcapozPJBXGMcY6DpTjBGy"
    "YKj6AVpRwfLgrnNSi3Ugjawx0oUmDgjE+xxbSCinnqRSGwTAGzA68CtRoNrABs1MkHyktTvcnkSOf+wRsQBHx60DTYuBtNbjRDdg"
    "+tAiUnAx+VLna2H7NMxUsImY/u+nTIqVNMhIP7vIPP3RW1HEFTOM+tL5JBBxWntGupPso9jDfS7aOJiI1L7TgFaK27qDdbPlckKf"
    "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"
    )
    return (MONA_B64,)


@app.cell
def _(Image, MONA_B64, base64, io, np, upload):
    # Load the target image (defaults to the embedded Mona Lisa so the demo needs no upload).
    SIZE = 100  # work at low res for speed

    def load_target():
        if upload.value:
            raw = upload.value[0].contents
        else:
            raw = base64.b64decode(MONA_B64)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        return img.resize((SIZE, SIZE), Image.LANCZOS)

    target_img = load_target()
    target_arr = np.asarray(target_img, dtype=np.float32) / 255.0
    return SIZE, target_arr, target_img


@app.cell
def _(Image, ImageDraw, SIZE, np):
    RAD_MAX = 0.13  # max circle radius as a fraction of the image size

    def render_rgba(genome, opacity, base=None, shape="circle", size=None):
        """Composite shapes onto `base` (an RGBA image) or a black canvas. Returns RGBA.

        Shapes are drawn largest-first so small ones land on top and carry detail.
        `shape` is "circle" (ellipse) or "square" (rectangle). The genome is
        resolution-independent, so pass `size` to render crisply at any resolution
        (only valid with base=None; the incremental base is always at SIZE).
        """
        px = size or SIZE
        img = Image.new("RGBA", (px, px), (0, 0, 0, 255)) if base is None else base.copy()
        alpha = int(opacity * 255)
        order = np.argsort(genome[:, 2])[::-1] if len(genome) else []
        for cx, cy, r, R, G, B in genome[order]:
            x, y = cx * px, cy * px
            rad = r * px * RAD_MAX + 0.012 * px
            overlay = Image.new("RGBA", (px, px), (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            box = [x - rad, y - rad, x + rad, y + rad]
            fill = (int(R * 255), int(G * 255), int(B * 255), alpha)
            (draw.rectangle if shape == "square" else draw.ellipse)(box, fill=fill)
            img = Image.alpha_composite(img, overlay)
        return img

    def render(genome, opacity, base=None, shape="circle"):
        """genome: (N, 6) in [0,1] -> cx, cy, r, R, G, B. Returns HxWx3 float array."""
        return np.asarray(render_rgba(genome, opacity, base, shape).convert("RGB"), dtype=np.float32) / 255.0

    def mse(genome, opacity, target_arr, base=None, shape="circle"):
        return np.mean((render(genome, opacity, base, shape) - target_arr) ** 2)

    return mse, render, render_rgba


@app.cell
def _():
    # Persistent memo of finished runs, keyed by (circles, iterations, waves, opacity, image).
    # Survives re-runs of the compute cell, so pressing Evolve only recomputes the variant
    # whose inputs actually changed.
    SIM_CACHE = {}
    return (SIM_CACHE,)


@app.cell
def _(SIZE, mse, np, opacity, render_rgba, target_arr, time):
    def target_color(cx, cy):
        px = min(SIZE - 1, max(0, int(cx * SIZE)))
        py = min(SIZE - 1, max(0, int(cy * SIZE)))
        return target_arr[py, px]

    def seed(n, rng):
        """Random positions/radii, but each shape takes the target's color at its center."""
        g = rng.random((n, 6))
        for i in range(n):
            g[i, 3:6] = target_color(g[i, 0], g[i, 1])
        return g

    def run_variant(n_shapes, n_waves, total_iters, shape, on_step=None):
        """Fit n_shapes in n_waves frozen batches sharing a total step budget.

        Each step perturbs 1-3 shapes and keeps it only if the pixel MSE drops.
        Frozen batches are rendered once into a cached base, so a wave only
        re-renders its own shapes. Returns per-checkpoint `history` (MSE) and
        `times` (cumulative compute seconds, excluding the live-draw callback).
        `on_step(best, mse, done, total)` fires at ~30 checkpoints per wave.
        """
        rng = np.random.default_rng(0)
        sizes = [n_shapes // n_waves + (1 if i < n_shapes % n_waves else 0) for i in range(n_waves)]
        per = max(1, total_iters // n_waves)  # equal budget per wave -> same total compute
        total_steps = per * n_waves
        frozen, history, times, done, compute_s = np.empty((0, 6)), [], [], 0, 0.0
        for size in sizes:
            base = render_rgba(frozen, opacity.value, shape=shape) if len(frozen) else None
            full = lambda g: np.concatenate([frozen, g]) if len(frozen) else g
            g = seed(size, rng)
            cur = mse(g, opacity.value, target_arr, base, shape)
            history.append(cur)
            times.append(compute_s)
            draw_every = max(1, per // 30)
            for it in range(per):
                t0 = time.perf_counter()
                cand = g.copy()
                idx = rng.integers(0, size, rng.integers(1, 4))
                for j in idx:
                    cand[j, :3] = np.clip(cand[j, :3] + rng.normal(0, 0.12, 3), 0, 1)  # move + resize
                    if rng.random() < 0.5:
                        cand[j, 3:6] = target_color(cand[j, 0], cand[j, 1])  # snap color to target
                    else:
                        cand[j, 3:6] = np.clip(cand[j, 3:6] + rng.normal(0, 0.08, 3), 0, 1)
                trial = mse(cand, opacity.value, target_arr, base, shape)
                if trial < cur:
                    g, cur = cand, trial
                compute_s += time.perf_counter() - t0  # algorithm time only, not the redraw
                done += 1
                if it % draw_every == 0 or it == per - 1:
                    history.append(cur)
                    times.append(compute_s)
                    if on_step:
                        on_step(full(g), cur, done, total_steps)
            frozen = full(g)
        return {"best": frozen, "history": history, "times": times}

    return (run_variant,)


@app.cell(hide_code=True)
def _(
    Image,
    SIM_CACHE,
    circles,
    io,
    iterations,
    mo,
    np,
    opacity,
    plt,
    render,
    render_rgba,
    run,
    run_variant,
    squares,
    target_arr,
    target_img,
    wave_iterations,
    waves,
):
    HI_RES = 768  # final export/display resolution
    shape = "square" if squares.value else "circle"
    blank = Image.new("RGB", (240, 240), (32, 32, 32))
    target_disp = target_img.resize((240, 240))

    def as_img(genome):
        return Image.fromarray((render(genome, opacity.value, shape=shape) * 255).astype("uint8")).resize((240, 240))

    def board(cells):
        """Target on the left, then one panel (title, image, status) per strategy."""
        cols = [mo.vstack([mo.md("**Target**"), target_disp], align="center")]
        for title, img, status in cells:
            cols.append(mo.vstack([mo.md(f"**{title}**"), img, mo.md(status)], align="center"))
        return mo.hstack(cols, justify="center")

    if not run.value:
        mo.output.replace(mo.md("Press **Evolve** to compare *all at once* vs *waves*."))
    else:
        n, nw = circles.value, waves.value
        tkey = hash(target_arr.tobytes())
        # each strategy gets its own step budget (equal steps != equal time)
        variants = [("All at once", 1, iterations.value), (f"{nw} waves", nw, wave_iterations.value)]
        state = {title: (blank, "queued") for title, _, _ in variants}

        def redraw():
            mo.output.replace(board([(t, state[t][0], state[t][1]) for t, _, _ in variants]))

        redraw()
        results = {}
        for title, n_waves, budget in variants:
            key = (n, budget, n_waves, round(opacity.value, 3), shape, tkey)
            if key in SIM_CACHE:  # unchanged strategy -> reuse, no recompute
                results[title] = SIM_CACHE[key]
                state[title] = (as_img(SIM_CACHE[key]["best"]), "cached")
                redraw()
            else:
                def on_step(g, err, done, tot, title=title):
                    state[title] = (as_img(g), f"step {done:,}/{tot:,} · MSE {err * 255**2:,.0f}")
                    redraw()

                res = run_variant(n, n_waves, budget, shape, on_step)
                SIM_CACHE[key] = res
                results[title] = res

        # overlaid convergence: MSE vs actual steps (left) and vs wall-clock time (right)
        colors = ["#4C78A8", "#F58518"]
        fig, (ax_steps, ax_time) = plt.subplots(1, 2, figsize=(11, 3))
        for (title, _, budget), color in zip(variants, colors):
            h = [v * 255**2 for v in results[title]["history"]]
            ax_steps.plot(np.linspace(0, budget, len(h)), h, label=title, color=color)
            ax_time.plot(results[title]["times"], h, label=title, color=color)
        for ax in (ax_steps, ax_time):
            ax.set_yscale("log")
            ax.set_ylabel("MSE (log)")
            ax.legend()
        ax_steps.set_xlabel("steps")
        ax_steps.set_title("Convergence vs steps")
        ax_time.set_xlabel("compute time (s)")
        ax_time.set_title("Convergence vs time")
        fig.tight_layout()

        def final_panel(title, res):
            hi = render_rgba(res["best"], opacity.value, shape=shape, size=HI_RES).convert("RGB")
            buf = io.BytesIO()
            hi.save(buf, "PNG")
            download = mo.download(
                buf.getvalue(),
                filename=f"{title.replace(' ', '_').lower()}_{shape}_{HI_RES}px.png",
                mimetype="image/png",
                label=f"Download {HI_RES}px PNG",
            )
            return mo.vstack(
                [
                    mo.md(f"**{title}** · MSE {res['history'][-1] * 255**2:,.0f}"),
                    hi.resize((300, 300)),
                    download,
                ],
                align="center",
            )

        finals = mo.hstack(
            [mo.vstack([mo.md("**Target**"), target_disp], align="center")]
            + [final_panel(t, results[t]) for t, _, _ in variants],
            justify="center",
        )
        mo.output.replace(mo.vstack([finals, mo.as_html(fig)]))
    return


if __name__ == "__main__":
    app.run()
