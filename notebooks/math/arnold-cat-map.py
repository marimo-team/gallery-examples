# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["marimo", "numpy", "anywidget", "traitlets"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🐱 Arnold's cat map

    Take an $N\times N$ grid and move the tile at $(x, y)$ to

    $$\begin{pmatrix} x' \\ y' \end{pmatrix} = \begin{pmatrix} 2 & 1 \\ 1 & 1 \end{pmatrix}\begin{pmatrix} x \\ y \end{pmatrix} \pmod{N}.$$

    This operation has a fun twist: if you keep applying it the tiles must
    eventually return home.That recurrence **period depends only on $N$**,
    not on the picture.

    Fun aside: $\begin{pmatrix} 2 & 1 \\ 1 & 1
    \end{pmatrix} = \begin{pmatrix} 1 & 1 \\ 1 & 0 \end{pmatrix}^2$ and there are many (!) matrices that you can come up with that have this recurrence pattern.

    Hit **play** and watch the rainbow scramble — then snap back. Feel free to play around with your own matrices too!
    """)
    return


@app.cell(hide_code=True)
def _():
    import colorsys
    import math

    import anywidget
    import numpy as np
    import traitlets

    def hsv_hex(h, s, v):
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"

    return anywidget, hsv_hex, math, np, traitlets


@app.cell(hide_code=True)
def _(anywidget, traitlets):
    class CatMapWidget(anywidget.AnyWidget):
        n = traitlets.Int(29).tag(sync=True)
        colors = traitlets.List(traitlets.Unicode()).tag(sync=True)
        period = traitlets.Int(1).tag(sync=True)
        bijective = traitlets.Bool(True).tag(sync=True)
        a = traitlets.Float(2.0).tag(sync=True)
        b = traitlets.Float(1.0).tag(sync=True)
        c = traitlets.Float(1.0).tag(sync=True)
        d = traitlets.Float(1.0).tag(sync=True)
        speed = traitlets.Float(3.0).tag(sync=True)
        playing = traitlets.Bool(True).tag(sync=True)

        _esm = r"""
        function render({ model, el }) {
          el.style.textAlign = "center";
          const wrap = document.createElement("div");
          wrap.style.display = "inline-block";
          const canvas = document.createElement("canvas");
          wrap.appendChild(canvas);
          const label = document.createElement("div");
          label.style.fontFamily = "ui-monospace, monospace";
          label.style.marginTop = "14px";
          label.style.opacity = "0.85";
          el.appendChild(wrap);
          el.appendChild(label);
          const ctx = canvas.getContext("2d");

          const PX = 18;
          const SPLIT = 0.5;   // first half of a step = stretch, second half = fold
          const HOLD_MS = 1300; // pause on the reassembled image each full cycle
          let n, colors, gx, gy, px, py, iter, period, phase, hold;
          let rafId = null, lastT = 0;

          const ease = (t) => t * t * (3 - 2 * t);
          const mod = (a, m) => ((a % m) + m) % m;

          function commit() {
            // Advance every tile one full step: (x, y) -> A·(x, y) mod n.
            const a = model.get("a"), b = model.get("b");
            const c = model.get("c"), d = model.get("d");
            for (let i = 0; i < n * n; i++) {
              const x = gx[i], y = gy[i];
              gx[i] = mod(a * x + b * y, n);
              gy[i] = mod(c * x + d * y, n);
            }
            if (model.get("bijective")) {
              iter = (iter + 1) % period;
              if (iter === 0) hold = HOLD_MS;   // just landed back on the original
            } else {
              iter = iter + 1;                  // no return when A isn't invertible mod n
            }
          }

          function setup() {
            n = model.get("n");
            colors = model.get("colors");
            period = Math.max(model.get("period"), 1);
            const size = n * PX;
            canvas.width = size;
            canvas.height = size;
            gx = new Float64Array(n * n);
            gy = new Float64Array(n * n);
            px = new Float64Array(n * n);
            py = new Float64Array(n * n);
            for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
              const i = y * n + x;
              gx[i] = x;
              gy[i] = y;
            }
            iter = 0;
            phase = 0;
            hold = 0;
            draw();
          }

          function draw() {
            // Split each step in two: first the pure matmul (stretch, no wrap),
            // then the mod (fold the overhang back into the box by ±n).
            let stretch, fold;
            if (phase < SPLIT) { stretch = ease(phase / SPLIT); fold = 0; }
            else { stretch = 1; fold = ease((phase - SPLIT) / (1 - SPLIT)); }

            // Compute every tile's current position, tracking a bounding box so
            // we can auto-frame — this zooms out for the stretch and back in for
            // the fold, showing exactly where the folded-in pieces come from.
            const a = model.get("a"), b = model.get("b");
            const c = model.get("c"), d = model.get("d");
            let minx = 0, miny = 0, maxx = n, maxy = n;
            for (let i = 0; i < n * n; i++) {
              const x = gx[i], y = gy[i];
              const ux = a * x + b * y, uy = c * x + d * y;   // A·(x,y), un-wrapped
              let cx, cy;
              if (fold === 0) {
                cx = x + stretch * (ux - x);
                cy = y + stretch * (uy - y);
              } else {
                cx = ux - fold * (ux - mod(ux, n));      // slide overhang back by ±n
                cy = uy - fold * (uy - mod(uy, n));
              }
              px[i] = cx; py[i] = cy;
              if (cx < minx) minx = cx; else if (cx > maxx) maxx = cx;
              if (cy < miny) miny = cy; else if (cy > maxy) maxy = cy;
            }

            const V = Math.max(maxx - minx, maxy - miny) * 1.06;
            const sc = (n * PX) / V;
            const ox0 = (minx + maxx) / 2 - V / 2;
            const oy0 = (miny + maxy) / 2 - V / 2;

            ctx.setTransform(1, 0, 0, 1, 0, 0);
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.setTransform(sc, 0, 0, sc, -ox0 * sc, -oy0 * sc);

            const size = 0.76, g = (1 - size) / 2;
            for (let i = 0; i < n * n; i++) {
              ctx.fillStyle = colors[i];
              ctx.fillRect(px[i] + g, py[i] + g, size, size);
            }

            // The fixed [0,n)×[0,n) window — the actual image the map acts on.
            ctx.lineWidth = 1.5 / sc;
            ctx.strokeStyle = "rgba(255,255,255,0.6)";
            ctx.strokeRect(0, 0, n, n);

            ctx.setTransform(1, 0, 0, 1, 0, 0);
            const stage = phase < SPLIT ? "matmul (stretch)" : "mod n (fold)";
            if (model.get("bijective")) {
              const home = iter === 0 && phase < SPLIT ? "   ← back to the original!" : "";
              label.textContent = `iteration ${iter} / period ${period} · ${stage}${home}`;
            } else {
              label.textContent = `iteration ${iter} · ${stage} · det shares a factor with ${n} — not invertible, won't reassemble`;
            }
          }

          function frame(t) {
            if (!lastT) lastT = t;
            const dt = t - lastT;
            lastT = t;
            if (model.get("playing")) {
              if (hold > 0) {
                hold -= dt;               // linger on the reassembled image
              } else {
                const interval = 1000 / Math.max(model.get("speed"), 0.1);
                phase += dt / interval;
                while (phase >= 1) { phase -= 1; commit(); }
              }
            }
            draw();
            rafId = requestAnimationFrame(frame);
          }

          setup();
          rafId = requestAnimationFrame(frame);

          model.on("change:n", setup);
          model.on("change:colors", setup);
          model.on("change:period", setup);

          return () => { if (rafId) cancelAnimationFrame(rafId); };
        }
        export default { render };
        """

    return (CatMapWidget,)


@app.cell(hide_code=True)
def _(mo):
    n_slider = mo.ui.slider(5, 45, value=29, label="grid size N (= modulus)")
    speed_slider = mo.ui.slider(0.25, 6, value=1.5, step=0.25, label="steps / sec")
    play_switch = mo.ui.switch(value=True, label="play")
    matrix_ui = mo.ui.matrix(
        [[2, 1], [1, 1]], min_value=-6, max_value=6, step=1, label="matrix A"
    )
    return matrix_ui, n_slider, play_switch, speed_slider


@app.cell(hide_code=True)
def _(matrix_ui, mo, n_slider, play_switch, speed_slider):
    mo.hstack(
        [matrix_ui, mo.vstack([n_slider, speed_slider, play_switch])],
        justify="start",
        align="center",
        gap=2,
    )
    return


@app.cell
def _(hsv_hex, math, matrix_ui, n_slider, np):
    n = n_slider.value

    # A smooth diagonal rainbow so the scramble — and its return — is obvious.
    colors = [
        hsv_hex(((x + y) % n) / n, 0.6, 0.92)
        for y in range(n)
        for x in range(n)
    ]

    # The chosen matrix A (integer entries from the UI).
    (a, b), (c, d) = ((int(round(v)) for v in row) for row in matrix_ui.value)
    A = np.array([[a, b], [c, d]])

    # The map (x,y) -> A(x,y) mod n is a bijection — and eventually returns to
    # the identity — only when det(A) is invertible mod n, i.e. coprime to it.
    det = a * d - b * c
    bijective = math.gcd(det % n, n) == 1

    if bijective:
        M = A % n
        period = 1
        while not np.array_equal(M, np.eye(2, dtype=int)):
            M = (M @ A) % n
            period += 1
    else:
        period = 1  # unused when not bijective
    return a, b, bijective, c, colors, d, n, period


@app.cell
def _(CatMapWidget, a, b, bijective, c, colors, d, mo, n, period):
    cat = CatMapWidget(
        n=n, colors=colors, period=period, bijective=bijective,
        a=a, b=b, c=c, d=d,
    )
    widget = mo.ui.anywidget(cat)
    widget
    return (cat,)


@app.cell
def _(cat, play_switch, speed_slider):
    # Live controls that shouldn't rebuild (and reset) the animation.
    cat.speed = speed_slider.value
    cat.playing = play_switch.value
    return


if __name__ == "__main__":
    app.run()
