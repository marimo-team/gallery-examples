# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "marimo-cython",
#     "numpy",
#     "anywidget>=0.9",
#     "traitlets",
# ]
# ///

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import anywidget
    import cython
    import marimo as mo
    import numpy as np
    import traitlets

    from marimo_cython import cy

    return anywidget, cy, cython, mo, np, traitlets


@app.cell
def _(mo):
    mo.md("""
    # Mandelbrot — Cython vs Python

    Drag the slider and watch the difference.
    """)
    return


@app.cell
def _(mo):
    size_slider = mo.ui.slider(start=100, stop=800, step=100, value=300, label="Resolution (px)")
    size_slider
    return (size_slider,)


@app.cell
def _(cy, cython):
    @cy.compile(boundscheck=False, wraparound=False, cdivision=True)
    def mandelbrot(
        out: cython.int[:, :],
        xmin: cython.double,
        xmax: cython.double,
        ymin: cython.double,
        ymax: cython.double,
        max_iter: cython.int,
    ) -> None:
        rows: cython.Py_ssize_t = out.shape[0]
        cols: cython.Py_ssize_t = out.shape[1]
        i: cython.Py_ssize_t
        j: cython.Py_ssize_t
        n: cython.int
        cx: cython.double
        cy_: cython.double
        zx: cython.double
        zy: cython.double
        tmp: cython.double
        for i in range(rows):
            for j in range(cols):
                cx = xmin + (xmax - xmin) * j / cols
                cy_ = ymin + (ymax - ymin) * i / rows
                zx = 0.0
                zy = 0.0
                n = 0
                while zx * zx + zy * zy < 4.0 and n < max_iter:
                    tmp = zx * zx - zy * zy + cx
                    zy = 2.0 * zx * zy + cy_
                    zx = tmp
                    n += 1
                out[i, j] = n

    return (mandelbrot,)


@app.cell
def _(SideBySideWidget, mo):
    bench = SideBySideWidget(left_label="Cython", right_label="Python")
    widget = mo.ui.anywidget(bench)
    widget
    return bench, widget


@app.cell
async def _(bench, mandelbrot, np, size_slider):
    import asyncio
    import base64 as _b64
    import struct as _struct
    import time as _time
    import zlib as _zlib
    from concurrent.futures import ThreadPoolExecutor

    _size = size_slider.value
    _MAX_ITER = 200
    _XMIN, _XMAX = -2.0, 1.0
    _YMIN, _YMAX = -1.5, 1.5

    def _to_png(arr):
        a = arr.astype(np.float64)
        mx = a.max()
        a = (a / mx * 255).astype(np.uint8) if mx > 0 else np.zeros_like(a, dtype=np.uint8)
        h, w = a.shape

        def chunk(ct, data):
            c = ct + data
            return (
                _struct.pack(">I", len(data)) + c + _struct.pack(">I", _zlib.crc32(c) & 0xFFFFFFFF)
            )

        ihdr = _struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)
        raw = b"".join(b"\x00" + a[i].tobytes() for i in range(h))
        png = b"\x89PNG\r\n\x1a\n"
        png += chunk(b"IHDR", ihdr)
        png += chunk(b"IDAT", _zlib.compress(raw))
        png += chunk(b"IEND", b"")
        return _b64.b64encode(png).decode()

    def _py_mandelbrot(out, xmin, xmax, ymin, ymax, max_iter):
        rows, cols = out.shape
        for i in range(rows):
            for j in range(cols):
                cx = xmin + (xmax - xmin) * j / cols
                cy_ = ymin + (ymax - ymin) * i / rows
                zx, zy = 0.0, 0.0
                n = 0
                while zx * zx + zy * zy < 4.0 and n < max_iter:
                    zx, zy = zx * zx - zy * zy + cx, 2.0 * zx * zy + cy_
                    n += 1
                out[i, j] = n

    def _run_cython() -> tuple[str, float, str]:
        arr = np.zeros((_size, _size), dtype=np.intc)
        t0 = _time.perf_counter()
        mandelbrot(arr, _XMIN, _XMAX, _YMIN, _YMAX, _MAX_ITER)
        elapsed = _time.perf_counter() - t0
        return "left", elapsed, _to_png(arr)

    def _run_python() -> tuple[str, float, str]:
        arr = np.zeros((_size, _size), dtype=np.intc)
        t0 = _time.perf_counter()
        _py_mandelbrot(arr, _XMIN, _XMAX, _YMIN, _YMAX, _MAX_ITER)
        elapsed = _time.perf_counter() - t0
        return "right", elapsed, _to_png(arr)

    bench.left_image = ""
    bench.left_time = ""
    bench.left_done = False
    bench.right_image = ""
    bench.right_time = ""
    bench.right_done = False
    bench.summary = ""

    _loop = asyncio.get_event_loop()
    _pool = ThreadPoolExecutor(max_workers=2)

    _cy_future = _loop.run_in_executor(_pool, _run_cython)
    _py_future = _loop.run_in_executor(_pool, _run_python)

    _futures: list[asyncio.Future[tuple[str, float, str]]] = [_cy_future, _py_future]
    for coro in asyncio.as_completed(_futures):
        side, elapsed, b64 = await coro
        setattr(bench, f"{side}_image", b64)
        setattr(bench, f"{side}_time", f"{elapsed:.3f}s")
        setattr(bench, f"{side}_done", True)

    _pool.shutdown(wait=False)

    bench.summary = f"{_size}\u00d7{_size} pixels"
    return


@app.cell
def _(anywidget, traitlets):
    _SIDE_BY_SIDE_ESM = """
    function render({ model, el }) {
      el.innerHTML = "";
      const container = document.createElement("div");
      container.className = "sbs-root";
      container.innerHTML = `
        <style>
          .sbs-root {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            font-family: system-ui, -apple-system, sans-serif;
          }
          .sbs-panel {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 12px;
            background: #fafafa;
          }
          .sbs-header {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            margin-bottom: 8px;
          }
          .sbs-label { font-weight: 600; font-size: 14px; }
          .sbs-time  { font-size: 13px; color: #666; font-variant-numeric: tabular-nums; }
          .sbs-img-wrap {
            width: 100%;
            min-height: 120px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #fff;
            border-radius: 4px;
            overflow: hidden;
          }
          .sbs-img-wrap img {
            width: 100%;
            height: auto;
            display: block;
            image-rendering: pixelated;
          }
          .sbs-spinner {
            color: #999;
            font-size: 13px;
            animation: sbs-pulse 1.5s ease-in-out infinite;
          }
          @keyframes sbs-pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.4; }
          }
          .sbs-summary {
            grid-column: 1 / -1;
            text-align: center;
            font-size: 13px;
            color: #666;
            padding-top: 4px;
          }
        </style>
        <div class="sbs-panel" id="sbs-left">
          <div class="sbs-header">
            <span class="sbs-label" data-bind="left_label"></span>
            <span class="sbs-time" data-bind="left_time"></span>
          </div>
          <div class="sbs-img-wrap" data-panel="left">
            <span class="sbs-spinner">computing\\u2026</span>
          </div>
        </div>
        <div class="sbs-panel" id="sbs-right">
          <div class="sbs-header">
            <span class="sbs-label" data-bind="right_label"></span>
            <span class="sbs-time" data-bind="right_time"></span>
          </div>
          <div class="sbs-img-wrap" data-panel="right">
            <span class="sbs-spinner">computing\\u2026</span>
          </div>
        </div>
        <div class="sbs-summary" data-bind="summary"></div>
      `;
      el.appendChild(container);

      const refs = {
        left_label:  container.querySelector('[data-bind="left_label"]'),
        left_time:   container.querySelector('[data-bind="left_time"]'),
        left_wrap:   container.querySelector('[data-panel="left"]'),
        right_label: container.querySelector('[data-bind="right_label"]'),
        right_time:  container.querySelector('[data-bind="right_time"]'),
        right_wrap:  container.querySelector('[data-panel="right"]'),
        summary:     container.querySelector('[data-bind="summary"]'),
      };

      function updateText(key) {
        if (refs[key]) refs[key].textContent = model.get(key);
      }

      function updateImage(side) {
        const b64  = model.get(side + "_image");
        const done = model.get(side + "_done");
        const wrap = refs[side + "_wrap"];
        if (!wrap) return;
        if (b64) {
          const existing = wrap.querySelector("img");
          if (existing) {
            existing.src = "data:image/png;base64," + b64;
          } else {
            wrap.innerHTML = "";
            const img = document.createElement("img");
            img.src = "data:image/png;base64," + b64;
            wrap.appendChild(img);
          }
        } else if (!done) {
          wrap.innerHTML = '<span class="sbs-spinner">computing\\u2026</span>';
        }
      }

      for (const k of ["left_label", "left_time", "right_label", "right_time", "summary"]) {
        updateText(k);
      }
      updateImage("left");
      updateImage("right");

      for (const k of ["left_label", "left_time", "right_label", "right_time", "summary"]) {
        model.on("change:" + k, () => updateText(k));
      }
      for (const side of ["left", "right"]) {
        model.on("change:" + side + "_image", () => updateImage(side));
        model.on("change:" + side + "_done",  () => updateImage(side));
      }
    }
    export default { render };
    """

    class SideBySideWidget(anywidget.AnyWidget):
        left_label = traitlets.Unicode("Left").tag(sync=True)
        left_time = traitlets.Unicode("").tag(sync=True)
        left_image = traitlets.Unicode("").tag(sync=True)
        left_done = traitlets.Bool(False).tag(sync=True)
        right_label = traitlets.Unicode("Right").tag(sync=True)
        right_time = traitlets.Unicode("").tag(sync=True)
        right_image = traitlets.Unicode("").tag(sync=True)
        right_done = traitlets.Bool(False).tag(sync=True)
        summary = traitlets.Unicode("").tag(sync=True)
        _esm = _SIDE_BY_SIDE_ESM

    return (SideBySideWidget,)


if __name__ == "__main__":
    app.run()
