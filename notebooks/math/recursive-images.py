# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "anywidget==0.11.0",
#     "traitlets==5.16.1",
#     "numpy",
#     "pillow",
#     "wigglystuff==0.5.31",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import base64
    import io
    import math

    import marimo as mo
    import numpy as np
    from PIL import Image
    from wigglystuff import HoverSlider, Paint

    return HoverSlider, Image, Paint, base64, io, math, mo, np


@app.cell
def _(mo):
    mo.md("""
    # Recursive images

    1. Draw a **parent** below (transparent PNG).
    2. Place one or more **children** — copies of the parent you can drag
       around. The set of children is the recursive rule.
    """)
    return


@app.cell
def _(Paint, mo):
    paint = mo.ui.anywidget(Paint(width=300, height=300, store_background=False, rainbow_brush=True))
    return (paint,)


@app.cell
def _(RecursiveImageWidget, mo):
    raw_rule = RecursiveImageWidget(size=300)
    rule = mo.ui.anywidget(raw_rule)
    return raw_rule, rule


@app.cell
def _(paint, raw_rule):
    # Feed the current Paint drawing into the rule widget as the parent image.
    raw_rule.parent_image = paint.get_base64()
    return


@app.cell
def _(mo, paint, rule):
    mo.hstack([paint, rule], justify="start", gap=1)
    return


@app.cell
def _(HoverSlider, mo):
    depth = mo.ui.anywidget(
        HoverSlider(start=0, stop=20, step=1, value=6, label="depth")
    )
    angle_noise = mo.ui.anywidget(
        HoverSlider(start=0, stop=45, step=1, value=0, label="angle noise (°)")
    )
    size_noise = mo.ui.anywidget(
        HoverSlider(start=0, stop=0.5, step=0.01, value=0, label="size noise")
    )
    background = mo.ui.dropdown(
        ["transparent", "white", "black"], value="transparent", label="background"
    )
    mo.vstack([depth, angle_noise, size_noise, background], gap=0.5)
    return angle_noise, background, depth, size_noise


@app.cell(hide_code=True)
def _(Image, base64, io, math, np):
    # Must match BASE in recursive_image_widget.js (base-box world-width).
    BASE = 0.34

    def decode_png(data_url):
        if not data_url or "," not in data_url:
            return None
        raw = base64.b64decode(data_url.split(",", 1)[1])
        return Image.open(io.BytesIO(raw)).convert("RGBA")

    def inv_coeffs(scale, theta, c_out, c_in):
        # forward: out = c_out + scale * R(theta) * (in - c_in)
        # PIL wants the inverse map out -> in.
        inv = 1.0 / scale
        ct, st = math.cos(theta), math.sin(theta)
        a, b = inv * ct, inv * st
        d, e = -inv * st, inv * ct
        c = c_in[0] - (a * c_out[0] + b * c_out[1])
        f = c_in[1] - (d * c_out[0] + e * c_out[1])
        return (a, b, c, d, e, f)

    def warp(img, coeffs, W):
        return img.transform(
            (W, W), Image.AFFINE, coeffs, resample=Image.BILINEAR,
            fillcolor=(0, 0, 0, 0),
        )

    def seed_canvas(parent_img, parent, W):
        px, py, ps, pa = parent
        pw, ph = parent_img.size
        k = (BASE * ps * W) / pw
        if k <= 1e-9:
            return Image.new("RGBA", (W, W), (0, 0, 0, 0))
        coeffs = inv_coeffs(k, math.radians(pa), (px * W, py * W), (pw / 2, ph / 2))
        return warp(parent_img, coeffs, W)

    def render_ifs(
        parent_img, parent, children, depth,
        angle_noise=0.0, size_noise=0.0, W=640, seed=0,
    ):
        if parent_img is None:
            return None
        base = seed_canvas(parent_img, parent, W)
        px, py, ps, pa = parent
        kids = [(cx, cy, cs, ca) for cx, cy, cs, ca in children if ps > 1e-9 and cs > 1e-9]
        rng = np.random.default_rng(seed)
        canvas = base
        for _ in range(depth):
            if not kids:
                break
            acc = base.copy()
            for cx, cy, cs, ca in kids:
                # Perturb each child's angle/size afresh at every level (organic jitter).
                a = ca + rng.normal(0.0, angle_noise) if angle_noise else ca
                s = cs * max(1.0 + rng.normal(0.0, size_noise), 1e-3) if size_noise else cs
                coeffs = inv_coeffs(
                    s / ps, math.radians(a - pa), (cx * W, cy * W), (px * W, py * W)
                )
                acc = Image.alpha_composite(acc, warp(canvas, coeffs, W))
            canvas = acc
        return canvas

    def present(img, background="transparent", tile=20):
        W = img.size[0]
        if background == "white":
            bg = Image.new("RGBA", (W, W), (255, 255, 255, 255))
        elif background == "black":
            bg = Image.new("RGBA", (W, W), (0, 0, 0, 255))
        else:  # transparent -> checkerboard preview so alpha reads
            yy, xx = np.indices((W, W))
            cells = ((xx // tile) + (yy // tile)) % 2
            val = np.where(cells == 0, 245, 224).astype("uint8")
            bg = Image.fromarray(np.dstack([val, val, val])).convert("RGBA")
        out = Image.alpha_composite(bg, img)
        return out if background == "transparent" else out.convert("RGB")

    return decode_png, present, render_ifs


@app.cell
def _(
    angle_noise,
    background,
    decode_png,
    depth,
    mo,
    paint,
    present,
    render_ifs,
    rule,
    size_noise,
):
    parent_img = decode_png(paint.get_base64())
    result = render_ifs(
        parent_img, rule.parent, rule.children, int(depth.value["hover_value"]),
        angle_noise=angle_noise.value["hover_value"],
        size_noise=size_noise.value["hover_value"],
    )
    mo.stop(result is None, mo.md("*Draw a parent to see the recursion.*"))
    present(result, background.value)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## Appendix — the `RecursiveImageWidget`

    The rule-authoring widget, inlined as an anywidget. A *placement* is
    `[x, y, scale, angle_deg]`: the base box (world-width `BASE`, see the JS)
    scaled, rotated clockwise, centered at `(x, y)` in world coords `[0, 1]`.
    The `parent` placement defines the base frame; each child in `children` is
    another placement, and the recursive rule is the similarity mapping the
    parent box onto that child box.
    """)
    return


@app.cell
def _():
    import anywidget
    import traitlets

    _RIMG_JS = r"""
    // Rule-authoring widget: place similarity copies of a parent image.
    // A placement is [x, y, scale, angle_deg]: the base box (world-width BASE),
    // scaled, rotated clockwise, centered at (x, y) in the unit-square world.
    // `parent` is one placement (the base frame); `children` is a list of them.

    const BASE = 0.34; // parent base-box world-width (fraction of the stage)

    function render({ model, el }) {
      el.classList.add("rimg-root");

      const container = document.createElement("div");
      container.className = "rimg-container";

      const toolbar = document.createElement("div");
      toolbar.className = "rimg-toolbar";
      const addBtn = button("+ child", "add a child copy");
      const removeBtn = button("remove", "remove selected child");
      const clearBtn = button("clear", "remove all children");
      toolbar.append(addBtn, removeBtn, clearBtn);

      const stage = document.createElement("div");
      stage.className = "rimg-stage";
      container.append(toolbar, stage);
      el.append(container);

      function applySize() {
        const s = model.get("size") || 320;
        container.style.width = s + "px";
      }
      applySize();
      model.on("change:size", applySize);

      let parent = readList(model.get("parent"), [0.5, 0.5, 1.0, 0.0]);
      let children = readChildren();
      let sel = { kind: "parent" }; // {kind:'parent'} | {kind:'child', i} | null
      let aspect = 1;
      let active = null; // in-progress gesture

      // --- geometry ---
      function boxSize(scale) {
        return { w: BASE * scale, h: (BASE * scale) / aspect };
      }
      function placementOf(s) {
        return s && s.kind === "parent" ? parent : children[s.i];
      }

      function positionEl(node, p) {
        const [x, y, scale, angle] = p;
        const { w, h } = boxSize(scale);
        node.style.width = w * 100 + "%";
        node.style.height = h * 100 + "%";
        node.style.left = (x - w / 2) * 100 + "%";
        node.style.top = (y - h / 2) * 100 + "%";
        node.style.transform = `rotate(${angle}deg)`;
      }

      function makeEl(kind, i, p) {
        const isSel =
          sel && sel.kind === kind && (kind === "parent" || sel.i === i);
        const node = document.createElement("div");
        node.className =
          "rimg-el rimg-" + kind + (isSel ? " selected" : "");
        node.dataset.kind = kind;
        if (kind === "child") node.dataset.i = i;

        const img = document.createElement("img");
        img.className = "rimg-img";
        img.src = model.get("parent_image") || "";
        img.draggable = false;
        node.appendChild(img);

        if (isSel) {
          const rot = document.createElement("div");
          rot.className = "rimg-handle rimg-rotate";
          rot.dataset.role = "rotate";
          const sc = document.createElement("div");
          sc.className = "rimg-handle rimg-scale";
          sc.dataset.role = "scale";
          node.append(rot, sc);
        }
        positionEl(node, p);
        return node;
      }

      function draw() {
        stage.innerHTML = "";
        stage.appendChild(makeEl("parent", -1, parent));
        children.forEach((c, i) => stage.appendChild(makeEl("child", i, c)));
      }

      function nodeFor(s) {
        if (!s) return null;
        return s.kind === "parent"
          ? stage.querySelector(".rimg-parent")
          : stage.querySelector(`.rimg-child[data-i="${s.i}"]`);
      }

      // --- pointer gestures ---
      function worldOf(ev) {
        const r = stage.getBoundingClientRect();
        return [(ev.clientX - r.left) / r.width, (ev.clientY - r.top) / r.height];
      }

      stage.addEventListener("pointerdown", (ev) => {
        const elNode = ev.target.closest(".rimg-el");
        if (!elNode) {
          // click empty space -> deselect
          if (sel) {
            sel = null;
            draw();
          }
          return;
        }
        ev.preventDefault();
        const kind = elNode.dataset.kind;
        const i = kind === "child" ? parseInt(elNode.dataset.i) : -1;
        const role = ev.target.dataset.role || "pan";

        const newSel = kind === "parent" ? { kind } : { kind, i };
        const changedSel =
          !sel || sel.kind !== newSel.kind || sel.i !== newSel.i;
        sel = newSel;
        if (changedSel) draw();

        const p = placementOf(sel);
        const [wx, wy] = worldOf(ev);
        active = { role, sel, startPointer: [wx, wy], startPlacement: p.slice() };
        stage.setPointerCapture(ev.pointerId);
      });

      stage.addEventListener("pointermove", (ev) => {
        if (!active) return;
        const [wx, wy] = worldOf(ev);
        const p = placementOf(active.sel);
        const [sx, sy, sScale, sAngle] = active.startPlacement;

        if (active.role === "pan") {
          p[0] = clamp01(sx + (wx - active.startPointer[0]));
          p[1] = clamp01(sy + (wy - active.startPointer[1]));
        } else if (active.role === "scale") {
          const d0 = dist(active.startPointer, [sx, sy]) || 1e-6;
          const d1 = dist([wx, wy], [sx, sy]);
          p[2] = clampRange(sScale * (d1 / d0), 0.02, 3);
        } else if (active.role === "rotate") {
          const a = Math.atan2(wy - sy, wx - sx) * (180 / Math.PI) + 90;
          p[3] = a;
        }
        positionEl(nodeFor(active.sel), p);
      });

      stage.addEventListener("pointerup", () => {
        if (!active) return;
        active = null;
        commit();
      });

      // --- buttons ---
      addBtn.addEventListener("click", () => {
        const [px, py] = parent;
        children.push([clamp01(px), clamp01(py + 0.28), 0.55, 0]);
        sel = { kind: "child", i: children.length - 1 };
        commit();
        draw();
      });
      removeBtn.addEventListener("click", () => {
        if (!sel || sel.kind !== "child") return;
        children.splice(sel.i, 1);
        sel = children.length ? { kind: "child", i: Math.min(sel.i, children.length - 1) } : null;
        commit();
        draw();
      });
      clearBtn.addEventListener("click", () => {
        children = [];
        sel = { kind: "parent" };
        commit();
        draw();
      });

      // --- model sync ---
      function commit() {
        model.set("parent", parent.slice());
        model.set("children", children.map((c) => c.slice()));
        model.save_changes();
      }
      function readChildren() {
        const c = model.get("children");
        return Array.isArray(c) ? c.map((r) => r.slice()) : [];
      }
      model.on("change:children", () => {
        children = readChildren();
        if (sel && sel.kind === "child" && sel.i >= children.length)
          sel = children.length ? { kind: "child", i: children.length - 1 } : null;
        draw();
      });
      model.on("change:parent", () => {
        parent = readList(model.get("parent"), [0.5, 0.5, 1.0, 0.0]);
        draw();
      });
      model.on("change:parent_image", () => {
        // refresh aspect once the new image reports its natural size
        const probe = new Image();
        probe.onload = () => {
          if (probe.naturalHeight) aspect = probe.naturalWidth / probe.naturalHeight;
          draw();
        };
        probe.src = model.get("parent_image") || "";
        draw();
      });

      const ro = new ResizeObserver(() => draw());
      ro.observe(stage);

      // initial aspect probe
      const first = new Image();
      first.onload = () => {
        if (first.naturalHeight) aspect = first.naturalWidth / first.naturalHeight;
        draw();
      };
      first.src = model.get("parent_image") || "";
      draw();

      return () => ro.disconnect();
    }

    function button(label, title) {
      const b = document.createElement("button");
      b.className = "rimg-btn";
      b.textContent = label;
      if (title) b.title = title;
      return b;
    }
    function readList(v, fallback) {
      return Array.isArray(v) && v.length === 4 ? v.slice() : fallback.slice();
    }
    function clamp01(v) {
      return Math.max(0, Math.min(1, v));
    }
    function clampRange(v, lo, hi) {
      return Math.max(lo, Math.min(hi, v));
    }
    function dist(a, b) {
      return Math.hypot(a[0] - b[0], a[1] - b[1]);
    }

    export default { render };
    """

    _RIMG_CSS = r"""
    .rimg-container {
      color-scheme: light dark;
      --rimg-bg: #ffffff;
      --rimg-text: #111827;
      --rimg-border: #d1d5db;
      --rimg-toolbar-bg: #f9fafb;
      --rimg-btn-hover: #f3f4f6;
      --rimg-btn-active: #e5e7eb;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
        "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      box-sizing: border-box;
      border: 1px solid var(--rimg-border);
      border-radius: 0.5rem;
      overflow: hidden;
      background-color: var(--rimg-bg);
      color: var(--rimg-text);
      display: flex;
      flex-direction: column;
    }

    :where(.dark, .dark-theme, [data-theme="dark"]) .rimg-container {
      --rimg-bg: #1f2937;
      --rimg-text: #f9fafb;
      --rimg-border: #4b5563;
      --rimg-toolbar-bg: #111827;
      --rimg-btn-hover: #4b5563;
      --rimg-btn-active: #6b7280;
    }

    .rimg-toolbar {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 0.375rem;
      padding: 0.375rem 0.5rem;
      background-color: var(--rimg-toolbar-bg);
      border-bottom: 1px solid var(--rimg-border);
      flex-shrink: 0;
    }

    .rimg-btn {
      display: inline-flex;
      align-items: center;
      height: 2.25rem;
      padding: 0 0.875rem;
      border: 1px solid var(--rimg-border);
      border-radius: 0.375rem;
      background-color: var(--rimg-bg);
      color: var(--rimg-text);
      cursor: pointer;
      font-size: 14px;
      white-space: nowrap;
      transition: background-color 0.15s ease, border-color 0.15s ease;
    }

    .rimg-btn:hover {
      background-color: var(--rimg-btn-hover);
      border-color: var(--rimg-border);
    }

    .rimg-btn:active {
      background-color: var(--rimg-btn-active);
    }

    .rimg-stage {
      position: relative;
      width: 100%;
      aspect-ratio: 1 / 1;
      background: #ffffff
        repeating-conic-gradient(#eef2f7 0% 25%, #ffffff 0% 50%) 0 / 28px 28px;
      overflow: hidden;
      touch-action: none;
    }

    .rimg-el {
      position: absolute;
      transform-origin: center center;
      cursor: grab;
    }

    .rimg-img {
      width: 100%;
      height: 100%;
      display: block;
      pointer-events: none;
    }

    .rimg-parent {
      outline: 2px solid #334155;
    }
    .rimg-parent .rimg-img {
      opacity: 0.9;
    }

    .rimg-child {
      outline: 1px solid #60a5fa;
    }
    .rimg-child .rimg-img {
      opacity: 0.55;
    }

    .rimg-el.selected {
      outline-width: 2px;
      outline-color: #2563eb;
      z-index: 5;
    }
    .rimg-el.selected .rimg-img {
      opacity: 0.85;
    }

    .rimg-handle {
      position: absolute;
      width: 14px;
      height: 14px;
      box-sizing: border-box;
      background: #2563eb;
      border: 2px solid #ffffff;
      border-radius: 50%;
    }

    .rimg-scale {
      right: -7px;
      bottom: -7px;
      cursor: nwse-resize;
    }

    .rimg-rotate {
      left: 50%;
      top: -26px;
      margin-left: -7px;
      background: #16a34a;
      cursor: grab;
    }
    """

    class RecursiveImageWidget(anywidget.AnyWidget):
        """Author an IFS rule by placing transformed copies of a parent image.

        A *placement* is ``[x, y, scale, angle_deg]``: the base box (world-width
        ``BASE``, see JS) scaled by ``scale``, rotated ``angle_deg`` clockwise,
        and centered at ``(x, y)`` in world coords ``[0, 1]``. The ``parent``
        placement defines the base frame; each child in ``children`` is another
        placement. The recursive rule is, per child, the similarity mapping the
        parent box onto the child box.
        """

        _esm = _RIMG_JS
        _css = _RIMG_CSS

        # Stage size in pixels (square).
        size = traitlets.Int(320).tag(sync=True)

        # Parent PNG as a base64 data URL (set from wigglystuff Paint).
        parent_image = traitlets.Unicode("").tag(sync=True)

        # Parent placement [x, y, scale, angle_deg] — defines the base frame.
        parent = traitlets.List(
            traitlets.Float(), default_value=[0.5, 0.5, 1.0, 0.0]
        ).tag(sync=True)

        # One placement per child: [x, y, scale, angle_deg].
        children = traitlets.List(
            traitlets.List(traitlets.Float()), default_value=[]
        ).tag(sync=True)

    return (RecursiveImageWidget,)


if __name__ == "__main__":
    app.run()
