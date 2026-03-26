# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
#     "wigglystuff",
#     "mohtml==0.1.11",
#     "pillow",
#     "scipy",
# ]
# ///

import marimo

__generated_with = "0.21.1"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib

    matplotlib.rcParams["figure.dpi"] = 72
    import matplotlib.pyplot as plt
    from wigglystuff import ChartPuck

    return ChartPuck, mo, np, plt


@app.cell
def _(ChartPuck, mo, np, plt):
    x_bounds = (-5, 5)
    y_bounds = (-5, 5)


    def draw_circle(ax, widget):
        px, py = widget.x[0], widget.y[0]
        r = np.sqrt(px**2 + py**2)

        circle = plt.Circle((0, 0), r, fill=False, color="#e63946", linewidth=2)
        ax.add_patch(circle)

        ax.axhline(0, color="black", linewidth=0.5)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(x_bounds)
        ax.set_ylim(y_bounds)
        ax.set_aspect("equal")
        ax.set_title("")


    puck = mo.ui.anywidget(
        ChartPuck.from_callback(
            draw_fn=draw_circle,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
            figsize=(6, 6),
            x=2.0,
            y=0.0,
            puck_radius=6,
            throttle=100,
        )
    )
    return (puck,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Mapping between images

    This notebook is inspired by [this 3b1b video](https://www.youtube.com/watch?v=ldxFjLJ3rVY) and it serves as an interactive way to intuit what is happening.

    Let's start by comparing these two lines. You can move the dot on the left chart. Can you see the relationship?
    """)
    return


@app.cell
def _(mo, np, plt, puck):
    from mohtml import div

    px, py = puck.x[0], puck.y[0]
    puck_r = np.sqrt(px**2 + py**2)
    puck_theta = np.arctan2(py, px)
    puck_log_r = np.log(max(puck_r, 1e-9))

    # The circle (all points at radius r) maps to a vertical line at x = ln(r)
    theta_range = np.linspace(-np.pi, np.pi, 200)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(np.full_like(theta_range, puck_log_r), theta_range, color="#e63946", linewidth=2)
    ax.plot(puck_log_r, puck_theta, "o", color="#e63946", markersize=8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_xlabel("ln(r)")
    ax.set_ylabel("θ")
    ax.set_title("")

    mo.hstack([puck, mo.vstack([div(style="margin-top: 37px;"), fig])], justify="start")
    return (div,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    When you keep the circle in the same size and move the puck over it you'll see that is like changing an angle. And this is reflecting in the image to the right because the dot moves on the `y`-axis. When you change the radius though, you can see movement on the `x`-axis.

    But what if we apply a mapping of an entire image using this technique?
    """)
    return


@app.cell
def _(ChartPuck, log_img, map_log_to_complex, mo, np, plt):
    arr = np.array(log_img)
    mapped = map_log_to_complex(arr)


    def draw_mapped_with_circle(ax, widget):
        ax.imshow(mapped, extent=(-5, 5, -5, 5), origin="lower")
        px, py = widget.x[0], widget.y[0]
        r = np.sqrt(px**2 + py**2)
        circle = plt.Circle((0, 0), r, fill=False, color="#e63946", linewidth=2)
        ax.add_patch(circle)
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_aspect("equal")


    log_puck = mo.ui.anywidget(
        ChartPuck.from_callback(
            draw_fn=draw_mapped_with_circle,
            x_bounds=(-5, 5),
            y_bounds=(-5, 5),
            figsize=(6, 6),
            x=2.0,
            y=0.0,
            puck_radius=6,
            throttle=100,
        )
    )
    return arr, log_puck


@app.cell
def _(arr, div, log_puck, mo, np, plt):
    lp_x, lp_y = log_puck.x[0], log_puck.y[0]
    lp_r = np.sqrt(lp_x**2 + lp_y**2)
    log_r = np.log(max(lp_r, 1e-9))
    lp_theta = np.arctan2(lp_y, lp_x)
    angles = np.linspace(-np.pi, np.pi, 200)

    fig_log, ax_log = plt.subplots(figsize=(6, 6))
    ax_log.imshow(arr, extent=(-2, 2, -np.pi, np.pi), origin="lower", aspect="auto")
    ax_log.plot(np.full_like(angles, log_r), angles, color="#e63946", linewidth=2)
    ax_log.plot(log_r, lp_theta, "o", color="#e63946", markersize=10, zorder=5)
    ax_log.set_xlim(-2, 2)
    ax_log.set_ylim(-np.pi, np.pi)
    ax_log.set_xlabel("ln(r)")
    ax_log.set_ylabel("θ")

    mo.hstack([log_puck, mo.vstack([div(style="margin-top: 37px;"), fig_log])], justify="start")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Things can feel quite "trippy" but the main thing to remember here is that we're "just" doing a mapping. The chart on the right is transformed just like the red line.

    ## Different projections.

    By now you may recognize that the mapping has a relationship to complex numbers. These live in "polar coordinate"-land and can sometimes cause behavior that feels strange at first but make some sense when you start thinging about it.

    In the next section, you'll explore a point with a radius and you're able to select different functions in the complex plane to get an impression of how the underlying data is transformed.
    """)
    return


@app.cell
def _():
    from PIL import Image, ImageDraw
    from scipy.ndimage import map_coordinates

    return Image, ImageDraw, map_coordinates


@app.cell
def _(Image, ImageDraw):
    def make_checkerboard(width=400, height=400, tile_size=50):
        img = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(img)
        for row in range(0, height, tile_size):
            for col in range(0, width, tile_size):
                if (row // tile_size + col // tile_size) % 2 == 0:
                    draw.rectangle([col, row, col + tile_size, row + tile_size], fill="#b0b0b0")
        return img


    log_img = make_checkerboard()
    return (log_img,)


@app.cell
def _(map_coordinates, np):
    def map_log_to_complex(img_arr, output_size=400, plane_bounds=(-5, 5), log_bounds=(-2, 2)):
        h, w, _ = img_arr.shape
        lin = np.linspace(plane_bounds[0], plane_bounds[1], output_size)
        gx, gy = np.meshgrid(lin, lin)

        radius = np.sqrt(gx**2 + gy**2)
        angle = np.arctan2(gy, gx)
        log_radius = np.log(np.clip(radius, 1e-9, None))

        px_x = (log_radius - log_bounds[0]) / (log_bounds[1] - log_bounds[0]) * w
        px_y = (angle - (-np.pi)) / (2 * np.pi) * h

        channels = []
        for i in range(3):
            ch = map_coordinates(img_arr[..., i], [px_y % h, px_x % w], order=1)
            channels.append(ch)
        return np.stack(channels, axis=-1)

    return (map_log_to_complex,)


@app.cell
def _(mo):
    mapping_dropdown = mo.ui.dropdown(
        options={
            "z²": "z_squared",
            "1/z": "inversion",
            "z + 1/z (Joukowski)": "joukowski",
            "e^z": "exp",
            "log(z)": "log",
        },
        value="z²",
        label="Mapping",
    )
    return (mapping_dropdown,)


@app.cell
def _(np):
    def apply_mapping(z, name):
        if name == "z_squared":
            return z**2
        elif name == "inversion":
            return 1.0 / np.where(np.abs(z) < 1e-9, 1e-9, z)
        elif name == "joukowski":
            safe_z = np.where(np.abs(z) < 1e-9, 1e-9, z)
            return safe_z + 1.0 / safe_z
        elif name == "exp":
            return np.exp(z)
        elif name == "log":
            return np.log(np.where(np.abs(z) < 1e-9, 1e-9, z) + 0j)
        return z

    return (apply_mapping,)


@app.cell
def _(ChartPuck, mo, np):
    bounds = (-5, 5)
    radii = [0.3, 0.6, 1.0]
    colors = ["#e63946", "#457b9d", "#2a9d8f"]
    n_pts = 200


    def draw_input_circles(ax, widget):
        cx, cy = widget.x[0], widget.y[0]
        t = np.linspace(0, 2 * np.pi, n_pts)
        for rad, col in zip(radii, colors):
            ax.plot(cx + rad * np.cos(t), cy + rad * np.sin(t), color=col, linewidth=2)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(bounds)
        ax.set_ylim(bounds)
        ax.set_aspect("equal")
        ax.set_title("Input: z")


    mapping_puck = mo.ui.anywidget(
        ChartPuck.from_callback(
            draw_fn=draw_input_circles,
            x_bounds=bounds,
            y_bounds=bounds,
            figsize=(6, 6),
            x=2.0,
            y=1.0,
            puck_radius=6,
            throttle=100,
        )
    )
    return bounds, colors, mapping_puck, radii


@app.cell
def _(apply_mapping, np):
    def draw_mapped_circles(ax, cx, cy, radii, colors, mapping_name, n_pts=200):
        """Draw circles centered at (cx, cy) mapped through a function onto ax."""
        t = np.linspace(0, 2 * np.pi, n_pts)
        for rad, col in zip(radii, colors):
            z_in = (cx + rad * np.cos(t)) + 1j * (cy + rad * np.sin(t))
            z_out = apply_mapping(z_in, mapping_name)
            ax.plot(z_out.real, z_out.imag, color=col, linewidth=2)
        center_mapped = apply_mapping(cx + 1j * cy, mapping_name)
        ax.plot(
            center_mapped.real, center_mapped.imag, "o", color="#e63946", markersize=10, zorder=5
        )

    return (draw_mapped_circles,)


@app.cell
def _(
    bounds,
    colors,
    div,
    draw_mapped_circles,
    mapping_dropdown,
    mapping_puck,
    mo,
    plt,
    radii,
):
    fig_out, ax_out = plt.subplots(figsize=(6, 6))
    draw_mapped_circles(
        ax_out,
        mapping_puck.x[0],
        mapping_puck.y[0],
        radii,
        colors,
        mapping_dropdown.value,
    )
    ax_out.axhline(0, color="black", linewidth=0.5)
    ax_out.axvline(0, color="black", linewidth=0.5)
    ax_out.grid(True, alpha=0.3)
    ax_out.set_xlim(bounds)
    ax_out.set_ylim(bounds)
    ax_out.set_aspect("equal")
    ax_out.set_title(f"Output: f(z) = {mapping_dropdown.value}")

    mo.hstack(
        [mapping_puck, mo.vstack([div(style="padding-top: 22px;"), fig_out])], justify="start"
    )
    return


@app.cell
def _(mo):
    wrap_checkbox = mo.ui.checkbox(value=False, label="Wrap texture (tiling effect)")
    return (wrap_checkbox,)


@app.cell
def _(mapping_dropdown, mo, texture_dropdown, wrap_checkbox):
    mo.hstack([mapping_dropdown, texture_dropdown, wrap_checkbox], justify="start")
    return


@app.cell
def _(mo):
    texture_dropdown = mo.ui.dropdown(
        options={
            "Checkerboard": "checkerboard",
            "Color Spectrum": "spectrum",
            "Polar Grid": "polar_grid",
        },
        value="Checkerboard",
        label="Texture",
    )
    return (texture_dropdown,)


@app.cell
def _(Image, ImageDraw, np):
    def make_texture(name, width=400, height=400):
        if name == "checkerboard":
            tile_size = 50
            img = Image.new("RGB", (width, height), "white")
            draw = ImageDraw.Draw(img)
            for row in range(0, height, tile_size):
                for col in range(0, width, tile_size):
                    if (row // tile_size + col // tile_size) % 2 == 0:
                        draw.rectangle([col, row, col + tile_size, row + tile_size], fill="#b0b0b0")
            return np.array(img)
        elif name == "spectrum":
            x = np.linspace(0, 1, width)
            y = np.linspace(0, 1, height)
            gx, gy = np.meshgrid(x, y)
            r = (255 * gx).astype(np.uint8)
            g = (255 * gy).astype(np.uint8)
            b = (255 * (1 - gx)).astype(np.uint8)
            return np.stack([r, g, b], axis=-1)
        elif name == "polar_grid":
            img = Image.new("RGB", (width, height), "white")
            draw = ImageDraw.Draw(img)
            cx, cy = width // 2, height // 2
            for ring in range(20, max(width, height), 40):
                draw.ellipse(
                    [cx - ring, cy - ring, cx + ring, cy + ring], outline="#457b9d", width=1
                )
            for angle_deg in range(0, 360, 30):
                angle = np.radians(angle_deg)
                ex = int(cx + max(width, height) * np.cos(angle))
                ey = int(cy + max(width, height) * np.sin(angle))
                draw.line([(cx, cy), (ex, ey)], fill="#457b9d", width=1)
            return np.array(img)

    return (make_texture,)


@app.cell
def _(map_coordinates, np, wrap_checkbox):
    def map_image_through(img_arr, mapping_name, output_size=400, plane_bounds=(-5, 5)):
        h, w = img_arr.shape[:2]
        lin = np.linspace(plane_bounds[0], plane_bounds[1], output_size)
        gx, gy = np.meshgrid(lin, lin)
        z = gx + 1j * gy

        if mapping_name == "z_squared":
            src = np.sqrt(z + 0j)
        elif mapping_name == "inversion":
            safe_z = np.where(np.abs(z) < 1e-9, 1e-9, z)
            src = 1.0 / safe_z
        elif mapping_name == "joukowski":
            # Inverse of w = z + 1/z is z = (w ± sqrt(w²-4))/2
            disc = np.sqrt(z**2 - 4 + 0j)
            src = (z + disc) / 2
        elif mapping_name == "exp":
            src = np.log(z + 0j)
        elif mapping_name == "log":
            src = np.exp(z)
        else:
            src = z

        px_x = (src.real - plane_bounds[0]) / (plane_bounds[1] - plane_bounds[0]) * w
        px_y = (src.imag - plane_bounds[0]) / (plane_bounds[1] - plane_bounds[0]) * h

        if wrap_checkbox.value:
            # Wrap mode: image tiles seamlessly outside bounds
            channels = []
            for i in range(3):
                ch = map_coordinates(img_arr[..., i], [px_y, px_x], order=1, mode="wrap")
                channels.append(ch)
            return np.stack(channels, axis=-1)

        # Transparent mode: out-of-bounds pixels become transparent
        valid = (px_x >= 0) & (px_x < w) & (px_y >= 0) & (px_y < h) & np.isfinite(px_x) & np.isfinite(px_y)
        channels = []
        for i in range(3):
            ch = map_coordinates(img_arr[..., i], [px_y, px_x], order=1, mode="constant", cval=0)
            channels.append(ch)
        alpha = (valid * 255).astype(np.uint8)
        return np.dstack([np.stack(channels, axis=-1), alpha])

    return (map_image_through,)


@app.cell
def _(np):
    def draw_input_circles_on(ax, cx, cy, radii, colors, n_pts=200):
        """Draw concentric circles centered at (cx, cy) on ax."""
        t = np.linspace(0, 2 * np.pi, n_pts)
        for rad, col in zip(radii, colors):
            ax.plot(cx + rad * np.cos(t), cy + rad * np.sin(t), color=col, linewidth=2)
        ax.plot(cx, cy, "o", color="#e63946", markersize=10, zorder=5)

    return


@app.cell
def _(ChartPuck, colors, make_texture, mo, np, radii, texture_dropdown):
    texture = make_texture(texture_dropdown.value)


    def draw_texture_with_circles(ax, widget):
        ax.imshow(texture, extent=(-5, 5, -5, 5), origin="lower")
        cx, cy = widget.x[0], widget.y[0]
        t = np.linspace(0, 2 * np.pi, 200)
        for rad, col in zip(radii, colors):
            ax.plot(cx + rad * np.cos(t), cy + rad * np.sin(t), color=col, linewidth=2)
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_aspect("equal")
        ax.set_title("Input texture")


    texture_puck = mo.ui.anywidget(
        ChartPuck.from_callback(
            draw_fn=draw_texture_with_circles,
            x_bounds=(-5, 5),
            y_bounds=(-5, 5),
            figsize=(6, 6),
            x=2.0,
            y=1.0,
            puck_radius=6,
            throttle=100,
        )
    )
    return texture, texture_puck


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's now repeat the exercise but with an image at the bottom of it. The results can be quite unexpected!
    """)
    return


@app.cell
def _(
    colors,
    div,
    draw_mapped_circles,
    map_image_through,
    mapping_dropdown,
    mo,
    plt,
    radii,
    texture,
    texture_puck,
):
    mapped_texture = map_image_through(texture, mapping_dropdown.value)
    tcx, tcy = texture_puck.x[0], texture_puck.y[0]

    fig_dst, ax_dst = plt.subplots(figsize=(6, 6))
    ax_dst.imshow(mapped_texture, extent=(-5, 5, -5, 5), origin="lower")
    draw_mapped_circles(ax_dst, tcx, tcy, radii, colors, mapping_dropdown.value)
    ax_dst.set_xlim(-5, 5)
    ax_dst.set_ylim(-5, 5)
    ax_dst.set_aspect("equal")
    ax_dst.set_title(f"Mapped texture: {mapping_dropdown.value}")

    mo.hstack(
        [texture_puck, mo.vstack([div(style="padding-top: 21px;"), fig_dst])], justify="start"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    With `1/z` on a checkerboard, you can zoom in near the origin and see the pattern repeat infinitely — a hallmark of conformal inversion.
    """)
    return


@app.cell
def _(mo):
    zoom_slider = mo.ui.slider(0.1, 5.0, step=0.05, value=5.0, label="Zoom (half-width)")
    return (zoom_slider,)


@app.cell
def _(make_texture, map_image_through, mo, plt, zoom_slider):
    zoom_half = zoom_slider.value
    zoom_bounds = (-zoom_half, zoom_half)

    checker = make_texture("checkerboard")
    zoomed = map_image_through(checker, "inversion", output_size=500, plane_bounds=zoom_bounds)

    fig_zoom, ax_zoom = plt.subplots(figsize=(8, 8))
    ax_zoom.imshow(zoomed, extent=(*zoom_bounds, *zoom_bounds), origin="lower")
    ax_zoom.axhline(0, color="white", linewidth=0.5, alpha=0.5)
    ax_zoom.axvline(0, color="white", linewidth=0.5, alpha=0.5)
    ax_zoom.set_xlim(zoom_bounds)
    ax_zoom.set_ylim(zoom_bounds)
    ax_zoom.set_aspect("equal")
    ax_zoom.set_title(f"1/z on checkerboard — zoom: {zoom_half:.2f}")

    mo.vstack([zoom_slider, fig_zoom])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Appendix: What does "Wrap texture" do?

    When we map an image through a complex function, many output pixels trace back to source coordinates that fall outside the original image bounds. The **"Wrap texture"** checkbox controls what happens at those out-of-bounds pixels.

    - **Unchecked (default):** Out-of-bounds pixels become transparent. Only the region that maps back into the original image is visible, giving you a clear view of where the mapping is well-defined.
    - **Checked:** Out-of-bounds coordinates wrap around modulo the image size (e.g. coordinate 450 on a 400px image becomes 50, and -3 becomes 397). This effectively tiles the image infinitely in all directions, so the complex mapping produces a seamless, repeating pattern — which is where the "trippy" infinite-tiling effect comes from.

    Under the hood this is controlled by the `mode` parameter of `scipy.ndimage.map_coordinates`: `mode="wrap"` for tiling, `mode="constant"` with an alpha mask for transparency.
    """)
    return


if __name__ == "__main__":
    app.run()
