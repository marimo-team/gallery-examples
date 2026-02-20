# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy==2.3.5",
#     "matplotlib==3.10.8",
#     "wigglystuff",
# ]
# ///

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    from wigglystuff import ChartPuck


@app.cell
def _():
    matplotlib.rcParams["figure.dpi"] = 72
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    # Vector Operations Explorer

    Drag the two pucks to define vectors **A** (blue) and **B** (orange).
    Pick an operation from the dropdown to see the result (green).
    """)
    return


@app.cell
def _():
    operation_dropdown = mo.ui.dropdown(
        options={
            "Addition": "addition",
            "Subtraction": "subtraction",
            "Projection": "projection",
        },
        value="Addition",
        label="Operation",
    )
    operation_dropdown
    return (operation_dropdown,)


@app.cell(hide_code=True)
def _(operation_dropdown):
    x_bounds = (-5, 5)
    y_bounds = (-5, 5)

    initial_x = [1.0, 2.0]
    initial_y = [2.0, -1.0]

    def draw_vectors(ax, widget):
        ax_val, ay_val = widget.x[0], widget.y[0]
        bx_val, by_val = widget.x[1], widget.y[1]

        arrow_kw = dict(head_width=0.15, head_length=0.1, length_includes_head=True)

        # Vector A (blue)
        ax.arrow(0, 0, ax_val, ay_val, fc="#1f77b4", ec="#1f77b4", **arrow_kw)
        ax.text(ax_val / 2 - 0.3, ay_val / 2 + 0.2, "A", color="#1f77b4", fontsize=12, fontweight="bold")

        # Vector B (orange)
        ax.arrow(0, 0, bx_val, by_val, fc="#ff7f0e", ec="#ff7f0e", **arrow_kw)
        ax.text(bx_val / 2 + 0.2, by_val / 2 + 0.2, "B", color="#ff7f0e", fontsize=12, fontweight="bold")

        op = operation_dropdown.value

        if op == "addition":
            sx, sy = ax_val + bx_val, ay_val + by_val
            ax.arrow(0, 0, sx, sy, fc="#2ca02c", ec="#2ca02c", **arrow_kw)
            ax.text(sx / 2 + 0.2, sy / 2 + 0.2, "A+B", color="#2ca02c", fontsize=12, fontweight="bold")
            # Parallelogram dashed lines
            ax.plot([ax_val, sx], [ay_val, sy], "--", color="#ff7f0e", alpha=0.5, linewidth=1.5)
            ax.plot([bx_val, sx], [by_val, sy], "--", color="#1f77b4", alpha=0.5, linewidth=1.5)

        elif op == "subtraction":
            dx, dy = ax_val - bx_val, ay_val - by_val
            ax.arrow(0, 0, dx, dy, fc="#2ca02c", ec="#2ca02c", **arrow_kw)
            ax.text(dx / 2 + 0.2, dy / 2 + 0.2, "A-B", color="#2ca02c", fontsize=12, fontweight="bold")
            # Dashed line from tip of B to tip of A
            ax.plot([bx_val, ax_val], [by_val, ay_val], "--", color="#2ca02c", alpha=0.5, linewidth=1.5)

        elif op == "projection":
            dot_ab = ax_val * bx_val + ay_val * by_val
            dot_bb = bx_val**2 + by_val**2
            if dot_bb > 1e-9:
                scalar = dot_ab / dot_bb
                proj_x, proj_y = scalar * bx_val, scalar * by_val
                ax.arrow(0, 0, proj_x, proj_y, fc="#2ca02c", ec="#2ca02c", **arrow_kw)
                ax.text(proj_x / 2 + 0.2, proj_y / 2 + 0.2, "proj", color="#2ca02c", fontsize=12, fontweight="bold")
                # Dashed perpendicular from A's tip to projection
                ax.plot([ax_val, proj_x], [ay_val, proj_y], "--", color="#999999", alpha=0.7, linewidth=1.5)
                ax.plot(proj_x, proj_y, "o", color="#2ca02c", markersize=5)

        # Axis lines and grid
        ax.axhline(0, color="black", linewidth=0.5)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(x_bounds)
        ax.set_ylim(y_bounds)
        ax.set_aspect("equal")

    puck = mo.ui.anywidget(
        ChartPuck.from_callback(
            draw_fn=draw_vectors,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
            figsize=(6, 6),
            x=initial_x,
            y=initial_y,
            puck_radius=6,
            throttle=100,
            puck_color=["steelblue", "orange"]
        )
    )
    return (puck,)


@app.cell
def _(puck):
    puck
    return


@app.cell
def _(operation_dropdown, puck):
    ax_val, ay_val = puck.x[0], puck.y[0]
    bx_val, by_val = puck.x[1], puck.y[1]

    mag_a = np.sqrt(ax_val**2 + ay_val**2)
    mag_b = np.sqrt(bx_val**2 + by_val**2)
    angle_a = np.degrees(np.arctan2(ay_val, ax_val))
    angle_b = np.degrees(np.arctan2(by_val, bx_val))

    lines = [
        f"**A** = ({ax_val:.2f}, {ay_val:.2f}), |A| = {mag_a:.2f}, θ = {angle_a:.1f}°",
        f"**B** = ({bx_val:.2f}, {by_val:.2f}), |B| = {mag_b:.2f}, θ = {angle_b:.1f}°",
    ]

    op = operation_dropdown.value
    if op == "addition":
        sx, sy = ax_val + bx_val, ay_val + by_val
        mag_s = np.sqrt(sx**2 + sy**2)
        lines.append(f"**A + B** = ({sx:.2f}, {sy:.2f}), |A+B| = {mag_s:.2f}")
    elif op == "subtraction":
        dx, dy = ax_val - bx_val, ay_val - by_val
        mag_d = np.sqrt(dx**2 + dy**2)
        lines.append(f"**A - B** = ({dx:.2f}, {dy:.2f}), |A-B| = {mag_d:.2f}")
    elif op == "projection":
        dot_bb = bx_val**2 + by_val**2
        if dot_bb > 1e-9:
            scalar = (ax_val * bx_val + ay_val * by_val) / dot_bb
            proj_x, proj_y = scalar * bx_val, scalar * by_val
            mag_p = np.sqrt(proj_x**2 + proj_y**2)
            lines.append(f"**proj_B(A)** = ({proj_x:.2f}, {proj_y:.2f}), |proj| = {mag_p:.2f}")

    mo.md("\n\n".join(lines))
    return


if __name__ == "__main__":
    app.run()
