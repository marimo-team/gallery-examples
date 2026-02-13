# /// script
# requires-python = "==3.12"
# dependencies = [
#     "build123d==0.10.0",
#     "marimo>=0.19.7",
#     "marimo-cad==0.1.0",
# ]
# ///
import marimo

__generated_with = "0.19.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import tempfile
    from pathlib import Path

    import marimo as mo
    from build123d import (
        Axis,
        BuildLine,
        BuildPart,
        BuildSketch,
        Line,
        Plane,
        Spline,
        make_face,
        revolve,
    )

    import marimo_cad as cad
    from marimo_cad import export_stl

    return (
        Axis,
        BuildLine,
        BuildPart,
        BuildSketch,
        Line,
        Path,
        Plane,
        Spline,
        cad,
        export_stl,
        make_face,
        mo,
        revolve,
        tempfile,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Parametric Vase

    A smooth, organic vase created with spline curves and revolution.
    Adjust the sliders - camera position is preserved!
    """)
    return


@app.cell
def _(mo):
    # Shape parameters
    height = mo.ui.slider(80, 200, value=120, step=10, label="Height (mm)")
    base_radius = mo.ui.slider(20, 50, value=30, step=5, label="Base Radius (mm)")
    belly_radius = mo.ui.slider(30, 80, value=50, step=5, label="Belly Radius (mm)")
    neck_radius = mo.ui.slider(15, 40, value=25, step=5, label="Neck Radius (mm)")
    top_radius = mo.ui.slider(20, 60, value=35, step=5, label="Top Radius (mm)")

    # Profile control
    belly_height = mo.ui.slider(20, 80, value=40, step=5, label="Belly Height (%)")
    neck_height = mo.ui.slider(60, 95, value=80, step=5, label="Neck Height (%)")

    # Wall thickness
    wall = mo.ui.slider(2, 6, value=3, step=0.5, label="Wall Thickness (mm)")
    return (
        base_radius,
        belly_height,
        belly_radius,
        height,
        neck_height,
        neck_radius,
        top_radius,
        wall,
    )


@app.cell
def _(cad):
    # Create viewer once - it persists across slider changes
    viewer = cad.Viewer()
    return (viewer,)


@app.cell
def _(
    Axis,
    BuildLine,
    BuildPart,
    BuildSketch,
    Line,
    Plane,
    Spline,
    base_radius,
    belly_height,
    belly_radius,
    height,
    make_face,
    neck_height,
    neck_radius,
    revolve,
    top_radius,
    wall,
):
    def build_vase(
        h: float,
        r_base: float,
        r_belly: float,
        r_neck: float,
        r_top: float,
        belly_pct: float,
        neck_pct: float,
        thickness: float,
    ):
        """Build a smooth vase using spline revolution."""
        h_belly = h * (belly_pct / 100)
        h_neck = h * (neck_pct / 100)

        inner_r_base = max(r_base - thickness, 2)
        inner_r_belly = max(r_belly - thickness, 2)
        inner_r_neck = max(r_neck - thickness, 2)
        inner_r_top = r_top - thickness

        with BuildLine(Plane.XZ) as outer_profile:
            Line((0, 0), (r_base, 0))
            Spline(
                (r_base, 0),
                (r_belly, h_belly),
                (r_neck, h_neck),
                (r_top, h),
            )
            Line((r_top, h), (inner_r_top, h))
            Spline(
                (inner_r_top, h),
                (inner_r_neck, h_neck),
                (inner_r_belly, h_belly),
                (inner_r_base, thickness),
            )
            Line((inner_r_base, thickness), (0, thickness))
            Line((0, thickness), (0, 0))

        with BuildSketch(Plane.XZ) as profile_sketch:
            make_face(outer_profile.wires()[0])

        with BuildPart() as vase_part:
            revolve(profile_sketch.sketch, axis=Axis.Z, revolution_arc=360)

        vase = vase_part.part
        parts = [{"shape": vase, "name": "Vase", "color": "#E8D5B7"}]
        return vase, parts

    vase_shape, vase_parts = build_vase(
        h=height.value,
        r_base=base_radius.value,
        r_belly=belly_radius.value,
        r_neck=neck_radius.value,
        r_top=top_radius.value,
        belly_pct=belly_height.value,
        neck_pct=neck_height.value,
        thickness=wall.value,
    )
    return build_vase, vase_parts, vase_shape


@app.cell
def _(
    Path,
    base_radius,
    belly_height,
    belly_radius,
    export_stl,
    height,
    mo,
    neck_height,
    neck_radius,
    tempfile,
    top_radius,
    vase_parts,
    vase_shape,
    viewer,
    wall,
):
    # Update viewer - camera stays put!
    viewer.render(vase_parts)

    def _get_stl_bytes():
        with tempfile.TemporaryDirectory() as tmpdir:
            stl_path = Path(tmpdir) / "vase.stl"
            export_stl(vase_shape, stl_path)
            return stl_path.read_bytes()

    download_btn = mo.download(
        data=_get_stl_bytes,
        filename="vase.stl",
        mimetype="model/stl",
        label="Download STL",
    )

    mo.vstack(
        [
            mo.hstack(
                [
                    mo.vstack([mo.md("**Dimensions**"), height, base_radius, wall]),
                    mo.vstack([mo.md("**Shape**"), belly_radius, neck_radius, top_radius]),
                    mo.vstack([mo.md("**Profile**"), belly_height, neck_height]),
                    mo.vstack([mo.md("**Export**"), download_btn]),
                ],
                justify="start",
                gap=2,
            ),
            viewer,
        ]
    )
    return (download_btn,)


if __name__ == "__main__":
    app.run()
