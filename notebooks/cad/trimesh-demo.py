# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.19.7",
#     "matplotlib==3.10.8",
#     "networkx",
#     "numpy",
#     "rtree",
#     "scipy",
#     "shapely",
#     "trimesh",
# ]
# ///

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import urllib.request
    import networkx
    import numpy as np
    import rtree
    import scipy
    import shapely
    import trimesh
    from pathlib import Path


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Trimesh

    The [trimesh](https://trimesh.org/section.html) library lets you load in 3d objects with some extra utilities for Python.
    """)
    return


@app.cell
def _(mesh):
    mesh.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    One feature that demos particularily well is the ability to contruct a 2D slice out of a 3D object. Use the slider below to slice around.
    """)
    return


@app.cell
def _():
    slider = mo.ui.slider(-0.5, 0.5, 0.01, debounce=True, label="cutoff")
    slider
    return (slider,)


@app.cell
def _(mesh, slider):
    origin = mesh.centroid + np.array([0, 0, slider.value])
    mesh_slice = mesh.section(plane_origin=origin, plane_normal=[0, 0, 1])
    slice_2D, to_3D = mesh_slice.to_2D()
    slice_2D.show()
    return


@app.cell
def _():
    stl_path = Path(__file__).parent / "featuretype.STL"
    if not stl_path.exists():
        urllib.request.urlretrieve(
            "https://github.com/mikedh/trimesh/raw/main/models/featuretype.STL",
            stl_path,
        )

    mesh = trimesh.load_mesh(stl_path)
    return (mesh,)


if __name__ == "__main__":
    app.run()
