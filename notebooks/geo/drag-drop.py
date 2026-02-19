# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo",
#     "openlayers==0.1.6",
# ]
# ///

import marimo

__generated_with = "0.14.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import openlayers as ol
    return mo, ol


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Drag and drop GPX, GeoJSON, KML or TopoJSON files on to the map

        Download sample files from here:

        * [states.kml](https://openlayers.org/en/latest/examples/data/kml/states.kml)
        * [roads-seoul.geojson](https://openlayers.org/en/latest/examples/data/geojson/roads-seoul.geojson)
        * [fr-departments.json](https://openlayers.org/en/latest/examples/data/topojson/fr-departments.json)
        * [fells_loop.gpx](https://openlayers.org/en/latest/examples/data/gpx/fells_loop.gpx)
        """
    )
    return


@app.cell
def _(ol):
    m = ol.MapWidget()
    return (m,)


@app.cell
def _(m):
    m.add_drag_and_drop_interaction()
    return


@app.cell
def _(m):
    m
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()