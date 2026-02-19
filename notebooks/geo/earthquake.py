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


@app.cell
def _():
    data = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_month.geojson"
    return (data,)


@app.cell
def _(data, ol):
    vector = ol.VectorLayer(
        source=ol.VectorSource(url=data)
    )
    return (vector,)


@app.cell
def _(data, ol):
    radius = 8
    blur = 15

    heatmap = ol.HeatmapLayer(
        source=ol.VectorSource(url=data),
        opacity=0.5,
        weight=["get", "mag"],
        radius=radius,
        blur=blur
    )
    return blur, heatmap, radius


@app.cell
def _(heatmap, ol, vector):
    m = ol.MapWidget(layers=[ol.BasemapLayer(), vector, heatmap])
    m.add_tooltip()
    return (m,)


@app.cell
def _(m):
    m
    return


@app.cell(hide_code=True)
def _(heatmap, m, mo, radius):
    mo.ui.slider(start=0, stop=20, step=1, value=radius, label="radius", on_change=lambda r: m.add_layer_call(heatmap.id, "setRadius", r))

    return


@app.cell(hide_code=True)
def _(blur, heatmap, m, mo):
    mo.ui.slider(start=0, stop=30, step=1, value=blur, label="blur", on_change=lambda b: m.add_layer_call(heatmap.id, "setBlur", b))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()