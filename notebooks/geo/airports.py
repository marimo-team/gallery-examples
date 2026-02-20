# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo",
#     "openlayers==0.1.6",
# ]
# ///

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import openlayers as ol


@app.cell
def _():
    icon = "https://docs.maptiler.com/openlayers/examples/geojson-points/icon-plane-512.png"
    return (icon,)


@app.cell
def _():
    data = "https://d2ad6b4ur7yvpq.cloudfront.net/naturalearth-3.3.0/ne_10m_airports.geojson"
    return (data,)


@app.cell
def _(icon):
    style = ol.FlatStyle(
        icon_src=icon,
        icon_scale=["match", ["get", "type"], "major", 0.05, 0.03],
        # text_value=["get", "name"],
        # text_fill_color="steelblue"
    )
    return (style,)


@app.cell
def _(data, style):
    vector = ol.VectorLayer(source=ol.VectorSource(url=data), style=style)
    return (vector,)


@app.cell
def _(vector):
    m = ol.MapWidget(
        ol.View(center=(16.62662018, 49.2125578), zoom=5),
        layers=[ol.BasemapLayer(), vector],
    )
    m.add_tooltip()
    return (m,)


@app.cell
def _(m):
    m
    return


if __name__ == "__main__":
    app.run()
