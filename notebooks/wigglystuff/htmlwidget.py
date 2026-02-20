# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "altair",
#     "marimo",
#     "matplotlib",
#     "mohtml",
#     "numpy",
#     "polars",
#     "wigglystuff",
# ]
# ///

import marimo

__generated_with = "0.19.11"
app = marimo.App()

with app.setup:
    import marimo as mo
    import random
    import time
    import matplotlib.pylab as plt
    import numpy as np
    import polars as pl
    from mohtml import img
    from wigglystuff import HTMLRefreshWidget
    from wigglystuff import ImageRefreshWidget
    from wigglystuff import ProgressBar
    from wigglystuff.utils import refresh_matplotlib


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Updating `matplotlib` charts

    The easiest way to update matplotlib charts is to first write a function that can generate a chart. The most common way to use matplotlib is to use syntax like `plt.plot(...)` followed by a `plt.show(...)` and the best way to capture all of these layers is to wrap them all ina single function. Once you have such a function, you can use the `@refresh_matplotlib` decorator to turn this function into something that we can use in a refreshable-chart.
    """)
    return


@app.function
@refresh_matplotlib
def cumsum_linechart(data):
    y = np.cumsum(data)
    plt.plot(np.arange(len(y)), y)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The decorator takes the matplotlib image and turns it into a base64 encoded string that can be plotted by `<img>` tags in html. You can see this for yourself in the example below. The `img(src=...)` function call in `mohtml` is effectively a bit of syntactic sugar around `<img src="...">`.
    """)
    return


@app.cell
def _():
    img(src=cumsum_linechart([1, 2, 3, 2]))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Having a static image is great, but we want dynamic images! That's where our `ImageRefreshWidget` comes in. It allows you to trigger a streaming update to an image by running code from another cell. Try it out below!
    """)
    return


@app.cell
def _():
    widget = ImageRefreshWidget(
        src=cumsum_linechart([1,2,3,4])
    )
    widget
    return (widget,)


@app.cell
def _(widget):
    data = [random.random() - 0.5]

    for i in range(20):
        data += [random.random() - 0.5]
        # This one line over here causes the update!
        widget.src = cumsum_linechart(data)
        time.sleep(0.2)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    When you re-run the cell below you should see that the widget updates. This works because the widget knows how to respond to a change to the `widget.src` property. You only need to make sure that you pass along a base64 string that html images can handle, which is covered by the decorator that we applied earlier.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Updating `altair` charts

    This library can also deal with altair charts. This works by turning the chart into an SVG. This is a static representation that does not require any javascript to run, which means that we can apply a similar pattern as before!

    > Due to a required dependency to convert the altair chart to SVG we cannot run the altair demo in WASM. This code will run just fine locally on your machine but currently breaks on the Github pages deployment.

    ```python
    import altair as alt
    from wigglystuff import HTMLRefreshWidget
    from wigglystuff.utils import refresh_altair, altair2svg

    @refresh_altair
    def altair_cumsum_chart(data):
        df = pl.DataFrame({
            "x": range(len(data)), "y": np.array(data).cumsum()
        })
        return alt.Chart(df).mark_line().encode(x="x", y="y")

    svg_widget = HTMLRefreshWidget(html=altair_cumsum_chart([1, 2]))

    more_data = [random.random() - 0.5 for _ in range(10)]

    for _i in range(10):
        more_data += [random.random() - 0.5]
        svg_widget.html = altair_cumsum_chart(more_data)
        time.sleep(0.1)

    for _i in range(10):
        more_data += [random.random() - 0.5]
        svg_widget.html = altair_cumsum_chart(more_data)
        time.sleep(0.1)
    ```
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Unlike matplotlib charts though, altair is actually designed to give you objects back. That means that you don't need to use a decorated function for the update, you can also just convert the altair chart to SVG directly. This library supports utilities for both patterns.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Oh ... one more thing about that `HTMLRefreshWidget`

    We are injecting html now into that widget to allow us to draw altair charts. But why stop there? We can put in any HTML that we like!
    """)
    return


@app.cell
def _():
    html_widget = mo.ui.anywidget(HTMLRefreshWidget())
    html_widget
    return (html_widget,)


@app.cell
def _(html_widget):
    for _i in range(10):
        html_widget.html = f"<p>Counting {_i}</p>"
        time.sleep(0.1)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Progress bars

    We also provide progress bars that can update inplace without needing any 3rd party ipywidget tools. These should still work across all notebooks. These progress bars can also be updated from another cell in marimo, which the base progress bar does not allow.
    """)
    return


@app.cell
def _():
    progress = ProgressBar(value=0, max_value=100)
    progress
    return (progress,)


@app.cell
def _(progress):
    def slow_task():
        """Simulated task that takes time"""
        time.sleep(random.random() / 10)

    progress.value = 0 
    for _ in range(100):
        slow_task()
        progress.value += 1
    return


if __name__ == "__main__":
    app.run()
