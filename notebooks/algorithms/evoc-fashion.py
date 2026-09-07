# /// script
# requires-python = "==3.12.*"
# dependencies = [
#     "marimo",
#     "polars==1.39.3",
#     "numpy==2.4.4",
#     "scikit-learn==1.8.0",
#     "wigglystuff==0.3.5",
#     "matplotlib==3.10.8",
#     "pandas==3.0.1",
#     "umap-learn==0.5.11",
#     "evoc==0.3.1",
#     "altair==6.1.0",
#     "pyarrow==24.0.0",
# ]
# ///

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import polars as pl
    from sklearn.datasets import fetch_openml
    from sklearn.decomposition import PCA
    from umap import UMAP
    from evoc import EVoC
    from wigglystuff import ParallelCoordinates
    import matplotlib.pyplot as plt
    import altair as alt

    return (
        EVoC,
        PCA,
        ParallelCoordinates,
        UMAP,
        alt,
        fetch_openml,
        mo,
        np,
        pl,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Nested Clusters with EVoC

    This notebook loads the Fashion MNIST dataset, reduces the 784 pixel features
    down to a handful of components, and visualizes them with an interactive
    parallel coordinates plot. Use the brushes on each axis to filter and explore
    how different clothing categories separate in PCA/UMAP space.

    But why stop there? You can also explore clustering methods like [EVoC](https://github.com/TutteInstitute/evoc) that give you a view into nested clusters. These make the parallel coordinates more interesting, but you can also explore them with other widgets as well.
    """)
    return


@app.cell
def _(fetch_openml, np):
    mnist = fetch_openml("Fashion-MNIST", version=1, as_frame=False, parser="auto")
    images = mnist.data.astype(np.float32)
    labels = mnist.target.astype(int)

    label_names = {
        0: "T-shirt/top",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle boot",
    }
    return images, label_names, labels


@app.cell
def _(
    PCA,
    UMAP,
    checkbox,
    images,
    label_names,
    labels,
    n_components_slider,
    n_samples_slider,
    np,
    pl,
):
    rng = np.random.default_rng(42)
    idx = rng.choice(len(images), size=n_samples_slider.value, replace=False)

    if checkbox.value:
        pca = UMAP(n_components=n_components_slider.value)
    else:
        pca = PCA(n_components=n_components_slider.value)

    components = pca.fit_transform(images[idx])

    df = pl.DataFrame(
        {f"PC{i + 1}": components[:, i] for i in range(n_components_slider.value)}
    ).with_columns(pl.Series("label", [label_names[labels[i]] for i in idx]))
    return df, idx


@app.cell(hide_code=True)
def _(mo):
    n_samples_slider = mo.ui.slider(
        start=2500, stop=5000, step=500, value=2500, label="Number of samples"
    )
    n_components_slider = mo.ui.slider(start=3, stop=15, step=1, value=8, label="Components")
    checkbox = mo.ui.checkbox(label="UMAP")
    color_checkbox = mo.ui.checkbox(label="Color by label", value=True)
    [n_samples_slider, n_components_slider, checkbox, color_checkbox]
    return checkbox, color_checkbox, n_components_slider, n_samples_slider


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2D embedding

    Before diving into parallel coordinates, here's the first two components as a
    plain scatter. Brush a rectangle to pick a region of the embedding — the
    selected samples appear as thumbnails below. Toggle the **Color by label**
    checkbox above to see how the classes mix.
    """)
    return


@app.cell(hide_code=True)
def _(alt, color_checkbox, df, mo):
    chart_df = df.with_row_index("r").to_pandas()
    encodings = {"x": "PC1:Q", "y": "PC2:Q"}
    if color_checkbox.value:
        encodings["color"] = "label:N"
    base_chart = (
        alt.Chart(chart_df)
        .mark_circle(size=25, opacity=0.6)
        .encode(**encodings)
        .properties(width=600, height=400)
    )
    chart_2d = mo.ui.altair_chart(base_chart)
    chart_2d
    return (chart_2d,)


@app.cell(hide_code=True)
def _(chart_2d, idx, images, label_names, labels, np, plt):
    _selected = chart_2d.value
    _filtered = _selected["r"].to_list() if len(_selected) else []
    _sample_idx = np.array(_filtered[:10]) if len(_filtered) >= 10 else np.array(_filtered)

    if len(_sample_idx) == 0:
        _fig = None
    else:
        _fig, _axes = plt.subplots(1, len(_sample_idx), figsize=(2 * len(_sample_idx), 2))
        if len(_sample_idx) == 1:
            _axes = [_axes]
        for _ax, _si in zip(_axes, _sample_idx):
            _ax.imshow(images[idx[int(_si)]].reshape(28, 28), cmap="gray")
            _ax.set_title(label_names[labels[idx[int(_si)]]], fontsize=9)
            _ax.axis("off")
        plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(ParallelCoordinates, df, mo):
    widget = mo.ui.anywidget(ParallelCoordinates(df, height=500, color_by="label"))
    widget
    return


@app.cell(hide_code=True)
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Now to EVoCe a new trick!

    Let's now add the cluster layers to the chart. That already gives you an interesting idea on where you might be able to find clusters.
    """)
    return


@app.cell
def _(est):
    est.cluster_layers_
    return


@app.cell(hide_code=True)
def _(EVoC, ParallelCoordinates, df, idx, images, mo, np):
    est = EVoC(random_state=42)
    est.fit_predict(images[idx])

    pltr = df.with_columns(
        c0=est.cluster_layers_[0] + np.random.random(est.cluster_layers_[0].shape[0]) / 1.2,
        c1=est.cluster_layers_[1] + np.random.random(est.cluster_layers_[0].shape[0]) / 1.2,
        c2=est.cluster_layers_[2] + np.random.random(est.cluster_layers_[0].shape[0]) / 1.2,
    )

    evoc_widget = mo.ui.anywidget(ParallelCoordinates(pltr, height=500, color_by="label"))
    evoc_widget
    return est, evoc_widget


@app.cell(hide_code=True)
def _(evoc_widget, idx, images, label_names, labels, np, plt):
    _filtered = evoc_widget.selected_indices
    _sample_idx = np.array(_filtered[:10]) if len(_filtered) >= 10 else np.array(_filtered)

    _fig, _axes = plt.subplots(1, len(_sample_idx), figsize=(2 * len(_sample_idx), 2))
    if len(_sample_idx) == 1:
        _axes = [_axes]
    for _ax, _si in zip(_axes, _sample_idx):
        _ax.imshow(images[idx[_si]].reshape(28, 28), cmap="gray")
        _ax.set_title(label_names[labels[idx[_si]]], fontsize=9)
        _ax.axis("off")
    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Treemap

    You can also explore this data using a treemap. That's what we do below.
    """)
    return


@app.cell
def _(df, est, mo, pl):
    from wigglystuff import Treemap

    treemapped = df.select(c0=est.cluster_layers_[2], c1=est.cluster_layers_[1], c2=est.cluster_layers_[0], n=pl.lit(1), r=pl.row_index())

    _agg = treemapped.group_by("c0", "c1", "c2").len().sort("len", descending=True)

    treemap = mo.ui.anywidget(Treemap.from_dataframe(_agg, path_cols=["c0", "c1", "c2"], width="100%", height=500))
    treemap
    return treemap, treemapped


@app.cell(hide_code=True)
def _(idx, images, label_names, labels, np, plt, subset):
    _filtered = subset["r"].to_list()
    _sample_idx = np.array(_filtered[:10]) if len(_filtered) >= 10 else np.array(_filtered)

    _fig, _axes = plt.subplots(1, len(_sample_idx), figsize=(2 * len(_sample_idx), 2))
    if len(_sample_idx) == 1:
        _axes = [_axes]
    for _ax, _si in zip(_axes, _sample_idx):
        _ax.imshow(images[idx[_si]].reshape(28, 28), cmap="gray")
        _ax.set_title(label_names[labels[idx[_si]]], fontsize=9)
        _ax.axis("off")
    plt.tight_layout()
    _fig
    return


@app.cell
def _(pl, treemap, treemapped):
    subset = treemapped
    for col, val in enumerate(treemap.hovered_path[1:]):
        subset = subset.filter(pl.col(f"c{col}") == int(val))
    return (subset,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
