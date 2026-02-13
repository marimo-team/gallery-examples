# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "altair==6.0.0",
#     "marimo",
#     "matplotlib==3.10.8",
#     "pandas==3.0.0",
#     "pymde==0.2.3",
#     "torch==2.10.0",
# ]
# ///

import marimo

__generated_with = "0.19.8"
app = marimo.App(
    width="medium",
    css_file="/usr/local/_marimo/custom.css",
    auto_download=["html"],
)

with app.setup:
    import matplotlib.pyplot as plt
    import pymde
    import torch

    import marimo as mo


@app.cell(hide_code=True)
def _():
    mo.md("""
    # Visualizing embeddings
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    Here's an **embedding of MNIST**: each point represents a digit,
    with similar digits close to each other.

    **Try making a selection with your mouse!**
    """)
    return


@app.cell
def _(compute_embedding, constraint, embedding_dimension):
    with mo.persistent_cache("embedding"):
        embedding = compute_embedding(embedding_dimension, constraint)
    return (embedding,)


@app.cell
def _(alt, df):
    chart = mo.ui.altair_chart(
        alt.Chart(df)
        .mark_circle(size=4)
        .encode(
            x=alt.X("x:Q").scale(domain=(-2.5, 2.5)),
            y=alt.Y("y:Q").scale(domain=(-2.5, 2.5)),
            color=alt.Color("digit:N"),
        )
        .properties(width=500, height=500),
        chart_selection="interval",
    )
    chart
    return (chart,)


@app.cell
def _(chart):
    table = mo.ui.table(chart.value)
    return (table,)


@app.cell(hide_code=True)
def _(chart, show_images, table):
    # mo.stop() prevents this cell from running if the chart has
    # no selection
    mo.stop(not len(chart.value))

    # show 10 images: either the first 10 from the selection, or the first ten
    # selected in the table
    selected_images = (
        show_images(list(chart.value["index"]))
        if not len(table.value)
        else show_images(list(table.value["index"]))
    )

    mo.md(
        f"""
        **Here's a preview of the images you've selected**:

        {mo.as_html(selected_images)}

        Here's all the data you've selected.

        {table}
        """
    )
    return


@app.cell
def _():
    embedding_dimension = 2
    constraint = pymde.Standardized()
    return constraint, embedding_dimension


@app.cell
def _(embedding, mnist, pd):
    indices = torch.randperm(mnist.data.shape[0])[:20000].numpy()
    embedding_sampled = embedding.numpy()[indices]

    df = pd.DataFrame(
        {
            "index": indices,
            "x": embedding_sampled[:, 0],
            "y": embedding_sampled[:, 1],
            "digit": mnist.attributes["digits"][indices],
        }
    )
    return (df,)


@app.cell
def _(mnist):
    def compute_embedding(embedding_dim, constraint):
        mo.output.append(
            mo.md("Your embedding is being computed ... hang tight!").callout(kind="warn")
        )

        mde = pymde.preserve_neighbors(
            mnist.data,
            embedding_dim=embedding_dim,
            constraint=constraint,
            device="cuda" if torch.cuda.is_available() else "cpu",
            verbose=True,
        )
        X = mde.embed(verbose=True)
        mo.output.clear()
        return X

    return (compute_embedding,)


@app.cell
def _():
    mnist = pymde.datasets.MNIST()
    return (mnist,)


@app.cell
def _(mnist):
    def show_images(indices, max_images=10):
        indices = indices[:max_images]
        images = mnist.data.reshape((-1, 28, 28))[indices]
        fig, axes = plt.subplots(1, len(indices))
        fig.set_size_inches(12.5, 1.5)
        if len(indices) > 1:
            for im, ax in zip(images, axes.flat):
                ax.imshow(im, cmap="gray")
                ax.set_yticks([])
                ax.set_xticks([])
        else:
            axes.imshow(images[0], cmap="gray")
            axes.set_yticks([])
            axes.set_xticks([])
        plt.tight_layout()
        return fig

    return (show_images,)


@app.cell
def _():
    import altair as alt

    return (alt,)


@app.cell
def _():
    import pandas as pd

    return (pd,)


if __name__ == "__main__":
    app.run()
