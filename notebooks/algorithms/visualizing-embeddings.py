# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.8",
#     "pandas==3.0.0",
#     "pymde>=0.3.0",
#     "torch==2.10.0",
# ]
# ///

import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd
    import pymde
    import torch

    mnist = pymde.datasets.MNIST()


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


@app.function
@mo.persistent_cache
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


@app.cell
def _():
    embedding_dimension = 2
    constraint = pymde.Standardized()
    return constraint, embedding_dimension


@app.cell
def _(constraint, embedding_dimension):
    embedding = compute_embedding(embedding_dimension, constraint)
    return (embedding,)


@app.cell
def _(embedding):
    ax = pymde.plot(embedding, color_by=mnist.attributes["digits"])
    ax = mo.ui.matplotlib(ax)
    ax
    return (ax,)


@app.cell
def _(ax, embedding):
    mask = ax.value.get_mask(embedding[:, 0], embedding[:, 1])
    return (mask,)


@app.cell
def _(df, mask):
    table = mo.ui.table(df[mask])
    return (table,)


@app.cell(hide_code=True)
def _(mask, table):
    # mo.stop() prevents this cell from running if the ax has
    # no selection
    mo.stop(not mask.any())

    # show 10 images: either the first 10 from the selection, or the first ten
    # selected in the table
    selected_images = (
        show_images(list(mask.nonzero()[0]))
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


@app.function
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


@app.cell
def _(embedding):
    indices = torch.arange(mnist.data.shape[0]).numpy()

    df = pd.DataFrame(
        {
            "index": indices,
            "x": embedding[:, 0],
            "y": embedding[:, 1],
            "digit": mnist.attributes["digits"][indices],
        }
    )
    return (df,)


if __name__ == "__main__":
    app.run()
