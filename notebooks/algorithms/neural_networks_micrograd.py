# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.8",
#     "micrograd==0.1.0",
#     "numpy==2.4.2",
#     "scikit-learn==1.8.0",
# ]
# ///

import marimo

__generated_with = "0.20.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Neural Networks with Micrograd
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    This WASM-powered marimo notebook trains a tiny **neural network** using
    Andrej Karpathy's [micrograd
    library](https://github.com/karpathy/micrograd).
    Micrograd is an implementation of PyTorch-like automatic
    differentiation using only Python scalar operations. This notebook was
    adapted from a [demo
    notebook](https://github.com/karpathy/micrograd/blob/master/demo.ipynb) by
    Andrej.
    """)
    return


@app.cell(hide_code=True)
def _():
    import random
    import numpy as np
    import matplotlib.pyplot as plt

    np.random.seed(1337)
    random.seed(1337)
    return np, plt


@app.cell(hide_code=True)
async def _():
    import sys

    if "pyodide" in sys.modules:
        import micropip
        await micropip.install("micrograd")
    from micrograd.engine import Value
    from micrograd.nn import Neuron, Layer, MLP

    return MLP, Value


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    We start by generating a synthetic dataset of points labeled +1 or -1.
    Our goal is to train a network that can classify these points according
    to their labels, learning a decision boundary that separates them.
    """)
    return


@app.cell
def _():
    from sklearn.datasets import make_moons, make_blobs

    X, y = make_moons(n_samples=100, noise=0.1)
    y = y * 2 - 1  # make y be -1 or 1
    return X, y


@app.cell
def _(X, mo, plt, y):
    plt.figure(figsize=(5, 5))
    mo.center(plt.scatter(X[:, 0], X[:, 1], c=y, s=20, cmap=plt.cm.Spectral))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Note that the decision boundary must be **nonlinear**, which can be readily
    learned by neural networks. This could also be achieved by "shallow" or
    classical machine learning methods with the appropriate featurization or
    [kernelization](https://scikit-learn.org/stable/modules/svm.html).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **Try it!** Train a neural network by hitting the "Train" button. The
        learned decision boundary will be plotted below.

        _Try experimenting with the parameters. What happens if you change
        the number of layers and their sizes?_
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    n_layers = mo.ui.slider(
        start=1, stop=2, step=1, value=2, show_value=True,
        label="number of layers"
    )
    n_layers
    return (n_layers,)


@app.cell(hide_code=True)
def _(mo, n_layers):
    layer_sizes = mo.ui.array([
        mo.ui.slider(4, 16, step=1, value=12, show_value=True)
        for i in range(n_layers.value)
    ], label="layer sizes")

    iterations = mo.ui.slider(
        start=1,
        stop=40,
        step=1,
        value=20,
        show_value=True,
        label="gradient steps"
    )

    train_button = mo.ui.run_button(label="Train")

    mo.vstack([layer_sizes, iterations, train_button])
    return iterations, layer_sizes, train_button


@app.cell(hide_code=True)
def _(MLP, iterations, layer_sizes, mo, n_layers, train, train_button):
    mo.stop(
        not train_button.value,
        mo.md("Click the `Train` button to continue").callout(kind="warn")
    )

    model = MLP(n_layers.value, list(layer_sizes.value) + [1])
    print(model)
    print("number of parameters", len(model.parameters()))

    trained_model = train(
        model,
        iters=iterations.value
    )
    return (trained_model,)


@app.cell
def _(plot_decision_boundary, trained_model):
    plot_decision_boundary(trained_model)
    return


@app.cell
def _(Value, X, np, y):
    def loss(model, batch_size=None):

        # inline DataLoader :)
        if batch_size is None:
            Xb, yb = X, y
        else:
            ri = np.random.permutation(X.shape[0])[:batch_size]
            Xb, yb = X[ri], y[ri]
        inputs = [list(map(Value, xrow)) for xrow in Xb]

        # forward the model to get scores
        scores = list(map(model, inputs))

        # svm "max-margin" loss
        losses = [(1 + -yi * scorei).relu() for yi, scorei in zip(yb, scores)]
        data_loss = sum(losses) * (1.0 / len(losses))
        # L2 regularization
        alpha = 1e-4
        reg_loss = alpha * sum((p * p for p in model.parameters()))
        total_loss = data_loss + reg_loss

        # also get accuracy
        accuracy = [
            (yi > 0) == (scorei.data > 0) for yi, scorei in zip(yb, scores)
        ]
        return total_loss, sum(accuracy) / len(accuracy)

    return (loss,)


@app.cell
def _(loss, mo):
    def train(model, iters=20):
        for k in mo.status.progress_bar(range(iters)):

            # forward
            total_loss, acc = loss(model)

            # backward
            model.zero_grad()
            total_loss.backward()

            # update (sgd)
            learning_rate = 1.0 - 0.9 * k / 100
            for p in model.parameters():
                p.data -= learning_rate * p.grad

            if k % 1 == 0:
                print(f"step {k} loss {total_loss.data}, accuracy {acc*100}%")

        return model

    return (train,)


@app.cell
def _(Value, X, np, plt, y):
    def plot_decision_boundary(model):
        h = 0.25
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)
        )
        Xmesh = np.c_[xx.ravel(), yy.ravel()]
        inputs = [list(map(Value, xrow)) for xrow in Xmesh]
        scores = list(map(model, inputs))
        Z = np.array([1.0 if s.data > 0 else -1.0 for s in scores])
        Z = Z.reshape(xx.shape)

        fig = plt.figure()
        plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
        plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        return plt.gca()

    return (plot_decision_boundary,)


if __name__ == "__main__":
    app.run()
