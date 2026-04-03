# /// script
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.8",
#     "numpy==2.4.3",
# ]
# requires-python = ">=3.14"
# ///

import marimo

__generated_with = "0.22.0"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from numpy.linalg import eigh

    return eigh, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Spectral Graph Drawing

    This notebook shows how eigenvectors of the graph Laplacian can be used
    to draw graphs, positioning nodes in two dimensions so that the layout
    reflects the graph's connectivity.

    Given an undirected graph with adjacency matrix $A \in \mathbf{R}^{n 	imes n}$
    and degree matrix $D = \mathrm{diag}(A \mathbf{1})$, the **Laplacian matrix** is

    $$
    L = D - A.
    $$

    The eigenvectors of $L$ minimize the sum of squared distances between
    connected nodes. Specifically, if we assign each node $i$ a position
    $x_i \in \mathbf{R}$, the Laplacian satisfies

    $$
    x^T L x = \sum_{(i,j) \in E} (x_i - x_j)^2,
    $$

    so an eigenvector with a small eigenvalue places connected nodes close
    together.

    To draw a graph in two dimensions, we use the second and third smallest
    eigenvectors as $(x, y)$ coordinates for each node. The first eigenvector
    (constant, eigenvalue $0$) is discarded because it would place every node
    at the same point.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    graph_type = mo.ui.dropdown(
        options={
            "Path": "path",
            "Cycle": "cycle",
            "Grid": "grid",
            "Petersen": "petersen",
            "Random (Erdős–Rényi)": "random",
        },
        value="Path",
        label="Graph",
    )
    graph_type
    return (graph_type,)


@app.cell(hide_code=True)
def _(graph_type, mo):
    n_nodes = mo.ui.slider(
        start=10,
        stop=100,
        step=1,
        value=30,
        label="Number of nodes $n$",
        show_value=True,
    )
    edge_prob = mo.ui.slider(
        start=0.02,
        stop=0.4,
        step=0.02,
        value=0.1,
        label="Edge probability $p$",
        show_value=True,
    )
    mo.md(f"""
    ### Parameters

    {n_nodes}

    {edge_prob if graph_type.value == "random" else ""}
    """)
    return edge_prob, n_nodes


@app.cell(hide_code=True)
def _(edge_prob, graph_type, n_nodes, np):
    def build_adjacency(kind, n, p):
        if kind == "path":
            A = np.zeros((n, n))
            for i in range(n - 1):
                A[i, i + 1] = A[i + 1, i] = 1
            return A
        elif kind == "cycle":
            A = np.zeros((n, n))
            for i in range(n):
                A[i, (i + 1) % n] = A[(i + 1) % n, i] = 1
            return A
        elif kind == "grid":
            side = int(np.ceil(np.sqrt(n)))
            actual = side * side
            A = np.zeros((actual, actual))
            for i in range(actual):
                r, c = divmod(i, side)
                if c + 1 < side:
                    A[i, i + 1] = A[i + 1, i] = 1
                if r + 1 < side:
                    A[i, i + side] = A[i + side, i] = 1
            return A
        elif kind == "petersen":
            A = np.zeros((10, 10))
            for i in range(5):
                A[i, (i + 1) % 5] = A[(i + 1) % 5, i] = 1
                A[i, i + 5] = A[i + 5, i] = 1
            for i in range(5):
                A[5 + i, 5 + (i + 2) % 5] = A[5 + (i + 2) % 5, 5 + i] = 1
            return A
        else:
            rng = np.random.default_rng(42)
            A = (rng.random((n, n)) < p).astype(float)
            A = np.triu(A, 1)
            A = A + A.T
            return A


    adjacency_matrix = build_adjacency(
        graph_type.value, n_nodes.value, edge_prob.value
    )
    return (adjacency_matrix,)


@app.cell(hide_code=True)
def _(adjacency_matrix, eigh, np):
    degree_matrix = np.diag(np.sum(adjacency_matrix, axis=1))
    L = degree_matrix - adjacency_matrix
    eigenvalues, eigenvectors = eigh(L)
    return eigenvalues, eigenvectors


@app.cell(hide_code=True)
def _(adjacency_matrix, mo):
    _max_idx = min(adjacency_matrix.shape[0], 10)

    evec_x = mo.ui.slider(
        start=1,
        stop=_max_idx - 1,
        value=1,
        label="Eigenvector for $x$-axis",
        show_value=True,
    )
    evec_y = mo.ui.slider(
        start=1,
        stop=_max_idx - 1,
        value=2,
        label="Eigenvector for $y$-axis",
        show_value=True,
    )
    color_by = mo.ui.radio(
        options={"x-axis eigenvector": "x", "y-axis eigenvector": "y"},
        value="x-axis eigenvector",
        label="Color nodes by",
    )

    mo.md(f"""
    ### Eigenvector pair

    Choose which eigenvectors to use as coordinates. The default (second and
    third) gives the classical spectral drawing. Eigenvectors with larger
    eigenvalues vary more rapidly across the graph, so using them as
    coordinates emphasizes local differences between nearby nodes rather than
    global shape.

    {evec_x}

    {evec_y}

    Nodes are colored by the eigenvector you've selected below. For low
    eigenvectors the color varies smoothly across the graph; for higher ones
    it oscillates between neighbors.

    {color_by}
    """)
    return color_by, evec_x, evec_y


@app.cell(hide_code=True)
def _(adjacency_matrix, color_by, eigenvectors, evec_x, evec_y, np, plt):
    _x = eigenvectors[:, evec_x.value]
    _y = eigenvectors[:, evec_y.value]
    _n = adjacency_matrix.shape[0]
    _color = _x if color_by.value == "x" else _y

    _fig, _axes = plt.subplots(1, 2, figsize=(10, 5))

    _rng = np.random.default_rng(0)
    _rx, _ry = _rng.normal(size=_n), _rng.normal(size=_n)

    for _ax, (_px, _py), _title in [
        (_axes[0], (_rx, _ry), "Random layout"),
        (
            _axes[1],
            (_x, _y),
            f"Spectral layout (eigenvectors {evec_x.value}, {evec_y.value})",
        ),
    ]:
        for _i in range(_n):
            for _j in range(_i + 1, _n):
                if adjacency_matrix[_i, _j] > 0:
                    _ax.plot(
                        [_px[_i], _px[_j]],
                        [_py[_i], _py[_j]],
                        color="gray",
                        linewidth=0.5,
                        alpha=0.3,
                    )
        _ax.scatter(_px, _py, s=30, c=_color, cmap="coolwarm", zorder=3)
        _ax.set_title(_title)
        _ax.set_aspect("equal")
        _ax.axis("off")

    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Eigenvalue spectrum

    Each eigenvalue of $L$ measures how much the corresponding eigenvector
    varies across edges. An eigenvector with a small eigenvalue assigns similar
    values to connected nodes; one with a large eigenvalue changes sharply
    between neighbors. The plot below shows the spectrum, with the selected
    eigenvectors highlighted.
    """)
    return


@app.cell(hide_code=True)
def _(eigenvalues, evec_x, evec_y, plt):
    _fig, _ax = plt.subplots(figsize=(6, 3))
    _ax.bar(
        range(len(eigenvalues)),
        eigenvalues,
        color="lightgray",
        edgecolor="gray",
        linewidth=0.5,
    )
    for _idx in [evec_x.value, evec_y.value]:
        _ax.bar(
            _idx,
            eigenvalues[_idx],
            color="royalblue",
            edgecolor="gray",
            linewidth=0.5,
        )
    _ax.set_xlabel("index")
    _ax.set_ylabel(r"eigenvalue $\lambda_i$")
    _ax.set_title("Laplacian spectrum")
    plt.tight_layout()
    plt.gca()
    return


if __name__ == "__main__":
    app.run()
