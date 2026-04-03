# /// script
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.8",
#     "numpy==2.4.3",
#     "scikit-learn==1.8.0",
#     "scipy==1.17.1",
# ]
# requires-python = ">=3.14"
# ///

import marimo

__generated_with = "0.21.1"
app = marimo.App()

with app.setup(hide_code=True):
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from numpy.linalg import eigh
    import sklearn
    from sklearn.neighbors import NearestNeighbors


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Graph Laplacian

    This notebook shows a special way to represent graphs as matrices, capturing
    important characteristics of the graph's connectivity.

    The Laplacian matrix of an undirected graph with degree matrix $D$ and adjacency matrix $A$
    is the symmetric postive semidefinite matrix

    $$
    L = D - A.
    $$

    Here, $D$ is a diagonal matrix in $\mathbf{R}^{n \times n}$ with $D_{i,i}$ equal
    to the degree of node $i$, and

    $$
    A_{ij} \in \mathbf{R}^{n \times n} =
    \begin{cases}
    1, & \mbox{$i$ is connected to $j$} \\
    0, & \mbox{otherwise}.
    \end{cases}
    $$

    This _Laplacian matrix_ tells us
    how scalar potentials associated with nodes encoded in a vector $x$ vary across the graph, satisfying

    $$
    x^T L x = \sum_{i, j} (x_i - x_j)^2.
    $$

    **Spectral properties.** $L$ has many interesting spectral properties. For example, an
    eigenvector with small eigenvalue means that connected nodes will have similar
    values (or "potentials") in that eigenvector, a property we can exploit for
    applications such as clustering and embedding.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Spectral clustering

    In this notebook we show how the eigenvectors of a graph Laplacian can be used to cluster
    data points in settings where a naive application of $k$-means fails.
    """)
    return


@app.cell(hide_code=True)
def _():
    dataset_picker = mo.ui.dropdown(
        options={"Two Moons": "moons", "Concentric Circles": "circles"},
        value="Two Moons",
        label="Dataset",
    )

    dataset_picker
    return (dataset_picker,)


@app.cell
def _(dataset_picker):
    if dataset_picker.value == "moons":
        data, _ = sklearn.datasets.make_moons(n_samples=200, noise=0.06, random_state=0)
    else:
        data, _ = sklearn.datasets.make_circles(
            n_samples=200, noise=0.05, factor=0.4, random_state=0
        )
    return (data,)


@app.cell(hide_code=True)
def _(data):
    mo.hstack(
        [
            scatter(data, title="raw data"),
            color_by_clusters(
                data, compute_clusters(data), title="k-means on raw data"
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Neighborhood graph

    We can discover the natural clusters if we cluster the second eigenvector of a particular Laplacian matrix, interpreting the original data as a graph with two points connected if one is a nearest neighbor of the other. The number of neighbors is a parameter we can vary.

    ### Adjacency matrix

    First, we form the graph's **adjacency matrix.** Adjust the number of neighbors to see how it affects the matrix. In the visualization below, a black patch at index $(i, j)$ indicates that $A_{ij} = 1$, meaning nodes $i$ and $j$ are connected in the graph.
    """)
    return


@app.cell
def _():
    n_neighbors = mo.ui.slider(
        value=11, start=3, stop=20, step=1, label="Number of neighbors $k$", show_value=True
    )
    n_neighbors
    return (n_neighbors,)


@app.cell
def _(data, n_neighbors):
    neighbors = NearestNeighbors(n_neighbors=n_neighbors.value).fit(data)
    _A = neighbors.kneighbors_graph(data).toarray()
    adjacency_matrix = np.maximum(_A, _A.T)
    return (adjacency_matrix,)


@app.cell(hide_code=True)
def _(adjacency_matrix, data):
    plt.imshow(adjacency_matrix, cmap="gray_r")
    plt.gca()
    plt.xlabel("node index")
    plt.ylabel("node index")
    plt.title(f"Adjacency matrix A")
    _adjacency_ax = plt.gca()

    plt.figure()
    _ax = scatter(data, title="Neighbor graph")
    _n = adjacency_matrix.shape[0]
    for _i in range(_n):
        for _j in range(_i + 1, _n):
            if adjacency_matrix[_i, _j] > 0:
                _ax.plot(
                    [data[_i, 0], data[_j, 0]],
                    [data[_i, 1], data[_j, 1]],
                    color="gray",
                    linewidth=0.4,
                    alpha=0.5,
                )
    # Re-scatter on top so points aren't hidden behind edges
    _ax.scatter(data[:, 0], data[:, 1], s=10, color="royalblue", zorder=3)

    mo.hstack([_adjacency_ax, _ax])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Laplacian matrix

    Next, we form the associated **Laplacian matrix** and compute its eigenvectors.
    """)
    return


@app.function
def laplacian_matrix(adjacency_matrix):
    degree_matrix = np.diag(np.sum(adjacency_matrix, axis=1))
    L = degree_matrix - adjacency_matrix
    return L


@app.cell
def _(adjacency_matrix):
    L = laplacian_matrix(adjacency_matrix)
    return (L,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The bottom eigenvector of $L$ is the all-ones eigenvector with eigenvalue $0$. The next eigenvector, however, also has a small eigenvalue, meaning its associated eigenvector has connected nodes placed near each other. This second eigenvector is known as the **Fiedler eigenvector.**
    """)
    return


@app.cell
def _(L):
    eigenvalues, eigenvectors = eigh(L)
    fiedler_eigenvector = eigenvectors[:, 0].reshape(-1, 1)
    return (fiedler_eigenvector,)


@app.cell(hide_code=True)
def _(n_neighbors):
    mo.md(rf"""
    We can plot the entries of the Fiedler eigenvector. Notice how they sharply
    separate into two groups, suggesting that it may be useful in clustering the
    original data. The number of neighbors affects the structure of the graph
    and the distribution of values of the Fiedler eigenvalue.

    {n_neighbors}
    """)
    return


@app.cell(hide_code=True)
def _(fiedler_eigenvector):
    plt.scatter(
        range(len(fiedler_eigenvector)),
        fiedler_eigenvector,
        c=compute_clusters(fiedler_eigenvector),
        cmap="coolwarm",
        s=10,
    )
    plt.xlabel("indices")
    plt.ylabel("value")
    plt.gca()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Indeed, if we assign clusters based on a $k$-means clustering of the Fiedler eigenvector, we obtain the natural clustering on the moons dataset.
    """)
    return


@app.cell
def _(data, fiedler_eigenvector):
    color_by_clusters(data, compute_clusters(fiedler_eigenvector))
    return


@app.function
def color_by_clusters(X, cluster_labels, title=""):
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], s=2, c=cluster_labels, cmap='coolwarm')
    plt.axis("equal")
    plt.title(title)
    return plt.gca()


@app.function
def compute_clusters(X):
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=2, random_state=0)
    return kmeans.fit_predict(X)


@app.function
def scatter(X, title=""):
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], s=10)
    plt.axis("equal")
    plt.title(title)
    return plt.gca()


if __name__ == "__main__":
    app.run()
