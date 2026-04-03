# /// script
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.8",
#     "numpy==2.4.3",
#     "scikit-learn==1.8.0",
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
    from sklearn.neighbors import NearestNeighbors

    return NearestNeighbors, eigh, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Graph Signal Denoising

    A **graph signal** is a function that assigns a scalar value to each node
    of a graph. In practice, measured signals are often corrupted by noise. The
    goal of this notebook is to recover a clean signal from a noisy measurement,
    using the structure of the graph.

    The key idea is that meaningful signals tend to vary smoothly across edges:
    if two nodes are connected, their values are likely similar. Noise, on the
    other hand, has no such structure. The eigenvectors of the graph Laplacian

    $$
    L = D - A
    $$

    (where $A$ is the adjacency matrix and $D = \mathrm{diag}(A\mathbf{1})$ is
    the degree matrix) provide an orthonormal basis that separates smooth
    variation from rough. Eigenvectors with small eigenvalues vary smoothly
    across edges; eigenvectors with large eigenvalues oscillate sharply between
    neighbors.

    To denoise, we decompose the noisy signal in this basis, discard the
    high-eigenvalue components (where the noise lives), and reconstruct from
    the rest.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Graph construction

    We build a nearest-neighbor graph from points sampled on the unit circle.
    Two nodes are connected if one is among the other's $k$ nearest neighbors.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    signal_type = mo.ui.dropdown(
        options={
            "Sine wave": "sine",
            "Step function": "step",
            "Gaussian bump": "bump",
        },
        value="Sine wave",
        label="Signal",
    )
    signal_type
    return (signal_type,)


@app.cell(hide_code=True)
def _(mo):
    n_nodes = mo.ui.slider(
        start=50,
        stop=300,
        step=10,
        value=150,
        label="Number of nodes $n$",
        show_value=True,
    )
    n_neighbors = mo.ui.slider(
        start=3,
        stop=20,
        step=1,
        value=7,
        label="Number of neighbors $k$",
        show_value=True,
    )
    noise_level = mo.ui.slider(
        start=0.0,
        stop=1.0,
        step=0.05,
        value=0.3,
        label=r"Noise level $\sigma$",
        show_value=True,
    )
    n_components = mo.ui.slider(
        start=1,
        stop=30,
        step=1,
        value=5,
        label="Components to keep",
        show_value=True,
    )
    mo.md(f"""
    ### Parameters

    {n_nodes}

    {n_neighbors}

    {noise_level}

    {n_components}
    """)
    return n_components, n_neighbors, n_nodes, noise_level


@app.cell(hide_code=True)
def _(
    NearestNeighbors,
    eigh,
    n_neighbors,
    n_nodes,
    noise_level,
    np,
    signal_type,
):
    _rng = np.random.default_rng(42)
    _t = np.sort(_rng.uniform(0, 2 * np.pi, n_nodes.value))
    positions = np.column_stack([np.cos(_t), np.sin(_t)])

    neighbors = NearestNeighbors(n_neighbors=n_neighbors.value).fit(positions)
    _A = neighbors.kneighbors_graph(positions).toarray()
    adjacency_matrix = np.maximum(_A, _A.T)

    degree_matrix = np.diag(adjacency_matrix.sum(axis=1))
    L = degree_matrix - adjacency_matrix
    eigenvalues, eigenvectors = eigh(L)

    if signal_type.value == "sine":
        clean_signal = np.sin(2 * _t)
    elif signal_type.value == "step":
        clean_signal = np.sign(np.sin(_t))
    else:
        clean_signal = np.exp(-4 * (_t - np.pi) ** 2)

    noisy_signal = clean_signal + noise_level.value * _rng.standard_normal(
        len(clean_signal)
    )
    return (
        adjacency_matrix,
        clean_signal,
        eigenvectors,
        noisy_signal,
        positions,
    )


@app.cell(hide_code=True)
def _(eigenvectors, n_components, noisy_signal):
    coefficients = eigenvectors.T @ noisy_signal
    filtered_coefficients = coefficients.copy()
    filtered_coefficients[n_components.value :] = 0
    denoised_signal = eigenvectors @ filtered_coefficients
    return coefficients, denoised_signal


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Denoising by orthogonal projection

    To denoise the signal, we project it onto the subspace spanned by the
    first $k$ Laplacian eigenvectors:

    $$
    \hat{f} = \sum_{i=1}^{k} (v_i^T f)\, v_i.
    $$

    The effect is to keep the components of $f$ that vary smoothly across
    edges and discard the rest. Since noise has no reason to align with the
    smooth eigenvectors, it is largely removed.
    """)
    return


@app.cell(hide_code=True)
def _(
    adjacency_matrix,
    clean_signal,
    denoised_signal,
    n_components,
    noise_level,
    noisy_signal,
    plt,
    positions,
):
    _fig, _axes = plt.subplots(1, 3, figsize=(14, 4.5))
    _vmin = min(noisy_signal.min(), denoised_signal.min(), clean_signal.min())
    _vmax = max(noisy_signal.max(), denoised_signal.max(), clean_signal.max())
    _n = adjacency_matrix.shape[0]

    for _ax, _sig, _title in [
        (_axes[0], clean_signal, "Ground truth"),
        (
            _axes[1],
            noisy_signal,
            rf"Noisy signal ($\sigma$ = {noise_level.value:.2f})",
        ),
        (_axes[2], denoised_signal, f"Denoised ({n_components.value} components)"),
    ]:
        for _i in range(_n):
            for _j in range(_i + 1, _n):
                if adjacency_matrix[_i, _j] > 0:
                    _ax.plot(
                        [positions[_i, 0], positions[_j, 0]],
                        [positions[_i, 1], positions[_j, 1]],
                        color="gray",
                        linewidth=0.3,
                        alpha=0.2,
                    )
        _sc = _ax.scatter(
            positions[:, 0],
            positions[:, 1],
            c=_sig,
            cmap="coolwarm",
            s=20,
            vmin=_vmin,
            vmax=_vmax,
            zorder=3,
        )
        _ax.set_title(_title)
        _ax.set_aspect("equal")
        _ax.axis("off")

    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(
    clean_signal,
    denoised_signal,
    n_components,
    noise_level,
    noisy_signal,
    np,
    plt,
):
    _fig, _ax = plt.subplots(figsize=(10, 3.5))
    _idx = np.arange(len(clean_signal))
    _ax.plot(
        _idx, clean_signal, color="black", linewidth=1.5, label="Ground truth"
    )
    _ax.scatter(
        _idx,
        noisy_signal,
        color="gray",
        s=5,
        alpha=0.5,
        label=rf"Noisy ($\sigma$ = {noise_level.value:.2f})",
    )
    _ax.plot(
        _idx,
        denoised_signal,
        color="royalblue",
        linewidth=1.5,
        label=f"Denoised ({n_components.value} components)",
    )
    _ax.set_xlabel("node index")
    _ax.set_ylabel("signal value")
    _ax.legend(frameon=False)
    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Why is this the closest vector to $f$ in $\mathrm{span}\{v_1, \ldots, v_k\}$?**
    Let $S = \mathrm{span}\{v_1, \ldots, v_k\}$. For any other vector $g \in S$,

    $$
    \|f - g\|^2 = \|f - \hat{f} + \hat{f} - g\|^2 = \|f - \hat{f}\|^2 + \|\hat{f} - g\|^2
    $$

    where the cross term vanishes because $(f - \hat{f})$ is orthogonal to
    $(\hat{f} - g)$, which lies in $S$. Since $\|\hat{f} - g\|^2 \geq 0$,
    the minimum is achieved when $g = \hat{f}$.

    It remains to check that $f - \hat{f}$ is indeed orthogonal to $S$.
    Writing $\hat{f} = \alpha_1 v_1 + \cdots + \alpha_k v_k$ and requiring
    $v_j^T(f - \hat{f}) = 0$ for each $j$ gives

    $$
    v_j^T f = v_j^T \hat{f} = \alpha_j
    $$

    where the last step uses orthonormality ($v_j^T v_i = 0$ for $i \neq j$,
    and $1$ for $i = j$). So the coefficients are exactly the dot products
    $v_i^T f$, and we recover the formula above.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Spectral coefficients

    The bar chart below shows the signal's coefficients in the Laplacian
    eigenbasis. The dashed line marks the cutoff: coefficients to the left are
    kept, coefficients to the right are zeroed out. Noise tends to spread
    energy across all coefficients, while the underlying signal concentrates
    in the first few.
    """)
    return


@app.cell(hide_code=True)
def _(coefficients, n_components, np, plt):
    _fig, _ax = plt.subplots(figsize=(8, 3))
    _ax.bar(
        range(len(coefficients)),
        np.abs(coefficients),
        color="lightgray",
        edgecolor="gray",
        linewidth=0.5,
    )
    _ax.bar(
        range(n_components.value),
        np.abs(coefficients[: n_components.value]),
        color="royalblue",
        edgecolor="gray",
        linewidth=0.5,
    )
    _ax.axvline(
        n_components.value - 0.5, color="black", linestyle="--", linewidth=1
    )
    _ax.set_xlabel("eigenvector index")
    _ax.set_ylabel("$|$coefficient$|$")
    _ax.set_title("Spectral coefficients of noisy signal")
    plt.tight_layout()
    plt.gca()
    return


if __name__ == "__main__":
    app.run()
