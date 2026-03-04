# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.20.1",
#     "matplotlib>=3.10.8",
#     "pandas>=2.2.3",
#     "rdkit>=2025.9.5",
#     "scikit-learn>=1.6.1",
# ]
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    """Shared imports."""
    import os

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from rdkit import Chem, DataStructs, RDConfig, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    from sklearn.cluster import HDBSCAN
    from sklearn.manifold import TSNE
    from rdkit.Chem import Descriptors, Lipinski
    from rdkit.Chem.Draw import rdMolDraw2D

    def mol_to_svg(mol, width=200, height=150):
        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        return mo.Html(drawer.GetDrawingText())

    return (
        Chem,
        DataStructs,
        Descriptors,
        HDBSCAN,
        Lipinski,
        RDConfig,
        RDLogger,
        TSNE,
        mo,
        mol_to_svg,
        np,
        os,
        pd,
        plt,
        rdFingerprintGenerator,
    )


@app.cell(hide_code=True)
def _(mo):
    n_mols = mo.ui.number(
        start=100, stop=5000, step=100, value=2000, label="Number of molecules"
    )
    mo.md(f"""
    ## Data Preparation
    Load NCI molecules from RDKit sample data, compute Morgan fingerprints,
    build a Tanimoto distance matrix, and assemble a `pandas.DataFrame`
    with molecular properties.
    If you see stale data, re-run the table cell below.

    {n_mols}
    """)
    return (n_mols,)


@app.cell(hide_code=True)
def _(
    Chem,
    Descriptors,
    Lipinski,
    RDConfig,
    RDLogger,
    mo,
    mol_to_svg,
    n_mols,
    os,
    pd,
):
    smi_path = os.path.join(RDConfig.RDDataDir, "NCI", "first_5K.smi")
    RDLogger.DisableLog("rdApp.error")

    supplier = Chem.SmilesMolSupplier(
        smi_path,
        delimiter="\t",
        titleLine=False,
        smilesColumn=0,
        nameColumn=1,
    )
    mols = [mol for mol in supplier if mol is not None][: n_mols.value]

    props_df = pd.DataFrame(
        {
            "Mol": mols,
            "SMILES": [Chem.MolToSmiles(mol) for mol in mols],
            "AMW": [Descriptors.MolWt(mol) for mol in mols],
            "CLOGP": [Descriptors.MolLogP(mol) for mol in mols],
            "HBA": [Lipinski.NumHAcceptors(mol) for mol in mols],
            "HBD": [Lipinski.NumHDonors(mol) for mol in mols],
            "Rings": [Lipinski.RingCount(mol) for mol in mols],
            "RotBonds": [Lipinski.NumRotatableBonds(mol) for mol in mols],
        }
    )
    mo.ui.table(
        props_df,
        format_mapping={
            "Mol": mol_to_svg,
            "AMW": lambda x: f"{x:.2f}",
            "CLOGP": lambda x: f"{x:.2f}",
        },
        label=f"Rows : {len(props_df)}",
        page_size=5,
    )
    return mols, props_df


@app.cell(hide_code=True)
def _(mo):
    fp_radius = mo.ui.number(start=1, stop=6, step=1, value=2, label="Morgan radius")
    fp_size = mo.ui.number(
        start=512, stop=4096, step=512, value=2048, label="Fingerprint size"
    )
    fp_controls = mo.vstack([fp_radius, fp_size], align="start")
    mo.md(f"""
    ## Fingerprint Distance Matrix

    {fp_controls}
    """)
    return fp_radius, fp_size


@app.cell(hide_code=True)
def _(DataStructs, fp_radius, fp_size, mo, mols, np, rdFingerprintGenerator):
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=fp_radius.value, fpSize=fp_size.value
    )
    fps = [morgan_gen.GetFingerprint(mol) for mol in mols]

    dist_matrix = np.zeros((len(mols), len(mols)), dtype=np.float32)
    for i in mo.status.progress_bar(
        range(1, len(mols)), title="Building distance matrix"
    ):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dist_matrix[i, :i] = 1.0 - np.asarray(sims, dtype=np.float32)
        dist_matrix[:i, i] = dist_matrix[i, :i]
    return (dist_matrix,)


@app.cell(hide_code=True)
def _(mo):
    perplexity_input = mo.ui.number(
        start=5, stop=80, step=1, value=30, label="t-SNE perplexity"
    )
    early_exaggeration_input = mo.ui.number(
        start=4.0,
        stop=32.0,
        step=0.5,
        value=12.0,
        label="t-SNE early_exaggeration",
    )
    learning_rate_input = mo.ui.number(
        start=10.0,
        stop=2000.0,
        step=10.0,
        value=200.0,
        label="t-SNE learning_rate",
    )
    max_iter_input = mo.ui.number(
        start=250, stop=3000, step=50, value=1000, label="t-SNE max_iter"
    )
    random_state_input = mo.ui.number(
        start=0, stop=999, step=1, value=42, label="t-SNE random_state"
    )
    min_cluster_size_input = mo.ui.number(
        start=5, stop=100, step=1, value=15, label="HDBSCAN min_cluster_size"
    )
    min_samples_input = mo.ui.number(
        start=1, stop=20, step=1, value=3, label="HDBSCAN min_samples"
    )
    include_noise_checkbox = mo.ui.checkbox(
        value=True, label="Include cluster -1 (noise)"
    )

    tsne_controls = mo.vstack(
        [
            mo.md("**t-SNE**"),
            perplexity_input,
            early_exaggeration_input,
            learning_rate_input,
            max_iter_input,
            random_state_input,
        ]
    )
    hdbscan_controls = mo.vstack(
        [mo.md("**HDBSCAN**"), min_cluster_size_input, min_samples_input]
    )
    controls = mo.hstack([tsne_controls, hdbscan_controls], align="start")
    mo.vstack([mo.md("## Embedding and Clustering"), controls])
    return (
        early_exaggeration_input,
        include_noise_checkbox,
        learning_rate_input,
        max_iter_input,
        min_cluster_size_input,
        min_samples_input,
        perplexity_input,
        random_state_input,
    )


@app.cell(hide_code=True)
def _(
    HDBSCAN,
    TSNE,
    dist_matrix,
    early_exaggeration_input,
    learning_rate_input,
    max_iter_input,
    min_cluster_size_input,
    min_samples_input,
    mo,
    np,
    perplexity_input,
    random_state_input,
):
    tsne_embedding = TSNE(
        n_components=2,
        perplexity=perplexity_input.value,
        init="random",
        metric="precomputed",
        random_state=random_state_input.value,
        learning_rate=learning_rate_input.value,
        early_exaggeration=early_exaggeration_input.value,
        max_iter=max_iter_input.value,
        n_jobs=-1,
    ).fit_transform(dist_matrix)

    labels = HDBSCAN(
        min_cluster_size=min_cluster_size_input.value,
        min_samples=min_samples_input.value,
        copy=False,
    ).fit_predict(tsne_embedding)
    n_clusters = len(set(labels) - {-1})
    n_noise = int(np.sum(labels == -1))
    x_coords, y_coords = tsne_embedding[:, 0], tsne_embedding[:, 1]

    n_total = len(labels)
    noise_ratio = (100.0 * n_noise / n_total) if n_total else 0.0
    mo.stop(
        n_total == 0,
        mo.callout(mo.md("No molecules are available for embedding."), kind="warn"),
    )
    mo.stop(
        n_clusters == 0,
        mo.callout(
            mo.md(
                f"""
                **Embedding Summary**

                No HDBSCAN clusters were found with the current parameters.

                - t-SNE perplexity: `{perplexity_input.value}`
                - t-SNE early_exaggeration: `{early_exaggeration_input.value}`
                - t-SNE learning_rate: `{learning_rate_input.value}`
                - t-SNE max_iter: `{max_iter_input.value}`
                - t-SNE random_state: `{random_state_input.value}`
                - HDBSCAN min_cluster_size: `{min_cluster_size_input.value}`
                - HDBSCAN min_samples: `{min_samples_input.value}`
                - Molecules: `{n_total}`
                - Noise points (`-1`): `{n_noise}` ({noise_ratio:.1f}%)
                """
            ),
            kind="warn",
        ),
    )
    mo.callout(
        mo.md(
            f"""
            **Embedding Summary**

            - t-SNE perplexity: `{perplexity_input.value}`
            - t-SNE early_exaggeration: `{early_exaggeration_input.value}`
            - t-SNE learning_rate: `{learning_rate_input.value}`
            - t-SNE max_iter: `{max_iter_input.value}`
            - t-SNE random_state: `{random_state_input.value}`
            - HDBSCAN min_cluster_size: `{min_cluster_size_input.value}`
            - HDBSCAN min_samples: `{min_samples_input.value}`
            - Molecules: `{n_total}`
            - Clusters (excluding `-1`): `{n_clusters}`
            - Noise points (`-1`): `{n_noise}` ({noise_ratio:.1f}%)
            """
        ),
        kind="info",
    )
    return labels, n_clusters, n_noise, x_coords, y_coords


@app.cell(hide_code=True)
def _(mo):
    demo_gif = mo.center(
        mo.image(
            "https://raw.githubusercontent.com/N283T/chemspace-marimo/main/marimo-chemspace_169.gif",
            width=1200,
        )
    )
    mo.md(f"""
    ## Selection and Table View
    Select points on the scatter plot (box/lasso with Shift+drag) to inspect
    molecule structures and descriptors in a formatted table.

    {demo_gif}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    point_size_input = mo.ui.number(
        start=1, stop=100, step=1, value=20, label="Cluster point size"
    )
    noise_size_input = mo.ui.number(
        start=1, stop=100, step=1, value=15, label="Noise point size"
    )
    cluster_alpha_input = mo.ui.number(
        start=0.1, stop=1.0, step=0.1, value=0.8, label="Cluster alpha"
    )
    noise_alpha_input = mo.ui.number(
        start=0.1, stop=1.0, step=0.1, value=0.4, label="Noise alpha"
    )
    plot_controls = mo.hstack(
        [point_size_input, noise_size_input, cluster_alpha_input, noise_alpha_input],
        align="start",
    )
    mo.md(f"""
    ### Plot Style

    {plot_controls}
    """)
    return (
        cluster_alpha_input,
        noise_alpha_input,
        noise_size_input,
        point_size_input,
    )


@app.cell(hide_code=True)
def _(
    cluster_alpha_input,
    labels,
    mo,
    n_clusters,
    n_noise,
    noise_alpha_input,
    noise_size_input,
    plt,
    point_size_input,
    x_coords,
    y_coords,
):
    fig, ax = plt.subplots(figsize=(10, 6))
    noise_mask = labels == -1
    ax.scatter(
        x_coords[noise_mask],
        y_coords[noise_mask],
        c="lightgray",
        s=noise_size_input.value,
        alpha=noise_alpha_input.value,
        edgecolors="none",
        label="noise",
    )
    cluster_mask = ~noise_mask
    ax.scatter(
        x_coords[cluster_mask],
        y_coords[cluster_mask],
        c=labels[cluster_mask],
        cmap=plt.get_cmap("tab20", max(n_clusters, 1)),
        s=point_size_input.value,
        alpha=cluster_alpha_input.value,
        edgecolors="none",
    )
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(
        f"Chemical Space — {n_clusters} clusters, {n_noise} noise (n={len(labels)})"
    )
    plt.tight_layout()
    chart = mo.ui.matplotlib(ax, debounce=True)
    return (chart,)


@app.cell(hide_code=True)
def selection_display(
    chart,
    include_noise_checkbox,
    labels,
    mo,
    mol_to_svg,
    np,
    props_df,
    x_coords,
    y_coords,
):
    selected_indices = np.where(chart.value.get_mask(x_coords, y_coords))[0]
    mo.stop(
        len(selected_indices) == 0,
        mo.vstack(
            [
                chart,
                include_noise_checkbox,
                mo.callout(
                    mo.md(
                        "Select molecules on the scatter plot using "
                        "box or lasso (Shift+drag)."
                    ),
                    kind="warn",
                ),
            ],
            align="center",
        ),
    )

    base_table = props_df.copy()
    base_table.insert(0, "Cluster", labels.astype(int))
    table_data = base_table.iloc[selected_indices].copy()
    if not include_noise_checkbox.value:
        table_data = table_data[table_data["Cluster"] != -1]

    mo.stop(
        len(table_data) == 0,
        mo.vstack(
            [
                chart,
                include_noise_checkbox,
                mo.callout(
                    mo.md("No rows to show after filtering `cluster = -1` (noise)."),
                    kind="warn",
                ),
            ],
            align="center",
        ),
    )

    table_view = mo.ui.table(
        table_data,
        format_mapping={
            "Mol": mol_to_svg,
            "AMW": lambda x: f"{x:.2f}",
            "CLOGP": lambda x: f"{x:.2f}",
        },
        label=f"Rows shown: {len(table_data)}",
    )

    mo.vstack(
        [
            chart,
            include_noise_checkbox,
        ],
        align="center",
    )
    return (table_view,)


@app.cell
def _(table_view):
    table_view
    return


if __name__ == "__main__":
    app.run()
