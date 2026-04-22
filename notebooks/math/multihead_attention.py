# /// script
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
#     "anywidget",
#     "traitlets",
# ]
# ///

import marimo

__generated_with = "0.23.2"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import anywidget
    import traitlets

    return anywidget, mo, np, plt, traitlets


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Multi-Head Attention

    Multi-head attention runs $h$ independent attention operators in parallel, each
    with its own learned weight matrices, and concatenates their outputs. It is the
    standard attention block used in Transformers.

    This notebook builds the operator from the single-head definition.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Single-head self-attention

    Given input $X \in \mathbf{R}^{n \times d}$ (a sequence of $n$ tokens), single-head self-attention produces

    $$
    \operatorname{Attn}(Q, K, V) \;=\; \operatorname{softmax}\!\left(\frac{Q K^\top}{\sqrt{d}}\right) V,
    $$

    where $Q = X W_Q$, $K = X W_K$, $V = X W_V$, and $W_Q, W_K, W_V \in \mathbf{R}^{d \times d}$ are learned weight matrices. The softmax is taken row-wise, so each row of the output is a weighted average of rows of the value matrix $V$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Block structure

    Fix a number of heads $h$ dividing $d$. The three linear maps $W_Q, W_K, W_V \in \mathbf{R}^{d \times d}$ are written as $h$ column blocks, one per head:

    $$
    \begin{align*}
    W_Q & = \begin{bmatrix} W_Q^{(1)} & W_Q^{(2)} & \cdots & W_Q^{(h)} \end{bmatrix}, \\
    W_K & = \begin{bmatrix} W_K^{(1)} & W_K^{(2)} & \cdots & W_K^{(h)} \end{bmatrix}, \\
    W_V & = \begin{bmatrix} W_V^{(1)} & W_V^{(2)} & \cdots & W_V^{(h)} \end{bmatrix},
    \end{align*}
    $$

    with each block $W_Q^{(i)}, W_K^{(i)}, W_V^{(i)} \in \mathbf{R}^{d \times d/h}$. Multiplying by $X$ partitions $Q, K, V$ the same way:

    $$
    Q \;=\; X W_Q \;=\; \begin{bmatrix} X W_Q^{(1)} & \cdots & X W_Q^{(h)} \end{bmatrix}
         \;=\; \begin{bmatrix} Q_1 & Q_2 & \cdots & Q_h \end{bmatrix},
    $$

    and likewise

    $$
    K \;=\; \begin{bmatrix} K_1 & K_2 & \cdots & K_h \end{bmatrix}, \qquad
    V \;=\; \begin{bmatrix} V_1 & V_2 & \cdots & V_h \end{bmatrix},
    $$

    where $Q_i, K_i, V_i \in \mathbf{R}^{n \times d/h}$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The multi-head operator

    **Attention matrices.** Each head applies single-head self-attention to its own block of queries, keys, and values. Define the per-head **attention matrix**

    $$
    A^{(i)} \;=\; \operatorname{softmax}\!\left(\frac{Q_i K_i^\top}{\sqrt{d/h}}\right) \;\in\; \mathbf{R}^{n \times n},
    $$

    a row-stochastic matrix whose entry $A^{(i)}_{rj}$ is the weight head $i$ places
    on row $j$ of $X$ when forming the output at row $r$.

    **Heads.** The "head" output is then

    $$
    \operatorname{head}_i \;=\; A^{(i)} V_i \;=\; A^{(i)}\, X W_V^{(i)} \;\in\; \mathbf{R}^{n \times d/h}.
    $$

    **Multi-head attention.** The $h$ head outputs are concatenated along the feature dimension and passed through a learned output linear layer $W_O \in \mathbf{R}^{d \times d}$:

    $$
    \operatorname{MultiHead}(X) \;=\; \begin{bmatrix} \operatorname{head}_1 & \operatorname{head}_2 & \cdots & \operatorname{head}_h \end{bmatrix} W_O \;\in\; \mathbf{R}^{n \times d}.
    $$

    The block decomposition makes per-head independence visible: each head has its own attention matrix $A^{(i)}$, its own value readout $W_V^{(i)}$, and its own softmax. $W_O$ is the only place where information flows between heads.

    **A family of bilinear forms.** In other words, multi-head attention is a family of $h$ bilinear forms

    $$
    B_i(x, y) \;=\; x^\top\!\bigl(W_Q^{(i)} W_K^{(i)\top}\bigr) y, \qquad i = 1, \dots, h,
    $$

    on $\mathbf{R}^d \times \mathbf{R}^d$, parameterized by pairs $(W_Q^{(i)}, W_K^{(i)}) \in \mathbf{R}^{d \times d/h} \times \mathbf{R}^{d \times d/h}$. Evaluated pairwise on the rows of $X$ and composed with a row-wise softmax, each $B_i$ produces a row-stochastic matrix $A^{(i)} \in \mathbf{R}^{n \times n}$ that mixes the rows of the $i$-th column block of the value matrix $V = X W_V$.

    Concretely, for the attention logit between rows $r$ and $j$ of $X$ under head $i$, we
    evaluate $B_i$ at these rows to obtain

    $$
    B_i(x_r, x_j) \;=\; x_r^\top\!\bigl(W_Q^{(i)} W_K^{(i)\top}\bigr) x_j \;=\; q^{(i)\top}_r\, k^{(i)}_j,
    $$

    where $x_r, x_j \in \mathbf{R}^d$ are rows $r$ and $j$ of $X$ written as columns, and $q^{(i)}_r = W_Q^{(i)\top} x_r$, $k^{(i)}_j = W_K^{(i)\top} x_j$ are the per-head query and key. The full $n \times n$ matrix of these logits is $X\,W_Q^{(i)} W_K^{(i)\top} X^\top$, which is scaled by $1/\sqrt{d/h}$ and passed through a row-wise softmax to yield $A^{(i)}$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### A worked example

    Take a sequence of $n = 12$ tokens in $\mathbf{R}^8$ (three clusters of four), use $h = 4$ heads with $d/h = 2$ per head, and initialize the weight matrices randomly. Each head then computes its own attention matrix from its own $Q_i K_i^\top / \sqrt{d/h}$.
    """)
    return


@app.cell
def _(h_slider, np):
    n = 12
    d = 8
    h = int(h_slider.value)

    _rng = np.random.default_rng(0)

    _centers = _rng.normal(size=(3, d))
    _centers /= np.linalg.norm(_centers, axis=1, keepdims=True)
    X = np.vstack([c + 0.2 * _rng.normal(size=(4, d)) for c in _centers])

    W_Q = _rng.normal(size=(d, d)) / np.sqrt(d)
    W_K = _rng.normal(size=(d, d)) / np.sqrt(d)
    W_V = _rng.normal(size=(d, d)) / np.sqrt(d)
    W_O = _rng.normal(size=(d, d)) / np.sqrt(d)
    return W_K, W_O, W_Q, W_V, X, h


@app.cell
def _(np):
    def multi_head_attention(X, W_Q, W_K, W_V, W_O, h):
        n, d = X.shape

        Q = X @ W_Q
        K = X @ W_K
        V = X @ W_V

        heads = []
        attentions = []
        for i in range(h):
            Q_i = Q[:, i * (d // h) : (i + 1) * (d // h)]
            K_i = K[:, i * (d // h) : (i + 1) * (d // h)]
            V_i = V[:, i * (d // h) : (i + 1) * (d // h)]

            logits = Q_i @ K_i.T / np.sqrt((d // h))
            logits -= logits.max(axis=1, keepdims=True)
            A_i = np.exp(logits)
            A_i /= A_i.sum(axis=1, keepdims=True)

            heads.append(A_i @ V_i)
            attentions.append(A_i)

        concat = np.concatenate(heads, axis=1)
        output = concat @ W_O
        return output, np.stack(attentions)

    return (multi_head_attention,)


@app.cell(hide_code=True)
def _(mo):
    h_slider = mo.ui.dropdown(
        options=["1", "2", "4", "8"], value="4", label=r"number of heads $h$",
    )
    h_slider
    return (h_slider,)


@app.cell
def _(W_K, W_O, W_Q, W_V, X, h, multi_head_attention):
    output, attentions = multi_head_attention(X, W_Q, W_K, W_V, W_O, h)
    return (attentions,)


@app.cell(hide_code=True)
def _(attentions, h, plt):
    _cols = min(4, h)
    _rows = (h + _cols - 1) // _cols

    _fig, _axes_grid = plt.subplots(
        _rows, _cols,
        figsize=(3.4 * _cols + 1, 3.6 * _rows),
        squeeze=False,
        constrained_layout=True,
    )
    _axes = _axes_grid.flatten()

    for _i in range(h):
        _ax = _axes[_i]
        _im = _ax.imshow(attentions[_i], cmap="Purples", vmin=0, vmax=attentions.max())
        _ax.set_title(rf"$A^{{({_i + 1})}}$")
        _ax.set_xlabel(r"key row $j$")
        if _i % _cols == 0:
            _ax.set_ylabel(r"query row $r$")

    for _k in range(h, _rows * _cols):
        _axes[_k].axis("off")

    _fig.suptitle("Per-head attention matrices")
    _fig.colorbar(_im, ax=_axes_grid, fraction=0.04, pad=0.04, label=r"$A^{(i)}_{rj}$")
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The attention cube in $\mathbf{R}^{h \times n \times n}$ is rendered below as a stack of $h$ translucent heatmap planes, one per head.
    """)
    return


@app.cell(hide_code=True)
def _(anywidget, traitlets):
    class AttentionCube(anywidget.AnyWidget):
        _esm = r"""
        import * as THREE from "https://esm.sh/three@0.160.0";
        import { OrbitControls } from "https://esm.sh/three@0.160.0/examples/jsm/controls/OrbitControls.js";

        function render({ model, el }) {
            const A = model.get("attention_tensor");
            const h = A.length;
            const n = A[0].length;

            const wrap = document.createElement("div");
            wrap.className = "atten-cube-wrap";
            el.appendChild(wrap);

            const btnRow = document.createElement("div");
            btnRow.className = "atten-btn-row";
            wrap.appendChild(btnRow);

            const btns = [];
            const mkBtn = (label, idx) => {
                const b = document.createElement("button");
                b.textContent = label;
                b.className = "atten-btn";
                b.addEventListener("click", () => {
                    model.set("emphasized_head", idx);
                    model.save_changes();
                });
                btnRow.appendChild(b);
                return b;
            };
            btns.push(mkBtn("all", -1));
            for (let i = 0; i < h; i++) btns.push(mkBtn(`head ${i + 1}`, i));

            const sceneWrap = document.createElement("div");
            sceneWrap.className = "atten-scene";
            wrap.appendChild(sceneWrap);

            const width = 640, height = 480;
            const scene = new THREE.Scene();
            scene.background = new THREE.Color(0xffffff);

            const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 200);
            camera.position.set(14, 14, 18);

            const renderer = new THREE.WebGLRenderer({ antialias: true });
            renderer.setSize(width, height);
            sceneWrap.appendChild(renderer.domElement);

            const controls = new OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.target.set(0, 0, 0);

            scene.add(new THREE.AmbientLight(0xffffff, 0.9));

            const colormap = (v) => {
                const t = Math.max(0, Math.min(1, v));
                return [
                    Math.floor(255 - t * 200),
                    Math.floor(255 - t * 250),
                    Math.floor(255 - t * 155),
                ];
            };

            // Global max across all heads.
            let amax = 0;
            for (let i0 = 0; i0 < h; i0++)
                for (let r0 = 0; r0 < n; r0++)
                    for (let j0 = 0; j0 < n; j0++)
                        if (A[i0][r0][j0] > amax) amax = A[i0][r0][j0];

            const planes = [];
            const spacing = 2.2;
            const planeSize = 10;
            const yOffset = -((h - 1) * spacing) / 2;

            for (let i = 0; i < h; i++) {
                const data = new Uint8Array(n * n * 4);
                for (let r = 0; r < n; r++) {
                    for (let j = 0; j < n; j++) {
                        const v = A[i][r][j] / amax;
                        const [R, G, B] = colormap(v);
                        const idx = (r * n + j) * 4;
                        data[idx] = R; data[idx + 1] = G; data[idx + 2] = B; data[idx + 3] = 230;
                    }
                }
                const tex = new THREE.DataTexture(data, n, n, THREE.RGBAFormat);
                tex.needsUpdate = true;
                tex.magFilter = THREE.NearestFilter;
                tex.minFilter = THREE.NearestFilter;

                const geom = new THREE.PlaneGeometry(planeSize, planeSize);
                const mat = new THREE.MeshBasicMaterial({
                    map: tex, transparent: true, opacity: 0.92, side: THREE.DoubleSide,
                });
                const mesh = new THREE.Mesh(geom, mat);
                mesh.position.y = yOffset + i * spacing;
                mesh.rotation.x = -Math.PI / 2;
                scene.add(mesh);

                const edgeGeom = new THREE.EdgesGeometry(geom);
                const edgeMat = new THREE.LineBasicMaterial({
                    color: 0x444466, transparent: true, opacity: 1.0,
                });
                const edge = new THREE.LineSegments(edgeGeom, edgeMat);
                edge.rotation.x = -Math.PI / 2;
                edge.position.y = mesh.position.y + 0.01;
                scene.add(edge);

                const labelCanvas = document.createElement("canvas");
                labelCanvas.width = 128; labelCanvas.height = 32;
                const lctx = labelCanvas.getContext("2d");
                lctx.fillStyle = "rgba(20,20,30,0.95)";
                lctx.font = "bold 18px sans-serif";
                lctx.fillText(`A ${i + 1}`, 6, 22);
                const spriteMat = new THREE.SpriteMaterial({
                    map: new THREE.CanvasTexture(labelCanvas),
                    depthTest: false, transparent: true,
                });
                const sprite = new THREE.Sprite(spriteMat);
                sprite.scale.set(3, 0.8, 1);
                sprite.position.set(planeSize / 2 + 2, mesh.position.y, -planeSize / 2);
                scene.add(sprite);

                planes.push({ mesh, edge, sprite });
            }

            const cbar = document.createElement("div");
            cbar.className = "atten-colorbar";
            cbar.innerHTML = `
                <span class="atten-cbar-lbl">0</span>
                <div class="atten-cbar-gradient"></div>
                <span class="atten-cbar-lbl">${amax.toFixed(2)}</span>
                <span class="atten-cbar-title">attention weight</span>
            `;
            wrap.appendChild(cbar);

            const updateEmphasis = () => {
                const emph = model.get("emphasized_head");
                planes.forEach((p, i) => {
                    const active = emph === -1 || emph === i;
                    p.mesh.material.opacity = active ? 0.92 : 0.1;
                    p.edge.material.opacity = active ? 1.0 : 0.2;
                    p.sprite.material.opacity = active ? 1.0 : 0.25;
                });
                btns.forEach((b, bi) => {
                    const bidx = bi === 0 ? -1 : bi - 1;
                    b.classList.toggle("active", bidx === emph);
                });
            };
            model.on("change:emphasized_head", updateEmphasis);
            updateEmphasis();

            function animate() {
                requestAnimationFrame(animate);
                controls.update();
                renderer.render(scene, camera);
            }
            animate();
        }

        export default { render };
        """

        _css = r"""
        .atten-cube-wrap {
            display: flex; flex-direction: column; gap: 8px; padding: 4px;
        }
        .atten-btn-row { display: flex; gap: 6px; flex-wrap: wrap; }
        .atten-btn {
            padding: 6px 12px; border: 1px solid #bbb; border-radius: 4px;
            background: #ffffff; color: #222; cursor: pointer;
            font-size: 13px; font-family: inherit;
        }
        .atten-btn:hover { background: #f0f0f2; }
        .atten-btn.active {
            background: #553377; color: white; border-color: #553377;
        }
        .atten-scene {
            border-radius: 6px; overflow: hidden; background: #ffffff;
            border: 1px solid #ddd;
        }
        .atten-colorbar {
            display: flex; align-items: center; gap: 8px;
            font-size: 12px; font-family: inherit; color: #333;
            padding: 2px 4px;
        }
        .atten-cbar-gradient {
            width: 240px; height: 12px;
            background: linear-gradient(to right, rgb(255, 255, 255), rgb(55, 5, 100));
            border: 1px solid #999;
        }
        .atten-cbar-lbl { min-width: 1em; }
        .atten-cbar-title { margin-left: 8px; color: #555; }
        @media (prefers-color-scheme: dark) {
            .atten-btn { background: #2a2a2e; color: #eee; border-color: #555; }
            .atten-btn:hover { background: #3a3a3e; }
        }
        """

        attention_tensor = traitlets.List().tag(sync=True)
        emphasized_head = traitlets.Int(-1).tag(sync=True)

    return (AttentionCube,)


@app.cell(hide_code=True)
def _(AttentionCube, attentions, mo):
    attention_cube = mo.ui.anywidget(
        AttentionCube(attention_tensor=attentions.tolist(), emphasized_head=-1)
    )
    attention_cube
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Multi-head attention learns feature-dependent mixings

    **Single-head.** Single-head self-attention uses the same mixing coefficients for each column of $X$:

    $$
    \operatorname{Attn}(Q, K, V) \;=\; AXW_V \;=\; \begin{bmatrix} A X W_V^{(1)} & A X W_V^{(2)} & \cdots & A X W_V^{(h)} \end{bmatrix},
    $$

    where $A = \operatorname{softmax}(Q K^\top / \sqrt{d}) \in \mathbf{R}^{n \times
    n}$ and we have partitioned $W_V$ into the same $h$ column blocks used by
    multi-head. The same mixing matrix $A$ is applied to every column block.
    Equivalently, for each query $i$ the same weights $a_i^\top$ (row $i$ of $A$)
    mix the rows of $X W_V$ across all $d$ output features.

    **Multi-head.** Multi-head attention replaces this single $A$ with $h$ different mixing matrices, one per block:

    $$
    \begin{bmatrix} \operatorname{head}_1 & \operatorname{head}_2 & \cdots & \operatorname{head}_h \end{bmatrix}
    \;=\;
    \begin{bmatrix} A^{(1)} X W_V^{(1)} & A^{(2)} X W_V^{(2)} & \cdots & A^{(h)} X W_V^{(h)} \end{bmatrix},
    $$

    where each $A^{(i)} = \operatorname{softmax}(Q_i K_i^\top / \sqrt{d/h})$ is
    computed from its own query/key block. Different column blocks of the output now
    use different mixing weights. Head $i$ and head $j$ can produce different attention distributions
    over tokens, and each writes into a different slice of the feature space. $W_O$ afterwards mixes across blocks.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Inherited properties

    Permutation equivariance carries over from single-head attention. Each head is independently equivariant in $X$, concatenation preserves the permutation, and $W_O$ acts only on the feature axis. So

    $$
    \operatorname{MultiHead}(PX) \;=\; P \cdot \operatorname{MultiHead}(X)
    $$

    for every permutation matrix $P$. Positional information must still be injected into $X$ whenever sequence order matters.
    """)
    return


if __name__ == "__main__":
    app.run()
