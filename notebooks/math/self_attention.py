# /// script
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
# ]
# ///

import marimo

__generated_with = "0.23.1"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    return mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.outline()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Self-Attention

    Self-attention is the core operator of the Transformer architecture. It takes a
    sequence of vectors and rewrites each one as a **data-dependent linear
    combination** of the others.

    This notebook builds up the operator from scratch and offers two complementary
    interpretations:

    - **Soft lookup**: a differentiable relaxation of a key-value lookup.
    - **Row-stochastic mixing**: a learned graph over tokens, with entries coming from an exponentiated inner-product kernel.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Tokens

    A **token** is a single element of the input sequence, represented as a
    vector in $\mathbf{R}^d$ (for example, a word embedding in a language model,
    or a patch embedding in a vision model).

    The input to self-attention is a sequence of $n$ tokens stacked row-wise into $X
    \in \mathbf{R}^{n \times d}$; the $i$-th row $x_i^\top$ represents the $i$-th
    token.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Queries, keys, and values

    The self-attention operator uses three learned linear maps to produce a
    **query**, **key**, and **value** matrix:

    $$
    Q = X W_Q, \qquad K = X W_K, \qquad V = X W_V,
    $$

    with weight matrices $W_Q, W_K, W_V \in \mathbf{R}^{d \times d}$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Softmax

    The **softmax** operator turns a vector of real-valued logits $z \in \mathbf{R}^n$ into a probability distribution over $n$ positions:

    $$
    \operatorname{softmax}(z)_j \;=\; \frac{\exp(z_j)}{\sum_{k=1}^n \exp(z_k)}, \qquad j = 1, \dots, n.
    $$

    By construction $\operatorname{softmax}(z)_j > 0$ and $\sum_j \operatorname{softmax}(z)_j = 1$. It is a differentiable, order-preserving relaxation of $\arg\max$: the position with the largest $z_j$ receives the largest probability, and as the gap grows that probability approaches $1$.

    Move the slider below to see how changing a single logit reshapes the output distribution.
    """)
    return


@app.cell
def _(np):
    def softmax(z):
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    return (softmax,)


@app.cell(hide_code=True)
def _(mo):
    z3 = mo.ui.slider(
        start=-3.0, stop=5.0, step=0.1, value=2.0,
        label=r"$z_3$", show_value=True,
    )
    z3
    return (z3,)


@app.cell(hide_code=True)
def _(np, plt, softmax, z3):
    _z = np.array([0.5, 1.0, z3.value, 1.0, 0.5])
    _p = softmax(_z)

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(9, 3))
    _xs = range(1, len(_z) + 1)

    _ax1.bar(_xs, _z, color="steelblue")
    _ax1.set_title(r"logits $z$")
    _ax1.set_xlabel("$j$")
    _ax1.set_xticks(list(_xs))
    _ax1.axhline(0, color="black", linewidth=0.5)

    _ax2.bar(_xs, _p, color="darkorange")
    _ax2.set_title(r"$\operatorname{softmax}(z)$")
    _ax2.set_xlabel("$j$")
    _ax2.set_xticks(list(_xs))
    _ax2.set_ylim(0, 1.0)

    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The operator

    Self-attention is defined by

    $$
    \operatorname{Attn}(Q, K, V)
      \;=\; \underbrace{\operatorname{softmax}\!\left(\frac{Q K^\top}{\sqrt{d}}\right)}_{A \,\in\, \mathbf{R}^{n \times n}} V,
    $$

    where the softmax is taken row-wise and $A \in \mathbf{R}^{n \times n}$ is called the attention matrix. Note that it is a function of $X$, with three learned parameters $W_Q, W_K$ and $W_V$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Learned parameters

    The three separate learned matrices $W_Q$, $W_K$, and $W_V$ give attention three degrees of freedom.

    **A task-specific similarity.** The attention logit factors as

    $$
    q_i^\top k_j \;=\; x_i^\top \,(W_Q W_K^\top)\, x_j,
    $$

    a bilinear form on $\mathbf{R}^d$ with the learned matrix $W_Q W_K^\top$ in the middle. Setting $W_Q = W_K = I$ would commit to the raw inner product $x_i^\top x_j$; learning $W_Q, W_K$ lets the model choose which directions of the input space count toward similarity and which do not.

    **Selection independent from readout.** Setting $W_V = I$ would make the value $v_j$ equal to the input $x_j,$ so the vector used to decide which tokens to attend to would also be the vector transmitted from them. A distinct $W_V$ decouples the two: $(W_Q, W_K)$ choose which tokens contribute to each output; $W_V$ chooses what is extracted from those tokens once chosen.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Permutation equivariance

    Self-attention is **permutation-equivariant** in its input sequence. Let $P \in \{0,1\}^{n \times n}$ be a permutation matrix that reorders the $n$ rows of $X$. Then

    $$
    \operatorname{Attn}(PX W_Q,\; PX W_K,\; PX W_V) \;=\; P \cdot \operatorname{Attn}(Q, K, V).
    $$

    The operator uses only inner products between pairs of tokens and never references a position index, so its output depends on the **set** of tokens, not their order.

    Sketch: the attention matrix for the permuted input is
    $\operatorname{softmax}(PQK^\top P^\top / \sqrt{d}) = P A P^\top$ — softmax commutes with consistent row-and-column permutations. Multiplying by $PV$ then gives $P A P^\top P V = P A V$.

    This is why Transformers applied to ordered data like text add positional encodings to $X$: the equivariance becomes a liability when order matters, and the encoding breaks it deliberately.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    perm_seed = mo.ui.slider(
        start=0, stop=20, step=1, value=0,
        label=r"Permutation seed", show_value=True,
    )
    perm_seed
    return (perm_seed,)


@app.cell(hide_code=True)
def _(K, Q, V, attention, mo, np, perm_seed):
    _rng = np.random.default_rng(perm_seed.value)
    _perm = _rng.permutation(V.shape[0])

    # Permute inputs first, then run attention
    _Xp = V[_perm]
    _Ap = attention(_Xp, _Xp, _Xp, tau=1.0)
    _out_perm_first = _Ap @ _Xp

    # Run attention first, then permute output rows
    _out_attn_first = (attention(Q, K, V, tau=1.0) @ V)[_perm]

    _diff = np.max(np.abs(_out_perm_first - _out_attn_first))

    mo.md(rf"""
    With permutation $P$ corresponding to index order `{_perm.tolist()}`:

    $$
    \max_{{i,j}} \bigl|\, \operatorname{{Attn}}(PX)_{{ij}} \;-\; \bigl(P \cdot \operatorname{{Attn}}(X)\bigr)_{{ij}} \,\bigr| \;=\; {_diff:.2e}.
    $$

    The gap is at the level of floating-point rounding; the two outputs are equal as mathematical quantities.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interpretation I: soft lookup

    The the $i$-th row of the self-attention output is

    $$
    \operatorname{Attn}(Q, K, V)_i \;=\; \operatorname{softmax}\!\left(\frac{q_i^\top K^\top}{\sqrt{d}}\right) V.
    $$

    The softmax argument is the row of similarities between $q_i$ and every key:

    $$
    \frac{q_i^\top K^\top}{\sqrt{d}} \;=\; \frac{1}{\sqrt{d}}\bigl[\,q_i^\top k_1,\; q_i^\top k_2,\; \dots,\; q_i^\top k_n\,\bigr].
    $$

    The softmax turns this row into a **probability distribution over the $n$
    keys**: nonnegative entries summing to $1$, larger where the similarity is
    higher. Multiplying by $V$ then mixes the value rows $v_1^\top, \dots, v_n^\top$
    according to those weights.

    When one similarity dominates the rest, the softmax concentrates its mass on the single index

    $$
    j^{\star} \;=\; \arg\max_j\ q_i^\top k_j,
    $$

    and the output collapses to

    $$
    \operatorname{Attn}(Q, K, V)_i \;\approx\; v_{j^\star}^\top,
    $$

    the value row paired with the key most similar to $q_i$. In this sense, the
    self-attention operator can be interpreted as a differentiable lookup. When
    several similarities are close, the softmax spreads its mass and the
    output is a soft mix of value rows.

    **Temperature.** A temperature $\tau > 0$ scales the similarities,

    $$
    \operatorname{softmax}\!\left(\frac{q_i^\top K^\top}{\tau\sqrt{d}}\right),
    $$

    continuously interpolating between regimes:

    - As $\tau \to 0$, the distribution concentrates on $j^{\star}$ and attention becomes a hard lookup.
    - As $\tau \to \infty$, the distribution becomes uniform and the output is the mean of the value rows.

    Standard self-attention uses $\tau = 1$: differentiable everywhere, selective where it counts.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### A small worked example

    To see the operator in action, we construct $X$ with a clear geometric
    structure: three clusters of four tokens each in $\mathbf{R}^8$. We take
    $W_Q = W_K = W_V = I$, so $Q = K = V = X$; the attention pattern will then
    reflect the raw geometry of the input.
    """)
    return


@app.cell
def _(np):
    n_per_cluster= 4
    d = 8
    rng = np.random.default_rng(0)

    centers = rng.normal(size=(3, d))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    X = np.vstack([
        c + 0.2 * rng.normal(size=(n_per_cluster, d))
        for c in centers
    ])
    Q, K, V = X, X, X
    return K, Q, V


@app.cell
def _(np):
    def attention(Q, K, V, tau=1.0):
        d = Q.shape[-1]
        logits = (Q @ K.T) / (np.sqrt(d) * tau)
        logits -= logits.max(axis=1, keepdims=True)
        A = np.exp(logits)
        A /= A.sum(axis=1, keepdims=True)
        return A

    return (attention,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Use the temperature slider below to see the soft-to-hard transition. At
    $\tau = 1$ the attention matrix exhibits a block-diagonal pattern; each query
    attends most strongly to its own cluster. Small $\tau$ sharpens the pattern
    toward a hard lookup; large $\tau$ smooths it toward uniform mixing.
    """)
    return


@app.cell
def _(mo):
    tau_slider = mo.ui.slider(
        start=0.1, stop=3.0, step=0.05, value=1.0,
        label=r"Temperature $\tau$", show_value=True,
    )
    tau_slider
    return (tau_slider,)


@app.cell(hide_code=True)
def _(K, Q, V, attention, plt, tau_slider):
    A = attention(Q, K, V, tau=tau_slider.value)

    _fig, (_ax1, _ax2) = plt.subplots(
        1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1, 1.1]}
    )

    _im = _ax1.imshow(A, cmap="magma", vmin=0, vmax=max(A.max(), 1e-3))
    _ax1.set_title(rf"Attention matrix $A$  ($\tau = {tau_slider.value:.2f}$)")
    _ax1.set_xlabel("key index $j$")
    _ax1.set_ylabel("query index $i$")
    plt.colorbar(_im, ax=_ax1, fraction=0.046)

    _query_idx = 0
    _ax2.bar(range(A.shape[1]), A[_query_idx], color="royalblue")
    _ax2.set_title(rf"Row $i={_query_idx}$ of $A$: a distribution over keys")
    _ax2.set_xlabel("key index $j$")
    _ax2.set_ylabel(rf"$A_{{{_query_idx},\,j}}$")
    _ax2.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interpretation II: row-stochastic mixing

    Every row of $A$ is a probability distribution: the entries $A_{i,1}, \dots, A_{i,n}$ are nonnegative and sum to $1$. Entry $A_{ij}$ is the weight placed on value $v_j$ when forming row $i$ of the output, and is large when the key $k_j$ is similar to the query $q_i$.

    Row $i$ of the output is then the probability-weighted average of value rows:

    $$
    \operatorname{Attn}(Q, K, V)_i \;=\; \sum_{j=1}^n A_{ij}\, v_j^\top
      \;\in\; \operatorname{conv}\{v_1^\top, \dots, v_n^\top\}.
    $$

    Every output lies in the convex hull of the value rows: self-attention rewrites each token as a **data-dependent local average** of the others. Tokens that attend strongly to each other pull each other's representations together.

    Two useful reframings:

    - **Graph view.** Treat $A$ as the weighted adjacency matrix of a directed graph on the $n$ tokens: an edge from $i$ to $j$ has weight $A_{ij}$. One step of self-attention is one step of a random walk on this graph, smoothing the value signal along edges. Unlike the fixed Laplacian of a pre-specified graph, the edges here are recomputed at inference from $Q$ and $K$.

    - **Kernel view.** The unnormalized entries $\tilde A_{ij} = \exp(q_i^\top k_j / \sqrt{d})$ form an asymmetric exponentiated inner-product kernel. Row-normalization turns it into a transition matrix, and self-attention convolves $V$ with this kernel along the sequence dimension.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Extensions

    Three standard variations build directly on this operator:

    - **Masked attention.** Setting $(Q K^\top)_{ij} = -\infty$ for $j > i$ before
      the softmax zeros those entries of $A$, enforcing causality in autoregressive
      models.
    - **Multi-head attention.** Run $h$ independent attention operators in parallel
      with their own $W_Q^{(k)}, W_K^{(k)}, W_V^{(k)}$, concatenate the outputs, and
      apply a final linear map. Different heads learn to attend along
      different "directions".
    - **Cross-attention.** Take $Q$ from one sequence and $K, V$ from another.
      Self-attention is the special case where the two sequences coincide.
    """)
    return


if __name__ == "__main__":
    app.run()
