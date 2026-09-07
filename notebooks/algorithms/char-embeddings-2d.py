# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["marimo", "numpy", "matplotlib", "requests", "wigglystuff"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # 🔤 Watching letters find their place

    We learn a **2-dimensional embedding for each character** by training a tiny
    next-character predictor on ABC News headlines. Because the embedding is only
    2D, *the embedding itself is the plot* — no projection needed.

    Each letter `x` is looked up as a 2D point `E[x]`, a linear layer turns that
    point into a score for every possible *next* character, and we train it to
    predict the character that actually follows. As training proceeds, letters
    that behave similarly drift together. Hit **play** at the bottom to watch them
    move.
    """)
    return


@app.cell(hide_code=True)
def _():
    import numpy as np
    import requests
    import matplotlib.pyplot as plt
    from wigglystuff import FramePlayer

    return FramePlayer, np, plt, requests


@app.cell(hide_code=True)
def _(mo, requests):
    # Pull a sample of headlines from the ABC News Datasette instance.
    # Datasette caps each SQL response at 1000 rows, so we page a few times.
    url = "http://datasette.exe.xyz/abc_news.json"
    headlines = []
    for page in range(5):
        sql = f"select headline_text from headlines order by id limit 1000 offset {page * 1000}"
        resp = requests.get(url, params={"sql": sql, "_shape": "array"}, timeout=60)
        headlines.extend(row["headline_text"] for row in resp.json())
    mo.md(f"Fetched **{len(headlines):,}** headlines. Example: *{headlines[0]}*")
    return (headlines,)


@app.cell(hide_code=True)
def _(headlines, mo, np):
    # Vocabulary: 26 letters + space. Everything else is dropped, and each
    # headline is wrapped so a leading space marks "start of headline".
    vocab = " abcdefghijklmnopqrstuvwxyz"
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for ch, i in stoi.items()}
    V = len(vocab)

    def clean(text):
        text = text.lower()
        return "".join(ch if ch in stoi else " " for ch in text)

    # Build (current_char -> next_char) id pairs across all headlines.
    xs, ys = [], []
    for h in headlines:
        seq = " " + clean(h) + " "
        ids = [stoi[c] for c in seq]
        xs.extend(ids[:-1])
        ys.extend(ids[1:])

    X = np.array(xs, dtype=np.int64)
    Y = np.array(ys, dtype=np.int64)
    mo.md(f"Vocab size **{V}**, **{len(X):,}** character pairs to learn from.")
    return V, X, Y, itos


@app.cell(hide_code=True)
def _(V, X, Y, mo, np):
    # A next-char model with a 2D bottleneck and a hidden layer, trained by hand.
    #   emb    = E[x]                  (2D point per character)
    #   h      = tanh(emb @ W1 + b1)   (hidden layer, H units)
    #   logits = h @ W2 + b2           (score for every possible next char)
    #   loss   = cross-entropy against the true next char
    # We snapshot the embedding table E often so the motion between frames is small.
    def train_embeddings(X, Y, V, H=16, steps=1000, batch=4096, lr=0.02, snap_every=5, seed=0):
        rng = np.random.default_rng(seed)
        E = rng.normal(0, 0.3, size=(V, 2))
        W1 = rng.normal(0, 0.3, size=(2, H))
        b1 = np.zeros(H)
        W2 = rng.normal(0, 0.3, size=(H, V))
        b2 = np.zeros(V)
        params = [E, W1, b1, W2, b2]

        # Adam state, one (m, v) pair per parameter.
        adam = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        snapshots = []
        for t in range(1, steps + 1):
            idx = rng.integers(0, len(X), size=batch)
            xb, yb = X[idx], Y[idx]

            emb = E[xb]  # (m, 2)
            h_pre = emb @ W1 + b1  # (m, H)
            h = np.tanh(h_pre)
            logits = h @ W2 + b2  # (m, V)
            logits -= logits.max(1, keepdims=True)
            probs = np.exp(logits)
            probs /= probs.sum(1, keepdims=True)
            loss = -np.log(probs[np.arange(batch), yb] + 1e-9).mean()

            dlogits = probs
            dlogits[np.arange(batch), yb] -= 1
            dlogits /= batch
            dW2 = h.T @ dlogits  # (H, V)
            db2 = dlogits.sum(0)
            dh = dlogits @ W2.T  # (m, H)
            dh_pre = dh * (1 - h * h)
            dW1 = emb.T @ dh_pre  # (2, H)
            db1 = dh_pre.sum(0)
            demb = dh_pre @ W1.T  # (m, 2)
            dE = np.zeros_like(E)
            np.add.at(dE, xb, demb)
            grads = [dE, dW1, db1, dW2, db2]

            for p, g, (m, v) in zip(params, grads, adam):
                m *= beta1
                m += (1 - beta1) * g
                v *= beta2
                v += (1 - beta2) * (g * g)
                mhat = m / (1 - beta1**t)
                vhat = v / (1 - beta2**t)
                p -= lr * mhat / (np.sqrt(vhat) + eps)

            if t % snap_every == 0 or t == 1:
                snapshots.append((t, E.copy(), loss))

        steps_axis = [s for s, _, _ in snapshots]
        embeds = [e for _, e, _ in snapshots]
        losses = [l for _, _, l in snapshots]
        return steps_axis, embeds, losses

    steps_axis, snapshots, losses = train_embeddings(X, Y, V)
    mo.md(
        f"Trained a 2-layer model. Captured **{len(snapshots)}** frames. "
        f"Loss {losses[0]:.3f} → {losses[-1]:.3f}."
    )
    return losses, snapshots, steps_axis


@app.cell(hide_code=True)
def _(V, itos, losses, mo, np, plt, snapshots, steps_axis):
    # Each frame stacks the letter map (top) over the loss curve (bottom), with a
    # dot marking where we are in training. We encode to PNG and close immediately.
    from wigglystuff.chart_puck import fig_to_base64

    vowels = set("aeiou")

    def kind_color(ch):
        if ch == " ":
            return "#8b5cf6"  # space
        if ch in vowels:
            return "#ef4444"  # vowels
        return "#334155"  # consonants

    all_pts = np.vstack(snapshots)
    pad = 0.5
    xlim = (all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad)
    ylim = (all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad)

    def render_frame(k):
        E, step, loss = snapshots[k], steps_axis[k], losses[k]
        fig, (ax, axl) = plt.subplots(
            2,
            1,
            figsize=(6, 7),
            dpi=90,
            gridspec_kw={"height_ratios": [3, 1]},
        )
        for i in range(V):
            ch = itos[i]
            label = "␣" if ch == " " else ch
            ax.text(
                E[i, 0],
                E[i, 1],
                label,
                color=kind_color(ch),
                fontsize=18,
                fontweight="bold",
                ha="center",
                va="center",
            )
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"step {step:>4}   loss {loss:.3f}", fontsize=11, loc="left")
        for s in ax.spines.values():
            s.set_color("#e2e8f0")

        axl.plot(steps_axis, losses, color="#94a3b8", lw=1)
        axl.plot(steps_axis[: k + 1], losses[: k + 1], color="#2563eb", lw=1.8)
        axl.scatter([step], [loss], color="#2563eb", zorder=3, s=25)
        axl.set_xlabel("training step")
        axl.set_ylabel("loss")
        axl.spines[["top", "right"]].set_visible(False)

        fig.tight_layout()
        uri = fig_to_base64(fig)
        plt.close(fig)
        return uri

    frames = [render_frame(k) for k in range(len(snapshots))]
    mo.md(f"Rendered **{len(frames)}** frames.")
    return (frames,)


@app.cell(hide_code=True)
def _(FramePlayer, frames, mo):
    # ▶️ Press play to watch the letters organise themselves during training.
    player = mo.ui.anywidget(FramePlayer(frames, interval_ms=120, loop=True, width=460))
    player
    return


if __name__ == "__main__":
    app.run()
