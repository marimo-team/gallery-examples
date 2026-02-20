# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#     "marimo",
#     "anywidget",
#     "traitlets",
#     "numpy==2.3.5",
#     "matplotlib==3.10.8",
# ]
# ///

import marimo

__generated_with = "0.19.7"
app = marimo.App(
    css_file="/usr/local/_marimo/custom.css",
    auto_download=["html"],
)

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as _np
    import numpy as np
    from widget import GrpoGdpoWidget


@app.cell
def _():
    mo.md(r"""
 
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## GRPO vs GDPO Advantage Comparison

    This exploration was based on [this paper on alphaXiv](https://www.alphaxiv.org/abs/2601.05242).

    This widget demonstrates the difference between **GRPO** (Group Relative Policy Optimization)
    and **GDPO** (Group reward-Decoupled Policy Optimization) advantage calculations.

    **Click on the reward cells** (0 or 1) to toggle values and see how the advantages change.

    ### Formulas

    **GRPO** first aggregates rewards, then normalizes:

    $$r_j = r_j^{(1)} + r_j^{(2)} + \ldots + r_j^{(n)}$$

    $$A_j^{\text{GRPO}} = \frac{r_j - \mu(r)}{\sigma(r)}$$

    **GDPO** normalizes each reward dimension separately, then sums:

    $$A_j^{(i)} = \frac{r_j^{(i)} - \mu(r^{(i)})}{\sigma(r^{(i)})}$$

    $$A_j^{\text{GDPO}} = A_j^{(1)} + A_j^{(2)} + \ldots + A_j^{(n)}$$

    ### Key Insight

    When different reward combinations produce the same total (e.g., `[1,0,1]` and `[0,1,1]` both sum to 2),
    GRPO assigns identical advantages, while GDPO can distinguish them based on which specific
    rewards were achieved.

    The **Difference** column highlights when GDPO preserves information that GRPO loses.
    """)
    return


@app.cell
def _():
    widget = GrpoGdpoWidget()
    return (widget,)


@app.cell
def _(widget):
    widget_view = mo.ui.anywidget(widget)
    return (widget_view,)


@app.cell
def _(widget_view):
    widget_view
    return


@app.cell(hide_code=True)
def _(fig):
    fig
    return


@app.cell(hide_code=True)
def _():
    def normalize(arr):
        """Normalize array to zero mean and unit variance."""
        arr = np.array(arr, dtype=np.float64)
        std = arr.std()
        if std == 0:
            return np.zeros_like(arr)
        return (arr - arr.mean()) / std

    def compute_grpo_advantages(rewards):
        """GRPO: aggregate first, then normalize."""
        totals = rewards.sum(axis=1)
        return normalize(totals)

    def compute_gdpo_advantages(rewards):
        """GDPO: normalize each dimension, then sum."""
        advantages = np.zeros(len(rewards))
        for dim in range(rewards.shape[1]):
            advantages += normalize(rewards[:, dim])
        return advantages

    def train_policy(method, epochs=100, lr=0.1, batch_size=32, seed=41, fixed_rewards=None):
        """Train a 3-dimensional Bernoulli policy using GRPO or GDPO.

        Args:
            fixed_rewards: If provided (numpy array), use this fixed dataset every epoch.
                           If None, sample fresh data each epoch (default).

        Returns history of probabilities over epochs.
        """
        rng = np.random.default_rng(seed)

        # Policy parameters (log-odds, initialized to 0 -> prob 0.5)
        logits = np.zeros(3)

        history = []

        for _epoch in range(epochs):
            # Current probabilities
            probs = 1 / (1 + np.exp(-logits))
            history.append(probs.copy())

            # Get rewards: either fixed or freshly sampled
            if fixed_rewards is not None:
                rewards = fixed_rewards
            else:
                rewards = (rng.random((batch_size, 3)) < probs).astype(np.float64)

            # Compute advantages
            if method == 'grpo':
                advantages = compute_grpo_advantages(rewards)
            else:
                advantages = compute_gdpo_advantages(rewards)

            # Policy gradient update
            # grad log p(a) * advantage, where p(a) = prod of Bernoulli probs
            for i in range(3):
                # For Bernoulli: grad log p = (reward - prob) for that dimension
                grad = ((rewards[:, i] - probs[i]) * advantages).mean()
                logits[i] += lr * grad

        return np.array(history)
    return (train_policy,)


@app.cell(hide_code=True)
def _():
    reuse_toggle = mo.ui.switch(label="Train on widget data (instead of fresh samples each epoch)")
    reuse_toggle
    return (reuse_toggle,)


@app.cell
def _():
    def widget_rewards_to_array(rewards_list):
        """Convert widget rewards list to numpy array."""
        return _np.array([
            [r["correctness"], r["style"], r["conciseness"]]
            for r in rewards_list
        ], dtype=_np.float64)
    return (widget_rewards_to_array,)


@app.cell
def _(reuse_toggle, train_policy, widget_rewards_to_array, widget_view):
    if reuse_toggle.value:
        fixed = widget_rewards_to_array(widget_view.widget.rewards)
    else:
        fixed = None
    grpo_history = train_policy('grpo', epochs=150, lr=0.15, fixed_rewards=fixed)
    gdpo_history = train_policy('gdpo', epochs=150, lr=0.15, fixed_rewards=fixed)
    return gdpo_history, grpo_history


@app.cell
def _(gdpo_history, grpo_history):
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # blue, orange, green
    labels = ['correctness', 'style', 'conciseness']
    epochs = range(len(grpo_history))

    # Plot GDPO (solid) first, then GRPO (dotted) on top so dotted is visible
    for i, (color, label) in enumerate(zip(colors, labels)):
        ax.plot(epochs, gdpo_history[:, i], '-', color=color, linewidth=2,
                label=f'{label} (GDPO)')
    for i, (color, label) in enumerate(zip(colors, labels)):
        ax.plot(epochs, grpo_history[:, i], '--', color=color, linewidth=2,
                label=f'{label} (GRPO)')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Probability')
    ax.set_title('GRPO vs GDPO Convergence: Policy Probabilities Over Training')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right', ncol=2)
    ax.grid(True, alpha=0.3)
    return (fig,)


if __name__ == "__main__":
    app.run()
