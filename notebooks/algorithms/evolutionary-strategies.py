# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib==3.10.8",
#     "numpy==2.3.5",
#     "marimo",
# ]
# ///

import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    from functools import lru_cache
    return Ellipse, lru_cache, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Evolutionary Strategies: Watching the Search Space Adapt

    Evolutionary Strategies (ES) don't just find good solutions - they learn **how to search**.

    The key insight: ES maintains both a **mean** $\mu$ (where to search) and a **standard deviation** $\sigma$ (how wide to search). Both parameters adapt based on what the algorithm discovers.

    This notebook lets you watch $\sigma$ adapt in real-time across different challenging optimization landscapes.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The Two Learnable Parameters

    In ES, we sample $N$ candidate solutions from a Gaussian distribution:

    $$x_i \sim \mathcal{N}(\mu, \sigma) \quad \text{for } i = 1, 2, \ldots, N$$

    **Dimensionality in this notebook:**
    - $\mu \in \mathbb{R}^2$ — a 2D vector representing the center of our search
    - $\sigma \in \mathbb{R}$ — a scalar (same in all directions, "isotropic")
    - $x_i \in \mathbb{R}^2$ — each sample is a 2D point
    - $f(x_i) \in \mathbb{R}$ — the function value (fitness) at that point

    The magic of ES is that **both $\mu$ and $\sigma$ adapt based on what we find**:

    - If good solutions come from **far away** → increase $\sigma$ (explore more)
    - If good solutions come from **nearby** → decrease $\sigma$ (exploit locally)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The $\sigma$ Update Rule

    After sampling $N$ candidates and evaluating their fitness, we update $\sigma$:

    $$\Delta\sigma = \alpha_\sigma \cdot \frac{1}{N} \sum_{i=1}^{N} \left[ \tilde{f}_i \cdot \left( \frac{\|x_i - \mu\|^2}{\sigma^2} - 1 \right) \right]$$

    **Every term explained:**

    | Symbol | Meaning |
    |--------|---------|
    | $\Delta\sigma$ | The change to apply to $\sigma$ this iteration |
    | $\alpha_\sigma$ | Learning rate for $\sigma$ (typically 0.01–0.1). Controls how fast $\sigma$ adapts |
    | $N$ | Population size — number of samples per iteration |
    | $x_i$ | The $i$-th sampled candidate (a 2D point in this notebook) |
    | $\mu$ | Current mean of the search distribution (2D vector) |
    | $\|x_i - \mu\|^2$ | Squared Euclidean distance from sample to mean |
    | $\sigma$ | Current standard deviation (scalar) |
    | $\tilde{f}_i$ | **Normalized fitness** of sample $i$ (see below) |

    **What is $\tilde{f}_i$ (normalized fitness)?**

    We normalize the raw fitness values to have zero mean and unit variance:

    $$\tilde{f}_i = \frac{f(x_i) - \bar{f}}{\text{std}(f)}$$

    where $\bar{f} = \frac{1}{N}\sum_j f(x_j)$ is the mean fitness across all samples.

    For **minimization**, we flip the sign: $\tilde{f}_i = -\frac{f(x_i) - \bar{f}}{\text{std}(f)}$, so lower $f$ → higher $\tilde{f}$.

    **Why does this work?**

    The term $\frac{\|x_i - \mu\|^2}{\sigma^2} - 1$ measures how "far" a sample is relative to the current $\sigma$:
    - **Positive** when $\|x_i - \mu\| > \sigma$ (sample is far from center)
    - **Negative** when $\|x_i - \mu\| < \sigma$ (sample is close to center)
    - **Zero** when $\|x_i - \mu\| = \sigma$ (exactly one standard deviation away)

    Multiplying by $\tilde{f}_i$ creates a **correlation signal**:
    - Good solutions far away → positive contribution → $\sigma$ grows
    - Good solutions nearby → negative contribution → $\sigma$ shrinks
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The $\mu$ Update Rule

    The mean $\mu$ moves toward regions where good solutions were found:

    $$\Delta\mu = \alpha_\mu \cdot \frac{1}{N} \sum_{i=1}^{N} \left[ \tilde{f}_i \cdot \frac{(x_i - \mu)}{\sigma} \right]$$

    **Every term explained:**

    | Symbol | Meaning |
    |--------|---------|
    | $\Delta\mu$ | The change to apply to $\mu$ this iteration (a 2D vector) |
    | $\alpha_\mu$ | Learning rate for $\mu$ (typically 0.1–1.0). Controls step size |
    | $N$ | Population size |
    | $\tilde{f}_i$ | Normalized fitness of sample $i$ (same as in $\sigma$ update) |
    | $x_i - \mu$ | Direction vector from current mean to sample $i$ |
    | $\sigma$ | Current standard deviation (normalizes the step) |

    **How it works:**

    Each sample $x_i$ "votes" for the direction $(x_i - \mu)$ with weight $\tilde{f}_i$:
    - **Good samples** ($\tilde{f}_i > 0$) pull $\mu$ toward them
    - **Bad samples** ($\tilde{f}_i < 0$) push $\mu$ away from them

    The averaging over $N$ samples creates a **gradient-like signal** pointing toward better regions — but computed entirely from function evaluations, no actual gradients needed!

    Dividing by $\sigma$ normalizes the step size relative to the current search radius.
    """)
    return


@app.cell
def _(lru_cache, np):
    # Test functions - all 2D for consistent visualization

    # --- Smooth functions (good for understanding ES behavior) ---

    def sphere(x):
        """Simple quadratic bowl - the easiest function."""
        return np.sum(x**2)

    def rosenbrock(x):
        """Smooth banana-shaped valley. Easy to find, hard to follow."""
        return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

    def beale(x):
        """Smooth with a narrow curved valley."""
        return ((1.5 - x[0] + x[0]*x[1])**2 +
                (2.25 - x[0] + x[0]*x[1]**2)**2 +
                (2.625 - x[0] + x[0]*x[1]**3)**2)

    def booth(x):
        """Smooth quadratic with off-center minimum."""
        return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2

    # --- Multimodal functions (more challenging) ---

    def rastrigin(x):
        """Highly multimodal with regularly spaced local optima."""
        A = 10
        return A * 2 + np.sum(x**2 - A * np.cos(2 * np.pi * x))

    def ackley(x):
        """Nearly flat outer region with steep central funnel."""
        a, b, c = 20, 0.2, 2 * np.pi
        d = len(x)
        sum1 = np.sum(x**2)
        sum2 = np.sum(np.cos(c * x))
        return -a * np.exp(-b * np.sqrt(sum1/d)) - np.exp(sum2/d) + a + np.e

    def schwefel(x):
        """Deceptive: global minimum far from next best local optima."""
        d = len(x)
        return 418.9829 * d - np.sum(x * np.sin(np.sqrt(np.abs(x))))

    def himmelblau(x):
        """Multiple global optima - four symmetric minima."""
        return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

    # Function metadata for UI
    FUNCTIONS = {
        # Smooth functions first
        "Sphere": {
            "func": sphere,
            "bounds": (-5, 5),
            "optimum": (0, 0),
            "description": "Simple quadratic bowl. The easiest benchmark - sigma should shrink steadily."
        },
        "Rosenbrock": {
            "func": rosenbrock,
            "bounds": (-2, 2),
            "optimum": (1, 1),
            "description": "Smooth banana valley. Easy to find the valley, hard to follow it to the minimum."
        },
        "Beale": {
            "func": beale,
            "bounds": (-4.5, 4.5),
            "optimum": (3, 0.5),
            "description": "Smooth curved valley with flat regions. Tests sigma adaptation."
        },
        "Booth": {
            "func": booth,
            "bounds": (-10, 10),
            "optimum": (1, 3),
            "description": "Smooth quadratic plate. Simple but off-center minimum."
        },
        # Multimodal functions
        "Rastrigin": {
            "func": rastrigin,
            "bounds": (-5.12, 5.12),
            "optimum": (0, 0),
            "description": "Regular grid of local optima. Global minimum at origin."
        },
        "Ackley": {
            "func": ackley,
            "bounds": (-5, 5),
            "optimum": (0, 0),
            "description": "Flat outer region, steep central funnel. Tests exploration."
        },
        "Schwefel": {
            "func": schwefel,
            "bounds": (-500, 500),
            "optimum": (420.97, 420.97),
            "description": "Deceptive! Global optimum far from where gradients point."
        },
        "Himmelblau": {
            "func": himmelblau,
            "bounds": (-5, 5),
            "optimum": (3, 2),  # One of four optima
            "description": "Four global optima. ES may converge to different ones."
        }
    }

    # Cached landscape computation for faster slider updates
    @lru_cache(maxsize=16)
    def compute_landscape(func_name, n_points):
        """Compute the landscape grid for a function (cached)."""
        func_info = FUNCTIONS[func_name]
        f = func_info["func"]
        bounds = func_info["bounds"]
        x = np.linspace(bounds[0], bounds[1], n_points)
        y = np.linspace(bounds[0], bounds[1], n_points)
        X, Y = np.meshgrid(x, y)
        Z = np.array([[f(np.array([xi, yi])) for xi, yi in zip(xrow, yrow)]
                      for xrow, yrow in zip(X, Y)])
        return X, Y, Z
    return FUNCTIONS, compute_landscape


@app.cell
def _(FUNCTIONS, mo):
    func_dropdown = mo.ui.dropdown(
        options=list(FUNCTIONS.keys()),
        value="Sphere",
        label="Test Function"
    )
    func_dropdown
    return (func_dropdown,)


@app.cell(hide_code=True)
def _(FUNCTIONS, compute_landscape, func_dropdown, mo, plt):
    # Show the selected function's landscape (using cached computation)
    _func_name = func_dropdown.value
    _func_info = FUNCTIONS[_func_name]
    _optimum = _func_info["optimum"]

    _X, _Y, _Z = compute_landscape(_func_name, 200)

    _fig, _ax = plt.subplots(figsize=(8, 6))
    _cf = _ax.contourf(_X, _Y, _Z, levels=30, cmap='viridis')
    plt.colorbar(_cf, ax=_ax, label='f(x, y)')
    _ax.scatter([_optimum[0]], [_optimum[1]], c='red', s=100, marker='*',
                label=f'Global optimum', zorder=10)
    _ax.set_xlabel('x')
    _ax.set_ylabel('y')
    _ax.set_title(f'{_func_name} Function')
    _ax.legend()

    mo.vstack([
        _fig,
        mo.md(f"**{_func_name}**: {_func_info['description']}")
    ])
    return


@app.cell
def _(np):
    def es_with_history(f, mu0, sigma0, alpha_mu=0.5, alpha_sigma=0.1,
                        n_samples=50, steps=100, seed=42):
        """
        Evolutionary Strategy with full history for visualization.

        Returns a list of dicts, one per iteration, containing:
        - mu: current mean
        - sigma: current standard deviation
        - samples: population samples at this step
        - fitness: fitness values of samples
        - best_fitness: best fitness found so far
        """
        np.random.seed(seed)
        mu = np.array(mu0, dtype=float)
        sigma = float(sigma0)
        history = []
        best_fitness = float('inf')

        for step in range(steps):
            # Sample population
            samples = np.random.normal(mu, sigma, (n_samples, 2))
            f_vals = np.array([f(s) for s in samples])
            best_fitness = min(best_fitness, f_vals.min())

            # Store current state
            history.append({
                'mu': mu.copy(),
                'sigma': sigma,
                'samples': samples.copy(),
                'fitness': f_vals.copy(),
                'best_fitness': best_fitness
            })

            # Normalize fitness (minimization: lower is better)
            f_norm = -(f_vals - np.mean(f_vals)) / (np.std(f_vals) + 1e-8)

            # Update mu: move toward better samples
            d_mu = alpha_mu * np.mean(f_norm[:, None] * (samples - mu), axis=0) / sigma
            mu = mu + d_mu

            # Update sigma: expand if good solutions are far, contract if close
            normalized_dist_sq = np.sum((samples - mu)**2, axis=1) / (sigma**2)
            d_sigma = alpha_sigma * np.mean(f_norm * (normalized_dist_sq / 2 - 1))
            sigma = np.clip(sigma + d_sigma, 0.01, 100.0)

        return history
    return (es_with_history,)


@app.cell
def _(mo):
    # ES parameter controls
    n_samples_slider = mo.ui.slider(start=20, stop=200, step=10, value=50, label="Population Size")
    alpha_mu_slider = mo.ui.slider(start=0.1, stop=1.0, step=0.1, value=0.5, label="Learning Rate (mu)")
    alpha_sigma_slider = mo.ui.slider(start=0.01, stop=0.3, step=0.01, value=0.1, label="Learning Rate (sigma)")
    steps_slider = mo.ui.slider(start=50, stop=300, step=25, value=100, label="Iterations")

    # Starting position (relative to bounds, will be scaled per function)
    start_x_slider = mo.ui.slider(start=-0.8, stop=0.8, step=0.1, value=-0.5, label="Start X (relative)")
    start_y_slider = mo.ui.slider(start=-0.8, stop=0.8, step=0.1, value=-0.5, label="Start Y (relative)")
    init_sigma_slider = mo.ui.slider(start=0.1, stop=2.0, step=0.1, value=1.0, label="Initial Sigma (relative)")

    mo.vstack([
        mo.hstack([n_samples_slider, steps_slider], justify="start"),
        mo.hstack([alpha_mu_slider, alpha_sigma_slider], justify="start"),
        mo.hstack([start_x_slider, start_y_slider, init_sigma_slider], justify="start"),
    ])
    return (
        alpha_mu_slider,
        alpha_sigma_slider,
        init_sigma_slider,
        n_samples_slider,
        start_x_slider,
        start_y_slider,
        steps_slider,
    )


@app.cell
def _(
    FUNCTIONS,
    alpha_mu_slider,
    alpha_sigma_slider,
    es_with_history,
    func_dropdown,
    init_sigma_slider,
    n_samples_slider,
    start_x_slider,
    start_y_slider,
    steps_slider,
):
    # Run ES and store history
    _func_name = func_dropdown.value
    _func_info = FUNCTIONS[_func_name]
    _f = _func_info["func"]
    _bounds = _func_info["bounds"]
    _range = _bounds[1] - _bounds[0]
    _center = (_bounds[0] + _bounds[1]) / 2

    # Convert relative positions to absolute
    _start_x = _center + start_x_slider.value * _range / 2
    _start_y = _center + start_y_slider.value * _range / 2
    _init_sigma = init_sigma_slider.value * _range / 4

    es_history = es_with_history(
        _f,
        mu0=[_start_x, _start_y],
        sigma0=_init_sigma,
        alpha_mu=alpha_mu_slider.value,
        alpha_sigma=alpha_sigma_slider.value,
        n_samples=n_samples_slider.value,
        steps=steps_slider.value,
        seed=42
    )
    return (es_history,)


@app.cell
def _(es_history, mo):
    # Iteration slider for playback
    iteration_slider = mo.ui.slider(
        start=0,
        stop=len(es_history) - 1,
        step=1,
        value=0,
        label="Iteration",
        show_value=True
    )
    iteration_slider
    return (iteration_slider,)


@app.cell(hide_code=True)
def _(
    Ellipse,
    FUNCTIONS,
    compute_landscape,
    es_history,
    func_dropdown,
    iteration_slider,
    mo,
    np,
    plt,
):
    # Main visualization: population cloud with trajectory
    _func_name = func_dropdown.value
    _func_info = FUNCTIONS[_func_name]
    _bounds = _func_info["bounds"]
    _optimum = _func_info["optimum"]
    _iter = iteration_slider.value
    _state = es_history[_iter]

    # Use cached landscape grid
    _X, _Y, _Z = compute_landscape(_func_name, 150)

    _fig, (_ax_main, _ax_sigma) = plt.subplots(1, 2, figsize=(14, 5))

    # Left plot: landscape with population
    _cf = _ax_main.contourf(_X, _Y, _Z, levels=30, cmap='viridis', alpha=0.8)
    plt.colorbar(_cf, ax=_ax_main)

    # Draw trajectory up to current iteration
    _trajectory = np.array([h['mu'] for h in es_history[:_iter+1]])
    if len(_trajectory) > 1:
        _ax_main.plot(_trajectory[:, 0], _trajectory[:, 1], 'r-', linewidth=2, alpha=0.7, label='Trajectory')

    # Draw search ellipse (2-sigma)
    _ellipse = Ellipse(
        xy=_state['mu'],
        width=4*_state['sigma'],
        height=4*_state['sigma'],
        fill=True, alpha=0.2, color='red',
        edgecolor='red', linewidth=2
    )
    _ax_main.add_patch(_ellipse)

    # Draw current samples
    _ax_main.scatter(_state['samples'][:, 0], _state['samples'][:, 1],
                     c='white', s=15, alpha=0.6, edgecolors='black', linewidths=0.5)

    # Draw current mean
    _ax_main.scatter([_state['mu'][0]], [_state['mu'][1]],
                     c='yellow', s=200, marker='*', edgecolors='red', linewidths=2,
                     zorder=10, label=f'Current mu')

    # Draw global optimum
    _ax_main.scatter([_optimum[0]], [_optimum[1]], c='lime', s=100, marker='X',
                     edgecolors='black', linewidths=1, zorder=10, label='Global optimum')

    _ax_main.set_xlim(_bounds)
    _ax_main.set_ylim(_bounds)
    _ax_main.set_xlabel('x')
    _ax_main.set_ylabel('y')
    _ax_main.set_title(f'Iteration {_iter}: mu=({_state["mu"][0]:.2f}, {_state["mu"][1]:.2f}), sigma={_state["sigma"]:.3f}')
    _ax_main.legend(loc='upper right')

    # Right plot: sigma over time
    _sigmas = [h['sigma'] for h in es_history]
    _iterations = range(len(es_history))
    _ax_sigma.plot(_iterations, _sigmas, 'b-', linewidth=2)
    _ax_sigma.axvline(_iter, color='red', linestyle='--', linewidth=2, label=f'Current: {_iter}')
    _ax_sigma.scatter([_iter], [_state['sigma']], c='red', s=100, zorder=10)
    _ax_sigma.set_xlabel('Iteration')
    _ax_sigma.set_ylabel('Sigma (search radius)')
    _ax_sigma.set_title('Search Radius Over Time')
    _ax_sigma.legend()
    _ax_sigma.grid(True, alpha=0.3)

    plt.tight_layout()

    mo.vstack([
        _fig,
        mo.md(f"""
        **Iteration {_iter}**: The red ellipse shows the 2-sigma search region.
        White dots are the current population samples. Watch how the ellipse
        {'expands' if _iter > 0 and _state['sigma'] > es_history[_iter-1]['sigma'] else 'contracts'}
        as the algorithm {'explores' if _state['sigma'] > es_history[0]['sigma'] * 0.5 else 'exploits'}.
        """)
    ])
    return


@app.cell(hide_code=True)
def _(
    FUNCTIONS,
    compute_landscape,
    es_history,
    func_dropdown,
    iteration_slider,
    np,
    plt,
):
    # Per-sample contribution visualization with landscape background
    # All arrows have SAME LENGTH - color encodes magnitude of influence
    _func_name = func_dropdown.value
    _func_info = FUNCTIONS[_func_name]
    _bounds = _func_info["bounds"]
    _iter = iteration_slider.value
    _state = es_history[_iter]

    _mu = _state['mu']
    _sigma = _state['sigma']
    _samples = _state['samples']
    _fitness = _state['fitness']

    # Compute normalized fitness (same as in ES algorithm)
    _f_norm = -(_fitness - np.mean(_fitness)) / (np.std(_fitness) + 1e-8)

    # Compute per-sample sigma contributions
    _dist_sq = np.sum((_samples - _mu)**2, axis=1)
    _sigma_contrib = _f_norm * (_dist_sq / (_sigma**2) - 1)

    # Compute per-sample mu contributions (direction vectors)
    _mu_contrib = _f_norm[:, None] * (_samples - _mu) / _sigma

    # Use the full function bounds for consistent axes across all charts
    _xlim = _bounds
    _ylim = _bounds
    _plot_range = _bounds[1] - _bounds[0]

    # Fixed arrow length for all arrows - 5% of plot range
    _arrow_length = _plot_range * 0.05

    # Use cached landscape grid
    _X_local, _Y_local, _Z_local = compute_landscape(_func_name, 150)

    # Mu on left, Sigma on right
    _fig, (_ax_mu, _ax_sigma) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Left plot: Mu contributions with landscape background ---
    _ax_mu.contourf(_X_local, _Y_local, _Z_local, levels=25, cmap='viridis', alpha=0.6)

    # Compute mu contribution magnitudes for coloring
    _mu_magnitudes = np.linalg.norm(_mu_contrib, axis=1)
    _mu_mag_normalized = _mu_magnitudes / (_mu_magnitudes.max() + 1e-8)

    # Draw samples as points (no color here, arrows will have color)
    _ax_mu.scatter(_samples[:, 0], _samples[:, 1], c='white', s=60, alpha=0.9,
                   edgecolors='black', linewidths=1, zorder=5)

    # Draw uniform-length arrows with color encoding magnitude
    # Direction = contribution direction, Color = magnitude (yellow=strong, purple=weak)
    _cmap_mu = plt.cm.plasma
    for i in range(len(_samples)):
        _contrib_norm = np.linalg.norm(_mu_contrib[i]) + 1e-8
        _unit_dir = _mu_contrib[i] / _contrib_norm
        _arrow_vec = _unit_dir * _arrow_length
        _color = _cmap_mu(_mu_mag_normalized[i])
        _ax_mu.annotate('',
            xy=(_samples[i, 0] + _arrow_vec[0], _samples[i, 1] + _arrow_vec[1]),
            xytext=(_samples[i, 0], _samples[i, 1]),
            arrowprops=dict(arrowstyle='->', color=_color, lw=2.5, alpha=0.9,
                           mutation_scale=18), zorder=6)

    # Add colorbar for mu contributions
    _sm_mu = plt.cm.ScalarMappable(cmap=_cmap_mu, norm=plt.Normalize(0, 1))
    _cbar_mu = plt.colorbar(_sm_mu, ax=_ax_mu)
    _cbar_mu.set_label('Influence strength (yellow=high)')

    # Draw mu as star
    _ax_mu.scatter([_mu[0]], [_mu[1]], c='yellow', s=400, marker='*',
                   edgecolors='black', linewidths=2, zorder=10)

    # Draw net mu update direction (big orange arrow)
    _net_mu_update = np.mean(_mu_contrib, axis=0)
    _net_arrow_length = _plot_range * 0.1
    _net_arrow_norm = np.linalg.norm(_net_mu_update) + 1e-8
    _net_arrow = (_net_mu_update / _net_arrow_norm) * _net_arrow_length
    _ax_mu.annotate('',
        xy=(_mu[0] + _net_arrow[0], _mu[1] + _net_arrow[1]),
        xytext=(_mu[0], _mu[1]),
        arrowprops=dict(arrowstyle='->', color='orange', lw=5, alpha=0.95,
                       mutation_scale=25), zorder=11)

    _ax_mu.set_xlim(_xlim)
    _ax_mu.set_ylim(_ylim)
    _ax_mu.set_xlabel('x')
    _ax_mu.set_ylabel('y')
    _ax_mu.set_title(r'$\mu$ contributions: arrow direction = pull direction, color = strength')
    _ax_mu.set_aspect('equal')

    # --- Right plot: Sigma contributions with landscape background ---
    _ax_sigma.contourf(_X_local, _Y_local, _Z_local, levels=25, cmap='viridis', alpha=0.6)

    # Compute directions and normalized magnitudes for sigma
    _directions = _samples - _mu
    _distances = np.sqrt(np.sum(_directions**2, axis=1))
    _unit_dirs = _directions / (_distances[:, None] + 1e-8)
    _sigma_mag_normalized = np.abs(_sigma_contrib) / (np.abs(_sigma_contrib).max() + 1e-8)

    # Draw samples as points
    _ax_sigma.scatter(_samples[:, 0], _samples[:, 1], c='white', s=60, alpha=0.9,
                      edgecolors='black', linewidths=1, zorder=5)

    # Draw uniform-length arrows: outward (expand) or inward (contract)
    # Color uses diverging colormap: red=expand (high influence), blue=contract (high influence)
    # Lighter colors = weaker influence
    for i in range(len(_samples)):
        # Direction: outward if positive (expand), inward if negative (contract)
        _sign = 1 if _sigma_contrib[i] > 0 else -1
        _arrow_vec = _unit_dirs[i] * _arrow_length * _sign

        # Color: red for expand, blue for contract, intensity by magnitude
        if _sigma_contrib[i] > 0:
            # Red with intensity based on magnitude
            _color = (1.0, 0.2 + 0.6 * (1 - _sigma_mag_normalized[i]), 0.2 + 0.6 * (1 - _sigma_mag_normalized[i]))
        else:
            # Blue with intensity based on magnitude
            _color = (0.2 + 0.6 * (1 - _sigma_mag_normalized[i]), 0.2 + 0.6 * (1 - _sigma_mag_normalized[i]), 1.0)

        _ax_sigma.annotate('',
            xy=(_samples[i, 0] + _arrow_vec[0], _samples[i, 1] + _arrow_vec[1]),
            xytext=(_samples[i, 0], _samples[i, 1]),
            arrowprops=dict(arrowstyle='->', color=_color, lw=2.5, alpha=0.9,
                           mutation_scale=18), zorder=6)

    # Draw mu as star
    _ax_sigma.scatter([_mu[0]], [_mu[1]], c='yellow', s=400, marker='*',
                      edgecolors='black', linewidths=2, zorder=10)

    # Draw reference circle at 1 sigma
    _circle = plt.Circle(_mu, _sigma, fill=False, color='white', linestyle='--',
                         linewidth=3, alpha=0.9, zorder=7)
    _ax_sigma.add_patch(_circle)

    _ax_sigma.set_xlim(_xlim)
    _ax_sigma.set_ylim(_ylim)
    _ax_sigma.set_xlabel('x')
    _ax_sigma.set_ylabel('y')
    _ax_sigma.set_title(r'$\sigma$ contributions: outward=expand, inward=contract, saturation=strength')
    _ax_sigma.set_aspect('equal')

    # Add legend for sigma plot
    _ax_sigma.scatter([], [], c='red', s=80, label='Expand (dark red=strong)')
    _ax_sigma.scatter([], [], c='blue', s=80, label='Contract (dark blue=strong)')
    _ax_sigma.legend(loc='upper right')

    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Per-Sample Contributions

    The charts above show **how each sample contributes** to the $\mu$ and $\sigma$ updates. All arrows have the **same length** — color encodes the magnitude of influence.

    - **Left ($\mu$ contributions)**: Arrow **direction** shows where each sample pulls $\mu$. Arrow **color** (plasma colormap) shows influence strength: yellow = strong influence, purple = weak. The orange arrow shows the net update direction.

    - **Right ($\sigma$ contributions)**: Arrows point **outward** (expand $\sigma$) or **inward** (contract $\sigma$). Color encodes both direction and strength: **dark red** = strong expand, **dark blue** = strong contract, lighter colors = weaker influence. The dashed circle shows the current 1-$\sigma$ radius.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What to Notice

    As you scrub through the iterations, watch for these patterns:

    1. **Early iterations**: Large $\sigma$ allows broad exploration of the landscape
    2. **Finding the basin**: When ES finds a promising region, $\sigma$ starts shrinking
    3. **Fine-tuning**: Small $\sigma$ enables precise convergence to the optimum
    4. **Getting stuck**: On deceptive functions (try Schwefel!), watch if $\sigma$ can re-expand to escape local traps

    The $\sigma$ plot on the right tells the story: descending curves mean exploitation, rising curves mean the algorithm is searching for better regions.
    """)
    return


@app.cell(hide_code=True)
def _(
    Ellipse,
    FUNCTIONS,
    compute_landscape,
    es_history,
    func_dropdown,
    np,
    plt,
):
    # Small multiples: 5 key snapshots
    _func_name = func_dropdown.value
    _func_info = FUNCTIONS[_func_name]
    _bounds = _func_info["bounds"]

    _n_snapshots = 5
    _indices = [int(i * (len(es_history) - 1) / (_n_snapshots - 1)) for i in range(_n_snapshots)]

    _fig, _axes = plt.subplots(1, _n_snapshots, figsize=(15, 3))

    # Use cached landscape grid (coarser for speed)
    _X, _Y, _Z = compute_landscape(_func_name, 80)

    for _ax, _idx in zip(_axes, _indices):
        _state = es_history[_idx]

        _ax.contourf(_X, _Y, _Z, levels=20, cmap='viridis', alpha=0.8)

        # Trajectory up to this point
        _traj = np.array([h['mu'] for h in es_history[:_idx+1]])
        if len(_traj) > 1:
            _ax.plot(_traj[:, 0], _traj[:, 1], 'r-', linewidth=1.5, alpha=0.7)

        # Search ellipse
        _ellipse = Ellipse(
            xy=_state['mu'],
            width=4*_state['sigma'],
            height=4*_state['sigma'],
            fill=True, alpha=0.3, color='red',
            edgecolor='red', linewidth=1.5
        )
        _ax.add_patch(_ellipse)

        # Current mean
        _ax.scatter([_state['mu'][0]], [_state['mu'][1]],
                    c='yellow', s=80, marker='*', edgecolors='red', linewidths=1, zorder=10)

        _ax.set_xlim(_bounds)
        _ax.set_ylim(_bounds)
        _ax.set_title(f'Iter {_idx}\nσ={_state["sigma"]:.2f}', fontsize=9)
        _ax.set_xticks([])
        _ax.set_yticks([])

    _fig.suptitle('Evolution of the Search Space', fontsize=12)
    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    1. **$\sigma$ is learned exploration/exploitation balance**: Unlike fixed cooling schedules in simulated annealing, ES learns when to explore vs exploit based on feedback.

    2. **The adaptation is local and reactive**: $\sigma$ responds to what worked *recently*, not a predetermined schedule.

    3. **Starting $\sigma$ matters**: Too small → trapped in local optima. Too large → slow convergence. But adaptive $\sigma$ can recover from poor initialization.

    4. **Different landscapes, different $\sigma$ dynamics**:
       - Rastrigin: gradual shrinking as ES narrows onto the global basin
       - Ackley: may need large $\sigma$ to escape flat outer regions
       - Schwefel: tests whether adaptation can escape deceptive gradients
       - Himmelblau: may converge to different optima from different starts

    ## What's Next?

    This notebook showed **isotropic ES** (same $\sigma$ in all directions). Real-world problems often benefit from **direction-dependent** search:

    - **CMA-ES**: Learns a full covariance matrix (ellipses that rotate and stretch)
    - **Natural Evolution Strategies**: Uses natural gradients for more stable updates
    - **Separable ES**: Independent $\sigma$ per dimension

    The core insight remains: **let the search space adapt to the problem**.
    """)
    return


if __name__ == "__main__":
    app.run()
