# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "altair",
#     "pandas",
#     "numpy==2.2.5",
#     "scipy",
#     "matplotlib",
#     "pillow",
#     "qrcode==8.2",
# ]
# ///

import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.Html('''
    <style>
        body, .marimo-container {
            margin: 0 !important;
            padding: 0 !important;
            height: 100vh;
            overflow: hidden;
        }

        .app-header {
            padding: 8px 16px;
            border-bottom: 1px solid #dee2e6;
            background-color: #fff;
        }

        .app-layout {
            display: flex;
            height: calc(100vh - 80px);
            align-items: flex-start;
            justify-content: center;
            gap: 2em;
            padding: 1em 0.5em;
        }

        .app-plot {
            flex: 1;
            min-width: 0;
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
            align-items: stretch;
            z-index: 1;
            overflow: hidden;
        }

        .app-plot img,
        .app-plot svg {
            max-width: 100%;
            height: auto;
        }

        .app-sidebar-container {
            z-index: 10;
            position: relative;
            flex-shrink: 0;
            width: 320px;
        }

        .app-sidebar {
            display: flex;
            flex-direction: column;
            gap: 1.5em;
            padding: 1.5em;
            background-color: #f8f9fa;
            border-radius: 8px;
            border: 1px solid #dee2e6;
            width: 100%;
        }

        @media (max-width: 768px) {
            .app-layout {
                flex-direction: column;
                height: auto;
                overflow-y: auto;
            }
            .app-plot {
                max-width: 100%;
                width: 100%;
            }
            .app-sidebar-container {
                width: 100%;
            }
            .app-sidebar {
                width: 100%;
            }
        }

        .app-sidebar h4 {
            margin: 1em 0 0.5em 0;
            font-size: 0.9em;
            color: #495057;
            border-bottom: 1px solid #dee2e6;
            padding-bottom: 0.3em;
        }

        .app-sidebar h4:first-child {
            margin-top: 0;
        }

        .dual-plot-container {
            display: flex;
            gap: 1em;
            width: 100%;
        }

        .dual-plot-container > div {
            flex: 1;
            min-width: 0;
        }
    </style>
    ''')
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import altair as alt
    import matplotlib.pyplot as plt
    from scipy.stats import multivariate_normal, norm

    return alt, mo, multivariate_normal, norm, np, pd, plt


@app.cell(hide_code=True)
def _():
    import qrcode
    import io
    import base64

    # Generate QR code
    qr = qrcode.QRCode(version=1, box_size=10, border=4)
    qr.add_data('https://sciml.warwick.ac.uk/wasm/bayesian-regression-demo/')
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")

    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    qr_base64 = base64.b64encode(buffer.read()).decode()
    return (qr_base64,)


@app.cell(hide_code=True)
def _(mo, qr_base64):
    header = mo.Html(f'''
    <div class="app-header">
        <div style="display: flex; justify-content: space-between; align-items: center; margin: 0; padding: 0;">
            <div>
                <p style='font-size: 24px; margin: 0; padding: 0; line-height: 1.3;'><b>Sequential Bayesian Linear Regression</b>
                <br><span style="font-size: 16px;"><i>Live demos:</i>
                <a href="https://sciml.warwick.ac.uk/" target="_blank" style="color: #0066cc; text-decoration: none;">sciml.warwick.ac.uk</a>
                &nbsp;&nbsp;|&nbsp;&nbsp;
                <i>Code:</i>
                <a href="https://github.com/kermodegroup/demos" target="_blank" style="color: #0066cc; text-decoration: none;">github.com/kermodegroup/demos</a>
                </span></p>
            </div>
            <img src="data:image/png;base64,{qr_base64}" alt="QR Code" style="width: 100px; height: 100px; flex-shrink: 0;" />
        </div>
    </div>
    ''')
    return (header,)


@app.cell(hide_code=True)
def _(mo):
    # Prior precision (1/variance of weights)
    alpha_slider = mo.ui.slider(0.1, 5.0, 0.1, 2.0, label='Prior prec. $\\alpha$')

    # Noise precision (1/variance of observations)
    beta_slider = mo.ui.slider(1.0, 50.0, 1.0, 25.0, label='Noise prec. $\\beta$')

    # Number of posterior samples to draw
    n_samples_slider = mo.ui.slider(0, 20, 1, 5, label='$N$ samples')

    # True weights
    w0_slider = mo.ui.slider(-1.0, 1.0, 0.1, -0.3, label='True $w_0$')
    w1_slider = mo.ui.slider(-1.0, 1.0, 0.1, 0.5, label='True $w_1$')

    # Random observations
    n_random_slider = mo.ui.slider(0, 50, 1, 0, label='$N$ random obs.', debounce=True)
    noise_slider = mo.ui.slider(0.05, 0.5, 0.05, 0.2, label='Noise $\\sigma$', debounce=True)
    seed_slider = mo.ui.slider(0, 100, 1, 42, label='Random seed', debounce=True)

    return alpha_slider, beta_slider, n_samples_slider, w0_slider, w1_slider, n_random_slider, noise_slider, seed_slider


@app.cell(hide_code=True)
def _(mo):
    # State for clicked observations: list of (x, y) tuples
    get_observations, set_observations = mo.state([])

    return get_observations, set_observations


@app.cell(hide_code=True)
def _(np):

    class LinearBasis:
        """phi(x) = [1, x] for linear model y = w0 + w1*x"""
        num_basis = 2
        def __call__(self, x):
            return np.array([1.0, x])

    def design_matrix(X, phi):
        """Build design matrix Phi[i,j] = phi_j(x_i)"""
        return np.vstack([phi(x) for x in X])

    def prior(alpha, num_basis):
        """Prior: N(0, (1/alpha)*I)"""
        m0 = np.zeros(num_basis)
        S0 = (1.0 / alpha) * np.eye(num_basis)
        return m0, S0

    def posterior(Phi, y, alpha, beta):
        """Posterior: N(m_N, S_N) where:
        S_N = (alpha*I + beta*Phi.T@Phi)^{-1}
        m_N = beta * S_N @ Phi.T @ y
        """
        S_N_inv = alpha * np.eye(Phi.shape[1]) + beta * Phi.T @ Phi
        S_N = np.linalg.inv(S_N_inv)
        m_N = beta * S_N @ Phi.T @ y
        return m_N.ravel(), S_N

    def posterior_predictive(Phi_test, m_N, S_N, beta):
        """Predictive: y = m_N.T @ phi(x), var = phi.T @ S_N @ phi + 1/beta"""
        y_mean = (Phi_test @ m_N).ravel()
        y_epi = np.sum((Phi_test @ S_N) * Phi_test, axis=1)  # epistemic variance
        y_var = y_epi + 1.0/beta  # total variance (epistemic + aleatoric)
        return y_mean, np.sqrt(y_epi), np.sqrt(y_var)

    return (
        LinearBasis, design_matrix, prior, posterior, posterior_predictive,
    )


@app.cell(hide_code=True)
def _(np, pd):
    # Static click grid for point selection in data space
    _gx = np.linspace(-1.2, 1.2, 80)
    _gy = np.linspace(-2, 2, 80)
    click_grid_df = pd.DataFrame(
        [(x, y) for x in _gx for y in _gy],
        columns=['x', 'y']
    )
    return (click_grid_df,)


@app.cell(hide_code=True)
def _(mo, set_observations, get_observations):
    def clear_observations(_):
        set_observations([])

    def undo_observation(_):
        current = get_observations() or []
        if current:
            set_observations(current[:-1])

    clear_button = mo.ui.button(label="Clear All", on_click=clear_observations)
    undo_button = mo.ui.button(label="Undo Last", on_click=undo_observation)

    return clear_button, undo_button


@app.cell(hide_code=True)
def _(
    alt, np, pd, mo, plt, multivariate_normal, norm,
    alpha_slider, beta_slider, n_samples_slider,
    w0_slider, w1_slider, n_random_slider, noise_slider, seed_slider,
    get_observations, click_grid_df,
    LinearBasis, design_matrix, prior, posterior, posterior_predictive,
):
    # Get parameters
    alpha = alpha_slider.value
    beta = beta_slider.value
    n_samples = n_samples_slider.value

    # True weights from sliders
    TRUE_W0 = w0_slider.value
    TRUE_W1 = w1_slider.value

    # Get clicked observations
    clicked_observations = get_observations() or []

    # Generate random observations
    n_random = n_random_slider.value
    noise_std = noise_slider.value
    seed = seed_slider.value

    np.random.seed(seed)
    if n_random > 0:
        X_random = np.random.uniform(-1.0, 1.0, n_random)
        y_random = TRUE_W0 + TRUE_W1 * X_random + np.random.normal(0, noise_std, n_random)
        random_observations = list(zip(X_random, y_random))
    else:
        random_observations = []

    # Combine all observations
    all_observations = random_observations + clicked_observations
    n_obs = len(all_observations)
    n_clicked = len(clicked_observations)

    # Set up basis
    phi = LinearBasis()

    # Fixed axis limits
    w_min, w_max = -1.0, 1.0  # weight space
    x_min, x_max = -1.2, 1.2  # data space x
    y_min, y_max = -2.0, 2.0  # data space y

    # Compute prior
    m0, S0 = prior(alpha, phi.num_basis)

    # Compute posterior if we have observations
    if n_obs > 0:
        obs_arr = np.array(all_observations)
        X_train = obs_arr[:, 0]
        y_train = obs_arr[:, 1]
        Phi_train = design_matrix(X_train, phi)
        m_N, S_N = posterior(Phi_train, y_train, alpha, beta)
    else:
        m_N, S_N = m0, S0

    # --- Weight Space Plot (Left) using matplotlib ---
    n_grid = 100
    w0_grid = np.linspace(w_min, w_max, n_grid)
    w1_grid = np.linspace(w_min, w_max, n_grid)
    W0, W1 = np.meshgrid(w0_grid, w1_grid)
    pos = np.dstack((W0, W1))

    # Evaluate Gaussian PDF
    rv = multivariate_normal(m_N, S_N)
    Z = rv.pdf(pos)

    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(Z, extent=[w_min, w_max, w_min, w_max], origin='lower',
                   cmap='Blues', aspect='equal', interpolation='bilinear')
    ax.plot(TRUE_W0, TRUE_W1, 'r+', markersize=15, markeredgewidth=3, label='True weights')
    ax.plot(m_N[0], m_N[1], 'bo', markersize=10, label='Posterior mean')
    ax.set_xlabel('$w_0$ (intercept)', fontsize=14)
    ax.set_ylabel('$w_1$ (slope)', fontsize=14)
    ax.set_title('Weight Space', fontsize=18)
    ax.set_xlim(w_min, w_max)
    ax.set_ylim(w_min, w_max)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.tick_params(labelsize=12)
    plt.tight_layout()
    weight_chart = fig

    # --- Data Space Plot (Right) ---
    x_scale = alt.Scale(domain=[x_min, x_max])
    y_scale = alt.Scale(domain=[y_min, y_max])

    # Test points for plotting
    X_test = np.linspace(x_min, x_max, 200)
    Phi_test = design_matrix(X_test, phi)

    # Compute predictive distribution
    y_mean, y_std_epi, y_std_tot = posterior_predictive(Phi_test, m_N, S_N, beta)

    # Ground truth line
    y_true = TRUE_W0 + TRUE_W1 * X_test

    # --- Test metrics (comparing posterior predictive to true function on test grid) ---
    # RMSE: Root Mean Squared Error
    rmse_test = np.sqrt(np.mean((y_mean - y_true)**2))

    # MAE: Mean Absolute Error
    mae_test = np.mean(np.abs(y_mean - y_true))

    # CRPS: Continuous Ranked Probability Score for Gaussian
    # CRPS(N(μ,σ²), y) = σ * [z*(2*Φ(z) - 1) + 2*φ(z) - 1/√π]
    z_test = (y_true - y_mean) / y_std_tot
    crps_test = np.mean(y_std_tot * (z_test * (2 * norm.cdf(z_test) - 1) + 2 * norm.pdf(z_test) - 1 / np.sqrt(np.pi)))

    # Log likelihood: log p(y | μ, σ) = -0.5*log(2π) - log(σ) - 0.5*((y-μ)/σ)²
    ll_test = np.mean(-0.5 * np.log(2 * np.pi) - np.log(y_std_tot) - 0.5 * z_test**2)

    # --- Training metrics (comparing predictions at observation locations to observations) ---
    if n_obs > 0:
        # Compute predictive distribution at training points
        y_train_mean, y_train_std_epi, y_train_std_tot = posterior_predictive(Phi_train, m_N, S_N, beta)

        rmse_train = np.sqrt(np.mean((y_train_mean - y_train)**2))
        mae_train = np.mean(np.abs(y_train_mean - y_train))

        z_train = (y_train - y_train_mean) / y_train_std_tot
        crps_train = np.mean(y_train_std_tot * (z_train * (2 * norm.cdf(z_train) - 1) + 2 * norm.pdf(z_train) - 1 / np.sqrt(np.pi)))
        ll_train = np.mean(-0.5 * np.log(2 * np.pi) - np.log(y_train_std_tot) - 0.5 * z_train**2)
    else:
        rmse_train = np.nan
        mae_train = np.nan
        crps_train = np.nan
        ll_train = np.nan

    gt_df = pd.DataFrame({'x': X_test, 'y': y_true})
    gt_line = alt.Chart(gt_df).mark_line(
        color='black', strokeWidth=2, strokeDash=[5, 5], opacity=0.7
    ).encode(
        x=alt.X('x:Q', scale=x_scale, title='x'),
        y=alt.Y('y:Q', scale=y_scale, title='y')
    )

    # Uncertainty bands
    band_df = pd.DataFrame({
        'x': X_test,
        'y_mean': y_mean,
        'y_lower_ep': y_mean - 2 * y_std_epi,
        'y_upper_ep': y_mean + 2 * y_std_epi,
        'y_lower_tot': y_mean - 2 * y_std_tot,
        'y_upper_tot': y_mean + 2 * y_std_tot,
    })

    # Total uncertainty band (outer, lighter - includes aleatoric)
    total_band = alt.Chart(band_df).mark_area(
        opacity=0.15, color='#ff7f0e'
    ).encode(
        x=alt.X('x:Q', scale=x_scale),
        y=alt.Y('y_lower_tot:Q', scale=y_scale),
        y2='y_upper_tot:Q'
    )

    # Epistemic uncertainty band (inner, darker)
    epistemic_band = alt.Chart(band_df).mark_area(
        opacity=0.3, color='#1f77b4'
    ).encode(
        x=alt.X('x:Q', scale=x_scale),
        y=alt.Y('y_lower_ep:Q', scale=y_scale),
        y2='y_upper_ep:Q'
    )

    # Posterior mean line
    mean_df = pd.DataFrame({'x': X_test, 'y': y_mean})
    mean_line = alt.Chart(mean_df).mark_line(
        color='#1f77b4', strokeWidth=3
    ).encode(
        x=alt.X('x:Q', scale=x_scale),
        y=alt.Y('y:Q', scale=y_scale)
    )

    # Posterior samples
    samples_data = []
    if n_samples > 0:
        # Add jitter to S_N for numerical stability
        S_N_jitter = S_N + 1e-6 * np.eye(phi.num_basis)
        np.random.seed(42)
        w_samples = np.random.multivariate_normal(m_N, S_N_jitter, n_samples)
        for i, w in enumerate(w_samples):
            y_sample = w[0] + w[1] * X_test
            for x_val, y_val in zip(X_test, y_sample):
                samples_data.append({'x': x_val, 'y': y_val, 'sample': f'Sample {i+1}'})

    samples_df = pd.DataFrame(samples_data) if samples_data else pd.DataFrame(columns=['x', 'y', 'sample'])

    # Build layers list
    layers = [total_band, epistemic_band]

    if len(samples_df) > 0:
        samples_layer = alt.Chart(samples_df).mark_line(
            strokeWidth=1.5, opacity=0.5
        ).encode(
            x=alt.X('x:Q', scale=x_scale),
            y=alt.Y('y:Q', scale=y_scale),
            color=alt.Color('sample:N', legend=None)
        )
        layers.append(samples_layer)

    layers.extend([mean_line, gt_line])

    # Random observations (blue)
    if len(random_observations) > 0:
        random_df = pd.DataFrame(random_observations, columns=['x', 'y'])
        random_points = alt.Chart(random_df).mark_circle(
            color='#1f77b4', size=120, opacity=0.8
        ).encode(
            x=alt.X('x:Q', scale=x_scale),
            y=alt.Y('y:Q', scale=y_scale)
        )
        layers.append(random_points)

    # Clicked observations (red)
    if len(clicked_observations) > 0:
        clicked_df = pd.DataFrame(clicked_observations, columns=['x', 'y'])
        clicked_points = alt.Chart(clicked_df).mark_circle(
            color='#d62728', size=150, opacity=0.9
        ).encode(
            x=alt.X('x:Q', scale=x_scale),
            y=alt.Y('y:Q', scale=y_scale)
        )
        layers.append(clicked_points)

    # Click selection for adding points
    click_select = alt.selection_point(on='click', nearest=True, fields=['x', 'y'], name='click_select')

    # Invisible click layer (must be on top for interaction)
    click_layer = alt.Chart(click_grid_df).mark_point(
        opacity=0, size=100
    ).encode(
        x=alt.X('x:Q', scale=x_scale),
        y=alt.Y('y:Q', scale=y_scale)
    ).add_params(click_select)
    layers.append(click_layer)

    # Combine layers
    data_chart = alt.layer(*layers).properties(
        width='container', height=350,
        title='Data Space'
    ).configure_axis(
        grid=True, gridOpacity=0.3,
        labelFontSize=14, titleFontSize=16
    ).configure_title(fontSize=18)

    interactive_chart = mo.ui.altair_chart(data_chart)

    return interactive_chart, weight_chart, n_obs, n_clicked, n_random, TRUE_W0, TRUE_W1, rmse_train, mae_train, crps_train, ll_train, rmse_test, mae_test, crps_test, ll_test


@app.cell(hide_code=True)
def _(interactive_chart, weight_chart):
    # Pass-through display cell
    data_chart_display = interactive_chart
    weight_chart_display = weight_chart
    return data_chart_display, weight_chart_display


@app.cell(hide_code=True)
def _(interactive_chart, click_grid_df, get_observations, set_observations, pd):
    # Click handler - reads selection, updates state
    _current = get_observations() or []
    _filtered = interactive_chart.apply_selection(click_grid_df)

    # Check if valid DataFrame
    if _filtered is not None and isinstance(_filtered, pd.DataFrame) and len(_filtered) > 0 and len(_filtered) < len(click_grid_df):
        # User clicked on a point
        _new_x = float(_filtered['x'].iloc[0])
        _new_y = float(_filtered['y'].iloc[0])
        _new_point = (_new_x, _new_y)
        # Add the new point
        set_observations(_current + [_new_point])

    return ()


@app.cell(hide_code=True)
def _(
    mo,
    alpha_slider, beta_slider, n_samples_slider,
    w0_slider, w1_slider, n_random_slider, noise_slider, seed_slider,
    clear_button, undo_button,
    n_obs, n_clicked, n_random,
):
    # Ground truth section
    truth_section = mo.vstack([
        mo.Html("<h4>Ground Truth</h4>"),
        w0_slider,
        w1_slider,
        mo.Html("<small><span style='color: red;'>&#10010;</span> = true weights</small>"),
    ], gap="0.3em")

    # Prior section
    prior_section = mo.vstack([
        mo.Html("<h4>Prior</h4>"),
        alpha_slider,
    ], gap="0.3em")

    # Noise section
    noise_section = mo.vstack([
        mo.Html("<h4>Likelihood</h4>"),
        beta_slider,
    ], gap="0.3em")

    # Random data section
    random_section = mo.vstack([
        mo.Html("<h4>Random Data</h4>"),
        n_random_slider,
        noise_slider,
        seed_slider,
    ], gap="0.3em")

    # Click data section
    click_section = mo.vstack([
        mo.Html("<h4>Click to Add</h4>"),
        mo.hstack([clear_button, undo_button], gap="0.5em"),
        mo.Html(f"<small>Random: {n_random} | Clicked: {n_clicked} | Total: {n_obs}</small>"),
    ], gap="0.3em")

    # Sampling section
    sampling_section = mo.vstack([
        mo.Html("<h4>Visualization</h4>"),
        n_samples_slider,
        mo.Html("<small><span style='color: blue;'>&#9679;</span> = posterior mean</small>"),
    ], gap="0.3em")

    sidebar = mo.vstack([truth_section, prior_section, noise_section, random_section, click_section, sampling_section], gap="1em")

    sidebar_html = mo.Html(f'''
    <div class="app-sidebar">
        {sidebar}
        <p style="font-size: 0.85em; color: #666; margin-top: 1em;">
            <b>Tip:</b> Click on the right plot to add observations (red). Random data shown in blue.
        </p>
    </div>
    ''')
    return (sidebar_html,)


@app.cell(hide_code=True)
def _(mo, np, rmse_train, mae_train, crps_train, ll_train, rmse_test, mae_test, crps_test, ll_test):
    # Format value, handling NaN for when no training data
    def fmt(v):
        return '—' if np.isnan(v) else f'{v:.4f}'

    # Metrics table as custom HTML with train and test columns
    metrics_table = mo.Html(f'''
    <table style="width: 100%; border-collapse: collapse; font-size: 13px;">
        <thead>
            <tr style="border-bottom: 2px solid #dee2e6;">
                <th style="text-align: left; padding: 6px 8px;">Metric</th>
                <th style="text-align: right; padding: 6px 8px;">Train</th>
                <th style="text-align: right; padding: 6px 8px;">Test</th>
            </tr>
        </thead>
        <tbody>
            <tr style="border-bottom: 1px solid #dee2e6;">
                <td style="padding: 6px 8px;">RMSE</td>
                <td style="text-align: right; padding: 6px 8px; font-family: monospace;">{fmt(rmse_train)}</td>
                <td style="text-align: right; padding: 6px 8px; font-family: monospace;">{fmt(rmse_test)}</td>
            </tr>
            <tr style="border-bottom: 1px solid #dee2e6;">
                <td style="padding: 6px 8px;">MAE</td>
                <td style="text-align: right; padding: 6px 8px; font-family: monospace;">{fmt(mae_train)}</td>
                <td style="text-align: right; padding: 6px 8px; font-family: monospace;">{fmt(mae_test)}</td>
            </tr>
            <tr style="border-bottom: 1px solid #dee2e6;">
                <td style="padding: 6px 8px;">CRPS</td>
                <td style="text-align: right; padding: 6px 8px; font-family: monospace;">{fmt(crps_train)}</td>
                <td style="text-align: right; padding: 6px 8px; font-family: monospace;">{fmt(crps_test)}</td>
            </tr>
            <tr>
                <td style="padding: 6px 8px;">Log Lik.</td>
                <td style="text-align: right; padding: 6px 8px; font-family: monospace;">{fmt(ll_train)}</td>
                <td style="text-align: right; padding: 6px 8px; font-family: monospace;">{fmt(ll_test)}</td>
            </tr>
        </tbody>
    </table>
    ''')
    return (metrics_table,)


@app.cell(hide_code=True)
def _(mo, header, data_chart_display, weight_chart_display, sidebar_html, metrics_table):
    # Combined layout: header on top, two plots side by side, controls on right
    mo.Html(f'''
    {header}
    <div class="app-layout">
        <div class="app-plot">
            <div class="dual-plot-container">
                <div>{mo.as_html(weight_chart_display)}</div>
                <div>{mo.as_html(data_chart_display)}</div>
            </div>
        </div>
        <div class="app-sidebar-container">
            {sidebar_html}
            <div style="margin-top: 1em; padding: 1em; background-color: #f8f9fa; border-radius: 8px; border: 1px solid #dee2e6;">
                <h4 style="margin: 0 0 0.5em 0; font-size: 14px; color: #495057;">Metrics</h4>
                {mo.as_html(metrics_table)}
            </div>
        </div>
    </div>
    ''')
    return


if __name__ == "__main__":
    app.run()
