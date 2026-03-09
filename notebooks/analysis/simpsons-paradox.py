# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "arviz==0.23.4",
#     "graphviz==0.21",
#     "marimo>=0.20.4",
#     "matplotlib==3.10.8",
#     "numpy==2.4.2",
#     "pandas==3.0.1",
#     "pymc==5.28.1",
#     "seaborn==0.13.2",
#     "xarray==2026.2.0",
# ]
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App(
    auto_download=["html"],
)


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Simpson's paradox

    > This notebook was originally found on the PyMC docs [here](https://www.pymc.io/projects/examples/en/latest/causal_inference/GLM-simpsons-paradox.html) written by Benjamin T. Vincent, [DOI](https://zenodo.org/records/18677308). It was adapted for marimo.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [Simpson's Paradox](https://en.wikipedia.org/wiki/Simpson%27s_paradox) describes a situation where there might be a negative relationship between two variables within a group, but when data from multiple groups are combined, that relationship may disappear or even reverse sign. The gif below (from the Simpson's Paradox [Wikipedia](https://en.wikipedia.org/wiki/Simpson%27s_paradox) page) demonstrates this very nicely.

    ![](https://upload.wikimedia.org/wikipedia/commons/f/fb/Simpsons_paradox_-_animation.gif)

    Another way of describing this is that we wish to estimate the causal relationship $x \rightarrow y$. The seemingly obvious approach of modelling `y ~ 1 + x` will lead us to conclude (in the situation above) that increasing $x$ causes $y$ to decrease (see Model 1 below). However, the relationship between $x$ and $y$ is confounded by a group membership variable $group$. This group membership variable is not included in the model, and so the relationship between $x$ and $y$ is biased. If we now factor in the influence of $group$, in some situations (e.g. the image above) this can lead us to completely reverse the sign of our estimate of $x \rightarrow y$, now estimating that increasing $x$ causes $y$ to _increase_.

    In short, this 'paradox' (or simply omitted variable bias) can be resolved by assuming a causal DAG which includes how the main predictor variable _and_ group membership (the confounding variable) influence the outcome variable. We demonstrate an example where we _don't_ incorporate group membership (so our causal DAG is wrong, or in other words our model is misspecified; Model 1). We then show 2 ways to resolve this by including group membership as causal influence upon $x$ and $y$. This is shown in an unpooled model (Model 2) and a hierarchical model (Model 3).
    """)
    return


@app.cell
def _():
    import arviz as az
    import graphviz as gr
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import pymc as pm
    import seaborn as sns
    import xarray as xr

    return az, gr, np, pd, plt, pm, sns, xr


@app.cell
def _(az, np, plt):
    az.style.use("arviz-darkgrid")
    figsize = [12, 4]
    plt.rcParams["figure.figsize"] = figsize
    rng = np.random.default_rng(1234)
    return figsize, rng


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Generate data

    This data generation was influenced by this [stackexchange](https://stats.stackexchange.com/questions/479201/understanding-simpsons-paradox-with-random-effects) question. It will comprise observations from $G=5$ groups. The data is constructed such that there is a negative relationship between $x$ and $y$ within each group, but when all groups are combined, the relationship is positive.
    """)
    return


@app.cell
def _(np, pd, rng):
    def generate():
        group_list = ['one', 'two', 'three', 'four', 'five']
        trials_per_group = 20
        group_intercepts = rng.normal(0, 1, len(group_list))
        group_slopes = np.ones(len(group_list)) * -0.5
        group_mx = group_intercepts * 2
        group = np.repeat(group_list, trials_per_group)
        subject = np.concatenate([np.ones(trials_per_group) * i for i in np.arange(len(group_list))]).astype(int)
        intercept = np.repeat(group_intercepts, trials_per_group)
        slope = np.repeat(group_slopes, trials_per_group)
        mx = np.repeat(group_mx, trials_per_group)
        _x = rng.normal(mx, 1)
        y = rng.normal(intercept + (_x - mx) * slope, 1)
        data = pd.DataFrame({'group': group, 'group_idx': subject, 'x': _x, 'y': y})
        return (data, group_list)
    data, group_list = generate()
    return data, group_list


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To follow along, it is useful to clearly understand the form of the data. This is [long form](https://en.wikipedia.org/wiki/Wide_and_narrow_data) data (also known as narrow data) in that each row represents one observation. We have a `group` column which has the group label, and an accompanying numerical `group_idx` column. This is very useful when it comes to modelling as we can use it as an index to look up group-level parameter estimates. Finally, we have our core observations of the predictor variable `x` and the outcome `y`.
    """)
    return


@app.cell
def _(data):
    data
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And we can visualise this as below.
    """)
    return


@app.cell
def _(data, plt, sns):
    fig, _ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=data, x='x', y='y', hue='group', ax=_ax)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The rest of the notebook will cover different ways that we can analyse this data using linear models.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model 1: Pooled regression

    First we examine the simplest model - plain linear regression which pools all the data and has no knowledge of the group/multi-level structure of the data.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    From a causal perspective, this approach embodies the belief that $x$ causes $y$ and that this relationship is constant across all groups, or groups are simply not considered. This can be shown in the causal DAG below.
    """)
    return


@app.cell
def _(gr):
    # Cell tags: hide-input
    g = gr.Digraph()
    g.node(name="x", label="x")
    g.node(name="y", label="y")
    g.edge(tail_name="x", head_name="y")
    g
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We could describe this model mathematically as:

    $$
    \begin{aligned}
    \beta_0, \beta_1 &\sim \text{Normal}(0, 5) \\
    \sigma &\sim \text{Gamma}(2, 2) \\
    \mu_i &= \beta_0 + \beta_1 x_i \\
    y_i &\sim \text{Normal}(\mu_i, \sigma)
    \end{aligned}
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can also express Model 1 in Wilkinson notation as `y ~ 1 + x` which is equivalent to `y ~ x` as the intercept is included by default.

    * The `1` term corresponds to the intercept term $\beta_0$.
    * The `x` term corresponds to the slope term $\beta_1$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    So now we can express this as a PyMC model. We can notice how closely the model syntax mirrors the mathematical notation above.
    """)
    return


@app.cell
def _(data, pm):
    with pm.Model() as model1:
        _β0 = pm.Normal('β0', 0, sigma=5)
        _β1 = pm.Normal('β1', 0, sigma=5)
        _sigma = pm.Gamma('sigma', 2, 2)
        _x = pm.Data('x', data.x, dims='obs_id')
        _μ = pm.Deterministic('μ', _β0 + _β1 * _x, dims='obs_id')
        pm.Normal('y', mu=_μ, sigma=_sigma, observed=data.y, dims='obs_id')
    return (model1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And we can visualize the DAG which can be a useful way to check that our model is correctly specified.
    """)
    return


@app.cell
def _(model1, pm):
    pm.model_to_graphviz(model1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Conduct inference
    """)
    return


@app.cell
def _(model1, pm, rng):
    with model1:
        idata1 = pm.sample(random_seed=rng)
    return (idata1,)


@app.cell
def _(az, idata1):
    az.plot_trace(idata1, var_names=["~μ"]);
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Visualisation

    First we'll define a handy predict function which will do out of sample predictions for us. This will be handy when it comes to visualising the model fits.
    """)
    return


@app.cell
def _(az, pm, rng):
    def predict(model: pm.Model, idata: az.InferenceData, predict_at: dict) -> az.InferenceData:
        """Do posterior predictive inference at a set of out of sample points specified by `predict_at`."""
        with model:
            pm.set_data(predict_at)
            idata.extend(pm.sample_posterior_predictive(idata, var_names=["y", "μ"], random_seed=rng))
        return idata

    return (predict,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And now let's use that `predict` function to do out of sample predictions which we will use for visualisation.
    """)
    return


@app.cell
def _(data, idata1, model1, np, predict):
    # Cell tags: hide-output
    xi = np.linspace(data.x.min(), data.x.max(), 20)
    idata1_1 = predict(model=model1, idata=idata1, predict_at={'x': xi})
    return idata1_1, xi


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Finally, we can now visualise the model fit to data, and our posterior in parameter space.
    """)
    return


@app.cell
def _(az, data, idata1_1, plt, xi, xr):
    _fig, _ax = plt.subplots(figsize=(8, 6))

    def plot_band(xi, var: xr.DataArray, ax, color: str):
        _ax.plot(xi, var.mean(['chain', 'draw']), color=color)
        az.plot_hdi(xi, var, hdi_prob=0.6, color=color, fill_kwargs={'alpha': 0.2, 'linewidth': 0}, ax=_ax)
        az.plot_hdi(xi, var, hdi_prob=0.95, color=color, fill_kwargs={'alpha': 0.2, 'linewidth': 0}, ax=_ax)

    def _plot(idata: az.InferenceData):
        fig, _ax = plt.subplots(1, 3, figsize=(12, 4))
        _ax[0].scatter(data.x, data.y, color='k')
        plot_band(xi, idata.posterior_predictive.μ, ax=_ax[0], color='k')
        _ax[0].set(xlabel='x', ylabel='y', title='Conditional mean')
        _ax[1].scatter(data.x, data.y, color='k')
        plot_band(xi, idata.posterior_predictive.y, ax=_ax[1], color='k')
        _ax[1].set(xlabel='x', ylabel='y', title='Posterior predictive distribution')
        _ax[2].scatter(az.extract(idata, var_names=['β1']), az.extract(idata, var_names=['β0']), color='k', alpha=0.01, rasterized=True)
        _ax[2].set(xlabel='slope', ylabel='intercept', title='Parameter space')
        _ax[2].axhline(y=0, c='k')
        _ax[2].axvline(x=0, c='k')


    _plot(idata1_1)
    _ax
    return (plot_band,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The plot on the left shows the data and the posterior of the **conditional mean**. For a given $x$, we get a posterior distribution of the model (i.e. of $\mu$).

    The plot in the middle shows the conditional **posterior predictive distribution**, which gives a statement about the data we expect to see. Intuitively, this can be understood as not only incorporating what we know of the model (left plot) but also what we know about the distribution of error.

    The plot on the right shows our posterior beliefs in **parameter space**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    One of the clear things about this analysis is that we have credible evidence that $x$ and $y$ are _positively_ correlated. We can see this from the posterior over the slope (see right hand panel in the figure above) which we isolate in the plot below.
    """)
    return


@app.cell
def _(az, idata1_1):
    # Cell tags: hide-input
    _ax = az.plot_posterior(idata1_1.posterior['β1'], ref_val=0)
    _ax.set(title='Model 1 strongly suggests a positive slope', xlabel='$\\beta_1$')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model 2: Unpooled regression with counfounder included

    We will use the same data in this analysis, but this time we will use our knowledge that data come from groups. From a causal perspective we are exploring the notion that both $x$ and $y$ are influenced by group membership. This can be shown in the causal directed acyclic graph ([DAG](https://en.wikipedia.org/wiki/Directed_acyclic_graph)) below.
    """)
    return


@app.cell
def _(gr):
    # Cell tags: hide-input
    g_1 = gr.Digraph()
    g_1.node(name='x', label='x')
    g_1.node(name='g', label='group')
    g_1.node(name='y', label='y')
    g_1.edge(tail_name='x', head_name='y')
    g_1.edge(tail_name='g', head_name='x')
    g_1.edge(tail_name='g', head_name='y')
    g_1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    So we can see that $group$ is a [confounding variable](https://en.wikipedia.org/wiki/Confounding). So if we are trying to discover the causal relationship of $x$ on $y$, we need to account for the confounding variable $group$. Model 1 did not do this and so arrived at the wrong conclusion. But Model 2 will account for this.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    More specifically we will essentially fit independent regressions to data within each group. This could also be described as an unpooled model. We could describe this model mathematically as:

    $$
    \begin{aligned}
    \vec{\beta_0}, \vec{\beta_1} &\sim \text{Normal}(0, 5) \\
    \sigma &\sim \text{Gamma}(2, 2) \\
    \mu_i &= \vec{\beta_0}[g_i] + \vec{\beta_1}[g_i] x_i \\
    y_i &\sim \text{Normal}(\mu_i, g_i)
    \end{aligned}
    $$

    Where $g_i$ is the group index for observation $i$. So the parameters $\vec{\beta_0}$ and $\vec{\beta_1}$ are now length $G$ vectors, not scalars. And the $[g_i]$ acts as an index to look up the group for the $i^\text{th}$ observation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    :::{note}
    We can also express this Model 2 in Wilkinson notation as `y ~ 0 + g + x:g`.

    * The `g` term captures the group specific intercept $\beta_0[g_i]$ parameters.
    * The `0` means we do not have a global intercept term, leaving the group specific intercepts to be the only intercepts.
    * The `x:g` term captures group specific slope $\beta_1[g_i]$ parameters.
    :::
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's express Model 2 with PyMC code.
    """)
    return


@app.cell
def _(data, group_list, pm):
    coords = {'group': group_list}
    with pm.Model(coords=coords) as model2:
        _β0 = pm.Normal('β0', 0, sigma=5, dims='group')
        _β1 = pm.Normal('β1', 0, sigma=5, dims='group')
        _sigma = pm.Gamma('sigma', 2, 2)
        _x = pm.Data('x', data.x, dims='obs_id')
        g_2 = pm.Data('g', data.group_idx, dims='obs_id')
        _μ = pm.Deterministic('μ', _β0[g_2] + _β1[g_2] * _x, dims='obs_id')
        pm.Normal('y', mu=_μ, sigma=_sigma, observed=data.y, dims='obs_id')
    return coords, model2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    By plotting the DAG for this model it is clear to see that we now have individual intercept and slope parameters for each of the groups.
    """)
    return


@app.cell
def _(model2, pm):
    pm.model_to_graphviz(model2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Conduct inference
    """)
    return


@app.cell
def _(az, model2, pm, rng):
    with model2:
        idata2 = pm.sample(random_seed=rng)

    az.plot_trace(idata2, var_names=["~μ"]);
    return (idata2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Visualisation
    """)
    return


@app.cell
def _(data, np):
    _n_points = 10
    _n_groups = len(data.group.unique())
    xi_1 = np.concatenate([np.linspace(group[1].x.min(), group[1].x.max(), _n_points) for group in data.groupby('group_idx')])
    g_3 = np.concatenate([[i] * _n_points for i in range(_n_groups)]).astype(int)
    predict_at = {'x': xi_1, 'g': g_3}
    return g_3, predict_at, xi_1


@app.cell
def _(idata2, model2, predict, predict_at):
    # Cell tags: hide-output
    idata2_1 = predict(model=model2, idata=idata2, predict_at=predict_at)
    return (idata2_1,)


@app.cell
def _(az, data, g_3, group_list, idata2_1, plot_band, plt, xi_1):
    # Cell tags: hide-input
    def _plot(idata):
        fig, _ax = plt.subplots(1, 3, figsize=(12, 4))
        for i in range(len(group_list)):
            _ax[0].scatter(data.x[data.group_idx == i], data.y[data.group_idx == i], color=f'C{i}')
            plot_band(xi_1[g_3 == i], idata.posterior_predictive.μ.isel(obs_id=g_3 == i), ax=_ax[0], color=f'C{i}')
            _ax[1].scatter(data.x[data.group_idx == i], data.y[data.group_idx == i], color=f'C{i}')
            plot_band(xi_1[g_3 == i], idata.posterior_predictive.y.isel(obs_id=g_3 == i), ax=_ax[1], color=f'C{i}')
        _ax[0].set(xlabel='x', ylabel='y', title='Conditional mean')
        _ax[1].set(xlabel='x', ylabel='y', title='Posterior predictive distribution')
        for i, _ in enumerate(group_list):
            _ax[2].scatter(az.extract(idata, var_names='β1')[i, :], az.extract(idata, var_names='β0')[i, :], color=f'C{i}', alpha=0.01, rasterized=True, zorder=2)
        _ax[2].set(xlabel='slope', ylabel='intercept', title='Parameter space')
        _ax[2].axhline(y=0, c='k')
        _ax[2].axvline(x=0, c='k')
        return _ax
    _plot(idata2_1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In contrast to Model 1, when we consider groups we can see that now the evidence points toward _negative_ relationships between $x$ and $y$. We can see that from the negative slopes in the left and middle panels of the plot above, as well as from the majority of the posterior samples for the slope parameter being negative in the left panel above.

    The plot below takes a closer look at the group level slope parameters.
    """)
    return


@app.cell
def _(az, figsize, idata2_1):
    # Cell tags: hide-input
    _ax = az.plot_forest(idata2_1.posterior['β1'], combined=True, figsize=figsize)
    _ax[0].set(title='Model 2 suggests negative slopes for each group', xlabel='$\\beta_1$', ylabel='Group')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model 3: Partial pooling model with confounder included

    Model 3 assumes the same causal DAG as model 2 (see above). However, we can go further and incorporate more knowledge about the structure of our data. Rather than treating each group as entirely independent, we can use our knowledge that these groups are drawn from a population-level distribution. We could formalise this as saying that the group-level slopes and intercepts are modeled as deflections from a population-level slope and intercept, respectively.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And we could describe this model mathematically as:

    $$
      \begin{aligned}
      \beta_0 &\sim \text{Normal}(0, 5) \\
      \beta_1 &\sim \text{Normal}(0, 5) \\
      p_{0\sigma}, p_{1\sigma} &\sim \text{Gamma}(2, 2) \\
      \vec{u_0} &\sim \text{Normal}(0, p_{0\sigma}) \\
      \vec{u_1} &\sim \text{Normal}(0, p_{1\sigma}) \\
      \sigma &\sim \text{Gamma}(2, 2) \\
      \mu_i &= \overset{\text{intercept}}{
            \big(
                \underset{\text{pop}}{\beta_0}
                \; {+} \; \underset{\text{group}}{\vec{u_0}{[g_i]}}
            \big)
          }
          \; {+} \; \overset{\text{slope}}{
            \big(
                \underset{\text{pop}}{\beta_1 \cdot x_i}
                \; {+} \; \underset{\text{group}}{\vec{u_1}{[g_i]} \cdot x_i}
            \big)
          } \\ \\
      y_i &\sim \text{Normal}(\mu_i, \sigma)
      \end{aligned}
    $$


    where
    * $\beta_0$ and $\beta_1$ are the population level intercepts and slopes, respectively.
    * $\vec{u_0}$ and $\vec{u_1}$ are group level deflections from the population level parameters.
    * $p_{0\sigma}, p_{1\sigma}$ are the standard deviations of the intercept and slope deflections and can be thought of as 'shrinkage parameters'.
      * In the limt of $p_{0\sigma}, p_{1\sigma} \rightarrow \infty$ we recover the unpooled model.
      * In the limit of $p_{0\sigma}, p_{1\sigma} \rightarrow 0$ we recover the pooled model.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can also express this Model 3 in Wilkinson notation as `1 + x + (1 + x | g)`.

    * The `1` captures the global intercept, $\beta_0$.
    * The `x` captures the global slope, $\beta_1$.
    * The `(1 + x | g)` term captures group specific terms for the intercept and slope.
      * `1 | g` captures the group specific intercept deflections $\vec{u_0}$ parameters.
      * `x | g` captures the group specific slope deflections $\vec{u_1}[g_i]$ parameters.
    """)
    return


@app.cell
def _(coords, data, pm):
    with pm.Model(coords=coords) as model3:
        _β0 = pm.Normal('β0', 0, 5)
        _β1 = pm.Normal('β1', 0, 5)
        intercept_sigma = pm.Gamma('intercept_sigma', 2, 2)
        slope_sigma = pm.Gamma('slope_sigma', 2, 2)
        u0 = pm.Normal('u0', 0, intercept_sigma, dims='group')
        u1 = pm.Normal('u1', 0, slope_sigma, dims='group')
        _sigma = pm.Gamma('sigma', 2, 2)
        _x = pm.Data('x', data.x, dims='obs_id')
        g_4 = pm.Data('g', data.group_idx, dims='obs_id')
        _μ = pm.Deterministic('μ', _β0 + u0[g_4] + (_β1 * _x + u1[g_4] * _x), dims='obs_id')
        pm.Normal('y', mu=_μ, sigma=_sigma, observed=data.y, dims='obs_id')
    return (model3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The DAG of this model highlights the scalar population level parameters $\beta_0$ and $\beta_1$ and the group level parameters $\vec{u_0}$ and $\vec{u_1}$.
    """)
    return


@app.cell
def _(model3, pm):
    pm.model_to_graphviz(model3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For the sake of completeness, there is another equivalent way to write this model.

    $$
    \begin{aligned}
    p_{0\mu}, p_{1\mu} &\sim \text{Normal}(0, 5) \\
    p_{0\sigma}, p_{1\sigma} &\sim \text{Gamma}(2, 2) \\
    \vec{\beta_0} &\sim \text{Normal}(p_{0\mu}, p_{0\sigma}) \\
    \vec{\beta_1} &\sim \text{Normal}(p_{1\mu}, p_{1\sigma}) \\
    \sigma &\sim \text{Gamma}(2, 2) \\
    \mu_i &= \vec{\beta_0}[g_i] +  \vec{\beta_1}[g_i] \cdot x_i \\
    y_i &\sim \text{Normal}(\mu_i, \sigma)
    \end{aligned}
    $$

    where $\vec{\beta_0}$ and $\vec{\beta_1}$ are the group-level parameters. These group level parameters can be thought of as being sampled from population level intercept distribution $\text{Normal}(p_{0\mu}, p_{0\sigma})$ and population level slope distribution $\text{Normal}(p_{1\mu}, p_{1\sigma})$. So these distributions would represent what we might expect to observe for some as yet unobserved group.

    However, this formulation of the model does not so neatly map on to the Wilkinson notation. For this reason, we have chosen to present the model in the form given above. For an interesting discussion on this topic, see [Discussion #808](https://github.com/bambinos/bambi/discussions/808) in the [`bambi`](https://github.com/bambinos/bambi) repository.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The hierarchical model we are considering contains a simplification in that the population level slope and intercept are assumed to be independent. It is possible to relax this assumption and model any correlation between these parameters by using a multivariate normal distribution.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In one sense this move from Model 2 to Model 3 can be seen as adding parameters, and therefore increasing model complexity. However, in another sense, adding this knowledge about the nested structure of the data actually provides a constraint over parameter space.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Conduct inference
    """)
    return


@app.cell
def _(model3, pm, rng):
    with model3:
        idata3 = pm.sample(target_accept=0.95, random_seed=rng)
    return (idata3,)


@app.cell
def _(az, idata3):
    az.plot_trace(idata3, var_names=["~μ"]);
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Visualise
    """)
    return


@app.cell
def _(data, idata3, model3, np, predict):
    # Cell tags: hide-output
    _n_points = 10
    _n_groups = len(data.group.unique())
    xi_2 = np.concatenate([np.linspace(group[1].x.min(), group[1].x.max(), _n_points) for group in data.groupby('group_idx')])
    g_5 = np.concatenate([[i] * _n_points for i in range(_n_groups)]).astype(int)
    predict_at_1 = {'x': xi_2, 'g': g_5}
    idata3_1 = predict(model=model3, idata=idata3, predict_at=predict_at_1)
    return g_5, idata3_1, xi_2


@app.cell
def _(az, data, g_5, group_list, idata3_1, plot_band, plt, xi_2):
    # Cell tags: hide-input
    def _plot(idata):
        fig, _ax = plt.subplots(1, 3, figsize=(12, 4))
        for i in range(len(group_list)):
            _ax[0].scatter(data.x[data.group_idx == i], data.y[data.group_idx == i], color=f'C{i}')
            plot_band(xi_2[g_5 == i], idata.posterior_predictive.μ.isel(obs_id=g_5 == i), ax=_ax[0], color=f'C{i}')
            _ax[1].scatter(data.x[data.group_idx == i], data.y[data.group_idx == i], color=f'C{i}')
            plot_band(xi_2[g_5 == i], idata.posterior_predictive.y.isel(obs_id=g_5 == i), ax=_ax[1], color=f'C{i}')
        _ax[0].set(xlabel='x', ylabel='y', title='Conditional mean')
        _ax[1].set(xlabel='x', ylabel='y', title='Posterior predictive distribution')
        for i, _ in enumerate(group_list):
            _ax[2].scatter(az.extract(idata, var_names='β1') + az.extract(idata, var_names='u1')[i, :], az.extract(idata, var_names='β0') + az.extract(idata, var_names='u0')[i, :], color=f'C{i}', alpha=0.01, rasterized=True, zorder=2)
        _ax[2].set(xlabel='slope', ylabel='intercept', title='Parameter space')
        _ax[2].axhline(y=0, c='k')
        _ax[2].axvline(x=0, c='k')
        return _ax
    _ax = _plot(idata3_1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The panel on the right shows the group level posterior of the slope and intercept parameters as a contour plot. We can also just plot the marginal distribution below to see how much belief we have in the slope being less than zero.
    """)
    return


@app.cell
def _(az, figsize, idata2_1):
    # Cell tags: hide-input
    _ax = az.plot_forest(idata2_1.posterior['β1'], combined=True, figsize=figsize)[0]
    _ax.set(title='Model 3 suggests negative slopes for each group', xlabel='$\\beta_1$', ylabel='Group')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary
    Using Simpson's paradox, we've walked through 3 different models. The first is a simple linear regression which treats all the data as coming from one group. This amounts to a causal DAG asserting that $x$ causally influences $y$ and $\text{group}$ was ignored (i.e. assumed to be causally unrelated to $x$ or $y$). We saw that this lead us to believe the regression slope was positive.

    While that is not necessarily wrong, it is paradoxical when we see that the regression slopes for the data _within_ a group is negative.

    This paradox is resolved by updating our causal DAG to include the group variable. This is what we did in the second and third models. Model 2 was an unpooled model where we essentially fit separate regressions for each group.

    Model 3 assumed the same causal DAG, but adds the knowledge that each of these groups are sampled from an overall population. This added the ability to make inferences not only about the regression parameters at the group level, but also at the population level.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Authors
    * Authored by [Benjamin T. Vincent](https://github.com/drbenvincent) in July 2021
    * Updated by [Benjamin T. Vincent](https://github.com/drbenvincent) in April 2022
    * Updated by [Benjamin T. Vincent](https://github.com/drbenvincent) in February 2023 to run on PyMC v5
    * Updated to use `az.extract` by [Benjamin T. Vincent](https://github.com/drbenvincent) in February 2023 ([pymc-examples#522](https://github.com/pymc-devs/pymc-examples/pull/522))
    * Updated by [Benjamin T. Vincent](https://github.com/drbenvincent) in September 2024 ([pymc-examples#697](https://github.com/pymc-devs/pymc-examples/pull/697) and [pymc-examples#709](https://github.com/pymc-devs/pymc-examples/pull/709))
    """)
    return


if __name__ == "__main__":
    app.run()
