# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "anywidget",
#     "wandb",
#     "marimo",
#     "python-dotenv",
#     "wigglystuff==0.2.30",
# ]
# ///

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import math
    import random
    import time
    import wandb
    from dotenv import load_dotenv
    from wigglystuff import EnvConfig, WandbChart


@app.cell(hide_code=True)
def _():
    mo.md("""
    # Live wandb charts

    This demo starts multiple wandb runs with different learning rates, logs
    metrics with a delay, and the `WandbChart` widget polls the wandb GraphQL API
    from JavaScript to render live-updating line charts.

    Each widget shows **one metric** as a line chart. Pass multiple runs
    via `runs=` to compare them side by side in the same chart.
    """)
    return


@app.cell
def _():
    load_dotenv()


    logged_in = wandb.login()
    return (logged_in,)


@app.cell
def _(logged_in):
    logged_in
    return


@app.cell
def _(logged_in):
    config = None

    if not logged_in:
        config = mo.ui.anywidget(EnvConfig(["WANDB_API_KEY"]))

    config
    return (config,)


@app.cell
def _(config, logged_in):
    if not logged_in:
        config.require_valid()
        api_key = config["WANDB_API_KEY"]
        wandb.login(key=api_key)
    else:
        api_key = wandb.api.api_key
    return (api_key,)


@app.cell
def _():
    baseline = wandb.init(project="jupyter-projo", name="baseline")
    for _step in range(60):
        _t = _step / 59
        _noise = random.gauss(0, 0.15)
        baseline.log({"loss": math.exp(-2 * _t) + 0.05 + _noise, "acc": 1 - math.exp(-2 * _t) + _noise})
    baseline.finish()

    experiment = wandb.init(project="jupyter-projo", name="fast-lr", reinit=True)
    return baseline, experiment


@app.cell
def _(api_key, baseline, experiment):
    loss_chart = WandbChart(
        api_key=api_key,
        entity=baseline.entity,
        project=baseline.project,
        runs=[baseline, experiment],
        key="loss",
        smoothing_kind="exponential",
        smoothing_param=0.6,
        poll_seconds=1,
        width=400
    )
    return (loss_chart,)


@app.cell(hide_code=True)
def _(api_key, baseline, experiment):
    acc_chart = WandbChart(
        api_key=api_key,
        entity=baseline.entity,
        project=baseline.project,
        runs=[baseline, experiment],
        key="acc",
        smoothing_kind="gaussian",
        smoothing_param=2.0,
        poll_seconds=2,
        width=400
    )
    return (acc_chart,)


@app.cell
def _(acc_chart, loss_chart):
    mo.hstack([loss_chart, acc_chart])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    You can also have charts that only poll manually.
    """)
    return


@app.cell
def _(api_key, baseline, experiment):
    manual_chart = WandbChart(
        api_key=api_key,
        entity=baseline.entity,
        project=baseline.project,
        runs=[baseline, experiment],
        key="loss",
        smoothing_kind="rolling",
        smoothing_param=5,
        poll_seconds=None,
    )
    manual_chart
    return


@app.cell
def _(acc_chart, experiment, loss_chart):
    loss_chart
    acc_chart

    for _step in range(60):
        _t = _step / 59
        _noise = random.gauss(0, 0.25)
        experiment.log({"loss": math.exp(-5 * _t) + 0.01 + _noise, "acc": 1 - math.exp(-5 * _t) + _noise})
        time.sleep(2)

    experiment.finish()
    return


if __name__ == "__main__":
    app.run()
