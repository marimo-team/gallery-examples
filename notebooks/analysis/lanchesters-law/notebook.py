# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "altair==6.0.0",
#     "anywidget==0.9.21",
#     "marimo>=0.19.7",
#     "polars==1.36.1",
# ]
# ///

import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import polars as pl
    return (pl,)


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    controls = mo.md("""
    **HP:** {hp}

    **Damage:** {damage}

    **Attack Range:** {attack_range}

    **Move Speed:** {move_speed}

    **Speed Multiplier:** {speed_multiplier}

    **Spawn Mode:** {spawn_mode}
    """).batch(
        hp=mo.ui.slider(1, 100, value=20),
        damage=mo.ui.slider(1, 100, value=1),
        attack_range=mo.ui.slider(1, 200, value=10),
        move_speed=mo.ui.slider(1, 500, step=5, value=55),
        speed_multiplier=mo.ui.slider(1, 50, value=20),
        spawn_mode=mo.ui.dropdown(["sides", "mixed"], value="sides"),
    ).form()

    controls
    return (controls,)


@app.cell
def _(controls, mo):
    from battle_widget import BattleWidget

    mo.stop(controls.value is None, mo.md("Configure settings and submit the form to start the simulation."))

    battle = mo.ui.anywidget(
        BattleWidget(
            grid_spec={"n_blue": list(range(200, 1000, 10)), "n_red": [200]},
            runs_per_point=1,
            seed_mode="random",
            base_seed=20,
            arena_width=640,
            arena_height=420,
            unit_radius=4.0,
            spawn_mode=controls.value["spawn_mode"],
            step_dt=0.02,
            move_speed=float(controls.value["move_speed"]),
            attack_range=float(controls.value["attack_range"]),
            attack_cooldown=0.1,
            hit_chance=0.85,
            damage=int(controls.value["damage"]),
            hp=int(controls.value["hp"]),
            max_time=30.0,
            record_dt=0.2,
            render=True,
            speed_multiplier=int(controls.value["speed_multiplier"])
        )
    )
    battle
    return (battle,)


@app.cell
def _(battle, pl):
    pl.DataFrame(battle.results).with_columns(diff=pl.col("n_blue") - pl.col("n_red")).plot.line(x="time", y="n_blue", detail="run_id")
    return


@app.cell
def _(battle, pl):
    (
        pl.DataFrame(battle.results)
            .group_by("run_id")
            .agg(
                pl.col("n_blue").max().alias("blue_max"),
                pl.col("n_blue").min().alias("blue_min"),
                pl.col("n_red").max().alias("red_max"),
                pl.col("n_red").min().alias("red_min"),
            )
            .with_columns(
                blue_diff = pl.col("blue_max") - pl.col("blue_min"),
                red_diff = pl.col("red_max") - pl.col("red_min"),
            )
            .with_columns(
                start_diff = pl.col("blue_max") - pl.col("red_max"),
            )
            .plot.scatter("start_diff", "blue_min")
    )
    return


@app.cell
def _(battle, pl):
    (
        pl.DataFrame(battle.results)
            .group_by("run_id")
            .agg(
                pl.col("n_blue").max().alias("blue_max"),
                pl.col("n_blue").min().alias("blue_min"),
                pl.col("n_red").max().alias("red_max"),
                pl.col("n_red").min().alias("red_min"),
            )
            .with_columns(
                blue_diff = pl.col("blue_max") - pl.col("blue_min"),
                red_diff = pl.col("red_max") - pl.col("red_min"),
            )
            .with_columns(
                start_diff = pl.col("blue_max") - pl.col("red_max"),
            )
            .plot.scatter("blue_max", "blue_min")
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Lanchester's Laws

    Lanchester's laws are mathematical models that describe the power relationship between opposing forces in combat.
    The **Square Law** applies to aimed fire combat where each unit can engage any enemy:

    $$
    \frac{dB}{dt} = -\alpha R, \quad \frac{dR}{dt} = -\beta B
    $$

    Where:
    - $B(t)$ = number of Blue units at time $t$
    - $R(t)$ = number of Red units at time $t$
    - $\alpha$ = effectiveness of Red forces (kills per unit per time)
    - $\beta$ = effectiveness of Blue forces (kills per unit per time)

    ### Solving the Equations

    Dividing the two equations:

    $$
    \frac{dB}{dR} = \frac{\alpha R}{\beta B} \implies \beta B \, dB = \alpha R \, dR
    $$

    Integrating both sides:

    $$
    \beta B^2 - \beta B_0^2 = \alpha R^2 - \alpha R_0^2
    $$

    This gives us **Lanchester's Square Law**:

    $$
    \beta B^2 - \alpha R^2 = \beta B_0^2 - \alpha R_0^2 = \text{constant}
    $$

    ### Equal Strength Case ($\alpha = \beta$)

    When both armies have equal effectiveness, the equation simplifies to:

    $$
    B^2 - R^2 = B_0^2 - R_0^2
    $$

    If Blue wins (meaning $R \to 0$), the surviving Blue force is:

    $$
    B_{\text{final}} = \sqrt{B_0^2 - R_0^2}
    $$

    This only has a real solution when $B_0 > R_0$. The square law explains why **concentration of force** matters:
    doubling your army doesn't just double your advantage—it *quadruples* it.
    """)
    return


@app.cell
def _(battle, pl):
    (
        pl.DataFrame(battle.results)
            .group_by("run_id")
            .agg(
                pl.col("n_blue").max().alias("blue_max"),
                pl.col("n_blue").min().alias("blue_min"),
                pl.col("n_red").max().alias("red_max"),
                pl.col("n_red").min().alias("red_min"),
            )
            .with_columns(
                expected_blue_min = (pl.col("blue_max")**2 - pl.col("red_max")**2).sqrt(),
            )
            .plot.scatter("expected_blue_min", "blue_min")
    )
    return


@app.cell
def _(battle, pl):
    df = pl.DataFrame(battle.results).melt(id_vars=["run_id", "seed", "time"])
    return (df,)


@app.cell
def _(df, mo):
    dropdown = mo.ui.dropdown(df["run_id"].unique())
    dropdown
    return (dropdown,)


@app.cell
def _(df, dropdown, mo, pl):
    import altair as alt

    mo.stop(dropdown.value == "")

    run_id = dropdown.value

    _df = df.filter(pl.col("run_id") == run_id)

    chart = (
        alt.Chart(_df)
        .mark_line()
        .encode(x="time:Q", y="value:Q", color="variable:N", detail='run_id')
        .properties(width=640, height=320, title="Army sizes over time")
    )
    chart
    return


if __name__ == "__main__":
    app.run()
