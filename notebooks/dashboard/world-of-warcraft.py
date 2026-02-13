# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.19.7",
# ]
# ///
import marimo

__generated_with = "0.14.12"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell(hide_code=True)
def _(chart, max_session_threshold, mo):
    mo.vstack([
        mo.md(f"""
        ## Bot detection 

        For this work we use the [world of warcraft avatar dataset](https://github.com/koaning/wow-avatar-datasets). If we remove users that have had a session length that is too long then this has an effect on the number of users that we see over time. You can set the threshold below

        {max_session_threshold}

        And the chart below will update automatically
        """), 
        chart
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Code appendix""")
    return


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import altair as alt
    return mo, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Let's download the dataset first.""")
    return


@app.cell
def _():
    import urllib.request

    path, _ = urllib.request.urlretrieve(
        "https://github.com/koaning/wow-avatar-datasets/raw/refs/heads/main/wow-full.parquet", 
        "wow-full.parquet"
    )
    return (path,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Next, we read it into polars so that we may apply a pipeline.""")
    return


@app.cell
def _(path, pl):
    df = pl.read_parquet(path)
    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell(hide_code=True)
def _(pl):
    def set_types(dataf):
        return (dataf.with_columns([
                    pl.col("guild").is_not_null(),
                    pl.col("datetime").cast(pl.Int64).alias("timestamp")
                ]))

    def clean_data(dataf):
        return (
            dataf
            .filter(
                ~pl.col("class").is_in(["482", "Death Knight", "3485伊", "2400"]),
                pl.col("race").is_in(["Troll", "Orc", "Undead", "Tauren", "Blood Elf"])
            )
        )

    def sessionize(dataf, threshold=20 * 60 * 1000):
        return (dataf
                 .sort(["player_id", "timestamp"])
                 .with_columns(
                     (pl.col("timestamp").diff().cast(pl.Int64) > threshold).fill_null(True).alias("ts_diff"),
                     (pl.col("player_id").diff() != 0).fill_null(True).alias("char_diff"),
                 )
                 .with_columns(
                     (pl.col("ts_diff") | pl.col("char_diff")).alias("new_session_mark")
                 )
                 .with_columns(
                     pl.col("new_session_mark").cum_sum().alias("session")   
                 )
                 .drop(["char_diff", "ts_diff", "new_session_mark"]))

    def add_features(dataf):
        return (dataf
                 .with_columns(
                     pl.col("player_id").count().over("session").alias("session_length"),
                     pl.col("session").n_unique().over("player_id").alias("n_sessions_per_char")
                 ))

    def remove_bots(dataf, max_session_hours=24):
        # We're using some domain knowledge. The logger of our dataset should log
        # data every 10 minutes. That's what this line is based on.
        n_rows = max_session_hours * 6
        return (dataf
                .filter(pl.col("session_length").max().over("player_id") < n_rows))
    return add_features, clean_data, remove_bots, sessionize, set_types


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Part of the pipeline is "cached". This is the part of the pipeline that does not change if we change the slider value.""")
    return


@app.cell
def _(add_features, clean_data, df, sessionize, set_types):
    cached = (
        df
        .pipe(set_types)
        .pipe(clean_data)
        .pipe(sessionize, threshold=30 * 60 * 1000)
        .pipe(add_features)
    )
    return (cached,)


@app.cell
def _(mo):
    max_session_threshold = mo.ui.slider(2, 24, 1, value=24, label="Max session length (hours)")
    return (max_session_threshold,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""This way, only this function runs when we need it.""")
    return


@app.cell
def _(max_session_threshold, plot_per_date):
    chart = plot_per_date(max_session_threshold.value)
    return (chart,)


@app.cell(hide_code=True)
def _(cached, df, max_session_threshold, mo, pl, remove_bots):
    @mo.cache
    def plot_per_date(threshold):
        df_out = (
            cached.pipe(remove_bots, max_session_hours=max_session_threshold.value)
        )
        agg_orig = (
            df
            .with_columns(date=pl.col("datetime").dt.date())
            .group_by("date")
            .len()
            .with_columns(set=pl.lit("original"))
        )
        agg_clean = (
            df_out
            .with_columns(date=pl.col("datetime").dt.date())
            .group_by("date")
            .len()
            .with_columns(set=pl.lit("clean"))
        )
        return (
            pl.concat([agg_orig, agg_clean])
            .plot
            .line(x="date", y="len", color="set")
        )
    return (plot_per_date,)


@app.cell
def _(cached, max_session_threshold, remove_bots):
    df_out = (
        cached.pipe(remove_bots, max_session_hours=max_session_threshold.value)
    )
    return


if __name__ == "__main__":
    app.run()
