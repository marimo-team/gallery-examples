# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "polars==1.37.1",
#     "justetf-scraping @ git+https://github.com/druzsan/justetf-scraping.git",
# ]
# [tool.marimo.opengraph]
# title = "ETF vs All-World"
# description = "Compare country ETF returns against Vanguard FTSE All-World"
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import justetf_scraping
    from datetime import date
    from pathlib import Path

    # Anchor cached data next to this notebook regardless of the server's cwd.
    NOTEBOOK_DIR = mo.notebook_dir() or Path.cwd()
    return NOTEBOOK_DIR, date, justetf_scraping, mo, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Country ETF vs FTSE All-World

    Build a blended portfolio from single-country ETFs — pick the ETFs and set
    their mix weights. **Hover the chart to pick the start date**: every line
    re-bases to 0% at the hovered date, so you can read the return from any point
    against the **Vanguard FTSE All-World (VWCE)** benchmark. Daily NAVs come from
    [justETF](https://www.justetf.com/).
    """)
    return


@app.cell
def _():
    # Curated single-country ETFs (label -> ISIN). Each ISIN is verified to load
    # via justetf_scraping. VWCE is the always-on benchmark.
    COUNTRY_ETFS = {
        "Spain — Amundi IBEX 35": "LU1681046931",
        "UK — iShares Core FTSE 100": "IE0005042456",
        "France — Amundi CAC 40": "LU1834983550",
        "Germany — iShares Core DAX": "DE0005933931",
        "USA — iShares Core S&P 500": "IE00B5BMR087",
        "Canada — iShares MSCI Canada": "IE00B52SF786",
        "Australia — iShares MSCI Australia": "IE00B5377D42",
        "Singapore — iShares MSCI Singapore": "IE00B3VVMM84",
        "India — iShares MSCI India": "IE00BZCQB185",
        "Brazil — iShares MSCI Brazil": "IE00B0M63516",
    }
    VWCE_LABEL = "VWCE (Vanguard FTSE All-World)"
    VWCE_ISIN = "IE00BK5BQT80"
    return COUNTRY_ETFS, VWCE_ISIN, VWCE_LABEL


@app.cell
def _(COUNTRY_ETFS, mo):
    etf_select = mo.ui.multiselect(
        options=COUNTRY_ETFS,
        value=["USA — iShares Core S&P 500", "Germany — iShares Core DAX"],
        label="ETFs to compare",
    )
    etf_select
    return (etf_select,)


@app.cell
def _(COUNTRY_ETFS, etf_select, mo):
    # One slider per selected ETF (multiselect yields ISINs). Values are relative
    # and auto-normalized, so they express the mix rather than absolute amounts.
    isin_to_label = {isin: label for label, isin in COUNTRY_ETFS.items()}
    weights = mo.ui.dictionary(
        {
            isin: mo.ui.slider(
                0, 100, value=50, step=5, label=isin_to_label[isin], show_value=True
            )
            for isin in etf_select.value
        }
    )
    mo.vstack([mo.md("**Mix weights** (relative, auto-normalized)"), weights])
    return (weights,)


@app.cell
def _(NOTEBOOK_DIR, VWCE_ISIN, date, etf_select, justetf_scraping):
    parent_folder = NOTEBOOK_DIR / "etf-data"
    parent_folder.mkdir(exist_ok=True)

    def ensure_cached(isin):
        csv_path = parent_folder / f"{isin}.csv"
        is_stale = (
            not csv_path.exists()
            or date.fromtimestamp(csv_path.stat().st_mtime) < date.today()
        )
        if is_stale:
            df = justetf_scraping.load_chart(isin).reset_index()
            date_col = df.columns[0]
            df = df.rename(columns={date_col: "Date"})
            df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
            df["ISIN"] = isin
            df[["Date", "ISIN", "quote"]].to_csv(csv_path, index=False)
        return csv_path

    # Fetch only the selected ETFs plus the benchmark.
    for _isin in set(etf_select.value) | {VWCE_ISIN}:
        ensure_cached(_isin)
    return (parent_folder,)


@app.cell
def _(VWCE_ISIN, VWCE_LABEL, etf_select, mo, parent_folder, pl, weights):
    mo.stop(
        len(etf_select.value) == 0,
        mo.md("**Select at least one ETF to build a mix.**"),
    )

    def load_quotes(isin):
        return (
            pl.read_csv(parent_folder / f"{isin}.csv")
            .with_columns(Date=pl.col("Date").str.to_date())
            .select(["Date", pl.col("quote").alias(isin)])
            .sort("Date")
        )

    # Join every selected ETF and the benchmark on their common trading days so
    # each series has a value at every date the chart's hover can land on.
    cols = list(etf_select.value) + [VWCE_ISIN]
    wide = load_quotes(cols[0])
    for isin in cols[1:]:
        wide = wide.join(load_quotes(isin), on="Date", how="inner")

    # Rebase each price to 1.0 at the first common date so the mix weights are
    # meaningful and the pre-hover baseline reads as "return since the start".
    wide = wide.with_columns([(pl.col(c) / pl.col(c).first()).alias(c) for c in cols])

    # Normalize the raw slider values into weights that sum to 1 (equal if all zero).
    raw = {isin: weights.value[isin] for isin in etf_select.value}
    total = sum(raw.values())
    norm = (
        {isin: value / total for isin, value in raw.items()}
        if total > 0
        else {isin: 1 / len(raw) for isin in raw}
    )

    mix = pl.sum_horizontal([pl.col(isin) * norm[isin] for isin in etf_select.value])
    returns = wide.select(
        "Date",
        mix.alias("Portfolio mix"),
        pl.col(VWCE_ISIN).alias(VWCE_LABEL),
    ).unpivot(index="Date", variable_name="Series", value_name="value")
    return (returns,)


@app.cell(hide_code=True)
def _(pl, returns):
    # Vega-Lite "interactive index chart": a point selection on x follows the
    # pointer and every line re-bases to that date entirely in the browser (no
    # Python round-trip). We emit the raw spec because Altair's to_dict() hoists
    # the selection param to the top level, which Vega-Lite rejects for a lookup
    # ("cannot define and lookup in the same view") — the param must stay inside
    # the point layer. The initial value seeds the selection at the first date so
    # it renders on load and the lookup always has a row to join to.
    d0 = returns["Date"].min()
    values = returns.with_columns(Date=pl.col("Date").dt.strftime("%Y-%m-%d")).to_dicts()

    spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "data": {"values": values},
        "width": "container",
        "height": 380,
        "layer": [
            {
                "params": [
                    {
                        "name": "index",
                        "value": [
                            {"x": {"year": d0.year, "month": d0.month, "date": d0.day}}
                        ],
                        "select": {
                            "type": "point",
                            "encodings": ["x"],
                            "on": "pointerover",
                            "nearest": True,
                        },
                    }
                ],
                "mark": "point",
                "encoding": {
                    "x": {"field": "Date", "type": "temporal"},
                    "opacity": {"value": 0},
                },
            },
            {
                "transform": [
                    {"lookup": "Series", "from": {"param": "index", "key": "Series"}},
                    {
                        "calculate": "datum.index && datum.index.value > 0 ? (datum.value - datum.index.value) / datum.index.value * 100 : (datum.value - 1) * 100",
                        "as": "indexed",
                    },
                ],
                "mark": "line",
                "encoding": {
                    "x": {"field": "Date", "type": "temporal", "title": ""},
                    "y": {"field": "indexed", "type": "quantitative", "title": "Return (%)"},
                    "color": {"field": "Series", "type": "nominal", "title": ""},
                },
            },
            {
                "transform": [{"filter": {"param": "index"}}],
                "mark": {"type": "rule", "color": "gray"},
                "encoding": {"x": {"field": "Date", "type": "temporal"}},
            },
        ],
    }

    class VegaLiteChart:
        def _repr_mimebundle_(self, include=None, exclude=None):
            return {"application/vnd.vegalite.v5+json": spec}

    VegaLiteChart()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Downside risk (drawdown)

    How far each series sits below its own running peak — the "underwater" view of
    downside jumps. Deeper, longer troughs mean bigger crash exposure.
    """)
    return


@app.cell(hide_code=True)
def _(pl, returns):
    # Drawdown = value relative to the running peak so far, in % (always <= 0).
    drawdown = returns.sort("Series", "Date").with_columns(
        Drawdown=((pl.col("value") / pl.col("value").cum_max().over("Series")) - 1) * 100
    )
    dd_values = drawdown.with_columns(
        Date=pl.col("Date").dt.strftime("%Y-%m-%d")
    ).select(["Date", "Series", "Drawdown"]).to_dicts()

    dd_spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "data": {"values": dd_values},
        "width": "container",
        "height": 260,
        "mark": {"type": "area", "opacity": 0.5, "line": True},
        "encoding": {
            "x": {"field": "Date", "type": "temporal", "title": ""},
            "y": {"field": "Drawdown", "type": "quantitative", "title": "Drawdown (%)"},
            "color": {"field": "Series", "type": "nominal", "title": ""},
        },
    }

    class DrawdownChart:
        def _repr_mimebundle_(self, include=None, exclude=None):
            return {"application/vnd.vegalite.v5+json": dd_spec}

    DrawdownChart()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data for download
    """)
    return


@app.cell
def _(returns):
    returns
    return


if __name__ == "__main__":
    app.run()
