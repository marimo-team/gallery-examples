# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "altair==6.0.0",
#     "diskcache==5.6.3",
#     "marimo",
#     "pandas==3.0.0",
#     "polars==1.37.1",
#     "yfinance==1.1.0",
# ]
# ///

import marimo

__generated_with = "0.17.7"
app = marimo.App(
    width="medium",
    css_file="/usr/local/_marimo/custom.css",
    auto_download=["html"],
)


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    return mo, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Portfolio tracker

    Fill in your stock portfolio below and we will figure out how the value of your investment changes over time. We go back 72 months and use the `yfinance` SDK for the stock values.
    """)
    return


@app.cell
def _(mo, pd):
    investments = mo.ui.data_editor(
        pd.DataFrame(
            [
                {"Date": "2021-02-01", "Ticker": "msft", "Investment": 500},
                {"Date": "2023-02-01", "Ticker": "aapl", "Investment": 800},
                {"Date": "2024-02-01", "Ticker": "aapl", "Investment": 200},
            ]
        )
    )
    investments
    return (investments,)


@app.cell
def _(investments):
    type(investments.value)
    return


@app.cell
def _(parent_folder, pl):
    df = pl.read_csv(f"{parent_folder}/*", glob=True)
    return


@app.cell(hide_code=True)
def _(parent_folder, records):
    import altair as alt
    import polars as pl


    def clean_data(dataf, investment_records):
        # Convert investment records to polars DataFrame with uppercase ticker
        investments_df = pl.DataFrame(investment_records)

        return (
            dataf.drop(["Dividends", "Stock Splits", "Volume"])
            .join(investments_df, on=["Date", "Ticker"], how="left")
            .with_columns(pl.col("Investment").fill_null(0))
            .sort("Ticker", "Date")
        )


    def calculate_portfolio_value(dataf):
        # For each ticker, we need to track shares purchased at each investment
        return (
            dataf.with_columns(
                # Calculate cumulative investment per ticker
                CumInvestment=pl.col("Investment").cum_sum().over("Ticker")
            )
            # Filter to start from first investment
            .filter(pl.col("CumInvestment") > 0)
            .with_columns(
                # When investment happens, calculate shares bought at that price
                SharesBought=pl.when(pl.col("Investment") > 0)
                .then(pl.col("Investment") / pl.col("Close"))
                .otherwise(0)
            )
            .with_columns(
                # Calculate total shares owned (cumulative)
                TotalShares=pl.col("SharesBought").cum_sum().over("Ticker")
            )
            .with_columns(
                # Current portfolio value = total shares × current price
                PortfolioValue=pl.col("TotalShares") * pl.col("Close"),
                # Profit/Loss = portfolio value - cumulative investment
                PnL=(pl.col("TotalShares") * pl.col("Close")) - pl.col("CumInvestment"),
            )
        )


    def calculate_performance(dataf):
        return (
            dataf.group_by("Date")
            .agg(
                [
                    pl.sum("CumInvestment").alias("TotalInvested"),
                    pl.sum("PortfolioValue").alias("TotalValue"),
                    pl.sum("PnL").alias("TotalPnL"),
                ]
            )
            .with_columns(
                Date=pl.col("Date").str.to_date(),
                ReturnPct=((pl.col("TotalValue") / pl.col("TotalInvested")) - 1) * 100,
            )
            .sort("Date")
        )


    def make_chart(dataf):
        portfolio_chart = (
            alt.Chart(dataf)
            .mark_area(line=True, opacity=0.3)
            .encode(
                x=alt.X("Date:T", title="Date"),
                y=alt.Y("TotalValue:Q", title="Value ($)"),
            )
        )

        # Add invested amount line
        invested_line = (
            alt.Chart(dataf)
            .mark_line(strokeDash=[5, 5], color="black", strokeWidth=2)
            .encode(x="Date:T", y="TotalInvested:Q")
        )

        # Combine charts
        return portfolio_chart + invested_line


    cached = (
        pl.read_csv(f"{parent_folder}/*", glob=True)
        .pipe(clean_data, investment_records=records)
        .pipe(calculate_portfolio_value)
        .pipe(calculate_performance)
    )
    return cached, make_chart, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Investment value over time
    """)
    return


@app.cell
def _(cached, make_chart):
    cached.pipe(make_chart)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Returns over time
    """)
    return


@app.cell
def _(cached):
    cached.plot.line("Date", "ReturnPct")
    return


@app.cell
def _(investments):
    records = investments.value.assign(Ticker=lambda d: d["Ticker"].str.upper()).to_dict(
        orient="records"
    )
    # records
    return (records,)


@app.cell
def _(records):
    import yfinance as yf
    from pathlib import Path

    parent_folder = Path("invest-data")
    parent_folder.mkdir(exist_ok=True)


    def download_tickers(tickers):
        for record in tickers:
            ticker = record.upper()
            if not (parent_folder / f"{ticker}.csv").exists():
                (
                    yf.Ticker("MSFT")
                    .history(period="72mo")
                    .reset_index()
                    .assign(Date=lambda d: d["Date"].dt.strftime("%Y-%m-%d"), Ticker=ticker)
                    .to_csv(f"{parent_folder}/{ticker}.csv")
                )


    out = download_tickers(set(_["Ticker"] for _ in records))
    return (parent_folder,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Data for download
    """)
    return


@app.cell
def _(cached):
    cached
    return


if __name__ == "__main__":
    app.run()
