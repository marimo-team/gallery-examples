# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "altair==5.5.0",
#     "anthropic==0.71.0",
#     "marimo",
#     "polars==1.34.0",
# ]
# ///

import marimo

__generated_with = "0.17.3"
app = marimo.App(width="columns", auto_download=["html"], sql_output="polars")

with app.setup:
    import marimo as mo
    import altair as alt
    import polars as pl


@app.cell
def _():
    df = (
        pl.read_csv("lego_sets.csv")
        .filter(pl.col("category") == "Normal")
        .filter(~pl.col("US_retailPrice").is_null())
        .filter(pl.len().over(pl.col("theme")) >= 10)
    )
    return (df,)


@app.cell(hide_code=True)
def _(multi_select):
    mo.md(f"""
    # Exploring price differences in Lego sets

    Lego is well known for producing sets across a wide range of themes and price points. It's not just movie tie-ins like Star Wars or Harry Potter that are out there, we also have more generic themes like City or Technic.

    One might wonder, are there sets that are more expensive than other ones? Does it depend on the total number of pieces in the box? Or might a license fee also apply? 

    ## Select themes

    Start by selecting the lego themes that you want to explore first. 

    {multi_select}
    """)
    return


@app.cell
def _(final):
    mo.hstack(
        [
            mo.stat(caption="Average piece price", label=_["theme"], value=_["pieceprice"], bordered=True)
            for _ in final.group_by("theme").agg(pl.mean("pieceprice")).to_dicts()
        ], widths="equal", gap=1
    )
    return


app._unparsable_cell(
    r"""
    mo.md(\"
        r\"\"\"These are the average prices per piece per theme. But you may want to dive a bit deeper.\"\"\"
    )
    """,
    column=None, disabled=False, hide_code=True, name="_"
)


@app.cell
def _():
    checkbox = mo.ui.checkbox(label="Show line chart", value=False)
    check_inflation = mo.ui.checkbox(label="Assume inflation", value=False)
    return check_inflation, checkbox


@app.cell
def _(check_inflation, checkbox, date_range, pieceprice_range, price_range, xaxis, yaxis):
    mo.hstack([
        mo.vstack([
            mo.md("**Data Settings**"), 
            date_range, 
            price_range, 
            pieceprice_range
        ]), 
        mo.vstack([
            mo.md("**Chart Settings**"), 
            xaxis, 
            yaxis, 
            checkbox, 
            check_inflation
    ])])
    return


@app.cell
def _():
    date_range = mo.ui.range_slider(1970, 2022, 1, label="Year Range", value=[2001, 2022])
    return (date_range,)


@app.cell
def _(df):
    themes = df.group_by("theme").len()["theme"].to_list()

    multi_select = mo.ui.multiselect(
        themes, label="Select Themes", value=["Duplo", "Star Wars", "City"], max_selections=5
    )
    return (multi_select,)


@app.cell
def _(date_range):
    y1, y2 = date_range.value
    return y1, y2


@app.cell
def _(df, multi_select, y1, y2):
    subset = (
        df.filter(
            pl.col("year") >= y1, pl.col("year") <= y2, pl.col("theme").is_in(multi_select.value)
        )
        .rename(dict(US_retailPrice="price"))
        .with_columns(pl.col("price").cast(pl.Float64))
        .with_columns(inflation=pl.col("price") * 1.03**(pl.col("year") - 1970))
        .with_columns(pieceprice=pl.col("price") / pl.col("pieces"))
    )
    return (subset,)


@app.cell
def _(subset):
    max_price = subset.select(pl.col("price")).max()["price"].to_list()[0]
    max_piece_price = subset.select(pl.col("pieceprice")).max()["pieceprice"].to_list()[0]

    price_range = mo.ui.range_slider(0, max_price, label="Price Range", value=[0, 150])
    pieceprice_range = mo.ui.range_slider(0, max_piece_price, label="Piece Price Range", value=[0, 2])
    return pieceprice_range, price_range


@app.cell
def _():
    xaxis = mo.ui.dropdown(
        ["year", "pieces", "price", "pieceprice"], label="X-Axis", value="pieces"
    )
    yaxis = mo.ui.dropdown(["pieces", "price", "pieceprice"], label="Y-Axis", value="price")
    return xaxis, yaxis


app._unparsable_cell(
    r"""
    mo.md(\"
        r\"\"\"The chart below is interactive and you can make selections in it to see each set in more detail.\"\"\"
    )
    """,
    column=None, disabled=False, hide_code=True, name="_"
)


@app.cell
def _(check_inflation, checkbox, pieceprice_range, price_range, subset, xaxis, yaxis):
    alt.renderers.set_embed_options(actions=False)

    final = subset.filter(
        pl.col("price") >= price_range.value[0], pl.col("price") <= price_range.value[1], 
        pl.col("pieceprice") >= pieceprice_range.value[0], pl.col("pieceprice") <= pieceprice_range.value[1]
    )

    if check_inflation.value:
        final = final.with_columns(
            price=pl.col("price") * pl.col("inflation"), 
            pieceprice=pl.col("pieceprice") * pl.col("inflation")
        )

    chart = alt.Chart(final).mark_point().encode(
        x=alt.X(xaxis.value).scale(zero=False), y=yaxis.value, color="theme"
    )

    if checkbox.value: 
        chart = alt.Chart(final).transform_loess(xaxis.value, yaxis.value, groupby=["theme"]).mark_line().encode(
            x=alt.X(xaxis.value).scale(zero=False), 
            y=yaxis.value, 
            color="theme"
        ) + chart.mark_point(opacity=0.2)

    mochart = mo.ui.altair_chart(chart)

    mochart
    return final, mochart


@app.cell
def _(mochart):
    mochart.value.select("set_id", "name", "theme", "imageURL")
    return


@app.cell(hide_code=True)
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
