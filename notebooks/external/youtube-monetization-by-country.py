# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "altair==6.3.0",
#     "marimo>=0.25.1",
#     "pandas==3.0.6",
# ]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import altair as alt
    import marimo as mo
    import pandas as pd

    return alt, mo, pd


@app.cell
def _(pd):
    # Pinned to the commit of the September 2026 snapshot so the numbers in
    # this notebook stay the same even if the repository gets a newer file.
    CSV_URL = (
        "https://raw.githubusercontent.com/supertj/youtube-monetization-by-country/"
        "8d3ba4940d8e19f65527ac68cbde2b150177e56f/"
        "youtube-monetization-signals-by-country.csv"
    )
    raw = pd.read_csv(CSV_URL, keep_default_na=False)
    return (raw,)


@app.cell(hide_code=True)
def _(mo, raw):
    _channels = int(raw["channels_checked"].sum())
    _unable = int(raw["unable_to_determine"].sum())
    mo.vstack(
        [
            mo.md(
                f"""
                # YouTube monetization signals by country

                In November 2023 YouTube removed the public flag that said
                whether a channel is monetized. Unless you own the channel,
                you can now only infer it from what the channel shows in
                public. This open dataset did that for
                {_channels:,} trending channels in {len(raw)} countries in
                September 2026. The question here is narrow: how much of the
                gap between countries survives once you account for small
                samples and for the channels that couldn't be classified?
                """
            ),
            mo.hstack(
                [
                    mo.stat(f"{len(raw)}", label="Countries"),
                    mo.stat(f"{_channels:,}", label="Channels checked"),
                    mo.stat(
                        f"{int(raw['channels_checked'].median())}",
                        label="Median per country",
                    ),
                    mo.stat(
                        f"{_unable / _channels:.1%}",
                        label="Unable to determine",
                    ),
                ],
                justify="start",
                gap=2,
            ),
        ]
    )
    return


@app.cell
def _(pd):
    def wilson_interval(hits, n, z=1.96):
        """95% Wilson score interval for a binomial share.

        Takes pandas Series of counts. Better than the textbook p ± 1.96·SE
        at n ≈ 60 and near 0 or 1 (several countries here have 0 or 1
        "likely" channels).
        """
        share = hits / n
        denom = 1 + z**2 / n
        center = (share + z**2 / (2 * n)) / denom
        half = z * ((share * (1 - share) / n + z**2 / (4 * n**2)) ** 0.5) / denom
        # At 0 hits the lower bound is 0 in exact arithmetic, but floating point
        # can leave it a hair below zero, which then prints as "-0.0%".
        return (center - half).clip(lower=0), (center + half).clip(upper=1)

    # Sanity check against known values: 0 of 60 gives 0% to about 6%.
    _low, _high = wilson_interval(pd.Series([0]), pd.Series([60]))
    assert _low[0] == 0 and abs(_high[0] - 0.0602) < 0.001
    return (wilson_interval,)


@app.cell
def _(mo, raw):
    measure = mo.ui.dropdown(
        options={
            "Likely monetized": "likely",
            "Likely or possibly monetized": "likely_or_possibly",
        },
        value="Likely monetized",
        label="Count as monetized",
    )
    countries = mo.ui.multiselect(
        options=sorted(raw["country"]),
        label="Only these countries (empty = all)",
    )
    mo.hstack([measure, countries], justify="start", gap=2)
    return countries, measure


@app.cell
def _(countries, measure, raw, wilson_interval):
    shares = raw.copy()
    if countries.value:
        shares = shares[shares["country"].isin(countries.value)]
    shares["hits"] = shares["likely_monetized"]
    if measure.value == "likely_or_possibly":
        shares["hits"] = shares["hits"] + shares["possibly_monetized"]
    shares["n"] = shares["channels_checked"]
    shares["share"] = shares["hits"] / shares["n"]
    shares["low"], shares["high"] = wilson_interval(shares["hits"], shares["n"])
    shares["label"] = shares["country"] + " (" + shares["n"].astype(str) + ")"
    shares = shares.sort_values("share", ascending=False).reset_index(drop=True)
    return (shares,)


@app.cell(hide_code=True)
def _(alt, measure, mo, shares):
    _y = alt.Y(
        "label:N",
        sort=shares["label"].tolist(),
        title=None,
        axis=alt.Axis(labelLimit=220, ticks=False, domain=False),
    )
    _tooltip = [
        alt.Tooltip("country:N", title="Country"),
        alt.Tooltip("hits:Q", title="Channels with the signals"),
        alt.Tooltip("n:Q", title="Channels checked"),
        alt.Tooltip("share:Q", title="Share", format=".1%"),
        alt.Tooltip("low:Q", title="95% interval, low", format=".1%"),
        alt.Tooltip("high:Q", title="95% interval, high", format=".1%"),
    ]
    _x = alt.X(
        "low:Q",
        title=f"{measure.selected_key}, share of channels checked",
        axis=alt.Axis(format="%", grid=True, gridOpacity=0.4),
        scale=alt.Scale(domain=[0, 1]),
    )
    _interval = (
        alt.Chart(shares)
        .mark_rule(strokeWidth=2, color="#86b6ef")
        .encode(x=_x, x2="high:Q", y=_y, tooltip=_tooltip)
    )
    _dot = (
        alt.Chart(shares)
        .mark_circle(size=64, color="#1c5cab", opacity=1)
        .encode(x="share:Q", y=_y, tooltip=_tooltip)
    )
    share_chart = mo.ui.altair_chart(
        (_interval + _dot).properties(
            width="container", height=alt.Step(16)
        ),
        chart_selection="point",
        legend_selection=False,
    )
    mo.vstack(
        [
            mo.md(
                "## Share of channels with the signals, with 95% intervals\n\n"
                "Dot = share of the channels checked in that country. Line = "
                "95% Wilson interval. The number after each country is how "
                "many channels were checked. Click a dot (shift-click for "
                "more) to pull its row into the table below."
            ),
            share_chart,
        ]
    )
    return (share_chart,)


@app.cell(hide_code=True)
def _(mo, share_chart, shares):
    _picked = share_chart.apply_selection(shares)
    _rows = shares if len(_picked) == 0 else _picked
    mo.ui.table(
        _rows[
            [
                "country",
                "n",
                "hits",
                "share",
                "low",
                "high",
                "unable_to_determine",
                "median_subscribers",
            ]
        ].rename(
            columns={
                "n": "checked",
                "hits": "with signals",
                "share": "share",
                "low": "95% low",
                "high": "95% high",
                "unable_to_determine": "unable",
                "median_subscribers": "median subs",
            }
        ),
        format_mapping={
            "share": "{:.1%}".format,
            "95% low": "{:.1%}".format,
            "95% high": "{:.1%}".format,
            "median subs": "{:,}".format,
        },
        selection=None,
        page_size=10,
        show_column_summaries=False,
        show_data_types=False,
    )
    return


@app.cell(hide_code=True)
def _(mo, shares):
    _top = shares.iloc[0]
    _median_width = (shares["high"] - shares["low"]).median()
    _comparison = ""
    if len(shares) > 1:
        _overlapping = int((shares["high"] >= _top["low"]).sum()) - 1
        _comparison = (
            f"{_top['country']} comes out on top at {_top['share']:.0%}, but "
            f"{_overlapping} of the other {len(shares) - 1} countries shown "
            "have intervals that reach into its interval. "
        )
    mo.md(
        f"""
        **How to read it.** With about 60 channels per country, the median
        interval is {_median_width * 100:.0f} percentage points wide.
        {_comparison}Countries a few points apart can't be told apart with
        samples this size, so compare the intervals, not just the dots.

        Where you draw the line matters as much. Switch to "Likely or possibly
        monetized" and the order changes a lot. Tanzania has no channels in
        "likely" but 48 of 60 in "possibly", which puts it in the top third
        on the wider measure. Nigeria goes from 10 of 60 to 52 of 60.

        The interval only covers sampling noise. It treats the channels in a
        country as a random draw from a larger group of similar channels and
        says nothing about how the channels were picked in the first place
        (see the notes at the bottom).
        """
    )
    return


@app.cell(hide_code=True)
def _(alt, mo, shares):
    _verdicts = {
        "likely_monetized": "Likely",
        "possibly_monetized": "Possibly",
        "unable_to_determine": "Unable to determine",
        "likely_not_monetized": "Likely not",
    }
    _long = shares.melt(
        id_vars=["country", "label", "n"],
        value_vars=list(_verdicts),
        var_name="verdict",
        value_name="channels",
    )
    _long["verdict"] = _long["verdict"].map(_verdicts)
    _long["order"] = _long["verdict"].map(
        {name: i for i, name in enumerate(_verdicts.values())}
    )
    _long["share"] = _long["channels"] / _long["n"]
    # Blue for the two "yes" tiers, orange for "no", neutral gray for
    # "can't tell" so it reads as missing information rather than a result.
    _colors = alt.Scale(
        domain=list(_verdicts.values()),
        range=["#1c5cab", "#6da7ec", "#a8a7a0", "#eb6834"],
    )
    _chart = (
        alt.Chart(_long)
        .mark_bar(stroke="white", strokeWidth=1)
        .encode(
            x=alt.X(
                "sum(share):Q",
                stack="normalize",
                title="Share of channels checked",
                axis=alt.Axis(format="%"),
            ),
            y=alt.Y(
                "label:N",
                sort=shares["label"].tolist(),
                title=None,
                axis=alt.Axis(labelLimit=220, ticks=False, domain=False),
            ),
            color=alt.Color(
                "verdict:N",
                scale=_colors,
                title=None,
                legend=alt.Legend(orient="top", direction="horizontal"),
            ),
            order=alt.Order("order:Q"),
            tooltip=[
                alt.Tooltip("country:N", title="Country"),
                alt.Tooltip("verdict:N", title="Verdict"),
                alt.Tooltip("channels:Q", title="Channels"),
                alt.Tooltip("share:Q", title="Share", format=".1%"),
            ],
        )
        .properties(width="container", height=alt.Step(16))
    )
    mo.vstack(
        [
            mo.md(
                "## The full verdict mix\n\n"
                "Each checked channel got one of four verdicts. Same country "
                "order as above. Gray is the share that couldn't be classified."
            ),
            _chart,
        ]
    )
    return


@app.cell(hide_code=True)
def _(alt, measure, mo, shares):
    _bounds = shares.assign(
        if_none=shares["hits"] / shares["n"],
        if_all=(shares["hits"] + shares["unable_to_determine"]) / shares["n"],
        classified_only=shares["hits"]
        / (shares["n"] - shares["unable_to_determine"]),
    )
    _y = alt.Y(
        "label:N",
        sort=shares["label"].tolist(),
        title=None,
        axis=alt.Axis(labelLimit=220, ticks=False, domain=False),
    )
    _tooltip = [
        alt.Tooltip("country:N", title="Country"),
        alt.Tooltip("unable_to_determine:Q", title="Unable to determine"),
        alt.Tooltip("if_none:Q", title="If none of them qualify", format=".1%"),
        alt.Tooltip("if_all:Q", title="If all of them qualify", format=".1%"),
        alt.Tooltip(
            "classified_only:Q", title="Leaving them out", format=".1%"
        ),
    ]
    _range = (
        alt.Chart(_bounds)
        .mark_bar(height=8, cornerRadius=4, color="#c9c8c1")
        .encode(
            x=alt.X(
                "if_none:Q",
                title=f"{measure.selected_key}, share of channels",
                axis=alt.Axis(format="%", grid=True, gridOpacity=0.4),
                scale=alt.Scale(domain=[0, 1]),
            ),
            x2="if_all:Q",
            y=_y,
            tooltip=_tooltip,
        )
    )
    _dot = (
        alt.Chart(_bounds)
        .mark_circle(size=64, color="#1c5cab", opacity=1)
        .encode(x="classified_only:Q", y=_y, tooltip=_tooltip)
    )
    _widest = _bounds.sort_values("unable_to_determine").iloc[-1]
    mo.vstack(
        [
            mo.md(
                f"""
                ## What the unclassified channels could do

                The gray bar runs from "none of the unclassified channels
                qualify" to "all of them do". The dot is the share among
                classified channels, which is what you get if the unclassified
                ones split the same way as the rest. In
                {_widest["country"]}, {_widest["unable_to_determine"]} of
                {_widest["n"]} channels came back "unable to determine", so
                its share could sit anywhere from {_widest["if_none"]:.0%} to
                {_widest["if_all"]:.0%}.
                """
            ),
            (_range + _dot).properties(width="container", height=alt.Step(16)),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## About the data

    **Source.** *YouTube monetization signals by country*, September 2026
    snapshot, published by MonetizationKit under CC BY 4.0 (Zenodo DOI
    10.5281/zenodo.23401599). The notebook reads the CSV from the dataset's
    GitHub repository, `supertj/youtube-monetization-by-country`.

    **What a verdict means.** Each channel was checked once against public
    signals of being in the YouTube Partner Program and put in one of four
    groups: likely monetized, possibly monetized, likely not, or unable to
    determine. A verdict is an inference from what the channel shows in
    public, not a statement from YouTube.

    **Which channels.** The channels behind each region's most popular
    videos, pulled from the YouTube Data API across 111 regions and 10
    categories, then filled toward a target number per country. These are
    trending channels, most with hundreds of thousands of subscribers. A
    low share for a country means the big channels checked there showed
    fewer signals. It is not that country's monetization rate.

    **Counts.** The 50 published rows add up to 3,080 channels. The
    dataset's README gives 3,192 channels checked overall; countries with
    fewer than 30 checked channels are not published.

    **The live page reads differently.** MonetizationKit keeps a
    [live version of this table](https://monetizationkit.com/data/youtube-monetized-channels-by-country/)
    that is recomputed weekly. It mixes in channels people looked up on the
    site, and it uses scoring rules revised in late September and early
    October 2026, with older checks re-scored, so its "likely" shares are
    higher than in this snapshot. Use one or the other. Don't mix rows from
    the two.
    """)
    return


if __name__ == "__main__":
    app.run()
