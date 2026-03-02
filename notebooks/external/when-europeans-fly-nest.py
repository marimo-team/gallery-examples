# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "altair==6.0.0",
#     "marimo>=0.19.7",
#     "matplotlib==3.10.3",
#     "pandas==2.2.3",
#     "plotly==6.1.2",
#     "seaborn==0.13.2",
# ]
# ///

import marimo

__generated_with = "0.17.6"
app = marimo.App()


@app.cell
def _():
    # Libraries
    import marimo as mo
    import pandas as pd

    # dataviz
    import seaborn as sns
    import altair as alt
    import plotly.express as px
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    # '%matplotlib inline' command is supported automatically in marimo

    # plt.rcParams["font.family"] = "Arial"
    return alt, mcolors, mo, pd, px, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### **A Python Dataviz Makeover Series**
    ---
    # Episode 1 – **When Europeans fly nest**

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="https://raw.githubusercontent.com/martidossi/data_viz_makeover/main/pics/europeans_parental_household.png" width="500" style="border-radius: 10px;">
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Despite its simplicity and overall ease of readability, this chart presents a few questionable design choices that could be revisited to enhance clarity and effectiveness (_e.g., what drives the positioning of the bubbles? Why do all the circles share the same size, even though their underlying values differ? Is the chosen colormap truly effective in this context?_). It is a bubble chart representing the average age at which young people leave home in European countries. Each country is depicted by a bubble indicating the related average age, with the bubble color corresponding to an age group, as shown in the legend. See [references](#references).

    🔍 **The goal** of this notebook is to explore alternative visualization approaches that represent the same data (and more) more effectively and intuitively, eventually with more contextual information.

    💻 **How**: everything is developed in Python. You're currently in a [Marimo](https://marimo.io/) notebook, an interactive and lightweight Python environment designed for enhanced data exploration and storytelling. Depending on the specific need, visualizations are built using [Altair](https://altair-viz.github.io/) or [Plotly](https://plotly.com/python/). If you're interested in the **code** behind, you can click on the "Show code" toggle button in the top right corner of the notebook interface:

    <img src="https://raw.githubusercontent.com/martidossi/data_viz_makeover/main/pics/see_code.png" width="300">

    ⚠️ _**Note**: this notebook is best viewed on a desktop screen for full functionality and layout, some elements may not display properly on mobile._

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Index

    **Part 1**: [Chart makeover](#1-chart-makeover)

    - [1.1 Data](#11-data)
    - [1.2 Barplot](#12-barplot)
    - [1.3 Bubble chart](#13-bubble-chart)
    - [1.4 Choropleth map](#14-choropleth-map)

    **Part 2**: [Adding contextual data](#2-adding-contextual-data)

    - [2.1 Data](#21-data)
    - [2.2 Ranged dot plot](#22-ranged-dot-plot)
    - [2.3 Heatmaps](#23-heatmaps)

    [References](#references)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ----

    # 1. Chart makeover
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1.1 Data
    The first step to better understand and improve the visualization involves the data itself: we start by recreating the dataset from scratch based on the values shown in the image.

    🔢 **How is this metric calculated?** We're looking at the _estimated average age at which young people leave their parental home_ across European countries. According to Eurostat, this represents the age at which 50% of the population no longer live in a household with their parent(s) (see [references](#references) for more details).
    """)
    return


@app.cell
def _(pd):
    data = [
        ['Finland', 21.3, 'below 23'],
        ['Estonia', 22.7, 'below 23'],
        ['Sweden', 21.4, 'below 23'],
        ['Denmark', 21.7, 'below 23'],
        ['Netherlands', 23, '23-25'],
        ['Germany', 23.8, '23-25'],
        ['Lithuania', 24.7, '23-25'],
        ['France', 23.4, '23-25'],
        ['Belgium', 26.3, '25-27'],
        ['Ireland', 26.9, '25-27'],
        ['Luxembourg', 26.8, '25-27'],
        ['Latvia', 26.8, '25-27'],
        ['Austria', 25.3, '25-27'],
        ['Czechia', 25.9, '25-27'],
        ['Poland', 28.9, '27-29'],
        ['Hungary', 27.1, '27-29'],
        ['Cyprus', 27.5, '27-29'],
        ['Romania', 27.7, '27-29'],
        ['Slovenia', 29.4, '29-31'],
        ['Greece', 30.7, '29-31'],
        ['Malta', 30.1, '29-31'],
        ['Portugal', 29.7, '29-31'],
        ['Spain', 30.3, '29-31'],
        ['Italy', 30.0, '29-31'],
        ['Bulgaria', 30.3, '29-31'],
        ['Slovakia', 30.8, '29-31'],
        ['Croatia', 33.4, 'over 31'],
    ]

    df = pd.DataFrame(
        data=data,
        columns=['country', 'avg_age_22', 'age_cluster']
    ).sort_values(by='avg_age_22')

    df
    return (df,)


@app.cell
def _(df):
    print(f'Number of rows: {df.shape[0]}, number of columns: {df.shape[1]}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1.2 Barplot
    The easiest yet most effective way to redesign this visualization is by using a bar chart. In addition to changing the visual model, we also update the **color scheme**: rather than using the rainbow palette from the original data visualization, we apply a more perceptually friendly one from the Seaborn library, named `magma`. For the sake of readability, we also opt for a horizontal bar chart.
    """)
    return


@app.cell
def _(df, mcolors, sns):
    # Rather than the color palette used in the chart, we use this one:
    palette = sns.color_palette("rocket", n_colors=df.age_cluster.nunique())
    # (for more color palette: https://www.practicalpythonfordatascience.com/ap_seaborn_palette).

    # Get hex codes:
    hex_colors = [mcolors.to_hex(color) for color in palette]

    # Add a color column to the df:
    dict_age_color = dict(zip(df.age_cluster.unique()[::-1], hex_colors))
    df['color'] = df.age_cluster.map(dict_age_color)
    return (dict_age_color,)


@app.cell
def _(mo):
    # Dropdown to decide how to sort:
    sort_by = mo.ui.dropdown(
        options=["by average age", "by country name"],
        value="by average age",
        label="How do you want to sort the bars?",
    )
    mo.md(f"### Sort values\n\n{sort_by}")
    return (sort_by,)


@app.cell
def _(alt, df, dict_age_color, mo, pd, sort_by):
    sort_method = "ascending" if sort_by.value == "by country name" else "-x"

    bar = (
        alt.Chart(df)
            .mark_bar(opacity=0.85)
            .encode(
                x=alt.X("avg_age_22:Q", title="Average age"),
                y=alt.Y("country:N", title=None, sort=sort_method),
                color=alt.Color(
                    "age_cluster:N",
                    title="Age group", # legend title
                    scale=alt.Scale(
                        domain=list(dict_age_color.keys()),
                        range=list(dict_age_color.values())
                    ),
                    sort=list(dict_age_color.keys())
                ),
                tooltip=[alt.Tooltip('avg_age_22:Q', title='Avg age')]
            )
            .properties(
                title="When Europeans fly nest (2022)",
                height=500
            )
    )

    # Vertical line at x = 26.4 (avg EU value across the 27 countries)
    # This value is not in the dataset, but we know it from the chart
    avg_val = 26.4
    df_avg_val_22 = pd.DataFrame({'x': [avg_val], 'label': ['avg EU value']})

    vline_label = (
        alt.layer(
            alt.Chart(df_avg_val_22) # v line
                .mark_rule(color='black', strokeDash=[4, 2])
                .encode(x='x:Q'),
            alt.Chart(df_avg_val_22) # v line label
                .mark_text(align='left', dx=5, dy=475, fontSize=12)
                .encode(
                    x='x:Q',
                    y=alt.value(0),
                    text='label:N'
                )
        )
    )

    # Combine all layers
    barh_chart = (
        mo.ui.altair_chart(bar + vline_label)
            .properties(title="When Europeans fly nest (2022)", height=500)
            .configure_title(fontSize=20)
            .configure_axis(titleFontSize=12)
            .configure_legend(
                labelFontSize=10,
                titleFontSize=12
            )
    )

    barh_chart
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1.3 Bubble chart
    This visualization builds on the original bubble chart concept. Visually grouping the bubbles by cluster reinforces a sense of mental order and enables quick comparisons of cluster sizes. Although each bubble's size is now proportional to its encoded value, the differences are so small that the variations in area are barely noticeable.
    """)
    return


@app.cell
def _(df):
    # We add a few new columns that are functional to the creation of this chart
    df['idx'] = df.groupby('age_cluster').cumcount().add(1)

    dict_age_cluster_id = dict(zip(df.age_cluster.unique(), range(6)))
    df['age_cluster_id'] = df.age_cluster.map(dict_age_cluster_id)
    return (dict_age_cluster_id,)


@app.cell
def _(alt, df, dict_age_color, mo):
    cluster_labels = df['age_cluster'].unique().tolist()

    # Build JS expression for label replacement
    label_expr = "datum.value == null ? '' : " + " || ".join([f"(datum.value == {i} ? '{label}' : '')" for i, label in enumerate(cluster_labels)])

    scaling_factor = 100

    bubbles = (
        alt.Chart(df)
            .mark_circle()
            .encode(
                x=alt.X("idx:Q", title=None, axis=None),
                y = alt.Y(
                    "age_cluster_id:Q",
                    title="Age group",
                    axis=alt.Axis(
                        tickCount=len(cluster_labels),
                        labelExpr=label_expr,
                    )
                ),
                size=alt.Size(
                    "avg_age_22:Q",
                    scale=alt.Scale(
                        type='sqrt',
                        range=[
                            scaling_factor*df.avg_age_22.min(),
                            scaling_factor*df.avg_age_22.max()]
                    ),
                    legend=None
                ),
                color=alt.Color(
                    "age_cluster:N",
                    scale=alt.Scale(
                        domain=list(dict_age_color.keys()),
                        range=list(dict_age_color.values())
                    ),
                    legend=None
                ),
                tooltip=[
                    alt.Tooltip('country:N', title='Country'),
                    alt.Tooltip('avg_age_22:Q', title='Avg age'),
                ],
            )
            .properties(
                width=600,
                height=400,
                title="When Europeans fly nest (2022)",
            )
    )


    # Label over bubbles
    bubble_labels = (
        alt.Chart(df)
            .mark_text(dy=0, fontSize=11)
            .encode(
                x="idx:Q", # shift text above the bubble
                y="age_cluster_id:Q",
                text="country:N"
            )
    )

    # Combine and autosize
    scatter_plot = (
        mo.ui.altair_chart(bubbles + bubble_labels)
            .properties(
                title="When Europeans fly nest (2022)",
                height=400,
                width=600,
                autosize=alt.AutoSizeParams(
                    type="fit",  # or "fit-x", "pad"
                    contains="padding"
                )
            )
            .configure_axis(
                titleFontSize=12
            )
            .configure_title(fontSize=20)
            .configure_view(
                strokeWidth=0
            )
    )

    scatter_plot
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1.4 Choropleth map

    When dealing with geographic data, maps provide the most immediate and intuitive representation. In this case, coloring countries based on their age group bins allows us to easily visualize large-scale trends.
    """)
    return


@app.cell
def _(df, dict_age_color, mo, px):
    fig_map = px.choropleth(
        df,
        locations='country',
        locationmode='country names',
        color='age_cluster',
        scope='europe',
        color_discrete_map=dict_age_color,
        title='When Europeans fly nest (2022)',
        hover_data={'country': True, 'avg_age_22': True}
    )

    fig_map.update_geos(
        fitbounds="locations",
        visible=False,
        showframe=True,
        showcoastlines=True,
        projection_type='natural earth'
    )

    fig_map.update_layout(
        title_x=0.5, # centers the title
        height=400,
        margin={"r":0, "t": 50, "l": 0, "b": 0},
        legend_title_text="Age group"
    )

    fig_map.update_traces(
        hovertemplate='<b>%{location}</b><br>Avg age %{customdata[1]}<extra></extra>'
    )

    mo.ui.plotly(fig_map)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # 2. Adding contextual data
    Continuing with the exercise, it may be helpful to explore the original data source (available on the Eurostat [website](https://ec.europa.eu/eurostat/web/main/home), see [references](#references)) to gain additional context and insights. _This is not related to modifications of the original chart, but rather to possible extensions._ In particular, the Eurostat website provides more granular data by gender and over time —data are available from 2000, but the earliest datasets are partially incomplete, so we focus only on the last 10 years, from 2015 to 2024.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2.1 Data
    Here's the complete dataset which includes one column for each year covered (`2015`, ..., `2024`), plus a feature `sex` which takes 3 values (`Total`, `Males` and `Females`) to distinguish by gender.
    """)
    return


@app.cell
def _(pd):
    # df_context = pd.read_csv('public/yth_demo.csv', sep=';')
    df_context = pd.read_csv(
        ''.join([
            'https://raw.githubusercontent.com/',
            'martidossi/',
            'data_viz_makeover/main/data/',
            'yth_demo.csv'
        ]),
        sep=';'
    )

    df_context = df_context[df_context.country!='European Union - 27 countries (from 2020)']
    df_context
    return (df_context,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ✔️ **Quality checks**:

    - We check that for each combination of `sex` and `year`, the number of available countries is consistent (see chart below). In particular, UK left EU in 2020, and other countries such as North Macedonia, Montenegro, and Türkiye have been excluded from recent data collections to maintain focus on official EU members.
    - We verify that the values in 2022 match those in the original chart.
    """)
    return


@app.cell
def _(alt, df_context, mo):
    # Create bar chart
    dict_sex_color = {
        'Females': "#eb495a",
        'Males': '#2d59cd',
        'Total': "#d1d1d1"
    }

    df_countries = (
        df_context
            .drop(['country'], axis=1)
            .groupby('sex')
            .apply(lambda x: x.notna().sum(), include_groups=False)
            .transpose()
            .reset_index()
            .rename(columns={'index': 'year'})
    )
    df_countries_long = (
        df_countries
            .melt(
                id_vars='year',
                var_name='sex',
                value_name='n_countries'
            )
    )

    chart_n_countries = (
        alt.Chart(df_countries_long)
            .mark_bar(
                size=10,
                opacity=0.8
            )
            .encode(
                x=alt.X('year:N', title='Year', axis=alt.Axis(labelAngle=0)),
                y=alt.Y('n_countries:Q', title='Number of countries'),
                color=alt.Color(
                    'sex:N',
                    title='Sex',
                    scale=alt.Scale(
                        domain=list(dict_sex_color.keys()),
                        range=list(dict_sex_color.values())
                    )
                ),
                xOffset='sex:N',
                tooltip=[alt.Tooltip("n_countries:Q", title="N. of countries")]
            )
            .properties(
                title='Number of countries per year and sex',
                width=400,
                height=200
            )
    )

    mo.ui.altair_chart(chart_n_countries)
    return (dict_sex_color,)


@app.cell
def _(df_context, mo):
    # Check which countries
    out_text = []
    for year in range(2020, 2025):
        drop_out_countries = set(df_context[['country', str(year)]].loc[df_context[str(year)].isna(), 'country'])
        out_text.append(f"In **{year}**: {', '.join(drop_out_countries)}")

    mo.md("📌 **Countries no longer present by year**\n\n" + ". \n".join(out_text))
    return


@app.cell
def _(df, df_context, mo):
    df_tmp_22 = df_context[['2022', 'country']]
    df_tmp_22 = df_tmp_22[df_tmp_22['2022'].notna()]
    missing_countries_22 = set(df_tmp_22.country) - set(df.country)
    mo.md("📌 **Countries from 2022 that are not present in the original data**\n\n" + ", ".join(missing_countries_22))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    These are both candidate countries. For the analysis, we focus only on **official EU member states** as of 2024 (same as in 2022), which are 27. This ensures consistency in the evaluation over time and automatically excludes any potential missing values.
    """)
    return


@app.cell
def _(df, df_context):
    df_context_eu = df_context[df_context['country'].isin(df.country.tolist())]
    return (df_context_eu,)


@app.cell
def _():
    # Check of remaining missing vals
    # df_context_eu.isna().sum().sum()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    📌 **Do the 2022 values match?**

    Not perfectly —there are three countries with slight differences, as shown in the `diff` column of the table below (head), sorted by `diff`. Moving forward, we will keep referencing the **original** data source (the one from Eurostat).
    """)
    return


@app.cell
def _(df, df_context):
    df_check_vals = (
        df_context[df_context.sex=='Total'][['2022', 'country']]
            .rename(columns={'2022': 'avg_age_eurostat'})
            .merge(df[['country', 'avg_age_22']], on='country', how='inner')
    )
    df_check_vals['diff'] = df_check_vals['avg_age_eurostat'] - df_check_vals['avg_age_22']
    df_check_vals.sort_values('diff').head(5)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    🔧 **Just a quick technicality**: as it stands, the dataset doesn't exactly match the characteristics of either wide or long format. For the sake of clarity and easier processing, we convert it to long format (see table below), so that each variable is a column and each observation a row. This choice is typically preferable as it accommodates datasets of arbitrary complexity. More about this [here](https://seaborn.pydata.org/tutorial/data_structure.html).
    """)
    return


@app.cell
def _(df_context_eu, pd):
    df_long = pd.DataFrame()

    for sex in df_context_eu.sex.unique():
        df_tmp = (
            df_context_eu[df_context_eu['sex']==sex]
                .drop('sex', axis=1)
                .melt(id_vars='country', var_name='year', value_name='value')
        )
        df_tmp['year'] = df_tmp['year'].astype('int')
        df_tmp['sex'] = sex
        df_long = pd.concat([df_long, df_tmp])

    df_long
    return (df_long,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2.2 Ranged dot plot
    The goal of this first chart is to compare the average age at which young people leave their parental home, broken down by **gender** in 2022, the year we've focused on so far. The visualization makes it very clear how, on average, girls consistently leave earlier than boys. We can also sort the countries in different ways to see how the rankings shift and explore the patterns!
    """)
    return


@app.cell
def _(mo):
    # Dropdown to decide how to sort:
    sort_dot_by = mo.ui.dropdown(
        options=["by overall age distance", "by overall age", "by female age", "by male age"],
        value="by overall age distance",
        label="Choose how to sort countries (largest values at the top): ",
    )
    mo.md(f"### Sort values\n\n{sort_dot_by}")
    return (sort_dot_by,)


@app.cell
def _(df_long):
    df_plot_dot = df_long[df_long.year==2022]

    sort_total = (
        df_plot_dot[df_plot_dot.sex=='Total']
            .sort_values(by='value', ascending=False).country.tolist()
    )

    sort_sex_m = (
        df_plot_dot[df_plot_dot.sex=='Males']
            .sort_values(by='value', ascending=False).country.tolist()
    )

    sort_sex_f = (
        df_plot_dot[df_plot_dot.sex=='Females']
            .sort_values(by='value', ascending=False).country.tolist()
    )

    df_diff = (
        df_plot_dot.pivot(index='country', columns='sex', values='value')
        .assign(diff=lambda d: (d['Males'] - d['Females']).abs())
        .reset_index()
        .sort_values('diff', ascending=False)
    )
    sort_diff = df_diff.country.tolist()

    dict_sorting_dot = {
        'by overall age distance': sort_diff,
        'by overall age': sort_total,
        'by female age': sort_sex_f,
        'by male age': sort_sex_m,
    }
    return df_plot_dot, dict_sorting_dot


@app.cell
def _(alt, df_plot_dot, dict_sex_color, dict_sorting_dot, mo, sort_dot_by):
    # Chart base
    chart_dot = (
        alt.Chart(df_plot_dot)
            .transform_filter(
                alt.FieldOneOfPredicate(field="sex", oneOf=["Females", "Males"])
            )
            .encode(
                x=alt.X(
                    "value:Q",
                    title="Average age",
                    scale=alt.Scale(domain=[18, 38])
                ),
                y=alt.Y(
                    "country:N",
                    title=None,
                    sort=dict_sorting_dot[sort_dot_by.value],
                )
            )
    )

    # Lines between dots:
    lines_dot = (
        chart_dot.mark_line(color="black", strokeWidth=1)
            .encode(detail="country:N")
    )

    # Dots:
    points_dot = (
        chart_dot.mark_point(
            size=140,
            opacity=0.8,
            filled=True,
        )
            .encode(
                color=alt.Color("sex:N")
                    .title("Sex")
                    .scale(
                        domain=['Females', 'Males'],
                        range=[dict_sex_color['Females'], dict_sex_color['Males']]
                    ),
                stroke=alt.value("black"),
                strokeWidth=alt.value(1),
                tooltip=[alt.Tooltip("value:Q", title="Avg age", format=".1f")]
            )
    )

    # Combine all layers
    ranged_dot_plot = (
        mo.ui.altair_chart(lines_dot + points_dot)
            .properties(
                title="When Europeans fly nest (2022)",
                width=500,
                height=500
            )
            .configure_title(fontSize=20)
            .configure_axis(
                titleFontSize=12
            )
            .configure_legend(
                labelFontSize=10,
                titleFontSize=12
            )
    )

    ranged_dot_plot
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2.3 Heatmaps
    Line charts are typically the go-to option to visualize how a variable changes **over time**. However, in this case, we're dealing with multiple countries and very similar trends both across and within them. A classic line chart would quickly turn into a spaghetti plot —messy and hard to read. There are many alternatives: for instance, using small multiples to dedicate a single chart to each country.

    Here, we choose to try a heatmap, where the average age over time is not represented by lines and points, but instead encoded as color, which is always one of the most intuitive and compact ways to spot patterns at a glance. Countries are sorted by increasing average age over time.
    """)
    return


@app.cell
def _(mo):
    # Dropdown to decide how to sort:
    viz_by = mo.ui.dropdown(
        options=["Total", "Females", "Males"],
        value="Total",
        label="Visualize by: ",
    )
    mo.md(f"### Gender selection\n\n{viz_by}")
    return (viz_by,)


@app.cell
def _(df_long, dict_age_cluster_id, pd, viz_by):
    # Add age cluster for all ages
    age_bins = [0, 23, 25, 27, 29, 31, 35] # sx included
    df_long['age_cluster'] = pd.cut(
        df_long['value'],
        bins=age_bins,
        labels=dict_age_cluster_id.keys(),
        right=False
    )

    # Sorting cols
    df_tmp_heatmap = df_long[df_long.sex==viz_by.value][['country', 'age_cluster']].value_counts().reset_index()
    df_tmp_heatmap['age_cluster_id'] = df_tmp_heatmap.age_cluster.map(dict_age_cluster_id)

    df_tmp_heatmap = df_tmp_heatmap.sort_values(by=['age_cluster_id', 'count'], ascending=[True, False])
    country_sorting_heatmap = df_tmp_heatmap.country.unique()
    return (country_sorting_heatmap,)


@app.cell
def _(alt, country_sorting_heatmap, df_long, dict_age_color, mo, viz_by):
    heatmap_discrete = (
        alt.Chart(df_long[df_long.sex==viz_by.value])
            .mark_rect(opacity=0.85, stroke='white', strokeWidth=0.5)
            .encode(
                alt.X('year:O', title='Year', axis=alt.Axis(labelAngle=0)),
                alt.Y('country:N', title=None, sort=country_sorting_heatmap[::-1]),
                color=alt.Color(
                    'age_cluster:N',
                    scale=alt.Scale(
                        domain=list(dict_age_color.keys()),
                        range=list(dict_age_color.values())),
                    legend=alt.Legend(title="Age group")
                ),
                tooltip=[alt.Tooltip('value:Q', title='Avg age')]
            )
            .properties(
                width=df_long['year'].nunique() * 50, # n px per year
                height=df_long['country'].nunique() * 20
            )
    )

    heatmap_discrete_plot = mo.ui.altair_chart(heatmap_discrete)
    heatmap_discrete_plot = (
        mo.ui.altair_chart(heatmap_discrete)
            .properties(
                title="When Europeans fly nest over time (2015-2024)",
                width=500,
                height=500
            )
            .configure_title(fontSize=20)
            .configure_axis(
                titleFontSize=12
            )
            .configure_legend(
                labelFontSize=10,
                titleFontSize=12
            )
    )
    heatmap_discrete_plot
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The version below uses a continuous color scale instead of a discrete one, which can help highlight more subtle differences and patterns across countries and years.
    """)
    return


@app.cell
def _(alt, country_sorting_heatmap, df_long, mo, viz_by):
    heatmap_continuous = (
        alt.Chart(df_long[df_long.sex==viz_by.value])
            .mark_rect(opacity=0.85, stroke='white', strokeWidth=0.5)
            .encode(
                alt.X('year:O', title='Year', axis=alt.Axis(labelAngle=0)),
                alt.Y('country:N', title=None, sort=country_sorting_heatmap[::-1]),
                alt.Color(
                    'value:Q',
                    scale=alt.Scale(scheme='magma', reverse=True)
                ),
                tooltip=[alt.Tooltip('value:Q', title='Avg age')]
            )
            .properties(
                width=df_long['year'].nunique() * 50, # n px per year
                height=df_long['country'].nunique() * 20
            )
    )

    heatmap_continuous_plot = mo.ui.altair_chart(heatmap_continuous)
    heatmap_continuous_plot = (
        mo.ui.altair_chart(heatmap_continuous)
            .properties(
                title="When Europeans fly nest over time (2015-2024)",
                width=500,
                height=500
            )
            .configure_title(fontSize=20)
            .configure_axis(
                titleFontSize=12
            )
            .configure_legend(
                labelFontSize=10,
                titleFontSize=12
            )
    )
    heatmap_continuous_plot
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Finally, this last heatmap shifts the perspective to also include gender in a single view. Earlier, with the [ranged dot plot](#22-ranged-dot-plot), we saw that (at least) in 2022 girls tended to leave home earlier than boys in every country. But does this pattern hold _across all 10 years_ in the dataset? In this chart, the color helps answer that question: shades closer to **<span style="color:#2d59cd">blue</span>** indicate that boys tend to leave home later, while tones leaning toward **<span style="color:#eb495a">red</span>** suggest that girls do. It's a quick way to spot trends and exceptions over time and across countries —and what's most striking is just how consistent this pattern remains across both time and place.
    """)
    return


@app.cell
def _(alt, df_long, dict_sex_color):
    # Create df with age differences (Males - Females)
    df_age_diff = (
        df_long
            .pivot(index=['country', 'year'], columns='sex', values='value')
            .assign(age_diff=lambda d: (d['Males'] - d['Females']))
            .reset_index()
            .sort_values('age_diff', ascending=False)
    )

    # Create color scale
    min_val = df_age_diff['age_diff'].min()
    max_val = df_age_diff['age_diff'].max()
    scale_val = round(max(abs(min_val), abs(max_val)))
    # This way the color intensity is set at the same distance for both extremes

    color_scale = alt.Scale(
        domain=[-scale_val, 0, scale_val],
        range=[dict_sex_color['Females'], '#ffffff', dict_sex_color['Males']]
    )

    # Country sorting
    country_sorting_heatmap_2 = df_age_diff.groupby('country').agg({'age_diff': 'sum'}).sort_values('age_diff', ascending=False)
    return color_scale, country_sorting_heatmap_2, df_age_diff


@app.cell
def _(alt, color_scale, country_sorting_heatmap_2, df_age_diff, df_long, mo):
    heatmap_sex = (
        alt.Chart(df_age_diff)
            .mark_rect(opacity=0.85, stroke='white', strokeWidth=0.5)
            .encode(
                alt.X('year:N', title='Year', axis=alt.Axis(labelAngle=0)),
                alt.Y('country:N', title=None, sort=country_sorting_heatmap_2[::-1]),
                alt.Color(
                    'age_diff:Q',
                    scale=color_scale
                ),
                tooltip=[
                    alt.Tooltip('Males:Q', title='Males avg age'),
                    alt.Tooltip('Females:Q', title='Females avg age'),
                    alt.Tooltip('age_diff:Q', title='Age difference')
                ]
            )
            .properties(
                width=df_long['year'].nunique() * 50, # n px per year
                height=df_long['country'].nunique() * 20
            )
    )

    heatmap_sex_plot = mo.ui.altair_chart(heatmap_sex)
    heatmap_sex_plot = (
        mo.ui.altair_chart(heatmap_sex)
            .properties(
                title="When Europeans fly nest over time by gender (2015-2024)",
                width=500,
                height=500
            )
            .configure_title(fontSize=20)
            .configure_axis(
                titleFontSize=12
            )
            .configure_legend(
                labelFontSize=10,
                titleFontSize=12
            )
    )
    heatmap_sex_plot
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ----

    ## 🎈Bonus chart

    To wrap up, a nice idea is to combine one of the earlier visualizations with some of the new information. The map below is much like the previous [one](#14-choropleth-map), but with two updates:

    1. it includes an animation showing changes from 2014 to 2025,
    2. it uses a continuous color scale, making it easier to spot subtle differences.
    """)
    return


@app.cell
def _(df_long, mo, px):
    fig_an = px.choropleth(
        df_long,
        locations='country',
        locationmode='country names',
        color='value',
        scope='europe',
        color_continuous_scale='magma_r',
        animation_frame='year',
        title='When Europeans fly nest (2014-2025)',
    )

    fig_an.update_geos(
        fitbounds="locations",
        visible=False,
        showframe=True,
        showcoastlines=True,
        projection_type='natural earth'
    )

    fig_an.update_layout(
        title_x=0.5,
        height=500,
        margin={"r":0, "t":50, "l":0, "b":0},
        coloraxis_colorbar=dict(title="Average age")
    )

    mo.ui.plotly(fig_an)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## References

    - The dataset ::lucide:arrow-down-right:: [URL](https://ec.europa.eu/eurostat/databrowser/view/yth_demo_030/default/table?lang=en)

    - The viz source ::lucide:arrow-down-right:: [URL](https://www.luzmo.com/blog/bad-data-visualization)

    - Related article with data ::lucide:arrow-down-right:: [URL](https://ec.europa.eu/eurostat/web/products-eurostat-news/w/ddn-20230904-1#:~:text=In%202022%2C%20young%20people%20across,average%20varied%20among%20EU%20countries)

    - _Inspired by_ **Storytelling with Data Blog** by Cole Nussbaumer Knaflic ::lucide:arrow-down-right:: [URL](https://www.storytellingwithdata.com/makeovers)

    - GitHub repo ::lucide:arrow-down-right:: [URL](https://github.com/martidossi/data_viz_makeover)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Thanks for following along, see you in the next one! 🐱

    💬 Feel free to get in touch if you have any questions —the project is still in its early stages, and I'd really value your feedback! 🫶

    - _How would you visualize this data? Any other chart ideas pop into your mind?_
    - _Notice anything off or that could be done better?_
    - _Is there something else you'd be curious to dive into in the next notebooks?_

    **💌 Contacts** Martina Dossi [martinadossi.hello at gmail.com]
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    <img src="https://marimo.app/logotype-wide.svg" width="200"/>
    """)
    return


if __name__ == "__main__":
    app.run()
