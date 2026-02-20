# /// script
# requires-python = ">=3.10,<3.12"
# dependencies = [
#     "marimo",
#     "cvxpy",
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "plotly",
#     "scikit-learn",
#     "scipy",
#     "sig-decomp==0.3.2",
# ]
# ///

import marimo

__generated_with = "0.19.11"
app = marimo.App()

with app.setup:
    import marimo as mo
    import importlib
    import os
    import urllib.request
    import gfosd
    import gfosd.components as gfc
    import inputs.components as complib
    import inputs.explainer as explainer
    import inputs.intro_problem as intro_problem
    import inputs.problems as problems
    import matplotlib.pyplot as plt
    import numpy as np


@app.cell
def _():
    mo.image(
        src="https://bmeyers.github.io/assets/gismo-cropped.png",
    ).center()
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    # Signal Decomposition
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    This app is a hands-on introduction to _signal decomposition_, an
    age-old problem about breaking down a complex signal, also known as a
    time series, into the sum of simpler interpretable ones.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    The simpler signals that come out of a decomposition are called
    _components_. When doing a signal decomposition, we have to specify
    two things:

    1. How many components do we want?
    2. What kinds of components, or "component classes", do we want?
    """)
    return


@app.cell
def _():
    component_options = [
        complib.Components.TREND_LINE,
        complib.Components.PERIODIC,
        complib.Components.PIECEWISE_CONSTANT,
    ]

    # will be used to track which options the user has tried
    get_component_radio_tracker, set_component_radio_tracker = mo.state(set())

    component_radio = mo.ui.radio(
        component_options,
        label="**Component Class**",
        on_change=lambda w: set_component_radio_tracker(
            lambda v: v.union({w})
        ),
    )
    other_component_radio = mo.ui.radio(
        component_options, label="**Component Class**"
    )
    return (
        component_options,
        component_radio,
        get_component_radio_tracker,
        other_component_radio,
    )


@app.cell
def _(component_options, get_component_radio_tracker):
    def user_tried_all_components():
        return len(get_component_radio_tracker()) == len(component_options)

    return (user_tried_all_components,)


@app.cell
def _(component_radio, other_component_radio):
    radios = [component_radio, other_component_radio]
    return (radios,)


@app.cell
def _():
    intro = intro_problem.IntroProblem()

    mo.md(
        f"""
        ## Part 1: Understanding components

        Let's build a decomposition for this signal:

        {mo.as_html(intro.plot())}
        """
    )
    return (intro,)


@app.cell
def _(get_show_third_component):
    _n_components = "2" if not get_show_third_component() else "3"
    _three_component_text = " and third " if get_show_third_component() else ""

    mo.md(
        f"""
        - Every decomposition needs at least two components. We'll make
        **{_n_components}** components.

        - We have to choose the component classes out of a library of
        available classes.

        - _Noise component_: The first component always represents noise,
        or a residual; we typically want it to be small.

        - _Other components_: What to choose for
        the other components depends on what properties we suspect the
        underlying simpler signals to have.

        Use the radio buttons to try a few options for the second
        {_three_component_text} component.
        """
    )
    return


@app.cell
def _(get_show_third_component, intro, radios):
    # Show radios
    (
        mo.hstack(
            [
                radios[0],
                *intro.plot_decomp(
                    (radios[0].value,)
                    if radios[0].value is not None
                    else tuple(),
                    width=2.5,
                    height=2,
                    min_plots=2,
                ),
            ],
        )
        if not get_show_third_component()
        else mo.hstack(radios, justify="space-around")
    )
    return


@app.cell
def _(get_show_third_component, intro, radios):
    # Plot 3-component decomposition
    (
        None
        if not get_show_third_component()
        else mo.hstack(
            [
                *intro.plot_decomp(
                    tuple(r.value for r in radios if r.value is not None),
                    width=2.5,
                    height=2,
                    min_plots=3,
                )
            ]
        )
    )
    return


@app.cell
def _(get_show_third_component, radios):
    # Component explainer callout
    (
        mo.md(explainer.explainer(radios[0].value)).callout(kind="neutral")
        if not get_show_third_component() and radios[0].value is not None
        else None
    )
    return


@app.cell
def _():
    get_show_third_component, set_show_third_component = mo.state(False)

    add_component_button = mo.ui.button(
        label="Add another component 🔧",
        on_change=lambda _: set_show_third_component(True),
    )
    remove_component_button = mo.ui.button(
        label="Remove third component 🔧",
        on_change=lambda _: set_show_third_component(False),
    )
    return (
        add_component_button,
        get_show_third_component,
        remove_component_button,
    )


@app.cell
def _(
    add_component_button,
    get_show_third_component,
    solved,
    user_tried_all_components,
):
    # Add component callout
    (
        mo.md(
            f"""
            {add_component_button.center()}

            It looks like you tried all the component classes. No matter which
            one you tried, the noise signal that came out didn't really look
            small and random. _When that happens, that usually means we 
            need to add another component to our decomposition_.
            """
        ).callout(kind="warn")
        if user_tried_all_components()
        and not get_show_third_component()
        and not solved.now
        else None
    )
    return


@app.cell
def _():
    class StickyBool:
        value = False

        def set(self):
            self.value = True
            return self

        def __bool__(self):
            return self.value

    solved_ever = StickyBool()
    return (solved_ever,)


@app.cell
def _(radios, solved_ever):
    _chosen_components = set([r.value for r in radios])

    class Solved:
        def __init__(self, sticky_bool):
            self.ever = sticky_bool
            self.now = False

    solved = Solved(solved_ever)
    solved.now = _chosen_components == set(
        [complib.Components.TREND_LINE, complib.Components.PERIODIC]
    )
    if solved.now:
        solved_ever.set()
    return (solved,)


@app.cell
def _(get_show_third_component, solved):
    # Solved callout
    (
        mo.md(
            f"""
            🎉 **_You did it!_**

            The noise is small and looks random, and the decomposition has
            linear and seasonal components. In fact, the signal was
            generated by adding a line to a sine wave.

            In the real world, where signals are measurements of messy
            data, you won't ever know if you've "solved" a signal
            decomposition problem. Instead you'll have to use your own
            intuition to guide the selection of component classes.

            In this sense, signal decomposition is kind of like
            unsupervised machine learning tasks, like clustering or
            embedding: it's up to you to judge whether or not your
            decomposition is a good one.

            **Part 2 is now available**.
            """
        ).callout(kind="success")
        if get_show_third_component() and solved.now
        else None
    )
    return


@app.cell
def _(solved):
    (
        mo.md(
            """**Heads up!**

             In part 2, you'll encounter something new: component class 
             parameters. Parameters are knobs you can use to customize 
             components.

             You'll also encounter some new component classes. If you're
             ever unsure about what a component does, check out the reference
             at the bottom of the page.
             """
        ).callout(kind="warn")
        if solved.ever
        else None
    )
    return


@app.cell
def _(
    get_show_third_component,
    remove_component_button,
    solved,
    user_tried_all_components,
):
    # Remove component button: shown when 3 components active
    (
        mo.md(
            f"""
            {remove_component_button.center()}

            Try making a decomposition with three components.
            We'll tell you once you've found the "right" decomposition.

            Hint: we generated the signal by adding a seasonal fluctuation
            to a line with constant slope.
            """
        ).callout(kind="warn")
        if user_tried_all_components()
        and get_show_third_component()
        and not solved.now
        else None
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Part 2: More Decompositions
    """)
    return


@app.cell
def _():
    selected_problem = mo.ui.dropdown(
        {
            problems.MaunaLoa.name(): problems.MaunaLoa,
            problems.ChangePoint.name(): problems.ChangePoint,
            problems.SolarPower.name(): problems.SolarPower,
            problems.Soiling.name(): problems.Soiling,
            problems.CustomDataProblem.name(): problems.CustomDataProblem,
        },
        label="Choose a signal:",
    )
    return (selected_problem,)


@app.cell
def _(selected_problem, solved):
    # Solve part 1 callout
    (
        mo.md(
            f"""
            **Part 1** taught you the basics of signal decomposition. In
            **Part 2**, you'll apply what you learned to decompose some real-world
            signals, as well as some more synthetic ones.

            Start by choosing a signal. We
            recommend starting with the CO~2~ signal, which tracks atmospheric
            emissions at the Mauna Loa Observatory.

            {selected_problem.center()}
            """
        )
        if solved.ever
        else mo.md(
            """
            🛑 Part 2 isn't available yet. Keep experimenting with
            component classes in part 1 until you've "solved" the decomposition,
            then return here.
            """
        ).callout(kind="alert")
    )
    return


@app.cell
def _():
    data_uploader = mo.ui.file(filetypes=[".csv"], kind="area")
    csv_has_header = mo.ui.checkbox(value=True)
    return csv_has_header, data_uploader


@app.cell
def _(csv_has_header, data_uploader):
    CSV_READ_ERROR = False

    def read_uploaded_csv():
        from io import BytesIO
        import pandas as pd

        if data_uploader.value:
            header = "infer" if csv_has_header.value else None
            try:
                return pd.read_csv(
                    BytesIO(data_uploader.contents()), header=header
                )
            except Exception:
                global CSV_READ_ERROR
                CSV_READ_ERROR = True
        return None

    _uploaded_df = read_uploaded_csv()
    column_name = (
        mo.ui.dropdown({str(c): c for c in _uploaded_df.columns.tolist()})
        if _uploaded_df is not None
        else None
    )

    def show_csv_parameters():
        if _uploaded_df is not None:
            return f"""

            _Check if your data file has a header:_ {csv_has_header}

            _Column containing signal:_ {column_name}

            Here's a preview of what you uploaded:

            {mo.hstack([_uploaded_df.head()], justify="center")}
            """
        else:
            return ""

    return CSV_READ_ERROR, column_name, read_uploaded_csv, show_csv_parameters


@app.cell
def _(data_uploader, selected_problem, show_csv_parameters):
    mo.stop(selected_problem.value != problems.CustomDataProblem)

    mo.md(
        f"""
        **Upload a signal.**

        You can upload your own signal and use this app to build a 
        decomposition for it. Your signal should be a CSV file.

        {data_uploader}

        {show_csv_parameters()}
        """
    ).callout()
    return


@app.cell
def _(CSV_READ_ERROR):
    mo.stop(not CSV_READ_ERROR)

    mo.md("There was a problem reading your CSV!").callout(kind="alert")
    return


@app.cell
def _(
    CSV_READ_ERROR,
    column_name,
    data_uploader,
    read_uploaded_csv,
    selected_problem,
):
    def _construct_problem(problem_class):
        if problem_class == problems.CustomDataProblem:
            if (
                CSV_READ_ERROR
                or not data_uploader.value
                or column_name.value is None
            ):
                return None
            df = read_uploaded_csv()
            return problems.CustomDataProblem(
                df[column_name.value], column_name.value
            )
        elif problem_class is not None:
            return problem_class()
        else:
            return None

    problem = _construct_problem(selected_problem.value)

    # State associated with each problem
    # The number of components in the decomposition
    get_k, set_k = mo.state(2)

    # The components selected: maintain as state so we can pre-populate
    # them as components are added and removed
    get_selected_components, set_selected_components = mo.state([])
    (
        get_selected_aggregate_components,
        set_selected_aggregate_components,
    ) = mo.state({})

    # The parameters selected: maintain as state so we can pre-populate
    get_selected_params, set_selected_params = mo.state({})
    get_selected_aggregate_params, set_selected_aggregate_params = mo.state({})
    return (
        get_k,
        get_selected_aggregate_components,
        get_selected_aggregate_params,
        get_selected_components,
        get_selected_params,
        problem,
        set_k,
        set_selected_aggregate_components,
        set_selected_aggregate_params,
        set_selected_components,
        set_selected_params,
    )


@app.cell
def _(set_k):
    add_button = mo.ui.button(
        on_change=lambda _: set_k(lambda v: v + 1),
        label="Add a component",
    )

    remove_button = mo.ui.button(
        on_click=lambda _: set_k(lambda v: max(2, v - 1)),
        label="Remove a component",
    )
    return add_button, remove_button


@app.cell
def _(get_k, get_selected_components, set_selected_components):
    def _get_default_component_value(index):
        if index >= len(get_selected_components()):
            return None
        return get_selected_components()[index]

    _dropdowns = [
        mo.ui.dropdown(
            complib.RESIDUAL_COMPONENTS
            if i == 0
            else complib.COMPONENT_LIBRARY,
            value=_get_default_component_value(i),
            allow_select_none=True,
        )
        for i in range(get_k())
    ]

    component_array = mo.ui.array(
        _dropdowns, label="Components", on_change=set_selected_components
    )
    return (component_array,)


@app.cell
def _(component_array, get_selected_params, set_selected_params):
    component_params = mo.ui.dictionary(
        {
            f"{i}": complib.parameter_controls(
                c, get_selected_params().get(str(i), {})
            )
            for i, c in enumerate(component_array.value)
            if c is not None
        },
        label="Parameters",
        on_change=set_selected_params,
    )
    return (component_params,)


@app.cell
def _(problem):
    mo.stop(problem is None)

    mo.md(f"### {problem.name()}")
    return


@app.cell
def _(problem):
    mo.stop(problem is None)

    problem.description()
    return


@app.cell
def _(add_button, problem, remove_button):
    mo.stop(problem is None)

    mo.md(
        f"""
        ## {mo.md(f"{add_button} {remove_button}").center()}
        """
    )
    return


@app.cell
def _(component_array, component_params, problem):
    mo.stop(problem is None)

    mo.hstack([component_array, component_params])
    return


@app.cell
def _(
    component_array,
    component_params,
    get_selected_aggregate_components,
    set_selected_aggregate_components,
):
    _aggregates = {}
    _options = [v for v in complib.COMPONENT_LIBRARY if v != "Aggregate"]

    def _get_default_aggregate_component_value(key, index):
        if key not in get_selected_aggregate_components():
            return None
        selected_components = get_selected_aggregate_components()[key]
        if index >= len(selected_components):
            return None
        return selected_components[index]

    for _i, _component in enumerate(component_array.value):
        _key = str(_i)
        if _component == "Aggregate":
            _dropdowns = [
                mo.ui.dropdown(
                    _options,
                    _get_default_aggregate_component_value(_key, i),
                    allow_select_none=True,
                )
                for i in range(component_params.value[_key]["components"])
            ]
            _aggregates[_key] = mo.ui.array(_dropdowns, label="components")

    aggregates = mo.ui.dictionary(
        _aggregates,
        label="Aggregates",
        on_change=set_selected_aggregate_components,
    )
    return (aggregates,)


@app.cell
def _(
    aggregates,
    get_selected_aggregate_params,
    set_selected_aggregate_params,
):
    _aggregate_params = {}

    for _key, _components in aggregates.value.items():
        defaults = get_selected_aggregate_params().get(_key, {})
        _aggregate_params[_key] = mo.ui.dictionary(
            {
                f"{i}": complib.parameter_controls(c, defaults.get(str(i), {}))
                for i, c in enumerate(_components)
                if c is not None
            },
            label="Parameters",
        )

    aggregate_params = mo.ui.dictionary(
        _aggregate_params,
        label="Aggregate Parameters",
        on_change=set_selected_aggregate_params,
    )

    (mo.hstack([aggregates, aggregate_params]) if aggregates.value else None)
    return (aggregate_params,)


@app.cell
def _(aggregate_params, aggregates, component_params):
    def _rollup_aggregate_params(aggregates, aggregate_params, params_dict):
        params_dict = params_dict.copy()
        for component_key, components in aggregates.items():
            children = []
            params = aggregate_params[component_key]
            for i, component in enumerate(components):
                if component is not None:
                    children.append((component, params[str(i)]))
            params_dict[component_key]["children"] = children
        return tuple(params_dict.values())

    rolled_up_params = _rollup_aggregate_params(
        aggregates.value, aggregate_params.value, component_params.value
    )
    return (rolled_up_params,)


@app.cell
def _(component_array):
    noise_component_selected = component_array.value[0] is not None
    return (noise_component_selected,)


@app.cell
def _(component_array, noise_component_selected, problem):
    should_compute_decomposition = (
        noise_component_selected
        and sum(1 for v in component_array.value if v is not None) >= 2
    ) and problem is not None
    return (should_compute_decomposition,)


@app.cell
def _(
    component_array,
    noise_component_selected,
    problem,
    rolled_up_params,
    should_compute_decomposition,
):
    mo.stop(problem is None)

    def _feedback():
        if not noise_component_selected and any(
            v is not None for v in component_array.value
        ):
            return mo.md(
                """
                Make sure to set the first component, which represents noise.
                """
            ).callout(kind="alert")
        elif noise_component_selected and not should_compute_decomposition:
            return mo.md(
                """
                Great job choosing the noise component. Now choose at least one 
                more component.
                """
            ).callout(kind="neutral")
        elif should_compute_decomposition:
            return problem.feedback(
                [c for c in component_array.value if c is not None],
                rolled_up_params,
            )

    _feedback()
    return


@app.function
def construct_components(problem, names, parameters):
    center_periodic = isinstance(problem, problems.MaunaLoa)
    return list(
        filter(
            lambda v: v is not None,
            [
                complib.construct_component(
                    name, param_group, center_periodic
                )
                for name, param_group in zip(names, parameters)
            ],
        )
    )


@app.function
def decompose(problem, components, params):
    c = construct_components(problem, components, params)
    f = problem.decompose(c)
    f.set_figwidth(6.4)
    return f


@app.cell
def _(
    component_array,
    problem,
    rolled_up_params,
    should_compute_decomposition,
):
    mo.stop(not should_compute_decomposition)

    def _do_decomposition():
        components = tuple([v for v in component_array.value if v is not None])
        f = decompose(problem, components, rolled_up_params)
        plt.tight_layout()
        f.axes[0].set_title("Noise Component: %s" % components[0])
        for i, c in enumerate(components[1:]):
            if c is not None:
                f.axes[i + 1].set_title(c + " Component")
        f.axes[-1].set_title("Denoised Signal")
        return f

    _do_decomposition()
    return


@app.cell
def _():
    explainer_choice = mo.ui.dropdown(complib.COMPONENT_LIBRARY)
    return (explainer_choice,)


@app.cell
def _(explainer_choice, solved):
    mo.stop(not solved.ever)

    mo.md(
        f"""
        ## Reference

        Tell me more about the {explainer_choice} component class.
        """
    )
    return


@app.cell
def _(explainer_choice):
    mo.md(
        explainer.explainer(explainer_choice.value)
    ).callout() if explainer_choice.value is not None else ""
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## More about Signal Decomposition

    This tutorial is based on the research book, ["Signal Decomposition
    Using  Masked Proximal Operators"](https://web.stanford.edu/~boyd/papers/sig_decomp_mprox.html),
    by Bennet Meyers and Stephen Boyd. It uses the [`signal-decomp`](https://github.com/cvxgrp/signal-decomposition) Python library to
    compute decompositions.

    We hope this app shows that math can be intuitive, actionable,
    and fun.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    This material is based upon work supported by the U.S. Department of
            Energy's Office of Energy Efficiency and Renewable Energy (EERE)
            under the Solar Energy Technologies Office Award Number 38529.
    """)
    return


@app.cell
def _():
    _base_url = "https://raw.githubusercontent.com/marimo-team/gallery-examples/main/notebooks/education/inputs"
    _files = [
        "__init__.py",
        "components.py",
        "dataloaders.py",
        "explainer.py",
        "intro_problem.py",
        "layout.py",
        "problems.py",
        "solutions.py",
    ]
    _asset_files = [
        "solar_power_soiling_reference.png",
    ]

    os.makedirs("inputs/assets", exist_ok=True)
    for _f in _files:
        _path = os.path.join("inputs", _f)
        if not os.path.exists(_path):
            urllib.request.urlretrieve(f"{_base_url}/{_f}", _path)
    for _f in _asset_files:
        _path = os.path.join("inputs", "assets", _f)
        if not os.path.exists(_path):
            urllib.request.urlretrieve(f"{_base_url}/assets/{_f}", _path)
    return


@app.cell
def _():
    problems.configure_matplotlib()
    _ = importlib.reload(complib)
    _ = importlib.reload(explainer)
    _ = importlib.reload(problems)
    _ = importlib.reload(intro_problem)
    return


if __name__ == "__main__":
    app.run()
