# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#     "altair==6.0.0",
#     "anywidget==0.9.21",
#     "marimo",
#     "numpy==2.3.5",
#     "pandas==3.0.0",
#     "traitlets==5.14.3",
# ]
# ///

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import json
    import altair as alt
    import anywidget
    import numpy as np
    import pandas as pd
    import traitlets


@app.cell(hide_code=True)
def _():
    mo.md("""
    This notebook is a response to [this popular Veritasium video](https://www.youtube.com/watch?v=iSNsgj1OCLA). It's a great video that's certainly worth a watch, but we felt that an interactive notebook let's you explore the problem in a way that a video never could. So here goes:

    # The 100 Prisoners Riddle

    Prisoners are numbered 1 to $N$. In a room, $N$ boxes each contain a slip with a random prisoner's number.
    Each prisoner can open at most $K$ boxes to find their own number. If **all** prisoners succeed, they go free. Otherwise they do not.

    So what strategy could you apply here?

    **The clever strategy:** Prisoner $i$ starts at box $i$, then follows the chain (if box $i$ contains $j$, open box $j$ next). This works because permutations form cycles — a prisoner fails only if they're in a cycle longer than $K$.

    To highlight this, feel free to toy around with the graph simulation below to convince yourself of this. Each node represents a box. Arrows show where each box points. Cycles are colored — **red cycles are longer than $K$** (representing failure!).

    You can also change the number of prisoners and the number of boxes to open if you want to play around and get a feel of the influence of these numbers.
    """)
    return


@app.cell(hide_code=True)
def _():
    n_prisoners_input = mo.ui.number(
        value=100, start=10, stop=200, step=10, label="Number of prisoners"
    )
    max_boxes_input = mo.ui.number(value=50, start=5, stop=100, step=5, label="Max boxes to open")
    mo.hstack([n_prisoners_input, max_boxes_input])
    return max_boxes_input, n_prisoners_input


@app.function
def find_cycles(permutation):
    """Find all cycles in a permutation. Returns list of cycles, each cycle is a list of indices."""
    n = len(permutation)
    visited = [False] * n
    cycles = []

    for start in range(n):
        if visited[start]:
            continue
        cycle = []
        current = start
        while not visited[current]:
            visited[current] = True
            cycle.append(current)
            current = permutation[current]
        if cycle:
            cycles.append(cycle)

    return cycles


@app.function
def generate_permutation(n=100, seed=None):
    """Generate a random permutation of 0 to n-1."""
    if seed is not None:
        np.random.seed(seed)
    return list(np.random.permutation(n))


@app.function
def max_cycle_length(perm):
    """Find the maximum cycle length in a single permutation (1D array)."""
    n = len(perm)
    visited = np.zeros(n, dtype=bool)
    max_len = 0
    for start in range(n):
        if visited[start]:
            continue
        length = 0
        current = start
        while not visited[current]:
            visited[current] = True
            length += 1
            current = perm[current]
        if length > max_len:
            max_len = length
    return max_len


@app.function
def simulate_batch(n_prisoners, n_simulations):
    """Run many simulations at once using NumPy. Returns array of max cycle lengths."""
    # Generate all permutations at once: shape (n_simulations, n_prisoners)
    perms = np.array([np.random.permutation(n_prisoners) for _ in range(n_simulations)])
    # Compute max cycle length for each
    return np.array([max_cycle_length(p) for p in perms])


@app.cell
def _():
    regenerate_button = mo.ui.button(label="🎲 New Random Arrangement")
    return (regenerate_button,)


@app.cell
def _(regenerate_button):
    regenerate_button
    return


@app.cell
def _(max_boxes_input, n_prisoners_input):
    n_prisoners = n_prisoners_input.value
    max_boxes = max_boxes_input.value
    return max_boxes, n_prisoners


@app.cell
def _(max_boxes, n_prisoners, regenerate_button):
    # Regenerate when button is clicked
    _click_count = regenerate_button.value

    permutation = generate_permutation(n_prisoners)
    cycles = find_cycles(permutation)
    cycle_lengths = sorted([len(c) for c in cycles], reverse=True)
    max_cycle = max(cycle_lengths)
    success = max_cycle <= max_boxes
    return cycle_lengths, cycles, max_cycle, permutation, success


@app.class_definition
class CycleGraphWidget(anywidget.AnyWidget):
    _esm = """
    import * as d3 from "https://esm.sh/d3@7";

    function render({ model, el }) {
        const width = 800;
        const height = 600;

        // Create SVG
        const svg = d3.select(el)
            .append("svg")
            .attr("width", width)
            .attr("height", height)
            .attr("viewBox", [0, 0, width, height]);

        // Add arrow marker definition
        svg.append("defs").append("marker")
            .attr("id", "arrowhead")
            .attr("viewBox", "-0 -5 10 10")
            .attr("refX", 20)
            .attr("refY", 0)
            .attr("orient", "auto")
            .attr("markerWidth", 6)
            .attr("markerHeight", 6)
            .append("path")
            .attr("d", "M 0,-5 L 10,0 L 0,5")
            .attr("fill", "#999");

        // Container for zoomable content
        const g = svg.append("g");

        // Add zoom behavior
        const zoom = d3.zoom()
            .scaleExtent([0.3, 3])
            .on("zoom", (event) => {
                g.attr("transform", event.transform);
            });
        svg.call(zoom);

        function update() {
            const graphData = JSON.parse(model.get("graph_data"));
            const nodes = graphData.nodes;
            const links = graphData.links;
            const cycleColors = graphData.cycle_colors;
            const maxBoxes = graphData.max_boxes || 50;

            // Clear previous content
            g.selectAll("*").remove();

            // Color scale for cycles
            const colorScale = d3.scaleOrdinal(d3.schemeTableau10);

            // Create links
            const link = g.append("g")
                .selectAll("line")
                .data(links)
                .join("line")
                .attr("stroke", d => {
                    const cycleIdx = cycleColors[String(d.source)];
                    const cycleLen = graphData.cycle_lengths[String(cycleIdx)];
                    return cycleLen > maxBoxes ? "#e41a1c" : colorScale(cycleIdx);
                })
                .attr("stroke-opacity", 0.6)
                .attr("stroke-width", 1.5)
                .attr("marker-end", "url(#arrowhead)");

            // Create nodes
            const node = g.append("g")
                .selectAll("g")
                .data(nodes)
                .join("g")
                .call(d3.drag()
                    .on("start", dragstarted)
                    .on("drag", dragged)
                    .on("end", dragended));

            node.append("circle")
                .attr("r", 12)
                .attr("fill", d => {
                    const cycleIdx = cycleColors[String(d.id)];
                    const cycleLen = graphData.cycle_lengths[String(cycleIdx)];
                    return cycleLen > maxBoxes ? "#e41a1c" : colorScale(cycleIdx);
                })
                .attr("stroke", "#fff")
                .attr("stroke-width", 1.5);

            node.append("text")
                .text(d => d.id + 1)
                .attr("text-anchor", "middle")
                .attr("dy", "0.35em")
                .attr("font-size", "8px")
                .attr("fill", "white")
                .attr("pointer-events", "none");

            // Force simulation — middle ground settings
            const padding = 40;
            const simulation = d3.forceSimulation(nodes)
                .force("link", d3.forceLink(links).id(d => d.id).distance(35).strength(1.5))
                .force("charge", d3.forceManyBody().strength(-80))
                .force("center", d3.forceCenter(width / 2, height / 2))
                .force("collision", d3.forceCollide().radius(16))
                .force("x", d3.forceX(width / 2).strength(0.05))
                .force("y", d3.forceY(height / 2).strength(0.05));

            simulation.on("tick", () => {
                // Soft bounds
                nodes.forEach(d => {
                    if (d.x < padding) d.x += (padding - d.x) * 0.15;
                    if (d.x > width - padding) d.x -= (d.x - (width - padding)) * 0.15;
                    if (d.y < padding) d.y += (padding - d.y) * 0.15;
                    if (d.y > height - padding) d.y -= (d.y - (height - padding)) * 0.15;
                });

                link
                    .attr("x1", d => d.source.x)
                    .attr("y1", d => d.source.y)
                    .attr("x2", d => d.target.x)
                    .attr("y2", d => d.target.y);

                node.attr("transform", d => `translate(${d.x},${d.y})`);
            });

            function dragstarted(event, d) {
                if (!event.active) simulation.alphaTarget(0.3).restart();
                d.fx = d.x;
                d.fy = d.y;
            }

            function dragged(event, d) {
                d.fx = event.x;
                d.fy = event.y;
            }

            function dragended(event, d) {
                if (!event.active) simulation.alphaTarget(0);
                d.fx = null;
                d.fy = null;
            }
        }

        model.on("change:graph_data", update);
        update();
    }

    export default { render };
    """
    graph_data = traitlets.Unicode("{}").tag(sync=True)


@app.cell
def _(cycles, max_boxes, permutation):
    # Build graph data for D3
    nodes = [{"id": i} for i in range(len(permutation))]
    links = [{"source": i, "target": int(permutation[i])} for i in range(len(permutation))]

    # Map each node to its cycle index
    _cycle_colors = {}
    _cycle_lengths_map = {}
    for idx, cycle in enumerate(cycles):
        _cycle_lengths_map[str(idx)] = len(cycle)
        for node in cycle:
            _cycle_colors[str(int(node))] = idx

    graph_data = {
        "nodes": nodes,
        "links": links,
        "cycle_colors": _cycle_colors,
        "cycle_lengths": _cycle_lengths_map,
        "max_boxes": max_boxes,
    }

    widget = CycleGraphWidget(graph_data=json.dumps(graph_data))
    widget
    return


@app.cell
def _(cycle_lengths, max_cycle, success):
    status_color = "green" if success else "red"
    status_text = (
        "✅ All prisoners can find their number!" if success else "❌ One cycle is too long!"
    )

    mo.md(
        f"""
        - <span style="color: {status_color}">{status_text}</span>
        - **Longest cycle:** {max_cycle} boxes
        - **All cycle lengths:** {", ".join(map(str, cycle_lengths))}
        """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ---
    ## Simulation: Longest Cycle Distribution

    The longest cycle is important in this story, and we can simulate the distribution of that.
    """)
    return


@app.cell
def _(n_prisoners):
    # Run 100k simulations
    n_sims = 100_000

    _results = []
    for _ in range(n_sims):
        perm = np.random.permutation(n_prisoners)
        visited = np.zeros(n_prisoners, dtype=bool)
        max_len = 0
        for start in range(n_prisoners):
            if visited[start]:
                continue
            length = 0
            current = start
            while not visited[current]:
                visited[current] = True
                length += 1
                current = perm[current]
            if length > max_len:
                max_len = length
        _results.append(max_len)

    # Count occurrences and compute probabilities
    _series = pd.Series(_results)
    _counts = _series.value_counts().sort_index()
    _df = pd.DataFrame({"longest_cycle": _counts.index, "probability": _counts.values / n_sims})

    _chart = (
        alt.Chart(_df)
        .mark_bar()
        .encode(
            x=alt.X(
                "longest_cycle:Q", title="Longest Cycle Length", scale=alt.Scale(domain=[0, 100])
            ),
            y=alt.Y("probability:Q", title="Probability"),
        )
        .properties(width=700, height=300, title="Distribution of longest cycle")
    )

    mo.ui.altair_chart(_chart)
    return


@app.cell
def _():
    mo.md(r"""
    ## Extra detail

    At this point you might think, "well, that's all very nice", but what if I am prisoner 42. I would select box 42 and walk along the cycle. What guarantees mee that my number is on the cycle that I am on right now?

    For any position to be part of a cycle, something must point to it — that's what makes it a cycle rather than a dead end.

    $... \Rightarrow a \Rightarrow 42 \Rightarrow b \Rightarrow ...$

    The box that points to position 42 is, by definition, the box containing slip 42. Otherwise we would not be in a cycle!

    Therefore, when prisoner 42 traverses their cycle, they must pass through this predecessor box to complete the loop, and that's exactly where their slip is.
    """)
    return


if __name__ == "__main__":
    app.run()
