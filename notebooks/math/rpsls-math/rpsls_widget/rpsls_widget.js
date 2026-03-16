import * as d3 from "https://esm.sh/d3@7";

function getNames(n) {
  return Array.from({ length: n }, (_, i) => String.fromCharCode(65 + i));
}

function buildTournament(n) {
  const k = Math.floor((n - 1) / 2);
  const isBalanced = n % 2 === 1;
  const edges = [];

  for (let i = 0; i < n; i++) {
    for (let j = 1; j <= k; j++) {
      edges.push({ source: (i + j) % n, target: i });
    }
  }

  // For even n, add one extra edge per pair of diametrically opposite nodes
  if (!isBalanced) {
    for (let i = 0; i < n / 2; i++) {
      const opposite = i + n / 2;
      // Alternate direction to spread imbalance
      if (i % 2 === 0) {
        edges.push({ source: opposite, target: i });
      } else {
        edges.push({ source: i, target: opposite });
      }
    }
  }

  // Compute out-degrees
  const outDegree = new Array(n).fill(0);
  for (const e of edges) {
    outDegree[e.source]++;
  }

  return { edges, outDegree, isBalanced, k };
}

function render({ model, el }) {
  const container = document.createElement("div");
  container.className = "rpsls-container";
  el.appendChild(container);

  const width = 500;
  const height = 500;
  const centerX = width / 2;
  const centerY = height / 2 - 10;
  const radius = 180;

  const svg = d3
    .select(container)
    .append("svg")
    .attr("width", width)
    .attr("height", height)
    .attr("viewBox", `0 0 ${width} ${height}`);

  // Read CSS custom properties for theming
  const cs = getComputedStyle(container);
  const edgeColor = cs.getPropertyValue("--rpsls-edge-color").trim() || "#888";
  const edgeDim = cs.getPropertyValue("--rpsls-edge-dim").trim() || "#e0e0e0";
  const arrowFill = cs.getPropertyValue("--rpsls-arrow-fill").trim() || "#666";
  const highlightStroke = cs.getPropertyValue("--rpsls-highlight-stroke").trim() || "#555";

  const defs = svg.append("defs");

  const edgeGroup = svg.append("g");
  const nodeGroup = svg.append("g");
  const labelGroup = svg.append("g");
  const balanceText = svg
    .append("text")
    .attr("class", "balance-label")
    .attr("x", centerX)
    .attr("y", height - 15);

  function updateMarkers(nodeRadius, markerSize) {
    defs.selectAll("marker").remove();

    const refX = nodeRadius / 1.4 + 10;

    const markerConfigs = [
      { id: "arrowhead", fill: arrowFill },
      { id: "arrowhead-red", fill: "#e74c3c" },
      { id: "arrowhead-beats", fill: "#27ae60" },
      { id: "arrowhead-loses", fill: "#e67e22" },
    ];

    for (const { id, fill } of markerConfigs) {
      defs
        .append("marker")
        .attr("id", id)
        .attr("viewBox", "0 -5 10 10")
        .attr("refX", refX)
        .attr("refY", 0)
        .attr("markerWidth", markerSize)
        .attr("markerHeight", markerSize)
        .attr("orient", "auto")
        .append("path")
        .attr("d", "M0,-4L10,0L0,4")
        .attr("fill", fill);
    }
  }

  // Mutable references so trait listeners can call current highlight functions
  let currentHighlightNode = () => {};
  let currentResetHighlight = () => {};
  let prevN = model.get("n");

  // 12 o'clock position (top of circle) — spawn/despawn point
  const topX = centerX;
  const topY = centerY - radius;

  function update() {
    const n = model.get("n");
    const names = getNames(n);
    const { edges, outDegree, isBalanced, k } = buildTournament(n);
    const growing = n > prevN;
    const shrinking = n < prevN;
    prevN = n;

    const nodeDur = model.get("_node_duration") || 400;
    const edgeDur = model.get("_edge_duration") || 400;

    // Scale sizes down as n grows
    const nodeRadius = d3.scaleLinear().domain([3, 9]).range([22, 12]).clamp(true)(n);
    const markerSize = d3.scaleLinear().domain([3, 9]).range([8, 5]).clamp(true)(n);
    const strokeWidth = d3.scaleLinear().domain([3, 9]).range([2, 1]).clamp(true)(n);
    const labelOffset = nodeRadius + (n <= 5 ? 24 : 12);
    const fontSize = d3.scaleLinear().domain([3, 9]).range([13, 10]).clamp(true)(n);

    updateMarkers(nodeRadius, markerSize);

    // Node positions on a circle (start from top)
    const nodes = names.map((name, i) => {
      const angle = (2 * Math.PI * i) / n - Math.PI / 2;
      return {
        name,
        x: centerX + radius * Math.cos(angle),
        y: centerY + radius * Math.sin(angle),
        outDegree: outDegree[i],
      };
    });

    // Color scale based on balance
    const balancedColor = "#3498db";
    const colorScale = d3
      .scaleSequential(d3.interpolateRdYlGn)
      .domain([Math.min(...outDegree) - 0.5, Math.max(...outDegree) + 0.5]);

    // Edges
    const balancedEdgeCount = n * k;
    const edgeDataMarked = edges.map((e, i) => ({
      source: e.source,
      target: e.target,
      key: `${names[e.source]}->${names[e.target]}`,
      x1: nodes[e.source].x,
      y1: nodes[e.source].y,
      x2: nodes[e.target].x,
      y2: nodes[e.target].y,
      isExtra: i >= balancedEdgeCount,
    }));

    // Growing: nodes move + existing edges slide, then new edges grow from node
    // Shrinking: removed edges fade out, then nodes move + remaining edges slide
    const newestNode = nodes[n - 1];

    const lines = edgeGroup.selectAll("line").data(edgeDataMarked, (d) => d.key);

    // Exiting edges fade out
    const exitingLines = lines.exit();
    let exitCount = 0;
    const exitTotal = exitingLines.size();
    exitingLines
      .attr("marker-end", null)
      .transition()
      .duration(edgeDur)
      .attr("stroke-opacity", 0)
      .attr("x1", topX)
      .attr("y1", topY)
      .attr("x2", topX)
      .attr("y2", topY)
      .remove()
      .on("end", () => {
        exitCount++;
        if (shrinking && exitCount === exitTotal) {
          animateNodesAndEdges();
        }
      });

    // New edges start collapsed at the newest node's position
    const linesEnter = lines
      .enter()
      .append("line")
      .attr("stroke-opacity", 0)
      .attr("x1", topX)
      .attr("y1", topY)
      .attr("x2", topX)
      .attr("y2", topY);

    const allLines = linesEnter.merge(lines).attr("class", "edge");

    // Set marker-end only on existing (updating) lines — entering lines
    // are collapsed to a point so markers would be visible at 12 o'clock.
    // Entering lines get their markers in animateNewEdges().
    lines
      .attr("stroke-dasharray", (d) => (d.isExtra ? "5,4" : "none"))
      .attr("marker-end", (d) =>
        d.isExtra ? "url(#arrowhead-red)" : "url(#arrowhead)"
      );

    // Move existing edges + nodes simultaneously
    function animateNodesAndEdges() {
      lines
        .transition()
        .duration(nodeDur)
        .attr("x1", (d) => d.x1)
        .attr("y1", (d) => d.y1)
        .attr("x2", (d) => d.x2)
        .attr("y2", (d) => d.y2)
        .attr("stroke", (d) => (d.isExtra ? "#e74c3c" : edgeColor))
        .attr("stroke-width", strokeWidth)
        .attr("stroke-opacity", 1);

      // When shrinking, new edges from topology change also need to animate in
      if (shrinking && linesEnter.size() > 0) {
        linesEnter
          .attr("stroke-dasharray", (d) => (d.isExtra ? "5,4" : "none"))
          .attr("marker-end", (d) =>
            d.isExtra ? "url(#arrowhead-red)" : "url(#arrowhead)"
          )
          .transition()
          .duration(nodeDur)
          .attr("x1", (d) => d.x1)
          .attr("y1", (d) => d.y1)
          .attr("x2", (d) => d.x2)
          .attr("y2", (d) => d.y2)
          .attr("stroke", (d) => (d.isExtra ? "#e74c3c" : edgeColor))
          .attr("stroke-width", strokeWidth)
          .attr("stroke-opacity", 1);
      }

      moveNodes();
    }

    // Grow new edges from the new node's position
    function animateNewEdges() {
      // Snap to new node's resting position before growing outward
      linesEnter
        .attr("x1", newestNode.x)
        .attr("y1", newestNode.y)
        .attr("x2", newestNode.x)
        .attr("y2", newestNode.y);

      linesEnter
        .attr("stroke-dasharray", (d) => (d.isExtra ? "5,4" : "none"))
        .attr("marker-end", (d) =>
          d.isExtra ? "url(#arrowhead-red)" : "url(#arrowhead)"
        )
        .transition()
        .duration(edgeDur)
        .attr("x1", (d) => d.x1)
        .attr("y1", (d) => d.y1)
        .attr("x2", (d) => d.x2)
        .attr("y2", (d) => d.y2)
        .attr("stroke", (d) => (d.isExtra ? "#e74c3c" : edgeColor))
        .attr("stroke-width", strokeWidth)
        .attr("stroke-opacity", 1);
    }

    // Move all nodes (and trigger edge animation when done)
    function moveNodes() {
      circles
        .exit()
        .transition()
        .duration(nodeDur)
        .attr("cx", topX)
        .attr("cy", topY)
        .attr("r", 0)
        .remove();

      let nodesFinished = 0;
      const totalNodes = allCircles.size();
      allCircles
        .transition()
        .duration(nodeDur)
        .attr("cx", (d) => d.x)
        .attr("cy", (d) => d.y)
        .attr("r", nodeRadius)
        .attr("fill", (d) =>
          isBalanced ? balancedColor : colorScale(d.outDegree)
        )
        .on("end", () => {
          nodesFinished++;
          if (growing && nodesFinished === totalNodes) {
            animateNewEdges();
          }
        });

      allLabels
        .transition()
        .duration(nodeDur)
        .attr("x", (d) => {
          const dx = d.x - centerX;
          return d.x + (dx / radius) * labelOffset;
        })
        .attr("y", (d) => {
          const dy = d.y - centerY;
          return d.y + (dy / radius) * labelOffset + 5;
        })
        .attr("font-size", fontSize)
        .attr("opacity", 1);

      labels
        .exit()
        .transition()
        .duration(nodeDur)
        .attr("opacity", 0)
        .remove();
    }

    // Highlight helpers (with optional animation)
    function highlightNode(idx, animate = false) {
      const dur = model.get("_node_duration") || 400;
      const applyLines = animate ? allLines.transition().duration(dur) : allLines;
      const applyCircles = animate ? allCircles.transition().duration(dur) : allCircles;
      const applyLabels = animate ? allLabels.transition().duration(dur) : allLabels;

      applyLines
        .attr("stroke", (d) => {
          if (d.source === idx) return "#27ae60";
          if (d.target === idx) return "#e67e22";
          return edgeDim;
        })
        .attr("stroke-width", (d) =>
          d.source === idx || d.target === idx ? strokeWidth * 2 : strokeWidth * 0.5
        );

      // These can't be transitioned, apply immediately
      allLines
        .attr("stroke-dasharray", (d) =>
          d.source === idx || d.target === idx ? "none" : (d.isExtra ? "5,4" : "none")
        )
        .attr("marker-end", (d) => {
          if (d.source === idx) return "url(#arrowhead-beats)";
          if (d.target === idx) return "url(#arrowhead-loses)";
          return "url(#arrowhead)";
        });

      // Raise connected edges above dimmed ones in the DOM
      allLines.filter((d) => d.source === idx || d.target === idx).raise();

      // Dim unrelated nodes, highlight selected node border
      applyCircles
        .attr("opacity", (d, i) => {
          const connected = edgeDataMarked.some(
            (e) => (e.source === idx && e.target === i) || (e.target === idx && e.source === i)
          );
          return i === idx || connected ? 1 : 0.3;
        })
        .style("stroke", (d, i) => (i === idx ? highlightStroke : null))
        .style("stroke-width", (d, i) => (i === idx ? "4px" : null));
      applyLabels.attr("opacity", (d, i) => {
        const connected = edgeDataMarked.some(
          (e) => (e.source === idx && e.target === i) || (e.target === idx && e.source === i)
        );
        return i === idx || connected ? 1 : 0.3;
      });
    }

    function resetHighlight(animate = false) {
      const dur = model.get("_node_duration") || 400;
      const applyLines = animate ? allLines.transition().duration(dur) : allLines;
      const applyCircles = animate ? allCircles.transition().duration(dur) : allCircles;
      const applyLabels = animate ? allLabels.transition().duration(dur) : allLabels;

      applyLines
        .attr("stroke", (d) => (d.isExtra ? "#e74c3c" : edgeColor))
        .attr("stroke-width", strokeWidth);

      allLines
        .attr("stroke-dasharray", (d) => (d.isExtra ? "5,4" : "none"))
        .attr("marker-end", (d) =>
          d.isExtra ? "url(#arrowhead-red)" : "url(#arrowhead)"
        );
      applyCircles.attr("opacity", 1).style("stroke", null).style("stroke-width", null);
      applyLabels.attr("opacity", 1);
    }

    // Nodes — data join only; transitions handled by moveNodes()
    const circles = nodeGroup.selectAll("circle").data(nodes, (d) => d.name);
    const circlesEnter = circles
      .enter()
      .append("circle")
      .attr("cx", topX)
      .attr("cy", topY)
      .attr("r", 0);
    const allCircles = circlesEnter.merge(circles).attr("class", "node-circle");
    allCircles
      .style("cursor", "pointer")
      .on("mouseenter", (event, d) => {
        const idx = nodes.indexOf(d);
        highlightNode(idx);
      })
      .on("mouseleave", () => resetHighlight());

    // Labels — data join only; transitions handled by moveNodes()
    const labels = labelGroup.selectAll("text").data(nodes, (d) => d.name);
    const labelsEnter = labels.enter().append("text").attr("opacity", 0);
    const allLabels = labelsEnter.merge(labels).attr("class", "node-label");
    allLabels
      .attr("text-anchor", "middle")
      .text((d) => d.name);

    // Kick off the animation chain
    if (shrinking && exitTotal > 0) {
      // Shrinking: edge exit callback will trigger animateNodesAndEdges
    } else if (growing || shrinking) {
      animateNodesAndEdges();
    } else {
      // Initial render
      moveNodes();
      animateNewEdges();
    }

    // Balance text
    if (isBalanced) {
      balanceText
        .text(`Balanced: each element beats exactly ${k} others`)
        .attr("fill", "#27ae60");
    } else {
      const degrees = [...new Set(outDegree)].sort().join(" or ");
      balanceText
        .text(`Not balanced: elements beat ${degrees} others (dashed = extra edges)`)
        .attr("fill", "#e74c3c");
    }

    // Expose highlight functions for trait listeners
    currentHighlightNode = highlightNode;
    currentResetHighlight = resetHighlight;
  }

  update();
  model.on("change:n", update);

  model.on("change:highlighted_node", () => {
    const name = model.get("highlighted_node");
    if (!name) {
      currentResetHighlight(true);
      return;
    }
    const n = model.get("n");
    const names = getNames(n);
    const idx = names.indexOf(name);
    if (idx !== -1) {
      currentHighlightNode(idx, true);
    }
  });
}

export default { render };
