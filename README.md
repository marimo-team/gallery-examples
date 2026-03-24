# marimo gallery

A collection of [marimo](https://marimo.io) notebooks showcasing algorithms,
research paper implementations, interactive visualizations, dashboards,
widgets, and more. Interactively explore these notebooks
at our [online gallery](https://molab.marimo.io/gallery).

**Running notebooks online.**
The easiest way to run the notebooks is with [molab](https://molab.marimo.io);
just click the "Open in molab" links below.

**Running notebooks locally.**
[Install `uv`](https://docs.astral.sh/uv/getting-started/installation/), download the notebook you want to run, then run

```bash
uvx marimo edit --sandbox <notebook>
```

## Algorithms

| Notebook | Description | |
| -------- | ----------- | - |
| Embedding Visualiser | Select points in embedded space to explore clusters. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/algorithms/visualizing-embeddings.py) |
| Neural networks | Training a tiny neural network with micrograd | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/algorithms/neural_networks_micrograd.py/wasm) |
| Evolutionary Strategies | Interactive exploration of evolutionary optimization. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/algorithms/evolutionary-strategies.py/wasm) |
| Smoothed Gradient Descent | Visualize gradient descent with momentum and smoothing. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/algorithms/smoothed-gradient-descent.py/wasm) |
| Loss Landscape Visualization | Visualize neural network loss landscapes using local-plane blending. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/algorithms/loss-landscape-visualization.py/wasm) |
| Federated Learning | Simulate federated learning. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/library/federated.py) |
| Sklearn Classification | Bootstrap models with preprocessing and cross-validation. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/library/sklearn-clf.py/wasm) |
| FastAPI + GliNER | Zero-shot entity extraction as webapp, API, or CLI. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/library/fastapi-gliner.py) |
| Chemical Space Explorer | Explore chemical space with RDKit fingerprints, t-SNE, and HDBSCAN clustering. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/library/chemical-space-explorer.py) |
| Bayesian Regression | Interactive sequential Bayesian linear regression demo. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/algorithms/bayesian-regression-demo.py) |

## Research papers

| Notebook | Description | |
| -------- | ----------- | - |
| GDPO vs GRPO | A comparison of GDPO and GRPO methods. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/alphaxiv/gdpo/notebook.py/wasm) |
| LLM Unlearning | Explore LLM unlearning by overfitting unwanted data. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/alphaxiv/llm-unlearning.py) |

## Analysis

| Notebook | Description | |
| -------- | ----------- | - |
| Lanchester's Law | Interactive simulation of Lanchester's Laws of combat. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/analysis/lanchesters-law/notebook.py/wasm) |
| Suguru | Interactive Suguru puzzle solver and generator. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/analysis/suguru/notebook.py/wasm) |
| ARC-Easy LLM | Prompt repetition experiment on ARC-Easy multiple-choice questions. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/analysis/arc-easy-llm.py) |
| Simpson's Paradox | Bayesian analysis of Simpson's paradox with PyMC. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/analysis/simpsons-paradox.py) |

## 3D

| Notebook | Description | |
| -------- | ----------- | - |
| Bookshelf | Make a bookshelf, using sliders, with marimo-cad. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/cad/bookshelf.py) |
| Trimesh Demo | Grab 2D slices from 3D files with trimesh. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/cad/trimesh-demo.py) |
| Vase | Make a vase for 3D printing with marimo-cad. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/cad/vase.py) |

## Dashboard

| Notebook | Description | |
| -------- | ----------- | - |
| Lego Price Explorer | Explore Lego price-per-piece patterns. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/dashboard/lego/notebook.py/wasm) |
| Movies Dashboard | Explore movie data with a dashboard. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/dashboard/movies.py/wasm) |
| Portfolio Calculator | An interactive tool for portfolio analysis. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/dashboard/portfolio.py) |
| World of Warcraft Bot Detection | Detect bots in WoW using session length analysis. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/dashboard/world-of-warcraft.py/wasm) |
| Altair Reactive Plots | Altair charts with brush selection and reactive filtering. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/library/altair-demo.py/wasm) |
| Matplotlib Selection | Select data points on a matplotlib plot with `mo.ui.matplotlib`. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/library/matplotlib_selection.py/wasm) |

## SQL

| Notebook | Description | |
| -------- | ----------- | - |
| Connect to SQLite | Use marimo's SQL cells to read from and write to SQLite databases. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/sql/sqlite.py) |
| MotherDuck | Explore using MotherDuck inside marimo. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/sql/motherduck.py) |
| SQL Interpolation | Interactive SQL with slider-driven query interpolation. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/sql/interpolate.py/wasm) |

## External

| Notebook | Description | |
| -------- | ----------- | - |
| Neo4j Widget | Interactive graph explorer for Neo4j databases. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/external/neo4jwidget.py) |
| Wandb Chart | Live Weights & Biases chart integration. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/external/wandbchart.py) |
| When Europeans Fly Nest | Dataviz makeover exploring when young Europeans leave home. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/external/when-europeans-fly-nest.py) |
| Mandelbrot | Cython vs Python Mandelbrot set benchmark. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/external/mandelbrot.py) |
| Manim Slides | Trigonometric identity proof animated with manim-slides. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/external/manim-slides.py) |
| dltHub + Hugging Face | Curate Hugging Face datasets with dltHub pipelines and data quality checks. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/external/dlthub-huggingface.py) |

## Geo

| Notebook | Description | |
| -------- | ----------- | - |
| Airport Display | Display all airports using OpenStreetMap and OpenLayers. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/geo/airports.py/wasm) |
| Click and Zoom | Click and zoom maps to get Python coordinates. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/geo/click-zoom.py/wasm) |
| Drag and Drop | Drop GPX, GeoJSON, KML, or TopoJSON onto maps. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/geo/drag-drop.py/wasm) |
| Earthquakes | Highlight all known earthquakes on the OpenLayers map. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/geo/earthquake.py/wasm) |

## Math

| Notebook | Description | |
| -------- | ----------- | - |
| Spectral Decomposition | Interactively explore the spectral matrix decomposition | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/math/eigen.py/wasm) |
| Signal Decomposition | Decompose complex signals into simpler interpretable components. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/education/signal_decomposition.py/wasm) |
| 100 Prisoners Puzzle | Simulate the famous 100 prisoners probability puzzle. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/math/100-prisoners.py/wasm) |
| Attractor | Interactive strange attractor visualization with scatter plots. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/math/attractor.py) |
| Matrix Decomposition | An interactive exploration of matrix decomposition techniques. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/math/matrix-decompositions.py/wasm) |
| Seam Carving | Content-aware image resizing using dynamic programming. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/math/seam-carving/notebook.py) |
| Matrix | Interactive matrix editor and PCA demo using `mo.ui.matrix`. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/math/matrix.py/wasm) |
| Cellular Automaton Art | Voter-model cellular automaton that self-organises into paint-splatter art. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/math/cellular-automaton-art.py/wasm) |
| Bak-Sneppen Model | Interactive 3D visualization of the Bak-Sneppen self-organized criticality model. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/math/bak-sneppen-model.py/wasm) |
| Vector Puck | Explore vector arithmetic with pucks over matplotlib. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/math/vector-puck.py/wasm) |
| Thermodynamic Linear Algebra | Estimate a matrix inverse via overdamped Langevin dynamics. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/math/thermodynamic_linear_algebra.py/wasm) |
| Interactive ODE Solver | Drag to set initial conditions and explore solutions of an ODE with a direction field. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/math/ode-solver.py) |
| RPSLS Math | Explore balanced tournament graphs for Rock-Paper-Scissors variants. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/math/rpsls-math/notebook.py) |

## Custom UI elements with Anywidget

| Notebook | Description | |
| -------- | ----------- | - |
| CellTour | Create guided tours through notebook cells. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/wigglystuff/celltour.py/wasm) |
| ChartPuck | Interactive chart editor with draggable control points. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/wigglystuff/chartpuck.py/wasm) |
| ColorPicker | An interactive color selection widget. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/wigglystuff/colorpicker.py/wasm) |
| CopyToClipboard | A utility widget for copying content to the clipboard. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/wigglystuff/copytoclipboard.py/wasm) |
| DriverTour | Create guided page tours with the DriverTour widget. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/wigglystuff/drivertour.py/wasm) |
| EdgeDraw | Draw edges between nodes for graph creation. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/wigglystuff/edgedraw.py/wasm) |
| EnvConfig | Configure environment variables interactively. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/wigglystuff/envconfig.py/wasm) |
| Fashion MNIST Parallel Coords | Explore Fashion MNIST with PCA and interactive parallel coordinates. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/wigglystuff/fashion-mnist-parallel-coords.py) |
| GamepadWidget | Capture input from gaming controllers. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/wigglystuff/gamepad.py/wasm) |
| Greedy Search Pucks | Greedy sampled search visualization with draggable pucks. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/wigglystuff/greedy_search_pucks.py/wasm) |
| HTMLRefreshWidget | Render dynamic HTML content that auto-refreshes. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/wigglystuff/htmlwidget.py/wasm) |
| KeystrokeWidget | Capture keyboard input and key combinations. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/wigglystuff/keystroke.py/wasm) |
| Paint | A drawing and painting canvas widget for freeform input. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/wigglystuff/paint.py/wasm) |
| PulsarChart | Joy Division-style pulsar chart visualization. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/wigglystuff/pulsarchart.py/wasm) |
| ShortcutWidget | Keyboard shortcut capture widget. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/wigglystuff/shortcut.py/wasm) |
| Slider2D | A 2D slider control for selecting x/y coordinates interactively. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/wigglystuff/slider2d.py/wasm) |
| SortableList | A draggable list for reordering items interactively. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/wigglystuff/sortlist.py/wasm) |
| SpeechToText | Convert voice to text using browser speech recognition. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/wigglystuff/talk.py/wasm) |
| Tangle | Interactive number manipulation with draggable values in text. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/wigglystuff/tangle.py/wasm) |
| TextCompare | Side-by-side text comparison with highlighted matching passages. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/wigglystuff/textcompare.py/wasm) |
| ThreeWidget | 3D visualization widget using Three.js. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/wigglystuff/threewidget.py/wasm) |
| Drawdata Scatter | Draw scatter data and visualize with Altair histograms. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/drawdata/scatter-demo.py/wasm) |
| WebcamCapture | Capture images from the webcam. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/wigglystuff/webcam_capture.py/wasm) |
