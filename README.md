# gallery-examples

A collection of [marimo](https://marimo.io) notebooks showcasing algorithms,
research paper implementations, interactive visualizations, dashboards,
widgets, and more.

**View on our gallery webpage.** Interactively explore these notebooks
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
| Embedding Visualiser | Select points in embedded space to understand UMAP clusters better. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/algorithms/visualizing-embeddings.py) |
| Evolutionary Strategies | Interactive exploration of evolutionary optimization algorithms. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/algorithms/evolutionary-strategies.py) |
| Smoothed Gradient Descent | Visualize gradient descent with momentum and smoothing techniques. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/algorithms/smoothed-gradient-descent.py) |
| Federated Learning | Interactive simulation of federated learning with hospitals training local models and FedAvg aggregation. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/library/federated.py) |
| Sklearn Classification | Bootstrap scikit-learn classification models with automatic preprocessing and cross-validation. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/library/sklearn-clf.py) |
| FastAPI + GliNER | GliNER v2 for zero-shot entity extraction, runnable as webapp, API, or CLI. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/library/fastapi-gliner.py) |

## Research papers

| Notebook | Description | |
| -------- | ----------- | - |
| GDPO vs GRPO | A comparison of GDPO and GRPO methods. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/alphaxiv/gdpo/notebook.py) |
| LLM Unlearning | Exploring unlearning in large language models by overfitting what you don't want. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/alphaxiv/llm-unlearning.py) |

## Analysis

| Notebook | Description | |
| -------- | ----------- | - |
| Lanchester's Law | Interactive simulation of Lanchester's Laws of combat. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/analysis/lanchesters-law/notebook.py) |
| Suguru | Interactive Suguru puzzle solver and generator. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/analysis/suguru/notebook.py) |

## 3D

| Notebook | Description | |
| -------- | ----------- | - |
| Bookshelf | Make a bookshelf, using sliders, with marimo-cad. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/cad/bookshelf.py) |
| Trimesh Demo | Grab 2D slices using trimesh, a Python library to work with 3D files. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/cad/trimesh-demo.py) |
| Vase | Make a vase for 3D printing with marimo-cad. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/cad/vase.py) |

## Dashboard

| Notebook | Description | |
| -------- | ----------- | - |
| Lego Price Explorer | Explore price differences across Lego themes and analyze price-per-piece patterns. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/dashboard/lego/notebook.py) |
| Movies Dashboard | Explore movie data interactively with a dashboard constructed with marimo UI elements. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/dashboard/movies.py) |
| Portfolio Calculator | An interactive tool for portfolio analysis and investment calculations. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/dashboard/portfolio.py) |
| World of Warcraft Bot Detection | Interactive visualization for detecting bots in WoW player data using session length analysis. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/dashboard/world-of-warcraft.py) |
| Altair Reactive Plots | Interactive Altair charts with brush selection and reactive filtering. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/library/altair-demo.py) |

## External

| Notebook | Description | |
| -------- | ----------- | - |
| Neo4j Widget | Interactive graph explorer for Neo4j databases. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/external/neo4jwidget.py) |
| Wandb Chart | Live Weights & Biases chart integration. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/external/wandbchart.py) |

## Geo

| Notebook | Description | |
| -------- | ----------- | - |
| Airport Display | Use OpenStreetMap and OpenLayers to show all the airports out there. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/geo/airports.py) |
| Click and Zoom | Click and zoom into maps to explore and get coordinates back in Python. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/geo/click-zoom.py) |
| Drag and Drop | Drag and drop GPX, GeoJSON, KML or TopoJSON files on to the map. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/geo/drag-drop.py) |
| Earthquakes | Highlight all known earthquakes on the OpenLayers map. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/geo/earthquake.py) |

## Math

| Notebook | Description | |
| -------- | ----------- | - |
| Signal Decomposition | Breaking a complex signal into the sum of simpler interpretable components. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/education/signal_decomposition.py) |
| 100 Prisoners Puzzle | An interactive simulation of the famous 100 prisoners probability puzzle. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/math/100-prisoners.py) |
| Attractor | Interactive strange attractor visualization with scatter plots. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/math/attractor.py) |
| Matrix Decomposition | An interactive exploration of matrix decomposition techniques. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/math/matrix-decompositions.py) |
| Seam Carving | Content-aware image resizing that preserves important features using dynamic programming. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/math/seam-carving/notebook.py) |
| Vector Puck | Explore vector arithmetic with pucks over matplotlib. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/math/vector-puck.py) |

## Custom UI elements with Anywidget

| Notebook | Description | |
| -------- | ----------- | - |
| CellTour | Create guided tours through notebook cells. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/wigglystuff/celltour.py) |
| ChartPuck | Interactive chart editor with draggable control points. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/wigglystuff/chartpuck.py) |
| ChartSelect | Select data points on a chart interactively. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/wigglystuff/chartselect.py) |
| ColorPicker | An interactive color selection widget. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/wigglystuff/colorpicker.py) |
| CopyToClipboard | A utility widget for copying content to the clipboard. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/wigglystuff/copytoclipboard.py) |
| DriverTour | Create guided page tours with the DriverTour widget. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/wigglystuff/drivertour.py) |
| EdgeDraw | Draw edges between nodes for graph creation. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/wigglystuff/edgedraw.py) |
| EnvConfig | Configure environment variables interactively. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/wigglystuff/envconfig.py) |
| GamepadWidget | Capture input from gaming controllers. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/wigglystuff/gamepad.py) |
| Greedy Search Pucks | Greedy sampled search visualization with draggable pucks. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/wigglystuff/greedy_search_pucks.py) |
| HTMLRefreshWidget | Render dynamic HTML content that auto-refreshes. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/wigglystuff/htmlwidget.py) |
| KeystrokeWidget | Capture keyboard input and key combinations. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/wigglystuff/keystroke.py) |
| Matrix | An interactive matrix editor for manipulating grid data. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/wigglystuff/matrix.py) |
| Paint | A drawing and painting canvas widget for freeform input. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/wigglystuff/paint.py) |
| PulsarChart | Joy Division-style pulsar chart visualization. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/wigglystuff/pulsarchart.py) |
| ShortcutWidget | Keyboard shortcut capture widget. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/wigglystuff/shortcut.py) |
| Slider2D | A 2D slider control for selecting x/y coordinates interactively. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/wigglystuff/slider2d.py) |
| SortableList | A draggable list for reordering items interactively. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/wigglystuff/sortlist.py) |
| SpeechToText | Convert voice input to text using the browser's speech recognition. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/wigglystuff/talk.py) |
| Tangle | Interactive number manipulation with draggable values in text. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/wigglystuff/tangle.py) |
| TextCompare | Side-by-side text comparison with highlighted matching passages. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/wigglystuff/textcompare.py) |
| ThreeWidget | 3D visualization widget using Three.js. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/wigglystuff/threewidget.py) |
| Drawdata Scatter | Draw scatter data interactively and visualize it with Altair histograms. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/drawdata/scatter-demo.py) |
| WebcamCapture | Capture images from the webcam. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/wigglystuff/webcam_capture.py) |
