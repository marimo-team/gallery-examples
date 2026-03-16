import time
from pathlib import Path
import anywidget
import traitlets

_ASSET_DIR = Path(__file__).parent
_JS_SOURCE = (_ASSET_DIR / "rpsls_widget.js").read_text()
_CSS_SOURCE = (_ASSET_DIR / "rpsls_widget.css").read_text()


class RpslsWidget(anywidget.AnyWidget):
    """Interactive RPSLS tournament graph widget.

    Keyword Arguments:
        n: Number of elements in the tournament (default 3, minimum 3).
        highlighted_node: Name of the currently highlighted node (default "").
    """

    _esm = _JS_SOURCE
    _css = _CSS_SOURCE

    n = traitlets.Int(default_value=3).tag(sync=True)
    highlighted_node = traitlets.Unicode(default_value="").tag(sync=True)
    _node_duration = traitlets.Int(default_value=400).tag(sync=True)
    _edge_duration = traitlets.Int(default_value=400).tag(sync=True)

    def animate_node(self, increment=1, node_duration=400, edge_duration=400):
        """Add or remove nodes from the graph with animation.

        Args:
            increment: Number of nodes to add (positive) or remove (negative).
            node_duration: Duration in ms for node movement/appearance.
            edge_duration: Duration in ms for edge fade-in/out.
        """
        self._node_duration = node_duration
        self._edge_duration = edge_duration
        step = 1 if increment > 0 else -1
        delay = (node_duration + edge_duration) / 1000 + 0.1
        for _ in range(abs(increment)):
            self.n = max(3, self.n + step)
            time.sleep(delay)

    def animate_highlight(self, name, duration=400):
        """Highlight a node by name, or clear if already highlighted.

        Args:
            name: Name of the node to highlight (e.g. "Rock").
            duration: Animation duration in milliseconds.
        """
        self._node_duration = duration
        self._edge_duration = duration
        if self.highlighted_node == name:
            self.highlighted_node = ""
        else:
            self.highlighted_node = name

    def clear_highlight(self, duration=400):
        """Remove any active node highlight.

        Args:
            duration: Animation duration in milliseconds.
        """
        self._node_duration = duration
        self._edge_duration = duration
        self.highlighted_node = ""
