from pathlib import Path

import anywidget
import traitlets


_ASSET_DIR = Path(__file__).parent
_JS_SOURCE = (_ASSET_DIR / "widget.js").read_text()
_CSS_SOURCE = (_ASSET_DIR / "widget.css").read_text()


class GrpoGdpoWidget(anywidget.AnyWidget):
    """Interactive widget for comparing GRPO vs GDPO advantage calculations.

    Allows users to toggle binary rewards (correctness, style, conciseness)
    and see how the two normalization approaches differ.
    """
    _esm = _JS_SOURCE
    _css = _CSS_SOURCE

    rewards = traitlets.List(
        traitlets.Dict(),
        default_value=[
            {"correctness": 1, "style": 0, "conciseness": 0},
            {"correctness": 1, "style": 0, "conciseness": 1},
            {"correctness": 0, "style": 1, "conciseness": 1},
            {"correctness": 0, "style": 1, "conciseness": 1},
            {"correctness": 1, "style": 0, "conciseness": 0},
            {"correctness": 0, "style": 1, "conciseness": 0},
        ]
    ).tag(sync=True)

    def add_rollout(self):
        """Add a new rollout with default values."""
        self.rewards = self.rewards + [{"correctness": 0, "style": 0, "conciseness": 0}]

    def remove_rollout(self, index=-1):
        """Remove a rollout by index (default: last)."""
        if len(self.rewards) > 2:
            rewards = list(self.rewards)
            rewards.pop(index)
            self.rewards = rewards
