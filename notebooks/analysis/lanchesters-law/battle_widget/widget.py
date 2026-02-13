from __future__ import annotations

from pathlib import Path

import anywidget
import traitlets


def _read_text(relpath: str) -> str:
    return (Path(__file__).parent / relpath).read_text(encoding="utf-8")


class BattleWidget(anywidget.AnyWidget):
    """
    JS-driven batch battle simulator widget.

    Configure a parameter grid via `grid_spec` (cartesian product of lists) and
    `runs_per_point`; the simulation runs immediately when the widget mounts.

    Results land in `results` as a flat list of records:
    `{"run_id": str, "seed": int, "time": float, "n_blue": int, "n_red": int}`.
    When finished, `done=True` and `results_len` is set. If JS fails, `error`
    contains a message.
    """
    _esm = _read_text("battle_widget.js")
    _css = _read_text("battle_widget.css")

    grid_spec = traitlets.Dict(
        default_value={"n_blue": [50], "n_red": [50]},
        help="Cartesian grid spec; values must be lists.",
    ).tag(sync=True)
    runs_per_point = traitlets.Int(1).tag(sync=True)
    seed_mode = traitlets.Unicode("random").tag(sync=True)  # "random" | "base_plus_index"
    base_seed = traitlets.Int(1).tag(sync=True)

    arena_width = traitlets.Int(640).tag(sync=True)
    arena_height = traitlets.Int(420).tag(sync=True)
    unit_radius = traitlets.Float(4.0).tag(sync=True)
    spawn_mode = traitlets.Unicode("sides", help='"sides" spawns teams on opposite halves; "mixed" mixes both sides across the arena.').tag(sync=True)

    step_dt = traitlets.Float(0.02).tag(sync=True)  # seconds
    move_speed = traitlets.Float(55.0).tag(sync=True)  # pixels / second

    attack_range = traitlets.Float(10.0).tag(sync=True)  # pixels
    attack_cooldown = traitlets.Float(0.25).tag(sync=True)  # seconds
    hit_chance = traitlets.Float(0.85).tag(sync=True)
    damage = traitlets.Int(1).tag(sync=True)
    hp = traitlets.Int(1).tag(sync=True)
    max_time = traitlets.Float(60.0).tag(sync=True)
    record_dt = traitlets.Float(0.1).tag(sync=True)

    render = traitlets.Bool(True).tag(sync=True)
    done = traitlets.Bool(False).tag(sync=True)
    results = traitlets.List(traitlets.Dict()).tag(sync=True)
    results_len = traitlets.Int(0).tag(sync=True)
    error = traitlets.Unicode("").tag(sync=True)
