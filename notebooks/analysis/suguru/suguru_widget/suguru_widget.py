from pathlib import Path
import anywidget
import traitlets

_ASSET_DIR = Path(__file__).parent
_JS_SOURCE = (_ASSET_DIR / "suguru_widget.js").read_text()
_CSS_SOURCE = (_ASSET_DIR / "suguru_widget.css").read_text()


class SuguruGeneratorWidget(anywidget.AnyWidget):
    _esm = _JS_SOURCE
    _css = _CSS_SOURCE

    width = traitlets.Int(default_value=5).tag(sync=True)
    height = traitlets.Int(default_value=5).tag(sync=True)
    shapes = traitlets.List(traitlets.List(traitlets.Int()), default_value=[]).tag(sync=True)
