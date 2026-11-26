"""Light / dark theme shared by the Qt GUIs.

The plotting code keeps drawing with the usual light colours (black text,
white backgrounds, black or grey grids). ThemedFigure maps those to dark
equivalents each time it is drawn, so every plot follows the theme without
changing how it is made. The Qt widgets get a dark Fusion palette.

    from tlist_theme import ThemedFigure as Figure
    tlist_theme.add_menu(main_window)        # View > Dark Mode, remembered
"""

from matplotlib import colors as mcolors
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtGui, QtWidgets

DARK_BG = "#1e1e1e"
DARK_FG = "#e0e0e0"

# (light colour, dark colour) pairs, per kind of artist
_BG = [("white", DARK_BG)]
_FG = [("black", DARK_FG)]
_GRID = [("black", "#7a7a7a"), ("#b0b0b0", "#4a4a4a")]
_LEGEND_EDGE = [("0.8", "#555555")]

_state = {"dark": False, "original": None}
_settings = QtCore.QSettings("TLIST", "tlist")


def is_dark() -> bool:
    return _state["dark"]


def _map(color, pairs):
    """The theme's version of `color` (alpha kept), or None if not themed."""
    if color is None or (isinstance(color, str) and color.lower() == "none"):
        return None
    try:
        rgba = mcolors.to_rgba(color)
    except ValueError:
        return None
    for light, dark in pairs:
        src, dst = (light, dark) if _state["dark"] else (dark, light)
        if mcolors.same_color(rgba[:3], mcolors.to_rgb(src)):
            return (*mcolors.to_rgb(dst), rgba[3])
    return None


def _recolor(get, set_, pairs):
    new = _map(get(), pairs)
    if new is not None:
        set_(new)


def _text(t):
    _recolor(t.get_color, t.set_color, _FG)
    box = t.get_bbox_patch()
    if box is not None:
        _recolor(box.get_facecolor, box.set_facecolor, _BG)


def restyle(fig):
    """Recolour a figure's standard light/dark colours for the current theme."""
    _recolor(fig.get_facecolor, fig.set_facecolor, _BG)
    for t in fig.texts:
        _text(t)
    fg = DARK_FG if _state["dark"] else "black"
    for ax in fig.axes:
        _recolor(ax.get_facecolor, ax.set_facecolor, _BG)
        for sp in ax.spines.values():
            _recolor(sp.get_edgecolor, sp.set_edgecolor, _FG)
        ax.tick_params(which="both", colors=fg, labelcolor=fg)
        texts = [ax.title, ax.xaxis.label, ax.yaxis.label,
                 ax.xaxis.get_offset_text(), ax.yaxis.get_offset_text(), *ax.texts]
        texts += [t for t in (getattr(ax, "_left_title", None), getattr(ax, "_right_title", None))
                  if t is not None]
        for t in texts:
            _text(t)
        for line in ax.xaxis.get_gridlines() + ax.yaxis.get_gridlines():
            _recolor(line.get_color, line.set_color, _GRID)
        for line in ax.lines:
            _recolor(line.get_color, line.set_color, _FG)
        leg = ax.get_legend()
        if leg is not None:
            frame = leg.get_frame()
            _recolor(frame.get_facecolor, frame.set_facecolor, _BG)
            _recolor(frame.get_edgecolor, frame.set_edgecolor, _LEGEND_EDGE)
            for t in leg.get_texts():
                _text(t)


class ThemedFigure(Figure):
    """A Figure that follows the light/dark theme whenever it is drawn."""

    def draw(self, renderer):
        restyle(self)
        super().draw(renderer)


def _dark_palette():
    p = QtGui.QPalette()
    c = QtGui.QColor
    # Inputs (Base) a shade lighter than the window, so empty checkboxes and
    # fields stand out (Fusion outlines them in a darker window colour).
    for role, col in [(p.Window, "#262626"), (p.WindowText, DARK_FG), (p.Base, "#3a3a3a"),
                      (p.AlternateBase, "#424242"), (p.ToolTipBase, "#2b2b2b"),
                      (p.ToolTipText, DARK_FG), (p.Text, DARK_FG), (p.Button, "#3a3a3a"),
                      (p.ButtonText, DARK_FG), (p.BrightText, "#ff5555"),
                      (p.Link, "#4ea1ff"), (p.Highlight, "#2f65ca"),
                      (p.HighlightedText, "#ffffff"), (p.PlaceholderText, "#808080")]:
        p.setColor(role, c(col))
    for role in (p.WindowText, p.Text, p.ButtonText):
        p.setColor(p.Disabled, role, c("#7a7a7a"))
    p.setColor(p.Disabled, p.Base, c("#2e2e2e"))
    return p


def set_dark(dark: bool):
    """Switch the whole application (widgets and every plot) to dark or light."""
    app = QtWidgets.QApplication.instance()
    if _state["original"] is None:
        _state["original"] = (app.style().objectName(), QtGui.QPalette(app.palette()))
    _state["dark"] = bool(dark)
    if dark:
        app.setStyle("Fusion")
        app.setPalette(_dark_palette())
    else:
        style, palette = _state["original"]
        app.setStyle(style)
        app.setPalette(palette)
    _settings.setValue("dark_mode", bool(dark))
    for w in app.allWidgets():
        if isinstance(w, NavigationToolbar2QT):
            _refresh_toolbar_icons(w)
        elif isinstance(w, FigureCanvasQTAgg):
            w.draw_idle()
    for act in _actions:
        act.blockSignals(True)
        act.setChecked(bool(dark))
        act.blockSignals(False)


def _refresh_toolbar_icons(tb):
    """matplotlib picks icon colours for the background when the toolbar is
    made, so redo them after the palette changes."""
    try:
        for text, _tip, image, callback in tb.toolitems:
            if text is not None and image and callback in tb._actions:
                tb._actions[callback].setIcon(tb._icon(image + ".png"))
    except (AttributeError, TypeError, ValueError):
        pass                                    # different matplotlib internals


_actions = []


def add_menu(window: QtWidgets.QMainWindow):
    """Add View > Dark Mode to a main window and apply the saved choice."""
    menu = window.menuBar().addMenu("View")
    act = menu.addAction("Dark Mode")
    act.setCheckable(True)
    act.setShortcut(QtGui.QKeySequence("Ctrl+Shift+D"))
    act.toggled.connect(set_dark)
    _actions.append(act)
    saved = _settings.value("dark_mode", False, type=bool)
    if saved != _state["dark"] or saved:
        set_dark(saved)
    else:
        act.setChecked(_state["dark"])
    return act
