"""Qt widgets shared by tlist_gui.py and tlist_scanner.py."""

import numpy as np
from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

import tlist_core as core
import tlist_plot


def _spin(lo, hi, val, dec=0, step=1):
    w = QtWidgets.QDoubleSpinBox() if dec else QtWidgets.QSpinBox()
    if dec:
        w.setDecimals(dec)
    w.setRange(lo, hi)
    w.setSingleStep(step)
    w.setValue(val)
    w.setKeyboardTracking(False)
    return w


class ProjectionTab(QtWidgets.QWidget):
    """Slice the energy-energy matrix on one axis and project onto the other.

    The slice is set by centre/width sliders (or low/high boxes) and is drawn
    as a band on the 2D plot; the projection is cut from the exact pair values.
    """
    SCALE = 10                      # slider steps per unit (0.1 keV resolution)

    def __init__(self, status):
        super().__init__()
        self.status = status
        self.res = None
        self.log = True
        self._sync = False
        self.band = []
        self.proj = None            # (edges, counts, n) of the current slice

        self.axis = QtWidgets.QComboBox()
        self.axis.addItems(["Gate on A (x), project onto B", "Gate on B (y), project onto A"])
        self.axis.currentIndexChanged.connect(self.axis_changed)

        self.c_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.w_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.centre = _spin(-1e6, 1e6, 511, 1, 1)
        self.width = _spin(0.1, 1e6, 20, 1, 1)
        self.lo = _spin(-1e6, 1e6, 501, 1, 1)
        self.hi = _spin(-1e6, 1e6, 521, 1, 1)
        for w in (self.centre, self.width, self.lo, self.hi):
            w.setKeyboardTracking(False)
        self.c_slider.valueChanged.connect(lambda v: self.set_cw(v / self.SCALE, None))
        self.w_slider.valueChanged.connect(lambda v: self.set_cw(None, v / self.SCALE))
        self.centre.valueChanged.connect(lambda v: self.set_cw(v, None))
        self.width.valueChanged.connect(lambda v: self.set_cw(None, v))
        self.lo.valueChanged.connect(self.lohi_changed)
        self.hi.valueChanged.connect(self.lohi_changed)

        self.bins = _spin(1, 10000, 2)
        self.bins.setToolTip("Bins follow the projected detector's channel boundaries "
                             "(avoids a comb pattern); this sets channels per bin.")
        self.bins.valueChanged.connect(self.update_slice)
        self.show_total = QtWidgets.QCheckBox("Show ungated projection")
        self.show_total.setChecked(True)
        self.show_total.toggled.connect(self.update_slice)
        export = QtWidgets.QPushButton("Export projection…")
        export.clicked.connect(self.export)

        grid = QtWidgets.QGridLayout()
        grid.addWidget(QtWidgets.QLabel("Slice"), 0, 0)
        grid.addWidget(self.axis, 0, 1, 1, 2)
        grid.addWidget(QtWidgets.QLabel("Channels / bin"), 0, 3)
        grid.addWidget(self.bins, 0, 4)
        grid.addWidget(self.show_total, 0, 5)
        grid.addWidget(export, 0, 6)
        self.unit_lbl = []
        for row, (name, slider, spin, name2, spin2) in enumerate(
                [("Centre", self.c_slider, self.centre, "Low", self.lo),
                 ("Width", self.w_slider, self.width, "High", self.hi)], start=1):
            grid.addWidget(QtWidgets.QLabel(name), row, 0)
            grid.addWidget(slider, row, 1, 1, 2)
            grid.addWidget(spin, row, 3)
            grid.addWidget(QtWidgets.QLabel(name2), row, 4)
            grid.addWidget(spin2, row, 5)
            u = QtWidgets.QLabel("keV")
            self.unit_lbl.append(u)
            grid.addWidget(u, row, 6)
        grid.setColumnStretch(1, 1)

        self.fig = Figure(figsize=(10, 6), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.fig)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addLayout(grid)
        lay.addWidget(NavigationToolbar2QT(self.canvas, self))
        lay.addWidget(self.canvas, 1)
        self.ax2d = self.axp = None

    # ------------------------------------------------------------ state
    def gate_axis(self):
        return "a" if self.axis.currentIndex() == 0 else "b"

    def gate_range(self):
        r = self.res
        e = r.ea_edges if self.gate_axis() == "a" else r.eb_edges
        return float(e[0]), float(e[-1])

    def set_result(self, res, log):
        self.res, self.log = res, log
        self.configure_sliders()
        self.redraw()

    def axis_changed(self):
        if self.res is not None:
            self.configure_sliders()
            self.redraw()

    def configure_sliders(self):
        lo, hi = self.gate_range()
        ug = self.res.units_a if self.gate_axis() == "a" else self.res.units_b
        for u in self.unit_lbl:
            u.setText("keV" if ug == "energy" else "ch")
        self._sync = True
        self.c_slider.setRange(int(lo * self.SCALE), int(hi * self.SCALE))
        self.w_slider.setRange(1, int((hi - lo) * self.SCALE))
        self._sync = False
        c, w = self.centre.value(), self.width.value()
        if not (lo <= c <= hi):
            c = (lo + hi) / 2
        self.set_cw(c, min(w, hi - lo), redraw=False)

    def set_cw(self, c=None, w=None, redraw=True):
        """Single place that changes the slice; keeps all widgets in step."""
        if self._sync:
            return
        c = self.centre.value() if c is None else c
        w = self.width.value() if w is None else w
        self._sync = True
        self.centre.setValue(c)
        self.width.setValue(w)
        self.lo.setValue(c - w / 2)
        self.hi.setValue(c + w / 2)
        self.c_slider.setValue(int(round(c * self.SCALE)))
        self.w_slider.setValue(int(round(w * self.SCALE)))
        self._sync = False
        if redraw:
            self.update_slice()

    def lohi_changed(self):
        if self._sync:
            return
        lo, hi = self.lo.value(), self.hi.value()
        if hi <= lo:
            hi = lo + 0.1
        self.set_cw((lo + hi) / 2, hi - lo)

    # ------------------------------------------------------------ drawing
    def redraw(self):
        if self.res is None:
            return
        self.fig.clear()
        self.ax2d, self.axp = self.fig.subplots(1, 2, width_ratios=[1, 1.25])
        m = tlist_plot.draw_energy_energy(self.ax2d, self.res, log=self.log)
        self.ax2d.set_title("Energy vs energy (slice shaded)")
        self.fig.colorbar(m, ax=self.ax2d, label="Counts", shrink=0.8)
        self.band = []
        self.update_slice()

    def update_slice(self, *_):
        if self.res is None or self.ax2d is None:
            return
        res, ga = self.res, self.gate_axis()
        lo, hi = self.lo.value(), self.hi.value()

        for artist in self.band:
            artist.remove()
        span = self.ax2d.axvspan if ga == "a" else self.ax2d.axhspan
        line = self.ax2d.axvline if ga == "a" else self.ax2d.axhline
        self.band = [span(lo, hi, facecolor="white", alpha=0.35, edgecolor="none", zorder=3),
                     line(lo, color="k", lw=1, zorder=4), line(hi, color="k", lw=1, zorder=4)]

        gdev, pdev = (res.a, res.b) if ga == "a" else (res.b, res.a)
        ug, up = (res.units_a, res.units_b) if ga == "a" else (res.units_b, res.units_a)
        pe = res.eb_edges if ga == "a" else res.ea_edges
        edges = core.channel_edges(pdev.run, up, pe[0], pe[-1], self.bins.value())
        edges, counts, n = core.project(res, ga, lo, hi, edges)
        self.proj = (edges, counts, n)

        def uname(u):
            return "keV" if u == "energy" else "ch"
        ax = self.axp
        ax.clear()
        if self.show_total.isChecked():
            _, tot, _ = core.project(res, ga, -np.inf, np.inf, edges)
            ax.step(edges, np.append(tot, tot[-1]), where="post", color="0.6", lw=1,
                    label="Ungated (all pairs)")
        ax.step(edges, np.append(counts, counts[-1]), where="post", color="C3", lw=1.3,
                label=f"{gdev.run.device} in [{lo:g}, {hi:g}) {uname(ug)}")
        if self.log:
            ax.set_yscale("log")
            ax.set_ylim(bottom=0.8)
        ax.set_xlim(edges[0], edges[-1])
        ax.set_xlabel(f"{pdev.run.device} {'Energy (keV)' if up == 'energy' else 'Channel'}")
        ax.set_ylabel(f"Counts / {edges[1] - edges[0]:.3g} {uname(up)}")
        ax.set_title(f"Projection onto {pdev.run.device}: {n:,} pairs in slice")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
        self.canvas.draw_idle()

    def export(self):
        if self.proj is None:
            self.status("Nothing to export - press Graph first")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export projection", "projection.csv", "CSV (*.csv)")
        if path:
            edges, counts, _ = self.proj
            core.export_projection(path, self.res, self.gate_axis(), self.lo.value(),
                                   self.hi.value(), edges, counts)
            self.status(f"Wrote {path}")
