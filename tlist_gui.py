#!/usr/bin/env python3
"""Qt5 GUI for Prospect T-List coincidence analysis.

    python tlist_gui.py [directory]
"""

import html
import os
import sys

from PyQt5 import QtCore, QtWidgets
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from tlist_theme import ThemedFigure as Figure

import tlist_core as core
import tlist_plot
import tlist_theme
from tlist_widgets import ProjectionTab, SinglesTab, _spin


class Loader(QtCore.QThread):
    """Scans a directory or loads runs off the UI thread."""
    progress = QtCore.pyqtSignal(str)
    done = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, fn, *args):
        super().__init__()
        self.fn, self.args = fn, args

    def run(self):
        try:
            self.done.emit(self.fn(self.progress.emit, *self.args))
        except Exception as e:          # report anything to the status bar
            self.failed.emit(f"{type(e).__name__}: {e}")


class PlotTab(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.fig = Figure(figsize=(8, 6), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.fig)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(NavigationToolbar2QT(self.canvas, self))
        lay.addWidget(self.canvas)


class Main(QtWidgets.QMainWindow):
    def __init__(self, directory=None):
        super().__init__()
        self.setWindowTitle("T-List Coincidence Analysis")
        self.resize(1400, 850)
        self.runs = []
        self.skipped = []           # (path, reason) of unusable .ava files
        self.cache = {}             # run.key -> EventData
        self.result = None
        self.worker = None

        side = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(side)

        # Directory
        self.dir_edit = QtWidgets.QLineEdit(directory or "")
        browse = QtWidgets.QPushButton("Browse…")
        browse.clicked.connect(self.browse)
        scan = QtWidgets.QPushButton("Scan")
        scan.clicked.connect(self.scan)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.dir_edit)
        row.addWidget(browse)
        row.addWidget(scan)
        form.addRow("Directory", row)

        # Device / run selection
        self.dev = {}
        self.run_box = {}
        for ab in "AB":
            d = QtWidgets.QComboBox()
            r = QtWidgets.QComboBox()
            r.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
            d.currentTextChanged.connect(lambda _t, ab=ab: self.fill_runs(ab))
            self.dev[ab], self.run_box[ab] = d, r
            if ab == "A":
                r.currentIndexChanged.connect(lambda _i: self.match_b())
            form.addRow(f"Device {ab}", d)
            form.addRow(f"Acq. start {ab}", r)

        # Time settings
        self.t_lo = _spin(-1e9, 1e9, -15000, 1, 100)
        self.t_hi = _spin(-1e9, 1e9, 15000, 1, 100)
        self.t_bins = _spin(1, 100000, 150)
        self.offset = _spin(-1e9, 1e9, 0, 1, 100)
        form.addRow(QtWidgets.QLabel("<b>Delta time (ns)</b>"))
        form.addRow("Range low", self.t_lo)
        form.addRow("Range high", self.t_hi)
        form.addRow("Bins", self.t_bins)
        form.addRow("Offset added to B", self.offset)

        self.gate_on = QtWidgets.QCheckBox("Gate energy plot on Δt window")
        self.g_lo = _spin(-1e9, 1e9, -800, 1, 100)
        self.g_hi = _spin(-1e9, 1e9, 0, 1, 100)
        form.addRow(self.gate_on)
        form.addRow("Gate low", self.g_lo)
        form.addRow("Gate high", self.g_hi)

        # Energy settings per axis
        self.units, self.e_lo, self.e_hi, self.e_bins, self.e_width = {}, {}, {}, {}, {}
        for ab in "AB":
            form.addRow(QtWidgets.QLabel(f"<b>Axis {ab} ({'x' if ab == 'A' else 'y'})</b>"))
            u = QtWidgets.QComboBox()
            u.addItems(["energy", "channel"])
            self.units[ab] = u
            self.e_lo[ab] = _spin(-1e6, 1e6, 0, 1, 10)
            self.e_hi[ab] = _spin(-1e6, 1e6, 4000, 1, 10)
            self.e_bins[ab] = _spin(1, 100000, 200)
            w = self.e_width[ab] = _spin(0, 1e5, 0, 4, 0.5)
            w.setSpecialValueText("auto (use Bins)")
            w.setSuffix(" keV")
            w.setToolTip("Histogram bin width. 0 = auto: Bins equal bins from Low to High.\n"
                         "Otherwise the bins have this width and High is rounded up to fit.")
            w.valueChanged.connect(lambda v, ab=ab: self.e_bins[ab].setEnabled(v == 0))
            form.addRow("Units", u)
            form.addRow("Low", self.e_lo[ab])
            form.addRow("High", self.e_hi[ab])
            form.addRow("Bins", self.e_bins[ab])
            form.addRow("Bin width", w)

        self.log = QtWidgets.QCheckBox("Log scale")
        self.log.setChecked(True)
        form.addRow(self.log)

        self.graph_btn = QtWidgets.QPushButton("Graph")
        self.graph_btn.clicked.connect(self.graph)
        export_btn = QtWidgets.QPushButton("Export data…")
        export_btn.clicked.connect(self.export)
        form.addRow(self.graph_btn)
        form.addRow(export_btn)

        self.info = QtWidgets.QLabel()
        self.info.setWordWrap(True)
        form.addRow(self.info)

        # Re-plot automatically on setting changes once data are loaded
        for w in [self.t_lo, self.t_hi, self.t_bins, self.offset, self.g_lo, self.g_hi,
                  *self.e_lo.values(), *self.e_hi.values(), *self.e_bins.values(),
                  *self.e_width.values()]:
            w.valueChanged.connect(self.replot_if_loaded)
        for w in [self.gate_on, self.log]:
            w.toggled.connect(self.replot_if_loaded)
        for w in self.units.values():
            w.currentTextChanged.connect(self.units_changed)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(side)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(380)

        self.tabs = QtWidgets.QTabWidget()
        self.ee_tab, self.dt_tab = PlotTab(), PlotTab()
        self.tabs.addTab(self.ee_tab, "Energy vs Energy")
        self.tabs.addTab(self.dt_tab, "Delta time")
        self.proj_tab = ProjectionTab(self.status)
        self.tabs.addTab(self.proj_tab, "Projections")
        self.singles_tab = SinglesTab(self.status, self.load_runs)
        self.tabs.addTab(self.singles_tab, "Singles")

        split = QtWidgets.QSplitter()
        split.addWidget(scroll)
        split.addWidget(self.tabs)
        split.setStretchFactor(1, 1)
        self.setCentralWidget(split)

        tlist_theme.add_menu(self)

        if directory:
            QtCore.QTimer.singleShot(0, self.scan)

    # ---------------------------------------------------------------- helpers
    def status(self, msg):
        self.statusBar().showMessage(msg)

    def busy(self):
        return self.worker is not None and self.worker.isRunning()

    def start(self, fn, args, on_done):
        if self.busy():
            return
        self.graph_btn.setEnabled(False)
        self.worker = Loader(fn, *args)
        self.worker.progress.connect(self.status)
        self.worker.done.connect(on_done)
        self.worker.failed.connect(lambda m: self.status("Error: " + m))
        self.worker.finished.connect(lambda: self.graph_btn.setEnabled(True))
        self.worker.start()

    # ---------------------------------------------------------------- actions
    def browse(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Data directory", self.dir_edit.text())
        if d:
            self.dir_edit.setText(d)
            self.scan()

    def scan(self):
        d = self.dir_edit.text().strip()
        if not os.path.isdir(d):
            self.status(f"Not a directory: {d}")
            return

        def job(progress, d):
            progress(f"Scanning {d} …")
            skipped = []
            return core.scan_directory(d, skipped), skipped
        self.start(job, (d,), self.scanned)

    def scanned(self, out):
        runs, self.skipped = out
        self.runs, self.cache, self.result = runs, {}, None
        devs = core.devices(runs)
        for i, ab in enumerate("AB"):
            box = self.dev[ab]
            box.blockSignals(True)
            box.clear()
            box.addItems(devs)
            box.setCurrentIndex(min(i, len(devs) - 1))
            box.blockSignals(False)
            self.fill_runs(ab)
        self.singles_tab.data = {}
        self.offer_singles()
        msg = f"Found {len(runs)} acquisitions from {len(devs)} devices"
        warnings = core.scan_warnings(runs, self.skipped)
        self.info.setText(self.warning_html(warnings))
        if warnings:
            msg += f" - {len(warnings)} warning(s)"
        # Pop up only for skipped files or problems with the pair selected by default
        # (an old fragmentary run in the folder shouldn't nag on every scan)
        sel = [r for r in (self.run_box["A"].currentData(), self.run_box["B"].currentData()) if r]
        urgent = core.scan_warnings(runs, self.skipped, only=sel)
        if urgent:
            box = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, "Data may be incomplete",
                                        "Some data in this directory can't be used or looks "
                                        "incomplete. Check that every file was copied.", parent=self)
            box.setInformativeText("\n".join("• " + w for w in urgent[:6])
                                   + ("\n…" if len(urgent) > 6 else ""))
            box.setDetailedText("\n\n".join(urgent))
            box.open()
        self.status(msg)

    @staticmethod
    def warning_html(warnings):
        return "".join(f"<font color='red'>⚠ {html.escape(w)}</font><br>" for w in warnings)

    def fill_runs(self, ab):
        box = self.run_box[ab]
        box.clear()
        for r in self.runs:
            if r.device == self.dev[ab].currentText():
                box.addItem(f"{r.start_str}  ({len(r.files)} files)", r)
        box.setCurrentIndex(box.count() - 1)       # latest run
        if ab == "A":
            self.match_b()

    def match_b(self):
        """Select the B run closest in start time to A's."""
        ra = self.run_box["A"].currentData()
        boxb = self.run_box["B"]
        if ra is None or boxb.count() == 0:
            return
        best = min(range(boxb.count()), key=lambda i: abs(boxb.itemData(i).start_ms - ra.start_ms))
        boxb.setCurrentIndex(best)
        self.offer_singles()

    def offer_singles(self):
        """Singles tab: for every device, the run closest in start to A's run."""
        ra = self.run_box["A"].currentData()
        if ra is None:
            return
        runs = [min((r for r in self.runs if r.device == d),
                    key=lambda r: abs(r.start_ms - ra.start_ms))
                for d in core.devices(self.runs)]
        self.singles_tab.set_runs(runs)

    def load_runs(self, runs, then):
        """Load any runs not yet cached (in the background), then call
        then({run.key: EventData}) for all of them."""
        need = [r for r in runs if r.key not in self.cache]

        def job(progress, need):
            out = {}
            for r in need:
                out[r.key] = core.load_run(
                    r, lambda k, n, r=r: progress(f"Loading {r.device}: file {k}/{n}"))
            return out

        def loaded(out):
            self.cache.update(out)
            then({r.key: self.cache[r.key] for r in runs})
        if not need:
            loaded({})
        elif self.busy():
            self.status("Busy loading - try again in a moment")
        else:
            self.start(job, (need,), loaded)

    def units_changed(self):
        for ab, u in self.units.items():
            self.e_width[ab].setSuffix(" keV" if u.currentText() == "energy" else " ch")
        self.replot_if_loaded()

    def axis_binning(self, ab, run):
        """((lo, hi), bins) for an energy axis, honouring the Bin width field,
        plus a stripes warning ('' if none)."""
        lo, hi = self.e_lo[ab].value(), self.e_hi[ab].value()
        width = self.e_width[ab].value()
        energy = self.units[ab].currentText() == "energy"
        if width > 0:
            if not energy:
                width = max(1, round(width))
            edges = core.edges_for_width(lo, hi, width)
            rng, bins = (lo, float(edges[-1])), len(edges) - 1
        else:
            rng, bins = (lo, hi), self.e_bins[ab].value()
        note = (core.comb_note(run.device, (rng[1] - rng[0]) / bins, run.ecal[1])
                if energy else "")
        return rng, bins, note

    def graph(self):
        ra, rb = self.run_box["A"].currentData(), self.run_box["B"].currentData()
        if ra is None or rb is None:
            self.status("Select two acquisitions first")
            return
        self.load_runs([ra, rb], lambda _data: self.compute(ra, rb))

    def replot_if_loaded(self, *_):
        ra, rb = self.run_box["A"].currentData(), self.run_box["B"].currentData()
        if self.result and ra and rb and ra.key in self.cache and rb.key in self.cache:
            self.compute(ra, rb)

    def compute(self, ra, rb):
        if self.t_hi.value() <= self.t_lo.value():
            self.status("Time range high must exceed low")
            return
        gate = (self.g_lo.value(), self.g_hi.value()) if self.gate_on.isChecked() else None
        try:
            ea_range, ea_bins, note_a = self.axis_binning("A", ra)
            eb_range, eb_bins, note_b = self.axis_binning("B", rb)
        except ValueError as e:
            self.status(str(e))
            return
        self.result = res = core.analyse(
            self.cache[ra.key], self.cache[rb.key],
            t_range=(self.t_lo.value(), self.t_hi.value()), t_bins=self.t_bins.value(),
            ea_range=ea_range, ea_bins=ea_bins,
            eb_range=eb_range, eb_bins=eb_bins,
            units_a=self.units["A"].currentText(), units_b=self.units["B"].currentText(),
            gate=gate, offset_ns=self.offset.value())
        log = self.log.isChecked()

        f = self.ee_tab.fig
        f.clear()
        ax = f.add_subplot()
        m = tlist_plot.draw_energy_energy(ax, res, log=log)
        f.colorbar(m, ax=ax, label="Counts")
        self.ee_tab.canvas.draw_idle()

        f = self.dt_tab.fig
        f.clear()
        tlist_plot.draw_delta_time(f.add_subplot(), res, log=log)
        self.dt_tab.canvas.draw_idle()

        self.proj_tab.set_result(res, log)

        txt = tlist_plot.summary(res).replace("\n", "<br>")
        if abs(ra.start_ms - rb.start_ms) > 10_000:
            txt += "<br><font color='red'>Runs started &gt;10 s apart</font>"
        for note in (note_a, note_b):
            if note:
                txt += f"<br><font color='#d9534f'>{html.escape(note)}</font>"
        warnings = core.scan_warnings(self.runs, self.skipped, only=[ra, rb])
        if warnings:
            txt += "<br>" + self.warning_html(warnings)
        self.info.setText(txt)
        self.status(f"{int(res.t_counts.sum()):,} pairs in Δt window, "
                    f"{int(res.ee_counts.sum()):,} in energy plot")

    def export(self):
        if not self.result:
            self.status("Nothing to export - press Graph first")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export prefix (writes _time.csv and _energy.csv)", "coincidence", "CSV (*.csv)")
        if not path:
            return
        prefix = path[:-4] if path.lower().endswith(".csv") else path
        core.export_time(self.result, prefix + "_time.csv")
        core.export_energy(self.result, prefix + "_energy.csv")
        self.status(f"Wrote {prefix}_time.csv and {prefix}_energy.csv")


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = Main(sys.argv[1] if len(sys.argv) > 1 else None)
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
