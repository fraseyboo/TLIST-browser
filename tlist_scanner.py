#!/usr/bin/env python3
"""
T-List Data Scanner (training version).

A Qt5 GUI that follows the workflow of the original T-List Data Scanner and
its lab manual, step by step:

  Search Directories  ->  Add New Directory, press the green 'play' icon to
                          start processing (files are re-scanned every
                          "Search Frequency" seconds).
  Scan Results        ->  devices / acquisitions found, calibration, time
                          window; click a device name to see its (ungated)
                          singles spectrum, hover to read channel energies.
  Analysis            ->  pick two devices + acquisition starts, choose X/Y
                          units, Graph / Clear Graph / Export Data.
  Projections (extra) ->  slice the last energy/channel matrix from Analysis
                          on one device and project onto the other.

Uses the same reader and coincidence code as tlist_cli.py / tlist_gui.py.

    python tlist_scanner.py
"""

import dataclasses
import html
import os
import sys
import threading
import time
from datetime import datetime

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.colors import LogNorm
from tlist_theme import ThemedFigure as Figure

import tlist_core as core
import tlist_plot
import tlist_theme
from tlist_widgets import ProjectionTab

LOW_E_CHECK_KEV = 100.0      # Scan Results spectrum warns if nothing is below this energy

UNIT_ENERGY, UNIT_CHANNEL, UNIT_DT, UNIT_COUNTS = (
    "Energy (keV)", "Channel", "Delta Time (ns)", "Counts")


def fmt_time(ms):
    """Old-tool style local time: 11/19/2026 11:56:26 AM"""
    return datetime.fromtimestamp(ms / 1000).strftime("%m/%d/%Y %I:%M:%S %p")


def play_icon(color="#2db34a", stop=False):
    pm = QtGui.QPixmap(22, 22)
    pm.fill(QtCore.Qt.transparent)
    p = QtGui.QPainter(pm)
    p.setRenderHint(QtGui.QPainter.Antialiasing)
    if stop:
        p.setBrush(QtGui.QColor("#c62828"))
        p.setPen(QtGui.QPen(QtGui.QColor("#7f1010"), 1))
        p.drawRoundedRect(3, 3, 16, 16, 3, 3)
    else:
        p.setBrush(QtGui.QColor(color))
        p.setPen(QtGui.QPen(QtGui.QColor("#1b6e2d"), 1))
        p.drawEllipse(1, 1, 20, 20)
        p.setBrush(QtCore.Qt.white)
        p.setPen(QtCore.Qt.NoPen)
        p.drawPolygon(QtGui.QPolygon([QtCore.QPoint(8, 6), QtCore.QPoint(8, 16),
                                      QtCore.QPoint(16, 11)]))
    p.end()
    return QtGui.QIcon(pm)


# ==========================================================================
# Data store: everything found by the scanners, shared between threads
# ==========================================================================
class Acquisition:
    """One device acquisition (run), built up file by file."""

    def __init__(self, info: core.FileInfo):
        self.run = core.Run(info.device, info.ip, info.mac, info.start_ms)
        self.run.timebase_ns = info.timebase_ns
        self.run.ecal = info.ecal
        self.n_channels = info.n_channels
        self.chunks = {}            # seq -> raw (n, 2)
        self.last_real_s = 0.0
        self.version = 0            # bumps whenever new data arrive
        # Target ECAL (user editable, see Scan Results tab)
        self.target = {"low": 0.0, "high": 4000.0, "bins": 200, "offset": 0.0, "slope": 1.0}

    def add(self, info: core.FileInfo):
        self.chunks[info.seq] = info.raw
        self.run.files.append((info.seq, info.path))
        self.run.real_by_seq[info.seq] = info.elapsed_real_s
        self.last_real_s = max(self.last_real_s, info.elapsed_real_s)
        self.n_channels = max(self.n_channels, info.n_channels)
        self.version += 1

    @property
    def effective_ecal(self):
        """File ECAL with the Target Offset/Slope folded in."""
        o, s = self.target["offset"], self.target["slope"]
        c = list(self.run.ecal) + [0.0, 0.0, 0.0]
        return (o + s * c[0], s * c[1], s * c[2])

    @property
    def last_found_ms(self):
        return self.run.start_ms + int(self.last_real_s * 1000)


class Store(QtCore.QObject):
    changed = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.lock = threading.RLock()
        self.acqs = {}              # (device, ip, start_ms) -> Acquisition
        self.seen = set()           # file paths already processed
        self.skipped = {}           # path -> reason the file can't be used
        self._events = {}           # key -> (version, EventData)

    def add_file(self, info):
        with self.lock:
            k = (info.device, info.ip, info.start_ms)
            acq = self.acqs.get(k)
            if acq is None:
                acq = self.acqs[k] = Acquisition(info)
            acq.add(info)

    def warnings(self, only=None) -> list[str]:
        """Skipped files and signs of missing files (core.scan_warnings).
        only: acquisitions to check (default all)."""
        with self.lock:
            runs = [a.run for a in self.acqs.values()]
            sel = None if only is None else [a.run for a in only]
            skipped = list(self.skipped.items()) if only is None else []
            return core.scan_warnings(runs, skipped, sel)

    def sorted_acqs(self):
        with self.lock:
            return sorted(self.acqs.values(), key=lambda a: (a.run.start_ms, a.run.device))

    def events(self, acq) -> core.EventData:
        """Event data for an acquisition (cached until new files arrive)."""
        with self.lock:
            key = acq.run.key
            cached = self._events.get(key)
            if cached and cached[0] == acq.version:
                return cached[1]
            chunks = [acq.chunks[s] for s in sorted(acq.chunks)]
            version = acq.version
        ev = core.events_from_raw(acq.run, chunks)
        with self.lock:
            self._events[key] = (version, ev)
        return ev


class DirectoryScanner(QtCore.QThread):
    """Processes new .ava files in one directory every `interval` seconds."""
    status = QtCore.pyqtSignal(object)          # self

    def __init__(self, path, store: Store, interval_getter):
        super().__init__()
        self.path = path
        self.store = store
        self.interval = interval_getter
        self.state = "Stopped"
        self.last_scan = None
        self.next_scan = None       # seconds until next scan
        self.n_files = 0
        self.progress = ""
        self._stop = threading.Event()

    def acquisitions_found(self):
        with self.store.lock:
            return sum(1 for a in self.store.acqs.values()
                       if a.run.files and os.path.dirname(a.run.files[0][1]) == self.path)

    def stop(self):
        self._stop.set()

    def run(self):
        self._stop.clear()
        while not self._stop.is_set():
            self.state = "Scanning"
            self.next_scan = None
            self.status.emit(self)
            try:
                names = sorted(os.listdir(self.path))
            except OSError as e:
                self.state = f"Error: {e}"
                self.status.emit(self)
                return
            new = [os.path.join(self.path, n) for n in names if n.lower().endswith(".ava")]
            new = [p for p in new if p not in self.store.seen]
            t_emit = time.monotonic()
            for k, p in enumerate(new):
                if self._stop.is_set():
                    break
                try:
                    info = core.read_file(p)
                except core.SkippedFile as e:
                    self.store.skipped[p] = str(e)
                    if str(e) != core.SKIP_INCOMPLETE:  # incomplete: try again next scan
                        self.store.seen.add(p)
                    continue
                except Exception:       # unreadable right now: try again next scan
                    continue
                self.store.skipped.pop(p, None)
                self.store.seen.add(p)
                self.store.add_file(info)
                self.n_files += 1
                if time.monotonic() - t_emit > 0.5:
                    self.progress = f"{k + 1}/{len(new)} new files"
                    self.status.emit(self)
                    self.store.changed.emit()
                    t_emit = time.monotonic()
            self.progress = ""
            self.last_scan = datetime.now()
            self.store.changed.emit()
            if self._stop.is_set():
                break
            self.state = "Running"
            wait = max(1, int(self.interval()))
            for s in range(wait, 0, -1):
                self.next_scan = s
                self.status.emit(self)
                if self._stop.wait(1.0):
                    break
        self.state = "Stopped"
        self.next_scan = None
        self.status.emit(self)


# ==========================================================================
# Tab 1: Search Directories
# ==========================================================================
class SearchTab(QtWidgets.QWidget):
    COLS = ["", "Directory", "Status", "Last Scan", "Next Scan", "# Acquisitions Found"]

    def __init__(self, store: Store):
        super().__init__()
        self.store = store
        self.scanners = []

        bar = QtWidgets.QHBoxLayout()
        self.play_all = QtWidgets.QToolButton()
        self.play_all.setIcon(play_icon())
        self.play_all.setToolTip("Start all directories")
        self.play_all.clicked.connect(lambda: [self.start(s) for s in self.scanners])
        self.stop_all = QtWidgets.QToolButton()
        self.stop_all.setIcon(play_icon(stop=True))
        self.stop_all.setToolTip("Stop all directories")
        self.stop_all.clicked.connect(lambda: [s.stop() for s in self.scanners])
        add = QtWidgets.QPushButton("Add New Directory")
        add.clicked.connect(self.add_directory)
        self.freq = QtWidgets.QSpinBox()
        self.freq.setRange(1, 3600)
        self.freq.setValue(3)
        self.move_files = QtWidgets.QCheckBox("Move Files After Scan")
        self.move_files.setEnabled(False)
        self.move_files.setToolTip("Not available in this version: files are never moved.")
        for w in (self.play_all, self.stop_all, add):
            bar.addWidget(w)
        bar.addSpacing(12)
        bar.addWidget(QtWidgets.QLabel("Search Frequency:"))
        bar.addWidget(self.freq)
        bar.addWidget(QtWidgets.QLabel("(s)"))
        bar.addSpacing(12)
        bar.addWidget(self.move_files)
        bar.addStretch()

        self.table = QtWidgets.QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setAlternatingRowColors(True)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        hh.setSectionResizeMode(0, QtWidgets.QHeaderView.Fixed)
        self.table.setColumnWidth(0, 34)
        self.table.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.context_menu)

        self.warn = QtWidgets.QLabel()
        self.warn.setWordWrap(True)
        self.warn.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.warn.setStyleSheet("QLabel { border: 1px solid #d9534f; border-radius: 4px; "
                                "padding: 6px; color: #d9534f; }")
        self.warn.hide()
        store.changed.connect(self.update_warnings)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addLayout(bar)
        lay.addWidget(self.table)
        lay.addWidget(self.warn)

    def update_warnings(self):
        warnings = self.store.warnings()
        text = "".join(f"⚠ {html.escape(w)}<br>" for w in warnings)
        if text != self.warn.text():
            self.warn.setText("<b>Data may be incomplete – check that every file was "
                              "copied:</b><br>" + text if warnings else "")
            self.warn.setVisible(bool(warnings))

    def add_directory(self):
        dlg = QtWidgets.QFileDialog(self, "Select the data directory")
        dlg.setFileMode(QtWidgets.QFileDialog.Directory)
        dlg.setOption(QtWidgets.QFileDialog.ShowDirsOnly, True)
        if not dlg.exec_():
            return
        path = os.path.abspath(dlg.selectedFiles()[0])
        self.add_path(path)

    def add_path(self, path):
        if any(s.path == path for s in self.scanners):
            return
        sc = DirectoryScanner(path, self.store, self.freq.value)
        sc.status.connect(self.refresh_row)
        self.scanners.append(sc)
        r = self.table.rowCount()
        self.table.insertRow(r)
        btn = QtWidgets.QToolButton()
        btn.setAutoRaise(True)
        btn.setIcon(play_icon())
        btn.setToolTip("Start processing this directory")
        btn.clicked.connect(lambda _=False, sc=sc: self.toggle(sc))
        self.table.setCellWidget(r, 0, btn)
        self.refresh_row(sc)

    def start(self, sc):
        if not sc.isRunning():
            sc.start()

    def toggle(self, sc):
        if sc.isRunning():
            sc.stop()
        else:
            sc.start()

    def refresh_row(self, sc):
        r = self.scanners.index(sc)
        state = sc.state + (f" ({sc.progress})" if sc.progress else "")
        vals = [sc.path, state,
                sc.last_scan.strftime("%m/%d/%Y %I:%M:%S %p") if sc.last_scan else "",
                str(sc.next_scan) if sc.next_scan else "",
                str(sc.acquisitions_found())]
        for c, v in enumerate(vals, start=1):
            it = self.table.item(r, c)
            if it is None:
                it = QtWidgets.QTableWidgetItem()
                self.table.setItem(r, c, it)
            it.setText(v)
        btn = self.table.cellWidget(r, 0)
        running = sc.state != "Stopped"
        btn.setIcon(play_icon(stop=running))
        btn.setToolTip("Stop processing" if running else "Start processing this directory")

    def context_menu(self, pos):
        r = self.table.rowAt(pos.y())
        if r < 0:
            return
        menu = QtWidgets.QMenu(self)
        act = menu.addAction("Remove Directory")
        if menu.exec_(self.table.viewport().mapToGlobal(pos)) == act:
            sc = self.scanners.pop(r)
            sc.stop()
            sc.wait(3000)
            self.table.removeRow(r)

    def shutdown(self):
        for s in self.scanners:
            s.stop()
        for s in self.scanners:
            s.wait(3000)


# ==========================================================================
# Tab 2: Scan Results
# ==========================================================================
class SpectrumCanvas(FigureCanvasQTAgg):
    """Singles spectrum with a hover read-out of channel / energy / counts."""

    def __init__(self):
        self.fig = Figure(figsize=(8, 3), layout="constrained")
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot()
        self.ax.set_xlabel("Energy (keV)")
        self.ax.set_ylabel("Counts")
        self.ax.grid(True, color="k", lw=0.6)
        self.counts = None
        self.ecal = None
        self.vline = None
        self.label = None
        self.mpl_connect("motion_notify_event", self.on_move)

    def show_spectrum(self, name, counts, ecal):
        self.counts, self.ecal = counts, ecal
        ch = np.arange(len(counts) + 1) - 0.5
        e = ecal[0] + ecal[1] * ch + (ecal[2] * ch ** 2 if len(ecal) > 2 else 0)
        ax = self.ax
        ax.clear()
        ax.stairs(counts, e, fill=True, color="#1565ff")
        ax.set_xlim(0, max(e[-1], 1))      # always from 0 keV: an offset calibration shows as a gap
        centres = 0.5 * (e[:-1] + e[1:])
        if not counts.sum():
            warn = "⚠ No events recorded in this acquisition"
        elif not counts[centres < LOW_E_CHECK_KEV].sum():
            warn = f"⚠ No counts below {LOW_E_CHECK_KEV:g} keV: check the energy calibration"
        else:
            warn = None
        if warn:
            ax.text(0.01, 0.95, warn, transform=ax.transAxes, ha="left",
                    va="top", fontsize=10, color="#d9534f",
                    bbox=dict(facecolor="white", edgecolor="#d9534f", alpha=0.9))
        ax.set_ylim(0, max(counts.max() * 1.05, 1))
        ax.set_xlabel("Energy (keV)")
        ax.set_ylabel("Counts")
        ax.set_title(f"{name}: singles spectrum (no coincidence gating)  -  "
                     "hover to read channel energy", fontsize=10)
        ax.grid(True, color="k", lw=0.6)
        self.vline = ax.axvline(np.nan, color="r", lw=0.8)
        self.label = ax.text(0.99, 0.95, "", transform=ax.transAxes, ha="right", va="top",
                             fontsize=10, family="monospace",
                             bbox=dict(facecolor="white", edgecolor="0.5", alpha=0.9))
        self.draw_idle()

    def on_move(self, ev):
        if self.counts is None or ev.inaxes is not self.ax or ev.xdata is None:
            return
        c = self.ecal
        # invert calibration (linear part is enough to locate the channel)
        if len(c) > 2 and c[2]:
            roots = np.roots([c[2], c[1], c[0] - ev.xdata])
            roots = roots[np.isreal(roots)].real
            ch = roots[roots >= 0].min() if np.any(roots >= 0) else 0
        else:
            ch = (ev.xdata - c[0]) / c[1]
        ch = int(round(ch))
        if not 0 <= ch < len(self.counts):
            return
        e = c[0] + c[1] * ch + (c[2] * ch * ch if len(c) > 2 else 0)
        self.vline.set_xdata([e, e])
        self.label.set_text(f"Channel {ch}\nEnergy  {e:9.2f} keV\nCounts  {int(self.counts[ch]):9d}")
        self.draw_idle()


class ScanResultsTab(QtWidgets.QWidget):
    target_changed = QtCore.pyqtSignal(object)      # [Acquisition] whose Target ECAL changed
    COLS = ["Device Name", "Start Run Time", "Last Found Time",
            "Offset", "Slope", "Quadratic",
            "Energy Low (keV)", "Energy High (keV)", "# BINs", "Offset", "Slope"]
    TARGET = {6: "low", 7: "high", 8: "bins", 9: "offset", 10: "slope"}

    def __init__(self, store: Store):
        super().__init__()
        self.store = store
        self.rows = []               # Acquisition per row
        self._updating = False

        bar = QtWidgets.QHBoxLayout()
        lbl = QtWidgets.QLabel("<i><b>Time Calibration:</b></i>")
        lbl.setStyleSheet("font-size: 15px")
        self.t_lo = QtWidgets.QLineEdit("-15000")
        self.t_hi = QtWidgets.QLineEdit("15000")
        self.t_bins = QtWidgets.QLineEdit("150")
        v = QtGui.QDoubleValidator()
        for w in (self.t_lo, self.t_hi):
            w.setValidator(v)
            w.setMaximumWidth(90)
        self.t_bins.setValidator(QtGui.QIntValidator(1, 1_000_000))
        self.t_bins.setMaximumWidth(90)
        bar.addWidget(lbl)
        bar.addWidget(QtWidgets.QLabel("Time Range (ns)"))
        bar.addWidget(self.t_lo)
        bar.addWidget(QtWidgets.QLabel("-"))
        bar.addWidget(self.t_hi)
        bar.addSpacing(20)
        bar.addWidget(QtWidgets.QLabel("Time BINs"))
        bar.addWidget(self.t_bins)
        bar.addSpacing(30)
        self.gate_on = QtWidgets.QCheckBox("Coincidence Gate (ns)")
        self.gate_on.setToolTip(
            "When ticked, Energy/Channel plots on Analysis (and Projections) only use pairs "
            "with Gate low <= dt < Gate high. Delta Time plots still show the full Time "
            "Range, with the gate drawn as dashed lines.")
        self.g_lo = QtWidgets.QLineEdit("-800")
        self.g_hi = QtWidgets.QLineEdit("0")
        for w in (self.g_lo, self.g_hi):
            w.setValidator(v)
            w.setMaximumWidth(90)
            w.setEnabled(False)
        self.gate_on.toggled.connect(self.g_lo.setEnabled)
        self.gate_on.toggled.connect(self.g_hi.setEnabled)
        bar.addWidget(self.gate_on)
        bar.addWidget(self.g_lo)
        bar.addWidget(QtWidgets.QLabel("-"))
        bar.addWidget(self.g_hi)
        bar.addStretch()

        # Two-level header: group labels above the column labels
        groups = QtWidgets.QHBoxLayout()
        self.table = QtWidgets.QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.verticalHeader().hide()
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QtWidgets.QHeaderView.Interactive)
        hh.setStretchLastSection(True)
        hh.sectionResized.connect(lambda *_: self.align_groups())
        self.table.cellClicked.connect(self.row_clicked)
        self.table.itemChanged.connect(self.target_edited)
        self.group_bar = QtWidgets.QWidget()
        self.group_bar.setLayout(groups)
        groups.setContentsMargins(0, 0, 0, 0)
        groups.setSpacing(0)
        self.group_labels = []
        self.group_spans = [(0, 1), (1, 2), (3, 3), (6, 5)]
        for text in ["", "Acquisition Times", "Source ECAL", "Target ECAL"]:
            l = QtWidgets.QLabel(f"<b>{text}</b>")
            l.setAlignment(QtCore.Qt.AlignCenter)
            l.setFrameShape(QtWidgets.QFrame.StyledPanel)
            groups.addWidget(l)
            self.group_labels.append(l)
        groups.addStretch()

        self.spectrum = SpectrumCanvas()
        self.hint = QtWidgets.QLabel("Click a device name to display its energy spectrum.")
        self.shown = None            # (Acquisition, counts) of the displayed spectrum
        self.export_btn = QtWidgets.QPushButton("Export Spectrum")
        self.export_btn.setToolTip("Save the displayed spectrum: channel, energy, counts")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self.export_spectrum)
        self.recal_btn = QtWidgets.QPushButton("Recalibrate…")
        self.recal_btn.setToolTip("Correct this device's energy calibration (sets the Target "
                                  "ECAL Offset/Slope) from known peaks or typed coefficients")
        self.recal_btn.setEnabled(False)
        self.recal_btn.clicked.connect(self.recalibrate)

        top = QtWidgets.QWidget()
        tl = QtWidgets.QVBoxLayout(top)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.addLayout(bar)
        tl.addWidget(self.group_bar)
        tl.addWidget(self.table)
        bottom = QtWidgets.QWidget()
        bl = QtWidgets.QVBoxLayout(bottom)
        bl.setContentsMargins(0, 0, 0, 0)
        hint_row = QtWidgets.QHBoxLayout()
        hint_row.addWidget(self.hint, 1)
        hint_row.addWidget(self.recal_btn)
        hint_row.addWidget(self.export_btn)
        bl.addLayout(hint_row)
        bl.addWidget(NavigationToolbar2QT(self.spectrum, self))
        bl.addWidget(self.spectrum)
        split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        split.addWidget(top)
        split.addWidget(bottom)
        split.setSizes([350, 450])
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(split)

        store.changed.connect(self.refresh)

    def align_groups(self):
        """Keep the group labels above their columns."""
        hh = self.table.horizontalHeader()
        for lbl, (first, n) in zip(self.group_labels, self.group_spans):
            w = sum(hh.sectionSize(c) for c in range(first, first + n))
            if first + n == len(self.COLS):        # last group: absorb stretch
                w = self.table.viewport().width() - hh.sectionPosition(first)
            lbl.setFixedWidth(max(w, 0))
        self.group_bar.setContentsMargins(self.table.frameWidth(), 0, 0, 0)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        QtCore.QTimer.singleShot(0, self.align_groups)

    def time_settings(self):
        try:
            lo, hi, n = float(self.t_lo.text()), float(self.t_hi.text()), int(self.t_bins.text())
        except ValueError:
            raise ValueError("Time Range / Time BINs must be numbers")
        if hi <= lo or n < 1:
            raise ValueError("Time Range high must be greater than low, BINs >= 1")
        return lo, hi, n

    def gate_settings(self):
        """(low, high) of the coincidence gate in ns, or None if not ticked."""
        if not self.gate_on.isChecked():
            return None
        try:
            lo, hi = float(self.g_lo.text()), float(self.g_hi.text())
        except ValueError:
            raise ValueError("Coincidence Gate values must be numbers")
        if hi <= lo:
            raise ValueError("Coincidence Gate high must be greater than low")
        return lo, hi

    def refresh(self):
        acqs = self.store.sorted_acqs()
        self._updating = True
        for acq in acqs:
            if acq not in self.rows:
                self.rows.append(acq)
                r = self.table.rowCount()
                self.table.insertRow(r)
                for c in range(len(self.COLS)):
                    it = QtWidgets.QTableWidgetItem()
                    if c not in self.TARGET:
                        it.setFlags(it.flags() & ~QtCore.Qt.ItemIsEditable)
                    self.table.setItem(r, c, it)
        for r, acq in enumerate(self.rows):
            ec = list(acq.run.ecal) + [0, 0, 0]
            t = acq.target
            vals = [acq.run.device, fmt_time(acq.run.start_ms), fmt_time(acq.last_found_ms),
                    repr(ec[0]), repr(ec[1]), repr(ec[2]),
                    f"{t['low']:g}", f"{t['high']:g}", str(t["bins"]),
                    f"{t['offset']:g}", f"{t['slope']:g}"]
            for c, v in enumerate(vals):
                it = self.table.item(r, c)
                if c in self.TARGET and self.table.currentItem() is it and \
                        self.table.state() == QtWidgets.QAbstractItemView.EditingState:
                    continue
                if it.text() != v:
                    it.setText(v)
        if self.rows and not getattr(self, "_sized", False):
            self.table.resizeColumnsToContents()
            self._sized = True
        self._updating = False
        self.align_groups()

    def target_edited(self, item):
        if self._updating or item.column() not in self.TARGET:
            return
        acq = self.rows[item.row()]
        key = self.TARGET[item.column()]
        try:
            val = int(item.text()) if key == "bins" else float(item.text())
            if key == "bins" and val < 1:
                raise ValueError
            changed = acq.target[key] != val
            acq.target[key] = val
        except ValueError:
            QtWidgets.QToolTip.showText(QtGui.QCursor.pos(), "Please enter a number")
            changed = False
        self.refresh()
        if changed:
            if self.shown is not None and self.shown[0] is acq:     # redraw the spectrum
                self.spectrum.show_spectrum(acq.run.device, self.shown[1], acq.effective_ecal)
            self.target_changed.emit([acq])

    def row_clicked(self, r, c):
        if c != 0:
            return
        acq = self.rows[r]
        ev = self.store.events(acq)
        n = max(acq.n_channels, int(ev.channel.max()) + 1 if len(ev.channel) else 1)
        counts = np.bincount(ev.channel, minlength=n)[:n]
        self.hint.setText(f"{acq.run.device}  (start {fmt_time(acq.run.start_ms)}): "
                          f"{len(ev.t_ns):,} events, {n} channels")
        self.spectrum.show_spectrum(acq.run.device, counts, acq.effective_ecal)
        self.shown = (acq, counts)
        self.export_btn.setEnabled(True)
        self.recal_btn.setEnabled(True)

    def recalibrate(self):
        if self.shown is None:
            return
        acq, counts = self.shown
        same = [a for a in self.store.sorted_acqs()
                if a.run.device == acq.run.device and a.run.ecal == acq.run.ecal]
        dlg = RecalibrateDialog(acq, counts, len(same), self)
        if not dlg.exec_():
            return
        changed = same if dlg.apply_all.isChecked() else [acq]
        for a in changed:
            a.target.update(dlg.target_for(a))
        self.refresh()
        self.target_changed.emit(changed)
        r = self.rows.index(acq)
        self.row_clicked(r, 0)
        self.hint.setText(self.hint.text() + "  -  recalibrated: E = "
                          + " + ".join(f"{c:.5g}" + ("·ch" if k == 1 else "·ch²" if k == 2 else "")
                                       for k, c in enumerate(acq.effective_ecal) if c or k < 2))

    def export_spectrum(self):
        if self.shown is None:
            return
        acq, counts = self.shown
        stamp = datetime.fromtimestamp(acq.run.start_ms / 1000).strftime("%Y%m%d_%H%M%S")
        name = f"{acq.run.device}_{stamp}_spectrum.txt"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Spectrum", name, "Text (*.txt *.csv)")
        if not path:
            return
        core.export_channel_spectrum(path, acq.run, counts)
        self.hint.setText(f"Exported {acq.run.device} spectrum to {path}")


class RecalibrateDialog(QtWidgets.QDialog):
    """Fit a new linear calibration E = a + b*ch for one acquisition, from
    peaks (channel -> known energy) or typed coefficients, and express it as
    Target ECAL Offset/Slope on top of the file's calibration."""
    LINES = ("Common lines (keV):  Na-22: 511, 1274.5   Cs-137: 661.7   Co-60: 1173.2, 1332.5\n"
             "Eu-152: 121.8, 244.7, 344.3, 778.9, 964.1, 1085.8, 1112.1, 1408.0")

    def __init__(self, acq, counts, n_same, parent=None):
        super().__init__(parent)
        self.acq, self.counts = acq, counts
        self.is_reset = False
        self.setWindowTitle(f"Recalibrate {acq.run.device}")
        self.resize(560, 560)
        c = acq.run.ecal
        ea = acq.effective_ecal

        info = QtWidgets.QLabel(
            f"File calibration: E = {c[0]:.5g} + {c[1]:.5g}·ch"
            + (f" + {c[2]:.3g}·ch²" if len(c) > 2 and c[2] else "")
            + f"<br>Currently used: E = {ea[0]:.5g} + {ea[1]:.5g}·ch"
            + "<br><br>Enter the channel of each peak you can identify and its known energy, "
              "then press <b>Fit</b>. <b>Find Peaks</b> fills in the channels of the strongest "
              "peaks (look at the spectrum: not every one is a real line)."
            + ("<br><font color='#d9534f'>The file calibration has a quadratic term; only "
               "its linear part is corrected (the quadratic term is scaled with it).</font>"
               if len(c) > 2 and c[2] else ""))
        info.setWordWrap(True)
        lines = QtWidgets.QLabel(self.LINES)
        lines.setStyleSheet("color: gray")

        self.table = QtWidgets.QTableWidget(8, 3)
        self.table.setHorizontalHeaderLabels(["Channel", "Energy (keV)", "Residual (keV)"])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        for r in range(8):
            res = QtWidgets.QTableWidgetItem()
            res.setFlags(res.flags() & ~QtCore.Qt.ItemIsEditable)
            self.table.setItem(r, 2, res)
        find = QtWidgets.QPushButton("Find Peaks")
        find.clicked.connect(self.find_peaks)
        add = QtWidgets.QPushButton("Add Row")
        add.clicked.connect(self.add_row)
        fit = QtWidgets.QPushButton("Fit")
        fit.clicked.connect(self.fit)
        btns = QtWidgets.QHBoxLayout()
        for b in (find, add, fit):
            btns.addWidget(b)
        btns.addStretch()

        self.a = QtWidgets.QLineEdit(f"{ea[0]:.6g}")
        self.b = QtWidgets.QLineEdit(f"{ea[1]:.6g}")
        v = QtGui.QDoubleValidator()
        for w in (self.a, self.b):
            w.setValidator(v)
            w.setMaximumWidth(110)
        cal = QtWidgets.QHBoxLayout()
        cal.addWidget(QtWidgets.QLabel("<b>New calibration:</b>  E ="))
        cal.addWidget(self.a)
        cal.addWidget(QtWidgets.QLabel("+"))
        cal.addWidget(self.b)
        cal.addWidget(QtWidgets.QLabel("× channel  (keV)"))
        cal.addStretch()

        self.msg = QtWidgets.QLabel()
        self.msg.setWordWrap(True)
        self.apply_all = QtWidgets.QCheckBox(
            f"Apply to all {n_same} {acq.run.device} acquisitions with the same file calibration")
        self.apply_all.setChecked(True)
        self.apply_all.setVisible(n_same > 1)

        box = QtWidgets.QDialogButtonBox()
        ok = box.addButton("Apply", QtWidgets.QDialogButtonBox.AcceptRole)
        reset = box.addButton("Reset to File Calibration", QtWidgets.QDialogButtonBox.ResetRole)
        box.addButton(QtWidgets.QDialogButtonBox.Cancel)
        ok.clicked.connect(self.try_accept)
        reset.clicked.connect(self.reset)
        box.rejected.connect(self.reject)

        lay = QtWidgets.QVBoxLayout(self)
        for w in (info, lines):
            lay.addWidget(w)
        lay.addLayout(btns)
        lay.addWidget(self.table)
        lay.addLayout(cal)
        lay.addWidget(self.msg)
        lay.addWidget(self.apply_all)
        lay.addWidget(box)

    def say(self, text, error=False):
        self.msg.setText(f"<font color='#d9534f'>{text}</font>" if error else text)

    def add_row(self):
        r = self.table.rowCount()
        self.table.insertRow(r)
        it = QtWidgets.QTableWidgetItem()
        it.setFlags(it.flags() & ~QtCore.Qt.ItemIsEditable)
        self.table.setItem(r, 2, it)

    def find_peaks(self):
        peaks = core.find_peaks(self.counts)
        if not peaks:
            self.say("No clear peaks found.", True)
            return
        while self.table.rowCount() < len(peaks):
            self.add_row()
        ec = self.acq.effective_ecal
        for r in range(self.table.rowCount()):
            for col in (0, 1, 2):
                if self.table.item(r, col) is None:
                    self.table.setItem(r, col, QtWidgets.QTableWidgetItem())
                self.table.item(r, col).setText("")
            if r < len(peaks):
                self.table.item(r, 0).setText(f"{peaks[r]:.1f}")
                self.table.item(r, 2).setText(f"now {ec[0] + ec[1] * peaks[r]:.0f} keV")
        self.say(f"{len(peaks)} peaks found. Type the known energy next to the peaks you "
                 "recognise (leave the others blank), then press Fit.")

    def rows(self):
        """[(row, channel, energy)] for rows with both values filled in."""
        out = []
        for r in range(self.table.rowCount()):
            txt = [(self.table.item(r, c).text().strip() if self.table.item(r, c) else "")
                   for c in (0, 1)]
            if not all(txt):
                continue
            try:
                out.append((r, float(txt[0]), float(txt[1])))
            except ValueError:
                raise ValueError(f"Row {r + 1}: channel and energy must be numbers")
        return out

    def fit(self):
        try:
            rows = self.rows()
            a, b, res = core.fit_linear([x[1] for x in rows], [x[2] for x in rows])
        except ValueError as e:
            self.say(str(e), True)
            return
        for r in range(self.table.rowCount()):
            self.table.item(r, 2).setText("")
        for (r, _, _), d in zip(rows, res):
            self.table.item(r, 2).setText(f"{d:+.2f}")
        self.a.setText(f"{a:.6g}")
        self.b.setText(f"{b:.6g}")
        worst = np.abs(res).max()
        self.say(f"Fit to {len(rows)} peaks: E = {a:.4g} + {b:.5g}·ch"
                 + (f", largest residual {worst:.1f} keV." if len(rows) > 2 else
                    ". Use 3 or more peaks to check the fit."),
                 error=len(rows) > 2 and worst > 10)

    def target_for(self, acq):
        """Target Offset/Slope giving E = a + b*ch with acq's file calibration."""
        if self.is_reset:
            return {"offset": 0.0, "slope": 1.0}
        a, b = float(self.a.text()), float(self.b.text())
        c = acq.run.ecal
        slope = b / c[1]
        return {"offset": a - slope * c[0], "slope": slope}

    def try_accept(self):
        try:
            a, b = float(self.a.text()), float(self.b.text())
        except ValueError:
            self.say("Enter numbers for the new calibration.", True)
            return
        if b <= 0:
            self.say("The slope (keV per channel) must be positive.", True)
            return
        if not self.acq.run.ecal[1]:
            self.say("The file calibration has zero slope and can't be corrected.", True)
            return
        self.accept()

    def reset(self):
        self.is_reset = True
        self.accept()


# ==========================================================================
# Tab 3: Analysis
# ==========================================================================
class AnalysisTab(QtWidgets.QWidget):
    matrix_ready = QtCore.pyqtSignal(object, bool)     # core.Result, log scale

    def __init__(self, store: Store, scan_tab: ScanResultsTab):
        super().__init__()
        self.store = store
        self.scan_tab = scan_tab
        self.last = None             # (kind, data) of the last graph for export
        self.mode = None             # "1d" or "2d"
        self.graphed = None          # (acq X, acq Y, X units, Y units) of the last graph
        scan_tab.target_changed.connect(self.target_changed)

        self.fig = Figure(figsize=(10, 6), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.ax = None

        self.dev = [QtWidgets.QComboBox(), QtWidgets.QComboBox()]
        self.start = [QtWidgets.QComboBox(), QtWidgets.QComboBox()]
        self.xunits = QtWidgets.QComboBox()
        self.xunits.addItems([UNIT_ENERGY, UNIT_CHANNEL, UNIT_DT])
        self.yunits = QtWidgets.QComboBox()
        self.yunits.addItems([UNIT_ENERGY, UNIT_CHANNEL, UNIT_COUNTS])
        for i in range(2):
            self.dev[i].setMinimumWidth(170)
            self.start[i].setMinimumWidth(200)
            self.dev[i].currentTextChanged.connect(lambda _t, i=i: self.fill_starts(i))

        big = "font-size: 16px"
        grid = QtWidgets.QGridLayout()
        l = QtWidgets.QLabel("Device:"); l.setStyleSheet(big)
        grid.addWidget(l, 0, 0, 2, 1)
        grid.addWidget(self.dev[0], 0, 1)
        grid.addWidget(self.dev[1], 1, 1)
        l = QtWidgets.QLabel("Acq. Start:"); l.setStyleSheet(big)
        grid.addWidget(l, 0, 2, 2, 1)
        grid.addWidget(self.start[0], 0, 3)
        grid.addWidget(self.start[1], 1, 3)
        l = QtWidgets.QLabel("X Units:"); l.setStyleSheet(big)
        grid.addWidget(l, 0, 4)
        grid.addWidget(self.xunits, 0, 5)
        l = QtWidgets.QLabel("Y Units:"); l.setStyleSheet(big)
        grid.addWidget(l, 1, 4)
        grid.addWidget(self.yunits, 1, 5)
        # Optional bin widths (blank = auto: Target ECAL Bins / Time BINs)
        self.width = [QtWidgets.QLineEdit(), QtWidgets.QLineEdit()]
        self.width_unit = [QtWidgets.QLabel(), QtWidgets.QLabel()]
        for k, (w, ul) in enumerate(zip(self.width, self.width_unit)):
            w.setPlaceholderText("auto")
            w.setFixedWidth(70)
            w.setToolTip("Histogram bin width for this axis, in the axis units.\n"
                         "Leave blank to use the Bins from Scan Results.")
            l = QtWidgets.QLabel(f"{'XY'[k]} Bin Width:"); l.setStyleSheet(big)
            grid.addWidget(l, k, 6)
            row = QtWidgets.QHBoxLayout()
            row.addWidget(w)
            row.addWidget(ul)
            grid.addLayout(row, k, 7)
        self.xunits.currentTextChanged.connect(self.units_changed)
        self.yunits.currentTextChanged.connect(self.units_changed)
        self.units_changed()
        sep = QtWidgets.QFrame(); sep.setFrameShape(QtWidgets.QFrame.VLine)
        grid.addWidget(sep, 0, 8, 3, 1)
        b_graph = QtWidgets.QPushButton("Graph")
        b_clear = QtWidgets.QPushButton("Clear Graph")
        b_export = QtWidgets.QPushButton("Export Data")
        b_graph.clicked.connect(self.graph)
        b_clear.clicked.connect(self.clear)
        b_export.clicked.connect(self.export)
        grid.addWidget(b_graph, 0, 9)
        grid.addWidget(b_clear, 1, 9)
        grid.addWidget(b_export, 2, 9)
        self.log = QtWidgets.QCheckBox("Log scale")
        self.log.setChecked(True)
        self._warnings = []
        grid.addWidget(self.log, 2, 5)
        self.msg = QtWidgets.QLabel()
        grid.addWidget(self.msg, 2, 0, 1, 5)
        controls = QtWidgets.QHBoxLayout()
        controls.addStretch()
        controls.addLayout(grid)
        controls.addStretch()

        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(NavigationToolbar2QT(self.canvas, self))
        lay.addWidget(self.canvas, 1)
        lay.addLayout(controls)

        store.changed.connect(self.refresh_devices)
        self.clear()

    # ---------------------------------------------------------------- combos
    def refresh_devices(self):
        devs = sorted({a.run.device for a in self.store.sorted_acqs()})
        for i, box in enumerate(self.dev):
            if [box.itemText(k) for k in range(box.count())] == devs:
                self.fill_starts(i, keep=True)
                continue
            cur = box.currentText()
            box.blockSignals(True)
            box.clear()
            box.addItems(devs)
            if cur in devs:
                box.setCurrentText(cur)
            else:
                box.setCurrentIndex(-1)     # make the user choose (manual step 7)
            box.blockSignals(False)
            self.fill_starts(i, keep=True)

    def fill_starts(self, i, keep=False):
        box = self.start[i]
        cur = box.currentData() if keep else None
        box.blockSignals(True)
        box.clear()
        for a in self.store.sorted_acqs():
            if a.run.device == self.dev[i].currentText():
                box.addItem(fmt_time(a.run.start_ms), a)
        idx = box.findData(cur) if cur is not None else -1
        box.setCurrentIndex(idx if idx >= 0 else (0 if box.count() == 1 else -1))
        box.blockSignals(False)

    def target_changed(self, acqs):
        """A Target ECAL (calibration, range or bins) changed on Scan Results:
        redraw the current plot if it uses one of those acquisitions."""
        if self.graphed is None or self.mode is None:
            return
        a0, a1, xu, yu = self.graphed
        used = ([] if xu == UNIT_DT else [a0]) + ([] if yu == UNIT_COUNTS else [a1])
        if not any(a in acqs for a in used):
            return
        same = (self.start[0].currentData() is a0 and self.start[1].currentData() is a1
                and self.xunits.currentText() == xu and self.yunits.currentText() == yu)
        if self.mode == "2d" and same:
            self.graph()
            self.msg.setText(self.msg.text() + "<br><i>Redrawn with the new Target ECAL.</i>")
        else:   # 1D plots overlay, so redrawing would duplicate a curve
            self.say("Target ECAL changed since this plot was drawn: press Clear Graph, "
                     "then Graph, to use it.", True)

    # ---------------------------------------------------------------- graph
    def say(self, text, error=False):
        if error:
            text = f"<font color='red'>{text}</font>"
        elif self._warnings:
            text += "".join(f"<br><font color='#d9534f'>⚠ {html.escape(w)}</font>"
                            for w in self._warnings)
        self.msg.setText(text)

    def clear(self):
        self.fig.clear()
        self.ax = self.fig.add_subplot()
        self.ax.grid(True, color="k", lw=0.6)
        self.mode = None
        self.last = None
        self.canvas.draw_idle()

    UNIT_SUFFIX = {UNIT_ENERGY: "keV", UNIT_CHANNEL: "channels", UNIT_DT: "ns", UNIT_COUNTS: ""}
    MAX_BINS = 20000

    def units_changed(self, *_):
        for k, units in enumerate((self.xunits.currentText(), self.yunits.currentText())):
            self.width_unit[k].setText(self.UNIT_SUFFIX[units])
            self.width[k].setEnabled(units != UNIT_COUNTS)

    def bin_width(self, k):
        """Bin width typed for axis k (0 = X, 1 = Y), or None for auto."""
        txt = self.width[k].text().strip()
        if not txt or not self.width[k].isEnabled():
            return None
        try:
            w = float(txt)
        except ValueError:
            raise ValueError(f"{'XY'[k]} Bin Width must be a number (or blank for auto)")
        if not w > 0:
            raise ValueError(f"{'XY'[k]} Bin Width must be greater than zero")
        return w

    @classmethod
    def _edges(cls, lo, hi, bins, width):
        """Bin edges from lo to hi: `bins` equal bins, or bins of `width`."""
        if width is None:
            return np.linspace(lo, hi, bins + 1)
        return core.edges_for_width(lo, hi, width, cls.MAX_BINS)

    def _axis(self, acq, ev, units, width=None):
        """Values and bin edges for a device axis in the requested units."""
        t = acq.target
        if units == UNIT_CHANNEL:
            n = max(acq.n_channels, int(ev.channel.max()) + 1 if len(ev.channel) else 1)
            edges = (np.linspace(0, n, min(t["bins"], n) + 1) if width is None
                     else self._edges(0, n, 0, max(1, round(width))))
            return ev.channel.astype(float), edges
        vals = t["offset"] + t["slope"] * ev.energy
        return vals, self._edges(t["low"], t["high"], t["bins"], width)

    @staticmethod
    def _comb_note(acq, units, edges):
        """Red note if energy bins aren't a whole number of ADC channels."""
        if units != UNIT_ENERGY or len(edges) < 2:
            return ""
        note = core.comb_note(acq.run.device, edges[1] - edges[0],
                              acq.target["slope"] * acq.run.ecal[1])
        return f"<br><font color='#d9534f'>{note}</font>" if note else ""

    def graph(self):
        acq = [self.start[i].currentData() for i in range(2)]
        if any(a is None for a in acq):
            self.say("Select a device and an Acq. Start for both devices.", True)
            return
        if acq[0] is acq[1]:
            self.say("Choose two different devices / acquisitions.", True)
            return
        self._warnings = self.store.warnings(only=acq)
        xu, yu = self.xunits.currentText(), self.yunits.currentText()
        if xu == UNIT_DT and yu != UNIT_COUNTS:
            self.say("With X = Delta Time, choose Y Units = Counts.", True)
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            t_lo, t_hi, t_bins = self.scan_tab.time_settings()
            gate = self.scan_tab.gate_settings()
            wx, wy = self.bin_width(0), self.bin_width(1)
            ea, eb = self.store.events(acq[0]), self.store.events(acq[1])
            t_edges = self._edges(t_lo, t_hi, t_bins, wx if xu == UNIT_DT else None)
            xa = None if xu == UNIT_DT else self._axis(acq[0], ea, xu, wx)
            ya = None if yu == UNIT_COUNTS else self._axis(acq[1], eb, yu, wy)
            w_lo, w_hi = (min(t_lo, gate[0]), max(t_hi, gate[1])) if gate else (t_lo, t_hi)
            i_all, j_all, dt_all = core.coincidence_pairs(ea.t_ns, eb.t_ns, w_lo, w_hi)
        except ValueError as e:
            self.say(str(e), True)
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        # Delta-time plots: the whole Time Range.  Energy/channel plots: the gate.
        in_range = (dt_all >= t_lo) & (dt_all < t_hi)
        g_lo, g_hi = gate if gate else (t_lo, t_hi)
        in_gate = (dt_all >= g_lo) & (dt_all < g_hi)
        dt = dt_all[in_range]
        i, j = i_all[in_gate], j_all[in_gate]
        gate_txt = f" (gate {g_lo:g} to {g_hi:g} ns)" if gate else ""
        da, db = acq[0].run.device, acq[1].run.device
        log = self.log.isChecked()
        notes = ((self._comb_note(acq[0], xu, xa[1]) if xa else "")
                 + (self._comb_note(acq[1], yu, ya[1]) if ya else ""))
        tb = acq[0].run.timebase_ns
        if xu == UNIT_DT and wx and abs(wx / tb - round(wx / tb)) > 1e-6:
            notes += (f"<br><font color='#d9534f'>Δt bins of {wx:g} ns aren't a multiple of the "
                      f"{tb:g} ns clock: expect stripes.</font>")

        self.graphed = (acq[0], acq[1], xu, yu)
        if yu == UNIT_COUNTS:
            if xu == UNIT_DT:
                edges = t_edges
                counts, _ = np.histogram(dt, edges)
                label = f"{da} - {db}"
                xlabel = "Δ Time (ns)"
                header = "deltaTime, counts"
                plot_gate = gate
            else:
                vals, edges = xa
                counts, _ = np.histogram(vals[i], edges)
                label = f"{da} (coincident with {db})"
                xlabel = f"{xu.split(' ')[0]} {da}" + (" (keV)" if xu == UNIT_ENERGY else "")
                header = f"{xu}, counts"
                label += gate_txt
                plot_gate = None
            self.plot_1d(edges, counts, label, xlabel, log, plot_gate)
            self.last = ("1d", header, edges[:-1], counts)
            self.say(f"{int(counts.sum()):,} coincident events plotted"
                     f"{'' if xu == UNIT_DT else gate_txt}." + notes)
            return

        # 2D: X = device 1 (or delta time), Y = device 2
        yv, yedges = ya
        if xu == UNIT_DT:
            xv, xedges = dt, t_edges
            j = j_all[in_range]
            xlabel = "Δ Time (ns)"
            xcol = "deltaTime"
        else:
            xv, xedges = xa
            xv = xv[i]
            xlabel = f"{'Energy' if xu == UNIT_ENERGY else 'Channel'} {da}" + \
                     (" (keV)" if xu == UNIT_ENERGY else "")
            xcol = da
        h, _, _ = np.histogram2d(xv, yv[j], bins=[xedges, yedges])
        ylabel = f"{'Energy' if yu == UNIT_ENERGY else 'Channel'} {db}" + \
                 (" (keV)" if yu == UNIT_ENERGY else "")
        self.plot_2d(xedges, yedges, h, xlabel, ylabel, log)
        self.last = ("2d", f"{xcol}, {db}, counts", xedges[:-1], yedges[:-1], h.astype(np.int64))
        self.say(f"{int(h.sum()):,} coincident events plotted"
                 f"{'' if xu == UNIT_DT else gate_txt}." + notes)

        if xu != UNIT_DT:          # energy/channel matrix: hand it to the Projections tab
            t_edges = np.linspace(t_lo, t_hi, t_bins + 1)       # Scan Results Time BINs
            res = core.Result(
                a=self._target_events(acq[0], ea), b=self._target_events(acq[1], eb),
                units_a=self._core_units(xu), units_b=self._core_units(yu),
                t_edges=t_edges, t_counts=np.histogram(dt, t_edges)[0],
                ea_edges=xedges, eb_edges=yedges, ee_counts=h.astype(np.int64),
                gate=(g_lo, g_hi), n_pairs=len(i), pair_a=xv, pair_b=yv[j])
            self.matrix_ready.emit(res, log)

    @staticmethod
    def _core_units(u):
        return "channel" if u == UNIT_CHANNEL else "energy"

    @staticmethod
    def _target_events(acq, ev):
        """EventData whose calibration includes the Target ECAL offset/slope,
        so projection bins land on this device's channel boundaries."""
        t = acq.target
        c = list(acq.run.ecal) + [0.0, 0.0, 0.0]
        run = dataclasses.replace(acq.run, ecal=(t["offset"] + t["slope"] * c[0],
                                                 t["slope"] * c[1], t["slope"] * c[2]))
        return core.EventData(run, ev.t_ns, ev.channel)

    def plot_1d(self, edges, counts, label, xlabel, log, gate=None):
        if self.mode != "1d":
            self.fig.clear()
            self.ax = self.fig.add_subplot()
        ax = self.ax
        centres = 0.5 * (edges[:-1] + edges[1:])
        line, = ax.plot(centres, counts, "-o", ms=3.5, lw=1.8, label=label)
        if gate:
            for g in gate:
                ax.axvline(g, color=line.get_color(), ls="--", lw=1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Counts")
        ax.set_yscale("log" if log else "linear")
        if log:
            ax.set_ylim(bottom=0.8)
        ax.grid(True, alpha=0.4)
        ax.legend(loc="upper right")
        self.mode = "1d"
        self.canvas.draw_idle()

    def plot_2d(self, xedges, yedges, h, xlabel, ylabel, log):
        self.fig.clear()
        self.ax = ax = self.fig.add_subplot()
        hm = np.ma.masked_equal(h.T, 0)
        vmax = max(h.max(), 2)
        norm = LogNorm(vmin=1, vmax=vmax) if log else None
        m = ax.pcolormesh(xedges, yedges, hm, cmap="turbo", norm=norm, shading="flat")
        self.fig.colorbar(m, ax=ax, label="Counts")
        tlist_plot.add_bin_readout(ax, xedges, yedges, h)   # hover shows the bin's counts
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, color="k", lw=0.6, alpha=0.5)
        self.mode = "2d"
        self.canvas.draw_idle()

    def export(self):
        if not self.last:
            self.say("Nothing to export - press Graph first.", True)
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Data", "coincidence_data.txt", "Text (*.txt *.csv)")
        if not path:
            return
        with open(path, "w") as fh:
            fh.write("Coincident events\n")
            if self.last[0] == "1d":
                _, header, x, counts = self.last
                fh.write(header + "\n")
                for a, c in zip(x, counts):
                    fh.write(f"{a:g}, {c}\n")
            else:
                _, header, x, y, h = self.last
                fh.write(header + "\n")
                for a, b in zip(*np.nonzero(h)):
                    fh.write(f"{x[a]:g}, {y[b]:g}, {h[a, b]}\n")
        self.say(f"Exported to {path}")


# ==========================================================================
# Tab 4: Projections (extra section, after the steps in the manual)
# ==========================================================================
class ProjectionsSection(QtWidgets.QStackedWidget):
    """Shows a hint until Analysis has produced an energy/channel matrix."""

    def __init__(self, analysis: AnalysisTab, status):
        super().__init__()
        hint = QtWidgets.QLabel(
            "<h3>Gamma-gamma projections</h3>"
            "<p>On the <b>Analysis</b> tab, choose two devices and their Acq. Start, set "
            "<b>X Units</b> and <b>Y Units</b> to <i>Energy (keV)</i> (or <i>Channel</i>) "
            "and press <b>Graph</b>.</p>"
            "<p>The matrix will then appear here. Choose which device to gate on and "
            "use the <b>Centre</b> / <b>Width</b> sliders (or type <b>Low</b> / <b>High</b>, "
            "e.g. 500 - 520 keV) to select a slice. The slice is shaded on the matrix and "
            "the coincident spectrum in the other device is shown alongside.</p>")
        hint.setWordWrap(True)
        hint.setAlignment(QtCore.Qt.AlignCenter)
        hint.setMargin(60)
        self.proj = ProjectionTab(status)
        self.addWidget(hint)
        self.addWidget(self.proj)
        analysis.matrix_ready.connect(self.show_result)

    def show_result(self, res, log):
        self.setCurrentWidget(self.proj)
        self.proj.set_result(res, log)


# ==========================================================================
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, directories=()):
        super().__init__()
        self.setWindowTitle("T-List Data Scanner")
        self.resize(1500, 950)
        self.store = Store()
        self.search = SearchTab(self.store)
        self.results = ScanResultsTab(self.store)
        self.analysis = AnalysisTab(self.store, self.results)
        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self.search, "Search Directories")
        tabs.addTab(self.results, "Scan Results")
        tabs.addTab(self.analysis, "Analysis")
        self.projections = ProjectionsSection(self.analysis, self.statusBar().showMessage)
        tabs.addTab(self.projections, "Projections")
        self.setCentralWidget(tabs)
        self.tabs = tabs
        tlist_theme.add_menu(self)
        for d in directories:
            self.search.add_path(os.path.abspath(d))

    def closeEvent(self, ev):
        self.search.shutdown()
        super().closeEvent(ev)


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow(sys.argv[1:])
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
