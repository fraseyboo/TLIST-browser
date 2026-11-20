"""
Core reader / analysis for Mirion (Canberra) Prospect time-stamped list-mode
(.ava) files.

File layout (as reverse-engineered from Lynx and Osprey output):

* Each .ava file is an XML <mcadata> document. Filenames look like
      <Device><IP>_<MAC>_<runstampUTC>_<filestamp>_<seq>.ava
  e.g. Osprey5138.253.96.79_00-00-AF-02-04-B9_20261019115627_20261019115630_2.ava
  (Prospect writes the month 0-based in the filename, so 10 == November.)
* <list_data><entry start_time_ms=... timebase="100" units="nsec"> holds a
  stream of <event time="T" data="D"/> elements:
    - T <  32768 : a real event. T is the 15-bit fine time (in `timebase` ns
                   ticks) since the last rollover marker, D is the ADC channel.
    - T >= 32768 : a rollover marker (D == 0). T - 32768 is the rollover count;
                   one rollover = 32768 ticks (3.2768 ms at 100 ns).
  Absolute time = ((rollover_count << 15) + T) * timebase   [ns]
  The counter restarts at 32768 for every acquisition, and synchronised
  devices restart together, so timestamps of two devices in the same
  acquisition are directly comparable.
* An acquisition ("run") is identified by the entry's start_time_ms. The
  filename's run stamp is NOT reliable: Prospect sometimes writes the first
  file(s) of a new run under the previous run's name.
* Energy calibration: <calibration_details type="energy"> default polynomial
  model, E = c0 + c1*ch + c2*ch^2 [keV].
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

ROLLOVER_FLAG = 32768          # time values >= this are rollover markers
ROLLOVER_BITS = 15             # fine time is 15 bits

_OCTET = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
_FNAME = re.compile(
    rf"^(?P<device>.+?)(?P<ip>{_OCTET}(?:\.{_OCTET}){{3}})_(?P<mac>[^_]+)_"
    rf"(?P<runstamp>\d{{14}})_(?P<filestamp>\d{{14}})_(?P<seq>\d+)\.ava$",
    re.IGNORECASE,
)
_ENTRY = re.compile(rb"<entry\b[^>]*>")
_ATTR = re.compile(rb'(\w+)="([^"]*)"')
_EVENT = re.compile(rb'<event\s+time="(\d+)"\s+data="(\d+)"')
_ECAL = re.compile(
    rb'<calibration_details\s+type="energy"\s*>(.*?)</calibration_details>', re.S)
_MODEL = re.compile(rb"<model\b([^>]*)>(.*?)</model>", re.S)
_COEF = re.compile(rb'<coefficient\s+value="([^"]+)"')


# --------------------------------------------------------------------------
# Data classes
# --------------------------------------------------------------------------
@dataclass
class Run:
    """One acquisition of one device: an ordered set of .ava files."""
    device: str
    ip: str
    mac: str
    start_ms: int
    files: list = field(default_factory=list)      # [(seq, path)]
    timebase_ns: float = 100.0
    ecal: tuple = (0.0, 1.0, 0.0)

    @property
    def key(self) -> str:
        return f"{self.device}@{self.start_ms}"

    @property
    def start_str(self) -> str:
        return format_ms(self.start_ms)

    def __str__(self):
        return f"{self.device} ({self.ip})  start {self.start_str}  [{len(self.files)} files]"


@dataclass
class EventData:
    run: Run
    t_ns: np.ndarray        # int64, absolute ns since run start (monotonic)
    channel: np.ndarray     # int32 ADC channel

    @property
    def energy(self) -> np.ndarray:
        c = self.run.ecal
        ch = self.channel.astype(np.float64)
        return c[0] + c[1] * ch + (c[2] * ch * ch if len(c) > 2 and c[2] else 0.0)

    def values(self, units: str) -> np.ndarray:
        return self.channel.astype(np.float64) if units == "channel" else self.energy

    @property
    def duration_s(self) -> float:
        return float(self.t_ns[-1] - self.t_ns[0]) * 1e-9 if len(self.t_ns) > 1 else 0.0


def format_ms(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


# --------------------------------------------------------------------------
# Directory scanning
# --------------------------------------------------------------------------
def _entry_attrs(buf: bytes) -> dict:
    m = _ENTRY.search(buf)
    if not m:
        return {}
    return {k.decode(): v.decode() for k, v in _ATTR.findall(m.group(0))}


def _energy_cal(buf: bytes):
    m = _ECAL.search(buf)
    if not m:
        return None
    models = _MODEL.findall(m.group(1))
    for attrs, body in models:
        if b'is_default="true"' in attrs and b'polynomial' in attrs:
            return tuple(float(c) for c in _COEF.findall(body))
    for attrs, body in models:
        if b'polynomial' in attrs:
            return tuple(float(c) for c in _COEF.findall(body))
    return None


def scan_directory(directory: str) -> list[Run]:
    """Find all runs in a directory. Reads each file's <entry> header only."""
    runs: dict[tuple, Run] = {}
    for name in os.listdir(directory):
        m = _FNAME.match(name)
        if not m:
            continue
        path = os.path.join(directory, name)
        with open(path, "rb") as fh:
            buf = fh.read()
        attrs = _entry_attrs(buf)
        if "start_time_ms" not in attrs:
            continue
        start_ms = int(attrs["start_time_ms"])
        k = (m["device"], m["ip"], start_ms)
        run = runs.get(k)
        if run is None:
            run = runs[k] = Run(m["device"], m["ip"], m["mac"], start_ms)
            run.timebase_ns = float(attrs.get("timebase", 100))
            cal = _energy_cal(buf)
            if cal:
                run.ecal = cal
        run.files.append((int(m["seq"]), path))
    out = sorted(runs.values(), key=lambda r: (r.device, r.start_ms))
    for r in out:
        r.files.sort()
    return out


def devices(runs: list[Run]) -> list[str]:
    return sorted({r.device for r in runs})


def find_run(runs: list[Run], device: str, start: str | None = None) -> Run:
    """Select a run by device name (case-insensitive) and optional start time.

    `start` may be a substring of the displayed UTC start ("11:56:26",
    "2026-11-19 11:56"), or the raw start_time_ms. If omitted, the latest run
    is used.
    """
    cand = [r for r in runs if r.device.lower() == device.lower()]
    if not cand:
        raise LookupError(f"No device '{device}'. Available: {', '.join(devices(runs))}")
    if start:
        cand = [r for r in cand if start in r.start_str or start == str(r.start_ms)]
        if not cand:
            raise LookupError(f"No {device} run matching start '{start}'")
        if len(cand) > 1:
            raise LookupError(f"Start '{start}' is ambiguous for {device}: "
                              + ", ".join(r.start_str for r in cand))
    return max(cand, key=lambda r: r.start_ms)


def pair_runs(runs: list[Run], dev_a: str, dev_b: str,
              start_a: str | None = None, start_b: str | None = None):
    """Pick runs for two devices. Given only one start (or none), the other
    device's run is the one whose start time is closest."""
    if start_a or not start_b:
        ra = find_run(runs, dev_a, start_a)
        rb = (find_run(runs, dev_b, start_b) if start_b else
              min((r for r in runs if r.device.lower() == dev_b.lower()),
                  key=lambda r: abs(r.start_ms - ra.start_ms), default=None))
        if rb is None:
            raise LookupError(f"No device '{dev_b}'. Available: {', '.join(devices(runs))}")
    else:
        rb = find_run(runs, dev_b, start_b)
        ra = min((r for r in runs if r.device.lower() == dev_a.lower()),
                 key=lambda r: abs(r.start_ms - rb.start_ms), default=None)
        if ra is None:
            raise LookupError(f"No device '{dev_a}'. Available: {', '.join(devices(runs))}")
    return ra, rb


# --------------------------------------------------------------------------
# Event loading
# --------------------------------------------------------------------------
def _events_from_buf(buf: bytes) -> np.ndarray:
    i = buf.find(b"<list_data")
    if i < 0:
        return np.empty((0, 2), np.int64)
    pairs = _EVENT.findall(buf, i)
    if not pairs:
        return np.empty((0, 2), np.int64)
    return np.array(pairs, dtype=np.int64)


def _parse_file(path: str):
    with open(path, "rb") as fh:
        return _events_from_buf(fh.read())


_SPECTRUM = re.compile(rb"<spectrum\b[^>]*>")


@dataclass
class FileInfo:
    """Everything needed from one .ava file (used for incremental scanning)."""
    path: str
    device: str
    ip: str
    mac: str
    seq: int
    start_ms: int
    timebase_ns: float
    ecal: tuple
    elapsed_real_s: float       # acquisition real time at the end of this file
    n_channels: int
    raw: np.ndarray             # (n, 2) int64: raw time / data values incl. rollovers


def read_file(path: str) -> FileInfo | None:
    """Parse one .ava file completely. Returns None if it isn't a list-mode file."""
    m = _FNAME.match(os.path.basename(path))
    if not m:
        return None
    with open(path, "rb") as fh:
        buf = fh.read()
    attrs = _entry_attrs(buf)
    if "start_time_ms" not in attrs:
        return None
    sm = _SPECTRUM.search(buf)
    sattrs = {k.decode(): v.decode() for k, v in _ATTR.findall(sm.group(0))} if sm else {}
    units = attrs.get("units", "nsec")
    real = float(attrs.get("elapsed_real", 0))
    real_s = real * 1e-6 if units == "nsec" else real   # entry real time is in µs
    return FileInfo(
        path=path, device=m["device"], ip=m["ip"], mac=m["mac"], seq=int(m["seq"]),
        start_ms=int(attrs["start_time_ms"]), timebase_ns=float(attrs.get("timebase", 100)),
        ecal=_energy_cal(buf) or (0.0, 1.0, 0.0), elapsed_real_s=real_s,
        n_channels=int(sattrs.get("elapsed_comp_end_chan", 0) or 0),
        raw=_events_from_buf(buf))


def load_run(run: Run, progress=None) -> EventData:
    """Parse all events of a run into absolute timestamps.

    progress: optional callable(done, total).
    """
    chunks = []
    n = len(run.files)
    for k, (_, path) in enumerate(run.files):
        chunks.append(_parse_file(path))
        if progress and (k % 50 == 0 or k == n - 1):
            progress(k + 1, n)
    return events_from_raw(run, chunks)


def events_from_raw(run: Run, chunks) -> EventData:
    """Turn raw (time, data) chunks, in file-sequence order, into EventData."""
    raw = np.concatenate(chunks) if len(chunks) else np.empty((0, 2), np.int64)
    t, d = raw[:, 0], raw[:, 1]

    is_roll = t >= ROLLOVER_FLAG
    roll = t[is_roll] - ROLLOVER_FLAG
    # The 15-bit rollover counter could wrap after ~107 s; unwrap it.
    if len(roll) > 1:
        wraps = np.concatenate(([0], np.cumsum(np.diff(roll) < 0)))
        roll = roll + (wraps << ROLLOVER_BITS)
    # Rollover count in effect for every element (events before the first
    # marker belong to the period before it).
    ev = ~is_roll
    k = np.cumsum(is_roll)[ev] - 1          # index of the latest marker, -1 = none yet
    first = roll[0] - 1 if len(roll) else 0
    counter = np.where(k >= 0, roll[np.maximum(k, 0)] if len(roll) else 0, first)
    ticks = (counter << ROLLOVER_BITS) + t[ev]
    t_ns = (ticks * run.timebase_ns).astype(np.int64)
    chan = d[ev].astype(np.int32)
    if len(t_ns) > 1 and np.any(np.diff(t_ns) < 0):
        o = np.argsort(t_ns, kind="stable")
        t_ns, chan = t_ns[o], chan[o]
    return EventData(run, t_ns, chan)


# --------------------------------------------------------------------------
# Coincidence analysis
# --------------------------------------------------------------------------
def coincidence_pairs(ta: np.ndarray, tb: np.ndarray, lo_ns: float, hi_ns: float,
                      chunk: int = 2_000_000):
    """All index pairs (i, j) with lo <= ta[i] - tb[j] < hi  (half-open).

    Both arrays must be sorted. Returns (i, j, dt) with dt = ta[i] - tb[j].
    """
    I, J = [], []
    for s in range(0, len(ta), chunk):
        a = ta[s:s + chunk]
        first = np.searchsorted(tb, a - hi_ns, side="right")
        last = np.searchsorted(tb, a - lo_ns, side="right")
        n = last - first
        tot = int(n.sum())
        if tot == 0:
            continue
        i = np.repeat(np.arange(len(a)), n)
        off = np.arange(tot) - np.repeat(np.cumsum(n) - n, n)
        I.append(i + s)
        J.append(first[i] + off)
    if not I:
        e = np.empty(0, np.int64)
        return e, e, e
    i = np.concatenate(I)
    j = np.concatenate(J)
    return i, j, ta[i] - tb[j]


@dataclass
class Result:
    a: EventData
    b: EventData
    units_a: str
    units_b: str
    t_edges: np.ndarray
    t_counts: np.ndarray
    ea_edges: np.ndarray
    eb_edges: np.ndarray
    ee_counts: np.ndarray       # shape (len(ea_edges)-1, len(eb_edges)-1)
    gate: tuple
    n_pairs: int


def analyse(a: EventData, b: EventData, *,
            t_range=(-15000.0, 15000.0), t_bins=150,
            ea_range=(0.0, 4000.0), ea_bins=200,
            eb_range=(0.0, 4000.0), eb_bins=200,
            units_a="energy", units_b="energy",
            gate=None, offset_ns=0.0) -> Result:
    """Build the delta-time and energy-vs-energy histograms.

    dt = t_A - (t_B + offset_ns).  `gate` (lo, hi) in ns restricts which pairs
    go into the energy-energy histogram; default is the full time range.
    """
    tb = b.t_ns + np.int64(round(offset_ns))
    lo, hi = t_range
    if gate is not None:
        lo, hi = min(lo, gate[0]), max(hi, gate[1])
    i, j, dt = coincidence_pairs(a.t_ns, tb, lo, hi)

    t_edges = np.linspace(t_range[0], t_range[1], t_bins + 1)
    inside = (dt >= t_range[0]) & (dt < t_range[1])   # half-open, like every bin
    t_counts, _ = np.histogram(dt[inside], bins=t_edges)

    g = gate if gate is not None else t_range
    sel = (dt >= g[0]) & (dt < g[1])
    xa = a.values(units_a)[i[sel]]
    xb = b.values(units_b)[j[sel]]
    ea_edges = np.linspace(ea_range[0], ea_range[1], ea_bins + 1)
    eb_edges = np.linspace(eb_range[0], eb_range[1], eb_bins + 1)
    ee, _, _ = np.histogram2d(xa, xb, bins=[ea_edges, eb_edges])
    return Result(a, b, units_a, units_b, t_edges, t_counts,
                  ea_edges, eb_edges, ee.astype(np.int64), tuple(g), int(sel.sum()))


# --------------------------------------------------------------------------
# Export (CSV, same layout idea as the old tool's "Export Data")
# --------------------------------------------------------------------------
def _ulabel(u):
    return "keV" if u == "energy" else "ch"


def export_time(res: Result, path: str):
    with open(path, "w") as fh:
        fh.write(f"# Coincident events: dt = t({res.a.run.device}) - t({res.b.run.device})\n")
        fh.write("deltaTime_low_ns, deltaTime_high_ns, counts\n")
        for lo, hi, c in zip(res.t_edges[:-1], res.t_edges[1:], res.t_counts):
            fh.write(f"{lo:g}, {hi:g}, {c}\n")


def export_energy(res: Result, path: str, include_zeros=False):
    da, db = res.a.run.device, res.b.run.device
    with open(path, "w") as fh:
        fh.write(f"# Coincident events, gate {res.gate[0]:g}..{res.gate[1]:g} ns; bin low edges\n")
        fh.write(f"{da}_{_ulabel(res.units_a)}, {db}_{_ulabel(res.units_b)}, counts\n")
        ii, jj = (np.indices(res.ee_counts.shape).reshape(2, -1) if include_zeros
                  else np.nonzero(res.ee_counts))
        for i, j in zip(ii, jj):
            fh.write(f"{res.ea_edges[i]:g}, {res.eb_edges[j]:g}, {res.ee_counts[i, j]}\n")
