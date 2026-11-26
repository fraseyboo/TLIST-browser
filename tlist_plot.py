"""Matplotlib drawing shared by the CLI and the Qt GUI."""

import numpy as np
from matplotlib.colors import LogNorm, Normalize


def _ulabel(u):
    return "Energy (keV)" if u == "energy" else "Channel"


def add_bin_readout(ax, xedges, yedges, counts):
    """Make the toolbar's cursor read-out show the 2D histogram bin under the
    mouse and its count, not just the (x, y) position.  counts[i, j] is the
    bin xedges[i]..xedges[i+1], yedges[j]..yedges[j+1]."""
    xedges, yedges = np.asarray(xedges), np.asarray(yedges)

    def fmt(x, y):
        pos = f"x={x:.5g}  y={y:.5g}"
        i = np.searchsorted(xedges, x, "right") - 1
        j = np.searchsorted(yedges, y, "right") - 1
        if 0 <= i < counts.shape[0] and 0 <= j < counts.shape[1]:
            return (f"counts = {int(counts[i, j]):,}   in x [{xedges[i]:.5g}, {xedges[i + 1]:.5g}) "
                    f"× y [{yedges[j]:.5g}, {yedges[j + 1]:.5g})   ({pos})")
        return pos
    ax.format_coord = fmt


def draw_energy_energy(ax, res, log=True, cmap="turbo"):
    """2D histogram: x = device A, y = device B. Returns the mappable."""
    h = np.ma.masked_equal(res.ee_counts.T, 0)
    vmax = max(int(res.ee_counts.max()), 1)
    norm = LogNorm(vmin=1, vmax=max(vmax, 2)) if log else Normalize(vmin=0, vmax=vmax)
    m = ax.pcolormesh(res.ea_edges, res.eb_edges, h, cmap=cmap, norm=norm, shading="flat")
    ax.set_xlabel(f"{res.a.run.device} {_ulabel(res.units_a)}")
    ax.set_ylabel(f"{res.b.run.device} {_ulabel(res.units_b)}")
    ax.set_title(f"Energy vs energy  ({int(res.ee_counts.sum()):,} pairs, "
                 f"gate {res.gate[0]:g} to {res.gate[1]:g} ns)")
    ax.set_xlim(res.ea_edges[0], res.ea_edges[-1])
    ax.set_ylim(res.eb_edges[0], res.eb_edges[-1])
    ax.set_facecolor("white")
    add_bin_readout(ax, res.ea_edges, res.eb_edges, res.ee_counts)
    return m


def draw_delta_time(ax, res, log=True, color="C1"):
    centres = 0.5 * (res.t_edges[:-1] + res.t_edges[1:])
    ax.step(res.t_edges, np.append(res.t_counts, res.t_counts[-1]), where="post",
            color=color, lw=1.2)
    ax.plot(centres, res.t_counts, "o", color=color, ms=3)
    if log:
        ax.set_yscale("log")
        ax.set_ylim(bottom=0.8)
    ax.set_xlim(res.t_edges[0], res.t_edges[-1])
    ax.set_xlabel(f"Δt = t({res.a.run.device}) − t({res.b.run.device})  (ns)")
    ax.set_ylabel("Counts")
    peak = centres[np.argmax(res.t_counts)] if res.t_counts.any() else 0
    ax.set_title(f"Delta time  ({int(res.t_counts.sum()):,} pairs, peak at {peak:g} ns)")
    ax.grid(True, which="major", alpha=0.3)
    if res.gate != (res.t_edges[0], res.t_edges[-1]):
        for g in res.gate:
            ax.axvline(g, color="k", ls="--", lw=0.8, alpha=0.6)


def summary(res):
    a, b = res.a, res.b
    return (f"A: {a.run.device} start {a.run.start_str} UTC, {len(a.t_ns):,} events, "
            f"{a.duration_s:.1f} s\n"
            f"B: {b.run.device} start {b.run.start_str} UTC, {len(b.t_ns):,} events, "
            f"{b.duration_s:.1f} s")
