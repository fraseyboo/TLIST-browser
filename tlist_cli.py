#!/usr/bin/env python3
"""
Command-line coincidence analysis for Prospect time-stamped list-mode data.

  python tlist_cli.py list data
  python tlist_cli.py analyse data -a Lynx -b Osprey5
  python tlist_cli.py analyse data -a Lynx -b Osprey5 --start-a 11:56:26 \
        --gate -800 0 --export out/run1 --save out/run1.png --no-show
"""

import argparse
import sys
import time

import tlist_core as core


def cmd_list(args):
    t0 = time.perf_counter()
    runs = core.scan_directory(args.directory)
    if not runs:
        print(f"No .ava files found in {args.directory}")
        return 1
    print(f"{'Device':<12}{'IP':<17}{'Start (UTC)':<22}{'Files':>7}  ECAL (c0, c1, c2)")
    for r in runs:
        cal = ", ".join(f"{c:.6g}" for c in r.ecal)
        print(f"{r.device:<12}{r.ip:<17}{r.start_str:<22}{len(r.files):>7}  {cal}")
    print(f"\n({len(runs)} runs, scanned in {time.perf_counter() - t0:.2f} s)")
    return 0


def cmd_analyse(args):
    t0 = time.perf_counter()
    runs = core.scan_directory(args.directory)
    try:
        ra, rb = core.pair_runs(runs, args.a, args.b, args.start_a, args.start_b)
    except LookupError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"A = {ra}\nB = {rb}")
    if abs(ra.start_ms - rb.start_ms) > 10_000:
        print("warning: runs started more than 10 s apart - probably not the same acquisition",
              file=sys.stderr)
    if (args.trange[1] - args.trange[0]) / args.tbins % ra.timebase_ns:
        print(f"warning: time bin width is not a multiple of the {ra.timebase_ns:g} ns timebase; "
              "expect a comb pattern", file=sys.stderr)
    a = core.load_run(ra)
    b = core.load_run(rb)
    t1 = time.perf_counter()
    res = core.analyse(
        a, b,
        t_range=tuple(args.trange), t_bins=args.tbins,
        ea_range=tuple(args.erange_a or args.erange), ea_bins=args.ebins_a or args.ebins,
        eb_range=tuple(args.erange_b or args.erange), eb_bins=args.ebins_b or args.ebins,
        units_a=args.units_a or args.units, units_b=args.units_b or args.units,
        gate=tuple(args.gate) if args.gate else None, offset_ns=args.offset)
    t2 = time.perf_counter()

    import tlist_plot
    print(tlist_plot.summary(res))
    print(f"Coincident pairs in time window: {int(res.t_counts.sum()):,}; "
          f"in energy histogram: {int(res.ee_counts.sum()):,}")
    print(f"Timing: scan+load {t1 - t0:.2f} s, analysis {t2 - t1:.2f} s")

    if args.export:
        core.export_time(res, args.export + "_time.csv")
        core.export_energy(res, args.export + "_energy.csv")
        print(f"Exported {args.export}_time.csv and {args.export}_energy.csv")

    if args.no_show and not args.save:
        return 0
    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))
    m = tlist_plot.draw_energy_energy(ax1, res, log=not args.linear)
    fig.colorbar(m, ax=ax1, label="Counts")
    tlist_plot.draw_delta_time(ax2, res, log=not args.linear)
    fig.suptitle(f"{ra.device} {ra.start_str}  vs  {rb.device} {rb.start_str}  (UTC)")
    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=130)
        print(f"Saved {args.save}")
    if not args.no_show:
        plt.show()
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("list", help="list devices / acquisitions in a directory")
    pl.add_argument("directory")
    pl.set_defaults(func=cmd_list)

    pa = sub.add_parser("analyse", aliases=["analyze"], help="coincidence histograms for two devices")
    pa.add_argument("directory")
    pa.add_argument("-a", required=True, help="device A name, e.g. Lynx (x axis; dt = tA - tB)")
    pa.add_argument("-b", required=True, help="device B name, e.g. Osprey5 (y axis)")
    pa.add_argument("--start-a", help="run start of A (substring of 'YYYY-MM-DD HH:MM:SS' UTC). "
                                      "Default: latest run")
    pa.add_argument("--start-b", help="run start of B. Default: run closest in time to A's")
    pa.add_argument("--trange", nargs=2, type=float, default=[-15000, 15000], metavar=("LO", "HI"),
                    help="delta-time histogram range in ns (default -15000 15000)")
    pa.add_argument("--tbins", type=int, default=150, help="delta-time bins (default 150)")
    pa.add_argument("--gate", nargs=2, type=float, metavar=("LO", "HI"),
                    help="dt window (ns) for pairs in the energy histogram (default: whole trange)")
    pa.add_argument("--offset", type=float, default=0.0, help="ns added to B timestamps")
    pa.add_argument("--units", choices=["energy", "channel"], default="energy")
    pa.add_argument("--units-a", choices=["energy", "channel"])
    pa.add_argument("--units-b", choices=["energy", "channel"])
    pa.add_argument("--erange", nargs=2, type=float, default=[0, 4000], metavar=("LO", "HI"))
    pa.add_argument("--erange-a", nargs=2, type=float, metavar=("LO", "HI"))
    pa.add_argument("--erange-b", nargs=2, type=float, metavar=("LO", "HI"))
    pa.add_argument("--ebins", type=int, default=200)
    pa.add_argument("--ebins-a", type=int)
    pa.add_argument("--ebins-b", type=int)
    pa.add_argument("--linear", action="store_true", help="linear instead of log colour/y scale")
    pa.add_argument("--export", metavar="PREFIX", help="write PREFIX_time.csv and PREFIX_energy.csv")
    pa.add_argument("--save", metavar="FILE", help="save figure (png/pdf/svg)")
    pa.add_argument("--no-show", action="store_true", help="don't open a plot window")
    pa.set_defaults(func=cmd_analyse)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
