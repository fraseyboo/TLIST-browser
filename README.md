# T-List coincidence analysis

Fast replacement for the old "T-List Data Scanner". Reads a directory of Mirion
Prospect time-stamped list-mode `.ava` files, picks two devices and an
acquisition for each, and plots

* **Energy vs energy** (2D histogram of coincident pairs, A on x, B on y)
* **Delta time vs counts**, with `dt = t(A) - t(B)`

The full `data/` folder (2.7k files, ~300 MB, 770k events) takes about 1.5 s.

## Setup

```sh
/opt/homebrew/bin/python3 -m venv ~/.venvs/tlist
~/.venvs/tlist/bin/pip install -r requirements.txt
```

Keep the venv **outside** this folder. Qt 5 can't load its platform plugins
from a path containing `Fraser's MBP` (spaces/apostrophe), so the GUI fails with
"no Qt platform plugin could be initialized". The CLI is unaffected.

## Command line

```sh
PY=~/.venvs/tlist/bin/python
$PY tlist_cli.py list data                          # devices and acquisitions
$PY tlist_cli.py analyse data -a Lynx -b Osprey5    # latest A run + closest B run
$PY tlist_cli.py analyse data -a Lynx -b Osprey5 --start-a 11:56:26 \
    --gate -800 0 --export out/run1 --save out/run1.png --no-show
```

Main options (`analyse -h` for all):

| option | default | meaning |
|---|---|---|
| `--start-a/--start-b` | latest / closest | run start, substring of `YYYY-MM-DD HH:MM:SS` (UTC) |
| `--trange LO HI`, `--tbins` | -15000 15000, 150 | Δt histogram (ns) |
| `--gate LO HI` | whole trange | Δt window for pairs in the energy plot |
| `--erange`, `--ebins` (`-a`/`-b` variants) | 0 4000, 200 | energy axes |
| `--units energy\|channel` | energy | axis units (per-axis: `--units-a/-b`) |
| `--offset NS` | 0 | shift added to B timestamps |
| `--linear` | log | linear colour / y scale |
| `--export PREFIX` | | writes `PREFIX_time.csv`, `PREFIX_energy.csv` |

## GUI

```sh
~/.venvs/tlist/bin/python tlist_gui.py data
```

Data load in a background thread and are cached, so changing bins, ranges,
gate or units re-plots instantly.

## File format notes

The format was reverse-engineered from the files; details are in the
docstring of `tlist_core.py`. In short:

* `<event time="T" data="D"/>`: T < 32768 is a real event (15-bit fine time in
  100 ns ticks, D = ADC channel). T >= 32768 is a rollover marker with count
  T - 32768, one rollover being 3.2768 ms.
  Absolute time = `((count << 15) + T) * timebase`.
* Acquisitions are grouped by the `<entry start_time_ms>` inside each file, not
  by the filename. Prospect wrote the first Lynx file of the 11:56:26 run under
  the 11:47:40 name. The filename month is also 0-based (`10` = November).
* Synchronised devices restart their rollover counters together, so A and B
  timestamps are compared directly (the start_time_ms values differ by
  ~0.9 s and are not used for alignment).
* Energy = default polynomial from `<calibration_details type="energy">`.
* All windows and bins are half-open, `[lo, hi)`. Keep the Δt bin width a
  multiple of 100 ns, or the histogram shows a comb pattern.

## Files

* `tlist_core.py`: scanning, parsing, coincidence finding, histograms, CSV export
* `tlist_plot.py`: matplotlib drawing shared by CLI and GUI
* `tlist_cli.py`: command line
* `tlist_gui.py`: PyQt5 GUI
