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

Each axis has a **Bin width** field (keV, or channels in channel units).
0 means auto: *Bins* equal bins from Low to High. Otherwise the bins have this
width and High is rounded up to fit. A red note in the side panel appears when
an energy bin isn't a whole number of ADC channels, because uneven bins show up
as stripes.

**Projections tab**: slice the energy-energy matrix on one detector and see
the coincident spectrum in the other, e.g. gate Osprey on 500–520 keV to see
what Lynx records alongside a 511 keV hit. Choose which axis to gate, then set
the slice with the centre/width sliders or type low/high (`[low, high)`).
The slice is shaded on the 2D plot and the projection updates live. Cuts use
each pair's exact energy, not the 2D bins. Projection bins follow the
projected detector's channel boundaries (set *Channels / bin*), which avoids a
comb pattern. The grey curve is the ungated projection for comparison.
*Export projection…* writes a CSV. *Log scale* switches the projection's y axis
between log and linear.

On any 2D histogram (Energy vs Energy, the Projections matrix and the training
version's Analysis matrix), the toolbar read-out at the top right shows the
count in the bin under the mouse, with that bin's edges.

**Singles tab**: treats the data as singles, with no coincidence test: one
ungated 1D spectrum per detector, counting every event. For each device it
offers the acquisition closest in start time to Device A's; tick the detectors
and press *Plot singles* (loading any run not already cached). Choose energy
or channel units, *Full range* or a Low/High window, channels per bin (each
detector is binned on its own channel boundaries), *Overlay* or *One panel per
detector*, and *Counts / s* to compare detectors normalised by run duration.
*Export…* writes one CSV with a block per detector.

**Dark mode**: *View → Dark Mode* (Ctrl+Shift+D) in either GUI switches the
widgets and every plot between light and dark. The choice is remembered for
next time. Saving a figure from the plot toolbar saves it as it looks on
screen, so switch back to light mode for figures you want on a white
background.

## Training version (follows the lab manual)

```sh
~/.venvs/tlist/bin/python tlist_scanner.py
```

Same analysis, laid out like the original T-List Data Scanner so the
existing lab manual can be followed step by step:

1. **Search Directories**: *Add New Directory* adds the folder with status
   *Stopped*. The green play icon starts processing, and new files are
   re-scanned every *Search Frequency* seconds. The global play/stop buttons
   act on all directories; right-click a row to remove it. *Move Files After
   Scan* is shown but disabled, so files are never moved.
2. **Scan Results**: one row per device acquisition (start / last found time,
   source ECAL from the files, editable Target ECAL), plus the *Time
   Calibration* range and bins. Click a **device name** to show its ungated
   singles spectrum; hover to read channel, energy and counts.
   *Export Spectrum* saves the displayed spectrum as text, one row per
   channel: channel, energy (keV), counts.
   The optional **Coincidence Gate (ns)** (off by default; e.g. −800 to 0)
   restricts Energy/Channel plots and Projections to pairs inside the gate,
   while Delta Time plots keep the full Time Range and show the gate as
   dashed lines. With it off, the Time Range is the coincidence window.
3. **Analysis**: choose two *Devices* and *Acq. Start*s, then the units:
   * X = Energy/Channel, Y = Energy/Channel → energy-vs-energy matrix
   * X = Delta Time (ns), Y = Counts → Δt histogram (t(dev 1) − t(dev 2))
   * X = Energy/Channel, Y = Counts → coincident spectrum of device 1

   *Graph* plots. Successive 1D graphs overlay until *Clear Graph*.
   *Export Data* writes the old text format ("Coincident events" header).
   The optional **X/Y Bin Width** fields set the histogram bin width for
   each axis, in that axis's units (keV, channels or ns). Leave them blank to
   use the Bins from Scan Results. A red note appears when an energy bin isn't
   a whole number of ADC channels, because uneven bins show up as stripes. Use
   a multiple of the keV-per-channel value given in the note.

   **Recalibrate…** (on Scan Results, after clicking a device name) corrects a
   bad energy calibration. *Find Peaks* lists the strongest peaks' channels.
   Type the known energy next to the peaks you recognise and press *Fit*, or
   type the new calibration directly. *Apply* sets that device's Target ECAL
   Offset/Slope so that the energies used match the new calibration. The
   change can apply to every acquisition with the same file calibration.
   *Reset to File Calibration* undoes it. The spectrum shows the energies in
   use. An energy/channel matrix on Analysis redraws automatically when a
   plotted device's Target ECAL changes. 1D plots overlay, so they show a
   reminder to *Clear Graph* and *Graph* again instead.
4. **Projections** (extra, beyond the manual): picks up the last
   Energy/Channel × Energy/Channel matrix graphed on Analysis (with the Time
   Range and Target ECAL in force) and offers the same slice-and-project
   controls as the main GUI's Projections tab.

Target ECAL maps source energy to `offset + slope × E` before binning into
[Energy Low, Energy High) with *# BINs*. Defaults 0 / 4000 / 200 / 0 / 1 leave
the data unchanged. The coincidence window is the Time Range on Scan Results.

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

## Missing or unusable files

All three programs check each data folder and warn when files look missing or
unusable. The CLI prints the warnings to stderr. The GUI shows a popup and lists
them in the side panel. The training version lists them in a red box on Search
Directories and under the plot on Analysis.

| Warning | Usual cause |
|---|---|
| *spectrum only: no list-mode (T-List) event data* | The file was saved with Save Spectrum, so it has a spectrum but no pulse timestamps. Coincidence analysis needs list-mode files. |
| *list-mode file that has been renamed* | The device, run and file order are read from Prospect's original filename, so keep the original names. |
| *incomplete file* | The file is still being written or was copied badly (it has no closing `</mcadata>` tag). |
| *N file(s) missing from the middle* | There are gaps in the file sequence numbers of a run. |
| *first file is X s into the acquisition* | The files from the start of the run weren't copied. |
| *ends X s before …* | One detector's run stops early, so its last files weren't copied. |
| *no run from another device started at the same time* | The other detector's files weren't copied. |

## Files

* `tlist_core.py`: scanning, parsing, coincidence finding, histograms, CSV export
* `tlist_plot.py`: matplotlib drawing shared by CLI and GUI
* `tlist_cli.py`: command line
* `tlist_gui.py`: PyQt5 GUI (compact, all settings on one panel)
* `tlist_scanner.py`: PyQt5 GUI matching the old scanner / lab manual
* `tlist_widgets.py`: the Projections widget shared by both GUIs, and the Singles tab
* `tlist_theme.py`: light/dark theme for both GUIs (View → Dark Mode)
