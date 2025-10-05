# LuxNoise Sweep Analysis

This repository collects the artefacts generated while characterising the LuxNoise simulations across both the initial "Trail" timelapse runs and the larger RNAP/Ribo resource sweeps (Cond1–Cond3).

## Repository layout

- `data/Cond*/summary_cond*.csv` – 500-replicate sweep summaries covering RNAP totals {20, 35, 50, 75, 100, 150, 200} and ribosome totals {50, 100, 200, 300, 400, 500}. Each row stores mean GFP, SD, CV, and CV² computed over the steady-state window (t = 3000–4000).
- `data/trails/Trail*/Cond*/` – key outputs from the earlier Trail 1/2/3 runs: the per-condition summary CSV, mean-curve CSV, and timelapse plot PNG.
- `plots/` – pre-rendered sweep visualisations (heatmaps, contour maps, 3D surfaces, and line comparisons) produced by `analyze_sweeps.py`.
- `scripts/` – all simulation drivers used in this work:
  - `cond*_timelapse.py` – single-condition Gillespie simulators (defaults N=500, T=6000).
  - `cond1_sweep.py`, `cond2_sweep.py`, `cond3_sweep.py` – resource sweep drivers (updated to support the larger grids).
- `analyze_sweeps.py` – plotting utility that ingests the sweep summaries and regenerates everything in `plots/`.

## Reproducing the plots

```bash
python analyze_sweeps.py
```

`analyze_sweeps.py` accepts optional flags if you need to relocate the data or output directories, or focus the cross-condition comparison on a specific ribosome value:

```bash
python analyze_sweeps.py --data path/to/data --plots path/to/output --ribo 300
```

## Re-running the simulations

- Timelapse trails: execute `python scripts/cond*_timelapse.py` with the desired `-N`, `--T`, and output arguments.
- Resource sweeps: execute `python scripts/cond*_sweep.py` with custom `--RNAP`, `--Ribo`, `--load`, `-N`, and `--T` values. The current sweep dataset was generated with RNAP totals {20, 35, 50, 75, 100, 150, 200}, ribosome totals {50, 100, 200, 300, 400, 500}, `load=0`, `N=500`, and `T=4000`.

### Raw timelapse trajectories

Full per-replicate timelapse CSVs are large (tens of MB per grid point) and therefore omitted from version control. They remain available in the working directory under `Trail */Cond*/cond*_timelapse_raw.csv` and `Trail sweep 1/Cond*/cond*_RNAP*_Ribo*_raw.csv`. Re-run the relevant scripts if you need refreshed raw data before plotting.

## Next steps

- Extend `analyze_sweeps.py` with statistical overlays (e.g. confidence intervals) once additional replicates or alternative seeds are available.
- Promote the analysis into a notebook or automated reporting pipeline for future sweeps.
