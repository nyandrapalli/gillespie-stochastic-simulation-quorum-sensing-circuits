# LuxNoise Sweep Analysis

This repository packages the analysis artefacts for the RNAP/Ribo resource sweeps we ran over the three LuxNoise conditions (Cond1–Cond3).

## Contents

- `data/Cond*/summary_cond*.csv` – 500-replicate sweep summaries covering RNAP totals {20, 35, 50, 75, 100, 150, 200} and ribosome totals {50, 100, 200, 300, 400, 500}. Each row stores mean GFP, SD, CV, and CV² calculated over the steady-state window (t = 3000–4000).
- `plots/` – pre-rendered visualisations created from the summaries (heatmaps, contour maps, 3D surfaces, line slices, and condition comparisons).
- `analyze_sweeps.py` – the script that (re)generates the plots. It expects the summary CSVs described above.

## Reproducing the plots

```bash
python analyze_sweeps.py
```

`analyze_sweeps.py` accepts optional flags if the data or output directories need to be relocated:

```bash
python analyze_sweeps.py --data path/to/data --plots path/to/output --ribo 300
```

The `--ribo` argument controls which ribosome value is used for the cross-condition line comparisons (defaults to the middle value present in the data).

## Raw data

The raw timelapse trajectories (hundreds of MB) are not stored here; they live in the original simulation workspace under `Trail sweep 1/Cond*/cond*_RNAP*_Ribo*_raw.csv`. Regenerate them by running the appropriate sweep scripts if needed before using `analyze_sweeps.py` on new inputs.

## Next steps

- Extend `analyze_sweeps.py` with statistical overlays (e.g. confidence intervals) once additional replicates or alternative seeds are available.
- Integrate the script into a notebook or pipeline for automated reporting across future sweeps.
