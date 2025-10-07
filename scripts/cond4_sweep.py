#!/usr/bin/env python3
"""
Cond4 resource-limitation sweep (single-cell Lux/Lux/Las plasmid combination).
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cond4_timelapse import Params4, gillespie_cond4


def run_grid(RNAP_vals, Ribo_vals, N=20, T_end=3000, window=(2500, 3000), outdir="sweep_cond4_out", seed=404):
    os.makedirs(outdir, exist_ok=True)
    summary = []
    rng = np.random.default_rng(seed)
    w0, w1 = window

    for R in RNAP_vals:
        for B in Ribo_vals:
            P = Params4(
                RNAP_total=R,
                Ribo_total=B,
                n_pCon_luxI=10,
                n_pCon_luxR=10,
                n_pCon_lasR=10,
                n_pLas_gfp=10,
                n_pLux_gfp=10,
                n_pLux_lasI=10,
                T_end=T_end,
            )
            t_grid = np.arange(int(T_end) + 1)
            G_las_curves = np.zeros((N, len(t_grid)))
            G_lux_curves = np.zeros((N, len(t_grid)))
            seeds = rng.integers(1, 2**31 - 1, size=N, dtype=np.int64)
            for i in range(N):
                ti, G_las, G_lux = gillespie_cond4(P, seed=int(seeds[i]))
                idx = np.searchsorted(ti, t_grid, side='right') - 1
                idx[idx < 0] = 0
                G_las_curves[i, :] = G_las[idx]
                G_lux_curves[i, :] = G_lux[idx]

            total_curves = G_las_curves + G_lux_curves
            df = pd.DataFrame(G_las_curves, columns=[f'GLas_t{t}' for t in t_grid])
            df = df.join(pd.DataFrame(G_lux_curves, columns=[f'GLux_t{t}' for t in t_grid]))
            df = df.assign(replicate_id=lambda d: np.arange(N))
            df.to_csv(os.path.join(outdir, f"cond4_RNAP{R}_Ribo{B}_raw.csv"), index=False)

            block_las = G_las_curves[:, w0 : w1 + 1].mean(axis=1)
            block_lux = G_lux_curves[:, w0 : w1 + 1].mean(axis=1)
            block_total = total_curves[:, w0 : w1 + 1].mean(axis=1)

            def stats(arr):
                mean = float(arr.mean())
                sd = float(arr.std(ddof=0))
                cv = float(sd / mean) if mean > 0 else float('nan')
                cv2 = float(cv * cv) if mean > 0 else float('nan')
                return mean, sd, cv, cv2

            mean_tot, sd_tot, cv_tot, cv2_tot = stats(block_total)
            mean_las, sd_las, cv_las, cv2_las = stats(block_las)
            mean_lux, sd_lux, cv_lux, cv2_lux = stats(block_lux)
            summary.append({
                'condition': 'Cond4',
                'RNAP': R,
                'Ribo': B,
                'copies': 10,
                'N': N,
                'T_end': T_end,
                'window_start': w0,
                'window_end': w1,
                'mean_total': mean_tot,
                'sd_total': sd_tot,
                'CV_total': cv_tot,
                'CV2_total': cv2_tot,
                'mean_pLasGFP': mean_las,
                'sd_pLasGFP': sd_las,
                'CV_pLasGFP': cv_las,
                'CV2_pLasGFP': cv2_las,
                'mean_pLuxGFP': mean_lux,
                'sd_pLuxGFP': sd_lux,
                'CV_pLuxGFP': cv_lux,
                'CV2_pLuxGFP': cv2_lux,
                'mean': mean_tot,
                'sd': sd_tot,
                'CV': cv_tot,
                'CV2': cv2_tot,
            })

            mean_las_curve = G_las_curves.mean(axis=0)
            mean_lux_curve = G_lux_curves.mean(axis=0)
            total_mean_curve = total_curves.mean(axis=0)
            plt.figure(figsize=(8, 4))
            for i in range(min(N, 30)):
                plt.plot(t_grid, total_curves[i, :], alpha=0.15, linewidth=0.5)
    summary_df = pd.DataFrame(summary)
    summary_csv = os.path.join(outdir, "summary_cond4.csv")
    summary_df.to_csv(summary_csv, index=False)
    return summary_csv


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--RNAP", nargs="+", type=int, default=[35, 75], help="RNAP_total grid")
    ap.add_argument("--Ribo", nargs="+", type=int, default=[100, 300], help="Ribo_total grid")
    ap.add_argument("-N", type=int, default=20)
    ap.add_argument("--T", type=int, default=3000)
    ap.add_argument("--w0", type=int, default=2500)
    ap.add_argument("--w1", type=int, default=3000)
    ap.add_argument("--out", type=str, default="sweep_cond4_out")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summary_csv = run_grid(args.RNAP, args.Ribo, N=args.N, T_end=args.T, window=(args.w0, args.w1), outdir=args.out)
    print("Wrote summary to:", summary_csv)
