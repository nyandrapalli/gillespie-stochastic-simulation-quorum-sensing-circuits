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
            curves = np.zeros((N, len(t_grid)))
            seeds = rng.integers(1, 2**31 - 1, size=N, dtype=np.int64)
            for i in range(N):
                ti, G = gillespie_cond4(P, seed=int(seeds[i]))
                idx = np.searchsorted(ti, t_grid, side='right') - 1
                idx[idx < 0] = 0
                curves[i, :] = G[idx]

            pd.DataFrame(curves, columns=[f"t{t}" for t in t_grid]).assign(unit_id=lambda d: np.arange(N)).to_csv(
                os.path.join(outdir, f"cond4_RNAP{R}_Ribo{B}_raw.csv"), index=False
            )

            block = curves[:, w0 : w1 + 1].mean(axis=1)
            mean = float(block.mean())
            sd = float(block.std(ddof=0))
            cv = float(sd / mean) if mean > 0 else float("nan")
            cv2 = float(cv * cv) if mean > 0 else float("nan")
            summary.append(
                {
                    "condition": "Cond4",
                    "RNAP": R,
                    "Ribo": B,
                    "copies": 10,
                    "N": N,
                    "T_end": T_end,
                    "window_start": w0,
                    "window_end": w1,
                    "mean": mean,
                    "sd": sd,
                    "CV": cv,
                    "CV2": cv2,
                }
            )

            mean_curve = curves.mean(axis=0)
            p10 = np.percentile(curves, 10, axis=0)
            p90 = np.percentile(curves, 90, axis=0)
            plt.figure(figsize=(8, 4))
            for i in range(min(N, 30)):
                plt.plot(t_grid, curves[i, :], alpha=0.15, linewidth=0.5)
            plt.plot(t_grid, mean_curve, linewidth=2.0, label="Mean")
            plt.fill_between(t_grid, p10, p90, alpha=0.25, label="10–90%")
            plt.axvspan(w0, w1, alpha=0.1, label="Steady window")
            plt.xlabel("Time")
            plt.ylabel("Total GFP")
            plt.title(f"Cond4 RNAP={R}, Ribo={B}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"cond4_RNAP{R}_Ribo{B}_plot.png"), dpi=150)
            plt.close()

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
