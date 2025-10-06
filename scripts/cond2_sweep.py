#!/usr/bin/env python3
"""
Cond2 resource-limitation sweep (LuxI producer + LuxR responder with plasmids).
Outputs per grid point:
  - Raw CSV (GA & GB timelapse): cond2_RNAP{R}_Ribo{B}_raw.csv
  - Preview PNG (GB trajectories): cond2_RNAP{R}_Ribo{B}_plot.png
Summary CSV across grid:
  - summary_cond2.csv (Mean, SD, CV, CV2 of GFP_total over steady window)
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cond2_timelapse import Params2, gillespie_cond2


def run_grid(RNAP_vals, Ribo_vals, N=20, T_end=3000, window=(2500, 3000), outdir="sweep_cond2_out", seed=202):
    os.makedirs(outdir, exist_ok=True)
    summary = []
    rng = np.random.default_rng(seed)
    w0, w1 = window

    for R in RNAP_vals:
        for B in Ribo_vals:
            P = Params2(
                RNAP_total=R,
                Ribo_total=B,
                n_pCon_luxI_A=10,
                n_pCon_luxR_B=10,
                n_pLux_gfp_B=10,
                T_end=T_end,
            )
            t_grid = np.arange(int(T_end) + 1)
            GA_all = np.zeros((N, len(t_grid)))
            GB_all = np.zeros((N, len(t_grid)))
            seeds = rng.integers(1, 2**31 - 1, size=N, dtype=np.int64)
            for i in range(N):
                ti, GA, GB = gillespie_cond2(P, seed=int(seeds[i]))
                idx = np.searchsorted(ti, t_grid, side='right') - 1
                idx[idx < 0] = 0
                GA_all[i, :] = GA[idx]
                GB_all[i, :] = GB[idx]

            cols = [f"GA_t{t}" for t in t_grid] + [f"GB_t{t}" for t in t_grid]
            raw = np.hstack([GA_all, GB_all])
            raw_df = pd.DataFrame(raw, columns=cols).assign(replicate_id=lambda d: np.arange(N))
            raw_df.to_csv(os.path.join(outdir, f"cond2_RNAP{R}_Ribo{B}_raw.csv"), index=False)

            A_means = GA_all[:, w0 : w1 + 1].mean(axis=1)
            B_means = GB_all[:, w0 : w1 + 1].mean(axis=1)
            TOT = A_means + B_means
            mean = float(TOT.mean())
            sd = float(TOT.std(ddof=0))
            cv = float(sd / mean) if mean > 0 else float("nan")
            cv2 = float(cv * cv) if mean > 0 else float("nan")
            summary.append(
                {
                    "condition": "Cond2",
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

            meanB = GB_all.mean(axis=0)
            p10 = np.percentile(GB_all, 10, axis=0)
            p90 = np.percentile(GB_all, 90, axis=0)
            plt.figure(figsize=(8, 4))
            for i in range(min(N, 30)):
                plt.plot(t_grid, GB_all[i, :], alpha=0.2, linewidth=0.5)
            plt.plot(t_grid, meanB, linewidth=2.0, label="Mean (B)")
            plt.fill_between(t_grid, p10, p90, alpha=0.25, label="10–90% (B)")
            plt.axvspan(w0, w1, alpha=0.1, label="Steady window")
            plt.xlabel("Time")
            plt.ylabel("GFP (cell B)")
            plt.title(f"Cond2 RNAP={R}, Ribo={B}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"cond2_RNAP{R}_Ribo{B}_plot.png"), dpi=150)
            plt.close()

    summary_df = pd.DataFrame(summary)
    summary_csv = os.path.join(outdir, "summary_cond2.csv")
    summary_df.to_csv(summary_csv, index=False)
    return summary_csv


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--RNAP", nargs="+", type=int, default=[35, 75], help="RNAP_total grid")
    ap.add_argument("--Ribo", nargs="+", type=int, default=[100, 300], help="Ribo_total grid")
    ap.add_argument("-N", type=int, default=20, help="systems per grid point")
    ap.add_argument("--T", type=int, default=3000, help="T_end")
    ap.add_argument("--w0", type=int, default=2500, help="steady window start")
    ap.add_argument("--w1", type=int, default=3000, help="steady window end")
    ap.add_argument("--out", type=str, default="sweep_cond2_out", help="output folder")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summary_csv = run_grid(args.RNAP, args.Ribo, N=args.N, T_end=args.T, window=(args.w0, args.w1), outdir=args.out)
    print("Wrote summary to:", summary_csv)
