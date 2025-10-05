#!/usr/bin/env python3
"""Generate visualisations for RNAP/Ribo sweeps.

The script expects summary CSV files under ``data/Cond*/summary_cond*.csv``
containing at least the columns ``RNAP``, ``Ribo``, ``mean``, ``CV``, and ``CV2``.
Rendered plots are written to ``plots/`` (created if it does not exist).
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import pandas as pd


def load_data(data_root: Path) -> pd.DataFrame:
    frames = []
    for cond_dir in sorted(data_root.glob("Cond*/")):
        summary = cond_dir / f"summary_{cond_dir.name.lower()}.csv"
        if not summary.exists():
            raise FileNotFoundError(f"Missing summary file: {summary}")
        df = pd.read_csv(summary)
        df["condition"] = cond_dir.name
        frames.append(df)
    if not frames:
        raise RuntimeError(f"No summary files found under {data_root}")
    return pd.concat(frames, ignore_index=True)


def _pivot(df: pd.DataFrame, value: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pivot = df.pivot(index="Ribo", columns="RNAP", values=value)
    pivot = pivot.sort_index().sort_index(axis=1)
    x = pivot.columns.to_numpy()
    y = pivot.index.to_numpy()
    X, Y = np.meshgrid(x, y)
    Z = pivot.to_numpy()
    return X, Y, Z


def plot_heatmap(df: pd.DataFrame, value: str, out_path: Path) -> None:
    X, Y, Z = _pivot(df, value)
    fig, ax = plt.subplots(figsize=(7, 5))
    mesh = ax.pcolormesh(X, Y, Z, shading="auto", cmap="viridis")
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(value)
    ax.set_xlabel("RNAP total")
    ax.set_ylabel("Ribo total")
    ax.set_title(f"{df['condition'].iloc[0]}: {value} heatmap")
    ax.set_xticks(np.unique(X))
    ax.set_yticks(np.unique(Y))
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_surface(df: pd.DataFrame, value: str, out_path: Path) -> None:
    X, Y, Z = _pivot(df, value)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none", alpha=0.9)
    ax.set_xlabel("RNAP total")
    ax.set_ylabel("Ribo total")
    ax.set_zlabel(value)
    ax.set_title(f"{df['condition'].iloc[0]}: {value} surface")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_contour(df: pd.DataFrame, value: str, out_path: Path) -> None:
    X, Y, Z = _pivot(df, value)
    fig, ax = plt.subplots(figsize=(7, 5))
    contour = ax.contourf(X, Y, Z, levels=15, cmap="viridis")
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label(value)
    ax.set_xlabel("RNAP total")
    ax.set_ylabel("Ribo total")
    ax.set_title(f"{df['condition'].iloc[0]}: {value} contour")
    ax.set_xticks(np.unique(X))
    ax.set_yticks(np.unique(Y))
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_lines(df: pd.DataFrame, value: str, ylabel: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for rn in sorted(df["RNAP"].unique()):
        sub = df[df["RNAP"] == rn].sort_values("Ribo")
        ax.plot(sub["Ribo"], sub[value], marker="o", label=f"RNAP {rn}")
    ax.set_xlabel("Ribo total")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{df['condition'].iloc[0]}: {value} vs Ribo")
    ax.legend(title="RNAP", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_condition_panels(df: pd.DataFrame, value: str, out_path: Path) -> None:
    conditions = sorted(df["condition"].unique())
    fig, axes = plt.subplots(1, len(conditions), figsize=(5 * len(conditions), 4), sharey=True)
    if len(conditions) == 1:
        axes = [axes]
    for ax, cond in zip(axes, conditions):
        sub = df[df["condition"] == cond]
        for rn in sorted(sub["RNAP"].unique()):
            seg = sub[sub["RNAP"] == rn].sort_values("Ribo")
            ax.plot(seg["Ribo"], seg[value], marker="o", label=f"RNAP {rn}")
        ax.set_title(cond)
        ax.set_xlabel("Ribo total")
        ax.grid(True, linestyle="--", alpha=0.3)
    axes[0].set_ylabel(value)
    axes[-1].legend(title="RNAP", bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.suptitle(f"{value} vs Ribo across conditions")
    fig.tight_layout(rect=[0, 0, 0.92, 0.95])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_condition_comparison(df: pd.DataFrame, ribo: int, value: str, out_path: Path) -> None:
    sub = df[df["Ribo"] == ribo]
    if sub.empty:
        raise ValueError(f"No data for Ribo={ribo}")
    fig, ax = plt.subplots(figsize=(6, 4))
    for cond in sorted(sub["condition"].unique()):
        seg = sub[sub["condition"] == cond].sort_values("RNAP")
        ax.plot(seg["RNAP"], seg[value], marker="o", label=cond)
    ax.set_xlabel("RNAP total")
    ax.set_ylabel(value)
    ax.set_title(f"{value} vs RNAP at Ribo={ribo}")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate plots for sweep summaries")
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parent / "data",
                        help="Path to data directory that contains Cond*/summary_cond*.csv")
    parser.add_argument("--plots", type=Path, default=Path(__file__).resolve().parent / "plots",
                        help="Output directory for plots")
    parser.add_argument("--ribo", type=int, default=None,
                        help="Ribo value to use for cross-condition RNAP comparisons (defaults to middle value)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_data(args.data)
    args.plots.mkdir(exist_ok=True)

    for cond in sorted(data["condition"].unique()):
        sub = data[data["condition"] == cond]
        plot_heatmap(sub, "CV2", args.plots / f"{cond.lower()}_cv2_heatmap.png")
        plot_surface(sub, "CV2", args.plots / f"{cond.lower()}_cv2_surface.png")
        plot_contour(sub, "CV2", args.plots / f"{cond.lower()}_cv2_contour.png")
        plot_lines(sub, "CV2", "CV²", args.plots / f"{cond.lower()}_cv2_lines.png")
        plot_lines(sub, "mean", "Mean GFP", args.plots / f"{cond.lower()}_mean_lines.png")

    plot_condition_panels(data, "CV2", args.plots / "cv2_line_panels.png")

    ribo_values = sorted(data["Ribo"].unique())
    ribo_value = args.ribo if args.ribo is not None else ribo_values[len(ribo_values) // 2]
    plot_condition_comparison(data, ribo_value, "CV2", args.plots / f"cv2_vs_rnap_ribo{ribo_value}.png")
    plot_condition_comparison(data, ribo_value, "mean", args.plots / f"mean_vs_rnap_ribo{ribo_value}.png")


if __name__ == "__main__":
    main()
