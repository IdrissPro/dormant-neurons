from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, ci: float = 0.95, seed: int = 0) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    if n <= 1:
        v = float(values.mean()) if n == 1 else float("nan")
        return v, v
    means = []
    for _ in range(n_boot):
        samp = rng.choice(values, size=n, replace=True)
        means.append(float(np.mean(samp)))
    means = np.array(means)
    lo = np.quantile(means, (1 - ci) / 2)
    hi = np.quantile(means, 1 - (1 - ci) / 2)
    return float(lo), float(hi)


def plot_learning_curve(df: pd.DataFrame, out_path: str, tag: str, group_cols: List[str]) -> None:
    """
    Expects df columns: step, value, plus group_cols and seed-like separation.
    """
    # Determine groups (e.g., env+algo or env+algo+variant)
    grouped = df.groupby(group_cols)

    plt.figure()
    for gname, gdf in grouped:
        # Build mean curve with bootstrap CI at each step
        # Approach: pivot by run_dir so we have per-seed values at each step.
        pivot = gdf.pivot_table(index="step", columns="run_dir", values="value", aggfunc="mean").sort_index()
        steps = pivot.index.values
        means = pivot.mean(axis=1).values

        lo = []
        hi = []
        for i, s in enumerate(steps):
            vals = pivot.loc[s].dropna().values.astype(float)
            if len(vals) == 0:
                lo.append(np.nan)
                hi.append(np.nan)
            else:
                l, h = bootstrap_ci(vals, n_boot=500, ci=0.95, seed=0)
                lo.append(l)
                hi.append(h)

        label = " | ".join(map(str, gname)) if isinstance(gname, tuple) else str(gname)
        plt.plot(steps, means, label=label)
        plt.fill_between(steps, lo, hi, alpha=0.2)

    plt.xlabel("env steps")
    plt.ylabel(tag)
    plt.title(f"Learning curve: {tag}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_metric_trajectory(df: pd.DataFrame, out_path: str, tag: str, group_cols: List[str]) -> None:
    """
    Similar to learning curve but for any scalar metric.
    """
    plot_learning_curve(df, out_path, tag, group_cols)


def plot_layer_heatmap(df: pd.DataFrame, out_path: str, tag_prefix: str, group_filter: Optional[Dict[str, str]] = None) -> None:
    """
    Heatmap of per-layer dormancy fractions across steps for one selected group/run aggregation.
    tag_prefix example:
      "dormancy/activation/actor/layer_frac/"
    """
    d = df[df["tag"].str.startswith(tag_prefix)].copy()
    if group_filter:
        for k, v in group_filter.items():
            d = d[d[k] == v]

    if d.empty:
        return

    # Choose one representative group by taking mean over runs at each step/layer
    d["layer"] = d["tag"].str[len(tag_prefix):]
    piv = d.pivot_table(index="layer", columns="step", values="value", aggfunc="mean")

    plt.figure(figsize=(10, max(3, 0.35 * len(piv.index))))
    plt.imshow(piv.values, aspect="auto")
    plt.colorbar(label="dormant fraction")
    plt.yticks(range(len(piv.index)), piv.index)
    plt.xticks(range(len(piv.columns)), piv.columns, rotation=45, ha="right")
    plt.xlabel("env steps")
    plt.title(f"Heatmap: {tag_prefix}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, default="out", help="Directory containing scalars.csv and runs_meta.csv")
    ap.add_argument("--out_dir", type=str, default="figs", help="Where to save plots")
    ap.add_argument("--group_by", type=str, nargs="*", default=["env", "algo"], help="Columns to group curves by")
    args = ap.parse_args()

    scalars_path = os.path.join(args.in_dir, "scalars.csv")
    meta_path = os.path.join(args.in_dir, "runs_meta.csv")
    if not os.path.exists(scalars_path):
        raise SystemExit(f"Missing {scalars_path}. Run aggregate.py first.")

    df = pd.read_csv(scalars_path)
    if os.path.exists(meta_path):
        meta = pd.read_csv(meta_path)
        df = df.merge(meta, on="run_dir", how="left")
    else:
        # fallback: infer env/algo from path if possible
        df["env"] = df["run_dir"].apply(lambda p: os.path.normpath(p).split(os.sep)[-3] if len(os.path.normpath(p).split(os.sep)) >= 3 else "unknown")
        df["algo"] = df["run_dir"].apply(lambda p: os.path.normpath(p).split(os.sep)[-2] if len(os.path.normpath(p).split(os.sep)) >= 2 else "unknown")

    os.makedirs(args.out_dir, exist_ok=True)

    # Common tags to plot
    tags = [
        "charts/episodic_return",
        "charts/episodic_length",
        "dormancy/activation/global_frac",
        "dormancy/gradient/global_frac",
        "repr/effective_rank",
        "repr/cosine_diversity",
        "repr/cka_to_ref",
        "losses/td_loss",
        "losses/q_loss",
        "losses/actor_loss",
        "losses/approx_kl",
    ]

    # Filter for each tag and plot if present
    for tag in tags:
        d = df[df["tag"] == tag].copy()
        if d.empty:
            continue
        out_path = os.path.join(args.out_dir, f"{tag.replace('/', '__')}.png")
        plot_metric_trajectory(d, out_path, tag, group_cols=args.group_by)

    # Heatmaps (examples)
    heatmap_specs = [
        ("dormancy__activation__actor_heatmap.png", "dormancy/activation/actor/layer_frac/"),
        ("dormancy__activation__critic_heatmap.png", "dormancy/activation/critic/layer_frac/"),
        ("dormancy__activation__shared_heatmap.png", "dormancy/activation/shared/layer_frac/"),
        ("dormancy__activation__q_heatmap.png", "dormancy/activation/layer_frac/"),
    ]
    for fname, prefix in heatmap_specs:
        out_path = os.path.join(args.out_dir, fname)
        plot_layer_heatmap(df, out_path, tag_prefix=prefix)

    print(f"Saved plots to {args.out_dir}/")


if __name__ == "__main__":
    main()

