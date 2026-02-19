from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


@dataclass
class RunMeta:
    run_dir: str
    algo: Optional[str] = None
    env: Optional[str] = None
    seed: Optional[int] = None


def find_metrics_files(runs_dir: str) -> List[str]:
    out: List[str] = []
    for root, _dirs, files in os.walk(runs_dir):
        if "metrics.jsonl" in files:
            out.append(os.path.join(root, "metrics.jsonl"))
    return sorted(out)


def infer_meta_from_path(run_dir: str) -> RunMeta:
    # heuristic: runs/<env>/<algo>/<timestamp_seedX>...
    parts = os.path.normpath(run_dir).split(os.sep)
    meta = RunMeta(run_dir=run_dir)
    if len(parts) >= 3:
        meta.algo = parts[-2]
        meta.env = parts[-3]
    # seed sometimes embedded in dirname
    base = os.path.basename(run_dir)
    if "seed" in base:
        # find trailing digits
        import re
        m = re.search(r"seed(\d+)", base)
        if m:
            meta.seed = int(m.group(1))
    return meta


def read_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def aggregate_scalars(metrics_path: str) -> pd.DataFrame:
    """
    Returns tidy DF with columns:
      run_dir, time, step, tag, value
    Includes:
      - type="scalar": one record per line
      - type="scalars": dict flattened per line
    """
    run_dir = os.path.dirname(metrics_path)
    rows: List[Dict] = []

    for rec in read_jsonl(metrics_path):
        rtype = rec.get("type", "")
        if rtype == "scalar":
            rows.append(
                {
                    "run_dir": run_dir,
                    "time": rec.get("time"),
                    "step": rec.get("step"),
                    "tag": rec.get("tag"),
                    "value": rec.get("value"),
                }
            )
        elif rtype == "scalars":
            step = rec.get("step")
            t = rec.get("time")
            vals = rec.get("values", {}) or {}
            for tag, value in vals.items():
                rows.append({"run_dir": run_dir, "time": t, "step": step, "tag": tag, "value": value})

    return pd.DataFrame(rows)


def aggregate_text(metrics_path: str, tag_prefixes: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Collects type="text" records. Useful for storing svd_topk lists or config text.
    Columns: run_dir, time, step, tag, text
    """
    run_dir = os.path.dirname(metrics_path)
    rows: List[Dict] = []
    prefixes = tag_prefixes or []
    for rec in read_jsonl(metrics_path):
        if rec.get("type") != "text":
            continue
        tag = rec.get("tag", "")
        if prefixes and not any(tag.startswith(p) for p in prefixes):
            continue
        rows.append(
            {
                "run_dir": run_dir,
                "time": rec.get("time"),
                "step": rec.get("step"),
                "tag": tag,
                "text": rec.get("text"),
            }
        )
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", type=str, default="runs", help="Root dir containing run subfolders.")
    ap.add_argument("--out_dir", type=str, default="out", help="Output directory.")
    ap.add_argument("--include_text", action="store_true", help="Also export text logs (e.g., svd_topk).")
    ap.add_argument("--text_prefix", type=str, action="append", default=[], help="Tag prefix to include for text logs.")
    args = ap.parse_args()

    paths = find_metrics_files(args.runs_dir)
    if not paths:
        raise SystemExit(f"No metrics.jsonl found under {args.runs_dir}")

    os.makedirs(args.out_dir, exist_ok=True)

    # Aggregate scalars
    dfs = []
    metas = []
    for mp in paths:
        run_dir = os.path.dirname(mp)
        df = aggregate_scalars(mp)
        meta = infer_meta_from_path(run_dir)
        metas.append(meta.__dict__)
        dfs.append(df)

    scalars = pd.concat(dfs, ignore_index=True)
    scalars.to_csv(os.path.join(args.out_dir, "scalars.csv"), index=False)

    runs_meta = pd.DataFrame(metas).drop_duplicates(subset=["run_dir"])
    runs_meta.to_csv(os.path.join(args.out_dir, "runs_meta.csv"), index=False)

    # Optional text export
    if args.include_text:
        tdfs = []
        for mp in paths:
            tdfs.append(aggregate_text(mp, tag_prefixes=args.text_prefix))
        texts = pd.concat(tdfs, ignore_index=True) if tdfs else pd.DataFrame()
        texts.to_csv(os.path.join(args.out_dir, "texts.csv"), index=False)

    print(f"Wrote: {os.path.join(args.out_dir, 'scalars.csv')}")
    print(f"Wrote: {os.path.join(args.out_dir, 'runs_meta.csv')}")
    if args.include_text:
        print(f"Wrote: {os.path.join(args.out_dir, 'texts.csv')}")


if __name__ == "__main__":
    main()

