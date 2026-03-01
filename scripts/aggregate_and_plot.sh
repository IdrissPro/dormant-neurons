#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/aggregate_and_plot.sh
#
# Assumes you ran experiments and have logs under ./runs/
# Produces:
#   ./out/scalars.csv, ./out/runs_meta.csv
#   ./figs/*.png

RUNS_DIR="${RUNS_DIR:-runs}"
OUT_DIR="${OUT_DIR:-out}"
FIGS_DIR="${FIGS_DIR:-figs}"

echo "Aggregating JSONL logs from: ${RUNS_DIR}"
python -m src.analysis.aggregate --runs_dir "${RUNS_DIR}" --out_dir "${OUT_DIR}" --include_text --text_prefix "repr/svd_topk"

echo "Plotting figures from: ${OUT_DIR}"
python -m src.analysis.plots --in_dir "${OUT_DIR}" --out_dir "${FIGS_DIR}" --group_by env algo

echo "Done."
echo " - Scalars: ${OUT_DIR}/scalars.csv"
echo " - Meta:    ${OUT_DIR}/runs_meta.csv"
echo " - Figs:    ${FIGS_DIR}/"