#!/usr/bin/env bash
#
# Full matrix: every normalised strategy on every instrument its universe names.
# 815 specs x their admissible symbols = ~7 776 runs. This coverage has never
# been produced — every earlier stage ran three markets per strategy.
#
# Two phases, and the split is the point:
#
#   1. Run everything, write nothing but a compact row per task. A run directory
#      per task would be thousands of directories holding a few numbers each —
#      the cost P15 stage 1.5 already paid once.
#   2. Re-run a bounded, market-covering selection and store those in full, so
#      `ts report index` has something to catalogue. Re-running is deterministic
#      and each row's stored digest makes the reconstruction checkable.
#
# The selection is by cross-sectional z *within each instrument*, not by
# drawdown. Measured on the delivered screen: rows under 10% drawdown were
# positive at 42.2% against a base rate of 42.0%, with a worse median
# expectancy and a third of the trades — the threshold selects strategies that
# barely traded, and it moves with --risk rather than with the strategy.
#
# What this is: reconnaissance. One window, the parameters each spec names, no
# folds, no selection, no null. The project's holdout is already spent
# (CLAUDE.md, "Решения P22 этап 3"), so nothing here can be confirmed.
#
# Usage:
#   scripts/run_all.sh [WORKERS] [TOP_PER_MARKET] [BARS]
#
#   WORKERS         processes                              (default: CPU count)
#   TOP_PER_MARKET  runs kept per (instrument, timeframe)  (default: 5)
#   BARS            bars per task                          (default: 20000)
#
# Resumable: rows are appended as they land. Interrupting costs the task in
# flight; re-running the same command continues.

set -euo pipefail

cd "$(dirname "$0")/.."

WORKERS="${1:-$( (command -v nproc >/dev/null && nproc) || sysctl -n hw.ncpu )}"
TOP="${2:-5}"
BARS="${3:-20000}"
SPECS="strategies/test_strategies/specs"

# --- preconditions, each checked because each has actually gone wrong -------

[ -x .venv/bin/ts ] || {
  echo "No .venv/bin/ts — run 'make install' first." >&2; exit 1; }
[ -d "$SPECS" ] || {
  echo "No $SPECS. It is not in git; copy it over, or regenerate with:" >&2
  echo "  .venv/bin/ts strategy normalize strategies/test_strategies" >&2
  echo "(regenerating needs strategies/scraped_strategies_v3/, also not in git)" >&2
  exit 1; }
[ -d data/ohlcv ] || {
  echo "No data/ohlcv — the parquet store is gitignored and must be copied." >&2; exit 1; }

SPEC_COUNT=$(find "$SPECS" -name '*.json' | wc -l | tr -d ' ')
SYMBOLS=$(find data/ohlcv -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')

echo "==============================================================="
echo " full matrix: $SPEC_COUNT specs x all admissible symbols"
echo " store: $SYMBOLS symbols    workers: $WORKERS    bars/task: $BARS"
echo " keeping top $TOP per (instrument, timeframe) in phase 2"
echo "==============================================================="
echo

START=$(date +%s)

# --- phase 1: everything, compact rows only --------------------------------
# --symbols 0 = every instrument the spec's universe names and the store holds.
# No --keep-max-dd: nothing is written to runs/ here, on purpose.
echo ">>> phase 1: running the full matrix (rows only, no run directories)"
.venv/bin/ts strategy screen "$SPECS" \
  --symbols 0 \
  --workers "$WORKERS" \
  --bars "$BARS" \
  --returns-sample 200 \
  --out reports/screen_all.html

SCREEN_ID=$(ls -t runs/sweep | head -1)
echo
echo ">>> phase 1 done. screen id: $SCREEN_ID"
echo

# --- phase 2: store a bounded, market-covering selection -------------------
echo ">>> phase 2: re-running the top $TOP per (instrument, timeframe) to store them"
.venv/bin/ts strategy keep-top "$SPECS" \
  --screen-id "$SCREEN_ID" \
  --top "$TOP" \
  --min-trades 100 \
  --workers "$WORKERS"

echo
echo ">>> building the catalogue"
.venv/bin/ts report index --out reports/runs/index.html || {
  echo "report index failed; the screen page is still at reports/screen_all.html" >&2; }

ELAPSED=$(( $(date +%s) - START ))
echo
echo "==============================================================="
echo " done in $((ELAPSED / 60)) min $((ELAPSED % 60)) s"
echo "   screen page : reports/screen_all.html"
echo "   catalogue   : reports/runs/index.html"
echo "   all rows    : runs/sweep/$SCREEN_ID/rows.parquet"
echo "   kept rows   : runs/sweep/$SCREEN_ID/kept_top$TOP/rows.parquet"
echo "==============================================================="
echo
echo "Read the screen page's header before quoting any number from it:"
echo "these are hypotheses ordered for a walk-forward, not results."
