#!/usr/bin/env bash
# Run the depth-prior variants at 30k (none already finished).
set -Eeuo pipefail
cd "$(dirname "$0")/.."

PY=/root/miniconda3/bin/python
[ -x "$PY" ] || PY=python

$PY train_depr.py -s data/mipnerf360/bicycle -m output/bicycle_uniform \
  --gate uniform --depth-path data/mipnerf360/bicycle/vggt \
  > logs/30k_uniform.log 2>&1

$PY train_depr.py -s data/mipnerf360/bicycle -m output/bicycle_entropy \
  --gate entropy --depth-path data/mipnerf360/bicycle/vggt \
  > logs/30k_entropy.log 2>&1

$PY train_depr.py -s data/mipnerf360/bicycle -m output/bicycle_fixedw \
  --gate entropy --depth-path data/mipnerf360/bicycle/vggt \
  --no-adaptive-w --w-fixed 0.5 > logs/30k_fixedw.log 2>&1

echo "ALL_DONE $(date)" > logs/bicycle_30k_done.txt
