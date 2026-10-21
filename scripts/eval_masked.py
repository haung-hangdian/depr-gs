"""Masked PSNR/SSIM over rendered test views."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
from PIL import Image

from depr.metrics import masked_psnr, masked_ssim, psnr, report


def load(path, size=None):
    img = np.asarray(Image.open(path).convert("RGB")).astype(np.float32)
    if size is not None:
        img = np.asarray(Image.fromarray(img.astype(np.uint8)).resize(size,
                        Image.Resampling.BILINEAR)).astype(np.float32)
    return img / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--renders", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--mask-suffix", default="occluder")
    ap.add_argument("--out-csv", default="metrics.csv")
    args = ap.parse_args()

    rows = []
    for render_path in sorted(Path(args.renders).glob("*.png")):
        stem = render_path.stem
        gt_path = Path(args.gt) / f"{stem}.png"
        mask_path = Path(args.masks) / f"{stem}_{args.mask_suffix}.png"
        if not (gt_path.exists() and mask_path.exists()):
            continue
        pred, gt = load(render_path), load(gt_path)
        mask = np.asarray(Image.open(mask_path).convert("L")) > 127
        if mask.shape[:2] != pred.shape[:2]:
            mask = np.asarray(Image.fromarray(mask.astype(np.uint8)).resize(
                (pred.shape[1], pred.shape[0]), Image.Resampling.NEAREST)) > 0
        rows.append({
            "image": stem,
            "psnr": round(psnr(pred * 255, gt * 255), 4),
            "masked_psnr": round(masked_psnr(pred * 255, gt * 255, mask), 4),
            "masked_ssim": round(masked_ssim(pred * 255, gt * 255, mask), 4),
        })

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "psnr", "masked_psnr", "masked_ssim"])
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(report(r["image"], r["psnr"], r["masked_ssim"]))
    print(f"masked metrics -> {args.out_csv}")


if __name__ == "__main__":
    main()
