"""Generate object/occluder masks with SAM2 for masked evaluation.

Prompt points are passed per scene through a prompts.json:
  {"image_name": [{"label": "occluder", "points": [[x, y]]}, ...]}
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--source", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--checkpoint", default="checkpoints/sam2.1_hiera_large.pt")
    ap.add_argument("--model-config", default="sam2.1_hiera_l.yaml")
    args = ap.parse_args()
    args.out = args.out or os.path.join(args.source, "masks")

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    predictor = SAM2ImagePredictor(build_sam2(
        args.model_config, args.checkpoint, device="cuda"))
    with open(args.prompts, encoding="utf-8") as f:
        prompts = json.load(f)

    os.makedirs(args.out, exist_ok=True)
    for img_name, entries in prompts.items():
        img = np.asarray(Image.open(os.path.join(args.source, img_name)).convert("RGB"))
        predictor.set_image(img)
        for entry in entries:
            label = entry["label"]
            pts = np.asarray(entry["points"], dtype=np.float32)
            labels = np.ones(len(pts), dtype=np.int32)
            mask, _, _ = predictor.predict(point_coords=pts, point_labels=labels)
            mask = (mask[0] > 0).astype(np.uint8) * 255
            out = os.path.join(args.out, f"{Path(img_name).stem}_{label}.png")
            Image.fromarray(mask).save(out)
    print(f"masks saved to {args.out}")


if __name__ == "__main__":
    main()
