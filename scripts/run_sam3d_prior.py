"""Run SAM 3D for one object and export an aligned Gaussian PLY.

Requires SAM 3D object weights (gated). Pose alignment to the geometry bundle
is implemented by a similarity fit on visible depth (TODO: wire to your depth
bundle loader once checkpoints are available).
"""

from __future__ import annotations

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--mask", required=True)
    ap.add_argument("--out-ply", required=True)
    ap.add_argument("--config", required=True, help="path to pipeline.yaml")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import numpy as np
    from PIL import Image
    from sam3d_objects.inference import Inference

    image = np.asarray(Image.open(args.image).convert("RGB"))
    mask = np.asarray(Image.open(args.mask).convert("L")) > 127
    inference = Inference(args.config)
    output = inference(image, mask, seed=args.seed)
    output["gs"].save_ply(args.out_ply)
    print(f"prior saved to {args.out_ply}; alignment TODO once depth bundle is loaded")


if __name__ == "__main__":
    main()
