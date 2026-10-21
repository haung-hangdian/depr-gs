"""Run VGGT-Omega on a folder of images and dump a geometry bundle.

Requires the gated VGGT-Omega checkpoint on HuggingFace:
  hf download facebook/vggt-omega --include "checkpoints/*" --exclude "*.onnx"
Set HF_TOKEN before running.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    from vggt_omega.models.vggt_omega import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import pose_encoding_to_camera

    names = sorted(str(p) for p in Path(args.images).glob("*")
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not names:
        raise SystemExit(f"no images in {args.images}")

    device = args.device if torch.cuda.is_available() else "cpu"
    model = VGGTOmega.from_pretrained("facebook/vggt-omega")
    model = model.to(device).eval()

    images = load_and_preprocess_images(names).to(device)
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    with torch.no_grad():
        predictions = model(images, dtype=dtype)

    os.makedirs(args.out, exist_ok=True)
    depths = predictions["depth"]  # (N,H,W)
    confs = predictions["depth_conf"]
    pose_enc = predictions["pose_enc"]

    cameras = {}
    for i, name in enumerate(names):
        base = Path(name).stem
        np.save(os.path.join(args.out, f"{base}_depth.npy"),
                depths[i].cpu().numpy())
        np.save(os.path.join(args.out, f"{base}_conf.npy"),
                confs[i].cpu().numpy())
        extri, intri = pose_encoding_to_camera(pose_enc[i].unsqueeze(0))
        cameras[name] = {
            "extrinsics": extri[0].cpu().numpy().tolist(),
            "intrinsics": intri[0].cpu().numpy().tolist(),
            "width": int(images.shape[-1]),
            "height": int(images.shape[-2]),
        }

    import json
    with open(os.path.join(args.out, "cameras.json"), "w", encoding="utf-8") as f:
        json.dump(cameras, f, indent=2)
    if "camera_and_register_tokens" in predictions:
        np.save(os.path.join(args.out, "registers.npy"),
                predictions["camera_and_register_tokens"].cpu().numpy())
    print(f"bundle saved to {args.out} ({len(names)} views)")


if __name__ == "__main__":
    main()
