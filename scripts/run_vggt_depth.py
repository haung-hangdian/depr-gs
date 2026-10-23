"""Run VGGT-Omega (or original VGGT) on a folder of images and dump a geometry bundle.

Local checkpoint required; this script does not hit HuggingFace at runtime.

  VGGT-Omega (recommended):  --model omega --ckpt checkpoints/vggt_omega_1b_512.pt
  Original VGGT:             --model vggt  --ckpt checkpoints/model.pt

Downloading checkpoints (gated, one-time):
  hf download facebook/VGGT-Omega vggt_omega_1b_512.pt --local-dir checkpoints
  hf download facebook/VGGT-1B model.pt --local-dir checkpoints
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def _load_ckpt(path: str):
    import torch
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # torch < 2.0 has no weights_only argument
        return torch.load(path, map_location="cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default="",
                    help="local .pt checkpoint path (required unless --hf-repo is given)")
    ap.add_argument("--model", choices=["omega", "vggt"], default="omega",
                    help="omega: VGGT-Omega in third_party/vggt-omega; "
                         "vggt: original VGGT in third_party/vggt")
    ap.add_argument("--hf-repo", default="",
                    help="HF repo id, e.g. facebook/VGGT-1B (original VGGT only)")
    ap.add_argument("--res", type=int, default=512,
                    help="VGGT-Omega image resolution (512 / 416 / 256)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if not args.ckpt and not args.hf_repo:
        raise SystemExit("provide --ckpt (local checkpoint) or --hf-repo")

    import torch

    device = args.device if torch.cuda.is_available() else "cpu"
    names = sorted(str(p) for p in Path(args.images).glob("*")
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not names:
        raise SystemExit(f"no images in {args.images}")

    if args.model == "omega":
        from vggt_omega.models import VGGTOmega
        from vggt_omega.utils.load_fn import load_and_preprocess_images
        from vggt_omega.utils.pose_enc import encoding_to_camera

        model = VGGTOmega().to(device).eval()
        model.load_state_dict(_load_ckpt(args.ckpt))
        images = load_and_preprocess_images(
            names, image_resolution=args.res).to(device)
        with torch.inference_mode():
            predictions = model(images)
        extrinsics, intrinsics = encoding_to_camera(
            predictions["pose_enc"], predictions["images"].shape[-2:])
    else:
        import vggt
        from vggt.models.vggt import VGGT
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        if args.hf_repo:
            model = VGGT.from_pretrained(args.hf_repo).to(device).eval()
        else:
            # VGGT() resolves its config via a path relative to the repo root,
            # so retry once from third_party/vggt if plain construction fails.
            try:
                model = VGGT()
            except Exception:
                root = Path(vggt.__file__).resolve().parents[1]
                cwd = os.getcwd()
                os.chdir(str(root))
                try:
                    model = VGGT()
                finally:
                    os.chdir(cwd)
            model = model.to(device).eval()
            model.load_state_dict(_load_ckpt(args.ckpt))
        images = load_and_preprocess_images(names).to(device)
        with torch.no_grad():
            predictions = model(images)
        img_hw = (predictions.get("images", images).shape[-2:])
        extrinsics, intrinsics = pose_encoding_to_extri_intri(
            predictions["pose_enc"], img_hw)

    # squeeze(0) drops the batch dim when present and is a no-op otherwise
    depths = predictions["depth"].squeeze(0)
    confs = predictions["depth_conf"].squeeze(0)
    extr = extrinsics.squeeze(0)
    intr = intrinsics.squeeze(0)

    os.makedirs(args.out, exist_ok=True)
    cameras = {}
    for i, name in enumerate(names):
        base = Path(name).stem
        np.save(os.path.join(args.out, f"{base}_depth.npy"),
                np.squeeze(depths[i].cpu().numpy()))
        np.save(os.path.join(args.out, f"{base}_conf.npy"),
                np.squeeze(confs[i].cpu().numpy()))
        cameras[name] = {
            "extrinsics": extr[i].cpu().numpy().tolist(),
            "intrinsics": intr[i].cpu().numpy().tolist(),
            "width": int(images.shape[-1]),
            "height": int(images.shape[-2]),
        }

    with open(os.path.join(args.out, "cameras.json"), "w", encoding="utf-8") as f:
        json.dump(cameras, f, indent=2)
    if "camera_and_register_tokens" in predictions:
        np.save(os.path.join(args.out, "registers.npy"),
                predictions["camera_and_register_tokens"].cpu().numpy())
    print(f"bundle saved to {args.out} ({len(names)} views)")


if __name__ == "__main__":
    main()
