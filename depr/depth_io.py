"""Depth/confidence bundles on disk."""

from __future__ import annotations

import json
import os

import numpy as np


def save_bundle(out_dir: str, names: list[str], depths: list[np.ndarray],
                confs: list[np.ndarray], cameras: dict) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for name, dep, conf in zip(names, depths, confs):
        base = os.path.splitext(os.path.basename(name))[0]
        np.save(os.path.join(out_dir, f"{base}_depth.npy"), dep)
        np.save(os.path.join(out_dir, f"{base}_conf.npy"), conf)
    with open(os.path.join(out_dir, "cameras.json"), "w", encoding="utf-8") as f:
        json.dump(cameras, f, indent=2)


def load_bundle(out_dir: str, names: list[str]) -> tuple[dict, dict, dict]:
    depths, confs = {}, {}
    for name in names:
        base = os.path.splitext(os.path.basename(name))[0]
        depths[name] = np.load(os.path.join(out_dir, f"{base}_depth.npy"))
        confs[name] = np.load(os.path.join(out_dir, f"{base}_conf.npy"))
    with open(os.path.join(out_dir, "cameras.json"), encoding="utf-8") as f:
        cameras = json.load(f)
    return depths, confs, cameras
