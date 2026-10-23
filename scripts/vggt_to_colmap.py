"""Convert a VGGT-Omega bundle to COLMAP sparse/0 (cameras, poses, sparse points).

The point cloud for Gaussian initialization is built by unprojecting the
predicted depths and subsampling, so 3DGS can be trained without COLMAP.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from depr.colmap_io import write_model


def qvec_from_rot(r: np.ndarray) -> np.ndarray:
    # world-to-camera rotation -> quaternion (w,x,y,z)
    trace = np.trace(r)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
    else:
        s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z], dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vggt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-points", type=int, default=200000)
    ap.add_argument("--sample-step", type=int, default=4)
    args = ap.parse_args()

    with open(os.path.join(args.vggt, "cameras.json"), encoding="utf-8") as f:
        cams = json.load(f)

    names = sorted(cams.keys())
    cameras, images = {}, {}
    pts, cols = [], []
    rng = np.random.default_rng(0)

    for i, name in enumerate(names):
        meta = cams[name]
        e = np.asarray(meta["extrinsics"])  # (3,4) world-to-camera
        k = np.asarray(meta["intrinsics"])  # (3,3)
        r, t = e[:3, :3], e[:3, 3]
        w, h = int(meta["width"]), int(meta["height"])
        fx, fy, cx, cy = float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])
        cameras[i + 1] = (w, h, fx, fy, cx, cy)
        q = qvec_from_rot(r)
        images[i + 1] = (q, t, i + 1, os.path.basename(name))

        # VGGT-Omega saves depth as (H, W, 1); squeeze guards either layout.
        depth = np.squeeze(np.load(os.path.join(args.vggt, f"{Path(name).stem}_depth.npy")))
        conf = np.squeeze(np.load(os.path.join(args.vggt, f"{Path(name).stem}_conf.npy")))
        v, u = np.mgrid[0:h:args.sample_step, 0:w:args.sample_step]
        z = depth[::args.sample_step, ::args.sample_step]
        c = conf[::args.sample_step, ::args.sample_step]
        keep = (z > 1e-3) & (c > 0.3) & np.isfinite(z)
        x = (u[keep] - cx) * z[keep] / fx
        y = (v[keep] - cy) * z[keep] / fy
        cam = np.stack([x, y, z[keep]], axis=-1)
        world = cam @ r.T + t
        pts.append(world)
        cols.append(np.tile(np.array([200, 200, 200]), (world.shape[0], 1)))

    pts = np.concatenate(pts, axis=0)
    cols = np.concatenate(cols, axis=0)
    if len(pts) > args.max_points:
        idx = rng.choice(len(pts), args.max_points, replace=False)
        pts, cols = pts[idx], cols[idx]

    write_model(args.out, cameras, images, pts, cols.astype(np.uint8))
    print(f"COLMAP model written to {args.out}: {len(names)} views, {len(pts)} points")


if __name__ == "__main__":
    main()
