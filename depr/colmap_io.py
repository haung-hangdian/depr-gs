"""Minimal COLMAP sparse/0 binary writers (PINHOLE cameras)."""

from __future__ import annotations

import os
import struct

import numpy as np


def _write_next(f, fmt, *args):
    f.write(struct.pack("<" + fmt, *args))


def write_cameras_bin(path: str, cameras: dict[int, tuple[int, int, float, float, float, float]]):
    # id -> (width, height, fx, fy, cx, cy)
    with open(path, "wb") as f:
        _write_next(f, "Q", len(cameras))
        for cid, (w, h, fx, fy, cx, cy) in cameras.items():
            _write_next(f, "iiQQ", cid, 1, w, h)  # model PINHOLE = 1
            _write_next(f, "dddd", fx, fy, cx, cy)


def write_images_bin(path: str, images: dict[int, tuple]):
    # id -> (qvec[4], tvec[3], camera_id, name)
    with open(path, "wb") as f:
        _write_next(f, "Q", len(images))
        for iid, (qvec, tvec, cid, name) in images.items():
            _write_next(f, "I", iid)
            _write_next(f, "dddd", *qvec)
            _write_next(f, "ddd", *tvec)
            _write_next(f, "I", cid)
            # COLMAP stores the image name as a NUL-terminated byte string.
            f.write(name.encode())
            f.write(b"\x00")
            _write_next(f, "Q", 0)  # no 2D points


def write_points3D_bin(path: str, xyz: np.ndarray, rgb: np.ndarray):
    with open(path, "wb") as f:
        _write_next(f, "Q", len(xyz))
        for i, (p, c) in enumerate(zip(xyz, rgb)):
            _write_next(f, "Q", i)
            _write_next(f, "ddd", float(p[0]), float(p[1]), float(p[2]))
            _write_next(f, "BBB", int(c[0]), int(c[1]), int(c[2]))
            _write_next(f, "d", 1.0)
            _write_next(f, "Q", 0)


def write_model(out_dir: str, cameras: dict, images: dict, xyz: np.ndarray,
                rgb: np.ndarray) -> None:
    os.makedirs(out_dir, exist_ok=True)
    write_cameras_bin(os.path.join(out_dir, "cameras.bin"), cameras)
    write_images_bin(os.path.join(out_dir, "images.bin"), images)
    write_points3D_bin(os.path.join(out_dir, "points3D.bin"), xyz, rgb)
