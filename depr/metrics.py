"""Metric helpers used by eval scripts."""

from __future__ import annotations

import numpy as np


def psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    mse = float(np.mean((pred.astype(np.float64) - gt.astype(np.float64)) ** 2))
    if mse < 1e-10:
        return float("inf")
    return float(10.0 * np.log10(255.0**2 / mse))


def masked_psnr(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    m = mask > 0.5
    if m.sum() < 16:
        return float("nan")
    diff = (pred[m].astype(np.float64) - gt[m].astype(np.float64)) ** 2
    mse = float(diff.mean())
    if mse < 1e-10:
        return float("inf")
    return float(10.0 * np.log10(255.0**2 / mse))


def masked_ssim(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    """Box-filter SSIM on the masked region (luminance channel)."""
    m = mask > 0.5
    if m.sum() < 64:
        return float("nan")
    a = pred.astype(np.float64)[..., :3].mean(-1)
    b = gt.astype(np.float64)[..., :3].mean(-1)
    a[~m], b[~m] = 0.0, 0.0
    win = 7
    mu_a = _box(a, win)
    mu_b = _box(b, win)
    sig_a = _box(a * a, win) - mu_a * mu_a
    sig_b = _box(b * b, win) - mu_b * mu_b
    sig_ab = _box(a * b, win) - mu_a * mu_b
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ssim_map = ((2 * mu_a * mu_b + c1) * (2 * sig_ab + c2)) / (
        (mu_a * mu_a + mu_b * mu_b + c1) * (sig_a + sig_b + c2))
    return float(ssim_map[m].mean())


def _box(x: np.ndarray, k: int) -> np.ndarray:
    out = x.astype(np.float64)
    for axis in (0, 1):
        pad_l, pad_r = k // 2, (k - 1) - k // 2
        pad = ((pad_l, pad_r), (0, 0)) if axis == 0 else ((0, 0), (pad_l, pad_r))
        p = np.pad(out, pad, mode="edge")
        c = np.cumsum(p, axis=axis)
        if axis == 0:
            out = (c[k - 1:, :] - c[:-(k - 1), :]) / k
        else:
            out = (c[:, k - 1:] - c[:, :-(k - 1)]) / k
    return out


def report(name: str, psnr: float, ssim: float, lpips: float | None = None) -> str:
    row = f"{name:24s} PSNR {psnr:7.3f}  SSIM {ssim:6.4f}"
    if lpips is not None:
        row += f"  LPIPS {lpips:.4f}"
    return row
