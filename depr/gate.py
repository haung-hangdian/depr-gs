"""Dynamics-Entropy Prior Routing.

The gate reads its evidence from the optimization trajectory itself: per-region
photometric residuals are accumulated over a window of iterations, and their
entropy separates (i) stably observed regions, (ii) stably unobserved regions,
and (iii) unstable regions. All operations are numpy so the module can be
tested offline.
"""

from __future__ import annotations

from collections import deque

import numpy as np


def _entropy(values: np.ndarray) -> float:
    total = values.sum()
    if total <= 1e-12:
        return 0.0
    p = values / total
    p = p[p > 1e-12]
    return float(-(p * np.log(p)).sum())


def _downsample(x: np.ndarray, grid: tuple[int, int]) -> np.ndarray:
    """Mean-pool an (H, W) array into grid cells; borders handled by clip."""
    h, w = x.shape
    gy, gx = grid
    y = np.clip((np.arange(gy + 1) * h / gy).astype(int), 0, h)
    xb = np.clip((np.arange(gx + 1) * w / gx).astype(int), 0, w)
    out = np.zeros((gy, gx), dtype=np.float32)
    for i in range(gy):
        for j in range(gx):
            out[i, j] = x[y[i] : y[i + 1], xb[j] : xb[j + 1]].mean()
    return out


def _upsample(m: np.ndarray, hw: tuple[int, int], smooth: int = 0) -> np.ndarray:
    """Nearest-upsample a (gy, gx) map to (H, W); optional box smoothing."""
    h, w = hw
    out = np.repeat(np.repeat(m, int(np.ceil(h / m.shape[0])), axis=0),
                    int(np.ceil(w / m.shape[1])), axis=1)[:h, :w]
    if smooth > 0:
        out = _box_blur(out, smooth)
    return out.astype(np.float32)


def _box_blur(x: np.ndarray, k: int) -> np.ndarray:
    """Separable box filter via running sums (no scipy dependency)."""
    out = x.astype(np.float32)
    for axis in (0, 1):
        pad_l, pad_r = k // 2, (k - 1) - k // 2
        pad = ((pad_l, pad_r), (0, 0)) if axis == 0 else ((0, 0), (pad_l, pad_r))
        p = np.pad(out, pad, mode="edge")
        c = np.cumsum(p, axis=axis)
        if axis == 0:
            prev = np.concatenate([np.zeros((1, c.shape[1]), dtype=c.dtype),
                                   c[:-1, :]], axis=0)
            out = (c[(k - 1):, :] - prev[:c.shape[0] - (k - 1), :]) / k
        else:
            prev = np.concatenate([np.zeros((c.shape[0], 1), dtype=c.dtype),
                                   c[:, :-1]], axis=1)
            out = (c[:, (k - 1):] - prev[:, :c.shape[1] - (k - 1)]) / k
    return out


class DynamicsGate:
    """Per-region routing among photometric / geometric / prior supervision."""

    PHOTO, GEO, PRIOR = 0, 1, 2

    def __init__(self, grid=(32, 32), window=100, ema=0.9,
                 inst_z=1.0, mu_quantile=0.7, demand_z=2.5,
                 w_grid=(0.2, 0.4, 0.6, 0.8, 1.0),
                 refresh_every=100, smooth=3):
        self.grid = grid
        self.window = window
        self.ema = ema
        self.inst_z = inst_z
        self.mu_quantile = mu_quantile
        self.demand_z = demand_z
        self.w_grid = list(w_grid)
        self.refresh_every = refresh_every
        self.smooth = smooth
        self.demand_on = False
        self.traces: dict[tuple[int, int], deque] = {}
        self.mu: dict[tuple[int, int], float] = {}
        self.ent: dict[tuple[int, int], float] = {}
        self.state: dict[tuple[int, int], int] = {}
        self.w_prior: dict[tuple[int, int], float] = {}
        self._masks = None
        self.overhead_ms = 0.0

    def _keys(self):
        return [(i, j) for i in range(self.grid[0]) for j in range(self.grid[1])]

    def update(self, residual_abs: np.ndarray):
        """Absorb one iteration's residual map; refresh routing if due."""
        import time
        t0 = time.perf_counter()
        region = _downsample(residual_abs.astype(np.float32), self.grid)
        for key in self._keys():
            i, j = key
            v = float(region[i, j])
            if key not in self.mu:
                self.mu[key] = v
            else:
                self.mu[key] = self.ema * self.mu[key] + (1 - self.ema) * v
            tr = self.traces.setdefault(key, deque(maxlen=self.window))
            tr.append(v)
        self._refresh()
        self.overhead_ms = (time.perf_counter() - t0) * 1e3

    def _refresh(self):
        ent_arr = np.zeros((self.grid[0], self.grid[1]), dtype=np.float32)
        mu_arr = np.zeros_like(ent_arr)
        ref_scale = max(1e-6, max((max(tr, default=0.0) for tr in self.traces.values()),
                                  default=1e-6))
        for key in self._keys():
            tr = self.traces[key]
            if len(tr) < 2:
                ent_arr[key] = 0.0
            else:
                arr = np.asarray(tr, dtype=np.float32)
                # normalized temporal dispersion: instability = std/(mean + floor)
                inst = arr.std() / (arr.mean() + 0.05 * ref_scale)
                ent_arr[key] = float(inst)
            mu_arr[key] = self.mu[key]

        # Robust per-image normalization: instability is judged relative to the
        # image's own distribution via median/MAD z-scores.
        inst_median = float(np.median(ent_arr))
        mad = float(np.median(np.abs(ent_arr - inst_median)))
        z_inst = (ent_arr - inst_median) / (1.4826 * mad + 1e-8)

        # Global prior demand: only a heavy instability tail (a few regions far
        # more unstable than the rest) justifies switching the depth route on.
        z95 = float(np.quantile(z_inst, 0.95))
        self.demand_on = z95 >= self.demand_z

        mu_thr = float(np.quantile(mu_arr, self.mu_quantile))
        for key in self._keys():
            e, m = z_inst[key], mu_arr[key]
            self.ent[key] = float(e)
            if e > self.inst_z:
                self.state[key] = self.GEO
            elif m >= mu_thr:
                self.state[key] = self.PHOTO
            else:
                self.state[key] = self.PRIOR

        if not self.demand_on:
            for key in self._keys():
                self.w_prior[key] = 0.0
            return

        # residual distribution entropy S0 over region means
        s0 = _entropy(mu_arr.flatten() + 1e-8)
        # adaptive prior strength: pick w maximizing distribution shift (AIF Eq.5)
        best_w, best_shift = 0.5, -1.0
        for w in self.w_grid:
            candidate = mu_arr.copy()
            for key in self._keys():
                if self.state[key] == self.PRIOR:
                    candidate[key] = mu_arr[key] * w
            shift = abs(_entropy(candidate.flatten() + 1e-8) - s0)
            if shift > best_shift:
                best_shift, best_w = shift, w
        for key in self._keys():
            self.w_prior[key] = best_w if self.state[key] == self.PRIOR else 0.0

    def masks(self, hw: tuple[int, int]) -> dict[str, np.ndarray]:
        """Return image-sized float masks: photo (0/1), geo (0/1), prior (0/w)."""
        p = np.zeros(self.grid, dtype=np.float32)
        g = np.zeros(self.grid, dtype=np.float32)
        pr = np.zeros(self.grid, dtype=np.float32)
        for key in self._keys():
            s = self.state.get(key, self.PHOTO)
            if s == self.PHOTO:
                p[key] = 1.0
            elif s == self.GEO:
                g[key] = 1.0 if self.demand_on else 0.0
            else:
                pr[key] = self.w_prior.get(key, 0.5)
        return {
            "photo": _upsample(p, hw, self.smooth),
            "geo": _upsample(g, hw, self.smooth),
            "prior": _upsample(pr, hw, self.smooth),
        }

    def state_map(self) -> np.ndarray:
        out = np.zeros(self.grid, dtype=np.uint8)
        for key in self._keys():
            out[key] = self.state.get(key, self.PHOTO)
        return out
