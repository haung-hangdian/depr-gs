"""Offline self-tests for the dynamics-entropy gate."""

import numpy as np

from depr.gate import DynamicsGate


def test_three_state_routing():
    rng = np.random.default_rng(0)
    gate = DynamicsGate(grid=(8, 8), window=20)
    h, w = 256, 256
    for _ in range(25):
        residual = np.zeros((h, w), dtype=np.float32)
        # top rows: stably observed (large residual, stable)
        residual[:80] = 0.9 + 0.05 * rng.standard_normal((80, w))
        # middle rows: stably unobserved (tiny residual)
        residual[80:160] = 0.02 + 0.005 * rng.standard_normal((80, w))
        # bottom rows: unstable (alternating magnitudes)
        flip = 1.0 if (_ % 2 == 0) else 0.05
        residual[160:] = flip + 0.1 * rng.standard_normal((96, w))
        gate.update(residual)
    state = gate.state_map()
    assert (state[0, :] == DynamicsGate.PHOTO).all()
    assert (state[3, :] == DynamicsGate.PRIOR).all()
    assert (state[5, :] == DynamicsGate.GEO).all()
    masks = gate.masks((h, w))
    assert masks["photo"].shape == (h, w)
    assert 0.0 < masks["prior"].max() <= 1.0


def test_overhead_recorded():
    gate = DynamicsGate(grid=(4, 4))
    gate.update(np.zeros((64, 64), dtype=np.float32))
    assert gate.overhead_ms >= 0.0


def test_clean_scene_suppresses_depth_route():
    rng = np.random.default_rng(1)
    gate = DynamicsGate(grid=(8, 8), window=20)
    for _ in range(25):
        residual = 0.5 + 0.03 * rng.standard_normal((256, 256)).astype(np.float32)
        gate.update(residual)
    assert not gate.demand_on
    masks = gate.masks((256, 256))
    assert masks["geo"].max() == 0.0
    assert masks["prior"].max() == 0.0
    assert masks["photo"].max() == 1.0


if __name__ == "__main__":
    test_three_state_routing()
    test_overhead_recorded()
    test_clean_scene_suppresses_depth_route()
    print("gate tests passed")
