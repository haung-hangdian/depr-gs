# DEPR-GS

Dynamics-Entropy Prior Routing for occlusion-robust 3D Gaussian Splatting.
The gate decides, per image region, whether the optimizer should trust
photometric observations, geometric priors, or object-level generative priors,
using evidence read from the optimization trajectory itself.

> Status: research prototype. Upstream code is pulled by `setup.ps1`.
> Paper draft and plans: see [`docs/`](docs).

## Repository layout

```text
depr-gs/
  depr/                  # self-contained routing/IO/metrics code (numpy, no torch)
  scripts/               # geometry bundle, COLMAP export, masks, prior, eval
  configs/               # experiment configurations
  docs/                  # paper draft and work plans
  third_party/           # upstream repos, cloned by setup.ps1 (git-ignored)
  train_depr.py          # 3DGS training loop with the gate injected
  tests/                 # offline self-tests (no GPU required)
```

## Quickstart

```powershell
.\setup.ps1                 # clone upstream, create conda env, build rasterizer

python scripts/run_vggt_depth.py --images data/bicycle/images --out data/bicycle/vggt \
    --ckpt checkpoints/vggt_omega_1b_512.pt
# original VGGT weights: add --model vggt --ckpt checkpoints/model.pt
python scripts/vggt_to_colmap.py --vggt data/bicycle/vggt --out data/bicycle/sparse/0

# baselines (same scene, same protocol)
python train_depr.py -s data/bicycle -m output/bicycle_v0      --gate none
python train_depr.py -s data/bicycle -m output/bicycle_uniform --gate uniform --depth-path data/bicycle/vggt
python train_depr.py -s data/bicycle -m output/bicycle_gate    --gate entropy --depth-path data/bicycle/vggt

# masked evaluation (masks generated with SAM2)
python scripts/make_sam2_masks.py -s data/bicycle --prompts masks/prompts.json
python scripts/eval_masked.py --renders output/bicycle_gate/test/ours_30000 \
                              --gt data/bicycle/test --masks data/bicycle/masks
```

Linux (AutoDL / bare-metal):

```bash
bash setup.sh
```

Offline tests (no GPU, no upstream checkout needed):

```powershell
pip install -e . pytest
python tests/test_gate.py
```

## Checkpoints (HuggingFace, gated)

```text
hf download facebook/VGGT-Omega vggt_omega_1b_512.pt --local-dir checkpoints
hf download facebook/VGGT-1B model.pt --local-dir checkpoints
hf download facebook/sam-3d-objects --include "checkpoints/*"
```

SAM2 checkpoint: download `sam2.1_hiera_large.pt` from the official SAM2 release page.

## Upstream dependencies and licenses

| Repository | Role | License |
|---|---|---|
| graphdeco-inria/gaussian-splatting | base and rasterizer | INRIA License |
| facebookresearch/vggt-omega | poses / depth / confidence | see repo LICENSE (typically CC-BY-NC) |
| facebookresearch/sam-3d-objects | object prior | SAM License |
| facebookresearch/sam2 | masks | Apache-2.0 |

This repository itself is MIT licensed (see `LICENSE`). Upstream code and weights
remain under their original licenses; do not redistribute upstream checkpoints.

## Citation

```bibtex
@software{depr_gs,
  title = {DEPR-GS: Dynamics-Entropy Prior Routing for Occlusion-Robust 3D Gaussian Splatting},
  note = {Research prototype},
}
```
