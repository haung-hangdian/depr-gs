"""DEPR-GS training loop.

Reimplements the official 3DGS loop with the dynamics-entropy gate injected.
Upstream package is expected at third_party/gaussian-splatting (see setup.ps1).
Only the photometric / geometric routes are required; the object prior is
admitted through --prior-ply + --prior-mask and densification freezing.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

UPSTREAM = Path(__file__).resolve().parent / "third_party" / "gaussian-splatting"
sys.path.insert(0, str(UPSTREAM))
# simple-knn is built in-place by setup.sh; keep the source dir importable
sys.path.insert(0, str(UPSTREAM / "submodules" / "simple-knn"))

import torch
from random import randint
from tqdm import tqdm

from scene import GaussianModel, Scene
from gaussian_renderer import render
from utils.general_utils import safe_state
from utils.image_utils import psnr as _psnr
from utils.loss_utils import l1_loss, ssim

from depr.depth_io import load_bundle
from depr.gate import DynamicsGate


def load_depth_bundle(path, names):
    if not path or not os.path.isdir(path):
        return None, None
    depths, confs, _ = load_bundle(path, names)
    return depths, confs


def resize_depth(depth, h, w):
    d = torch.from_numpy(depth).float()[None, None]
    d = torch.nn.functional.interpolate(d, size=(h, w), mode="bilinear",
                                        align_corners=False)[0, 0]
    return d


def render_depth(pc: GaussianModel, viewpoint):
    """Alpha-composited depth via override_color = normalized view-space z."""
    means = pc.get_xyz
    w2c = viewpoint.world_view_transform
    z_cam = (means @ w2c[:3, :3].T + w2c[:3, 3])[:, 2]
    z_min, z_max = z_cam.min(), z_cam.max()
    z_norm = (z_cam - z_min) / (z_max - z_min + 1e-6)
    color = z_norm[:, None].expand(-1, 3).contiguous()
    depth_img = render(viewpoint, pc, None, override_color=color)["render"]
    return depth_img.mean(dim=0, keepdim=True) * (z_max - z_min) + z_min


def freeze_prior_densify(viewspace_points, proj2d, prior_mask):
    """Zero view-space gradients for gaussians projecting into prior regions."""
    if prior_mask is None or proj2d is None:
        return
    h, w = prior_mask.shape[-2:]
    x = proj2d[:, 0].long()
    y = proj2d[:, 1].long()
    valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    if viewspace_points.grad is None:
        return
    with torch.no_grad():
        viewspace_points.grad[valid & (prior_mask[y[valid], x[valid]] > 0.5)] = 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--source-path", required=True)
    ap.add_argument("-m", "--model-path", required=True)
    # Attributes required by the upstream Scene loader (graphdeco gaussian-splatting).
    ap.add_argument("--images", default="images")
    ap.add_argument("--depths", default="")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--train-test-exp", action="store_true")
    ap.add_argument("-r", "--resolution", type=int, default=1)
    ap.add_argument("--iterations", type=int, default=30000)
    ap.add_argument("--test-iterations", nargs="+", type=int,
                    default=[7000, 15000, 30000])
    ap.add_argument("--save-iterations", nargs="+", type=int, default=[30000])
    ap.add_argument("--data-device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--white-background", action="store_true")
    ap.add_argument("--position-lr-init", type=float, default=0.00016)
    ap.add_argument("--densify-from", type=int, default=500)
    ap.add_argument("--densify-until", type=int, default=15000)
    ap.add_argument("--densification-interval", type=int, default=100)
    ap.add_argument("--opacity-reset-interval", type=int, default=3000)
    ap.add_argument("--densify-grad-threshold", type=float, default=0.0002)
    # DEPR options
    ap.add_argument("--gate", default="entropy",
                    choices=["entropy", "mean_grad", "depth_conf", "uniform", "none"])
    ap.add_argument("--gate-grid", type=int, default=32)
    ap.add_argument("--gate-window", type=int, default=100)
    ap.add_argument("--depth-path", default="")
    ap.add_argument("--lambda-depth", type=float, default=0.1)
    ap.add_argument("--lambda-dssim", type=float, default=0.2)
    ap.add_argument("--adaptive-w", action=argparse.BooleanOptionalAction,
                    default=True)
    ap.add_argument("--w-fixed", type=float, default=0.5)
    ap.add_argument("--prior-ply", default="")
    ap.add_argument("--prior-mask", default="")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    safe_state(args.seed)
    torch.autograd.set_detect_anomaly(False)
    os.makedirs(args.model_path, exist_ok=True)

    gaussians = GaussianModel(3)
    scene = Scene(args, gaussians)
    gaussians.training_setup()

    # optional object prior: load gaussians from PLY and tag the appended range
    prior_ids = None
    if args.prior_ply:
        from scene.gaussian_model import GaussianModel as GM
        prior_pc = GM(3)
        prior_pc.load_ply(args.prior_ply)
        n0 = gaussians.get_xyz.shape[0]
        gaussians.densification_postfix(prior_pc, torch.zeros(n0, dtype=torch.int))
        prior_ids = torch.arange(n0, gaussians.get_xyz.shape[0], device="cuda")
        print(f"prior appended: {len(prior_ids)} gaussians")

    prior_mask = None
    if args.prior_mask:
        from PIL import Image
        prior_mask = torch.from_numpy(
            (np.asarray(Image.open(args.prior_mask).convert("L")) > 127).astype(np.float32))
        prior_mask = prior_mask[None].to(args.data_device)

    bg = torch.tensor([1, 1, 1] if args.white_background else [0, 0, 0],
                      dtype=torch.float32, device="cuda")
    viewpoint_stack = None
    ema_loss = 0.0
    gate = DynamicsGate(grid=(args.gate_grid, args.gate_grid),
                        window=args.gate_window,
                        refresh_every=100,
                        w_grid=[0.2, 0.4, 0.6, 0.8, 1.0])
    progress = tqdm(range(1, args.iterations + 1), desc="Training",
                    disable=args.quiet)

    for iteration in progress:
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        depths, confs = load_depth_bundle(args.depth_path,
                                          [viewpoint_cam.image_name])

        render_pkg = render(viewpoint_cam, gaussians, None, bg)
        image, viewspace_points, visibility_filter, radii = (
            render_pkg["render"], render_pkg["viewspace_points"],
            render_pkg["visibility_filter"], render_pkg["radii"])
        gt = viewpoint_cam.original_image.to(args.data_device)

        per_pixel = (image - gt).abs().mean(dim=0, keepdim=True)
        gate.update(per_pixel.squeeze(0).detach().cpu().numpy())
        masks = gate.masks((gt.shape[1], gt.shape[2]))
        w_photo = torch.from_numpy(masks["photo"]).float()[None].to(args.data_device)
        w_geo = torch.from_numpy(masks["geo"]).float()[None].to(args.data_device)
        w_prior = torch.from_numpy(masks["prior"]).float()[None].to(args.data_device)

        loss = 0.0
        # photometric route: always on, down-weighted outside photo regions
        photo_scale = 0.15
        w_p = 1.0 if args.gate in ("none", "uniform") else (
            w_photo + (1 - w_photo) * photo_scale)
        per_pixel_l1 = (image - gt).abs().mean(dim=0, keepdim=True)
        loss = loss + (per_pixel_l1 * w_p).sum() / w_p.sum()
        loss = loss + args.lambda_dssim * (1.0 - ssim(image, gt))

        # geometric route
        if depths is not None and args.gate not in ("none",):
            d_gt = resize_depth(depths[viewpoint_cam.image_name],
                                gt.shape[1], gt.shape[2]).to(args.data_device)
            conf = resize_depth(confs[viewpoint_cam.image_name],
                                gt.shape[1], gt.shape[2]).to(args.data_device)
            d_render = render_depth(gaussians, viewpoint_cam)
            if args.gate == "uniform":
                w_g = torch.ones_like(w_geo)
            else:
                w_g = w_geo + w_prior * (args.w_fixed if not args.adaptive_w else 1.0)
            d_err = ((d_render - d_gt).abs() * conf * w_g).sum() / w_g.sum().clamp(min=1e-6)
            loss = loss + args.lambda_depth * d_err

        loss.backward()

        with torch.no_grad():
            proj2d = None
            if args.prior_mask and viewspace_points is not None:
                xyz = gaussians.get_xyz
                p_h = xyz @ viewpoint_cam.full_proj_transform[:3, :3].T + \
                    viewpoint_cam.full_proj_transform[:3, 3]
                p_h = p_h / p_h[:, -1:].clamp(min=1e-8)
                proj2d = torch.stack([
                    (p_h[:, 0] + 1) * gt.shape[2] / 2,
                    (1 - p_h[:, 1]) * gt.shape[1] / 2], dim=-1)
                freeze_prior_densify(viewspace_points, proj2d, prior_mask)
            ema_loss = 0.4 * loss.item() + 0.6 * ema_loss if iteration > 1 \
                else loss.item()
            if iteration < args.densify_until:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_points, visibility_filter)
                if iteration > args.densify_from and \
                        iteration % args.densification_interval == 0:
                    size_threshold = 20 if iteration > args.opacity_reset_interval \
                        else None
                    gaussians.densify_and_prune(args.densify_grad_threshold, 0.005,
                                                scene.cameras_extent, size_threshold)
                if iteration % args.opacity_reset_interval == 0:
                    gaussians.reset_opacity()
            gaussians.update_learning_rate(iteration)
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)

        if iteration % 1000 == 0 and not args.quiet:
            progress.set_postfix({"loss": f"{ema_loss:.5f}",
                                  "gate": f"{gate.overhead_ms:.1f}ms"})

    print(f"training finished: {args.model_path}")


if __name__ == "__main__":
    main()
