"""Draw the DEPR-GS project pipeline figure (PNG + PDF)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams["axes.unicode_minus"] = False

ORANGE_FILL, ORANGE_EDGE = "#FDEBD2", "#E07B1A"
ORANGE_STRONG, ORANGE_DARK = "#FADCB4", "#C96A0E"
BLUE_FILL, BLUE_EDGE = "#E9F0FA", "#4A78B4"
GRAY_FILL, GRAY_EDGE = "#F3F3F3", "#8C8C8C"
GREEN_FILL, GREEN_EDGE = "#E8F5EC", "#3E8E5A"
BAND_FILL, BAND_EDGE = "#FDF5E9", "#D89B57"

fig, ax = plt.subplots(figsize=(18, 11.8))
ax.set_xlim(0, 180)
ax.set_ylim(0, 118)
ax.axis("off")


def box(x, y, w, h, title, body, fc, ec, fs=10.5, title_fs=11.5, dashed=False,
        lw=1.5):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.4,rounding_size=1.2",
        linewidth=lw, facecolor=fc, edgecolor=ec,
        linestyle="--" if dashed else "-", zorder=3))
    cx, cy = x + w / 2, y + h / 2
    if body:
        n = len(body) + 1
        ax.text(cx, cy + (n - 2) * 2.15, title, ha="center", va="center",
                fontsize=title_fs, fontweight="bold", color="#222", zorder=4)
        ax.text(cx, cy - 1.2, "\n".join(body), ha="center", va="center",
                fontsize=fs, color="#333", linespacing=1.45, zorder=4)
    else:
        ax.text(cx, cy, title, ha="center", va="center", fontsize=title_fs,
                fontweight="bold", color="#222", zorder=4)


def arrow(p0, p1, dashed=False, color="#555", lw=1.5, z=2):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=14,
        linestyle="--" if dashed else "-", color=color,
        linewidth=lw, shrinkA=1, shrinkB=1, zorder=z))


def elbow(points, dashed=False, color="#555", lw=1.5):
    for a, b in zip(points[:-2], points[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, linewidth=lw,
                linestyle="--" if dashed else "-", zorder=2)
    arrow(points[-2], points[-1], dashed=dashed, color=color, lw=lw)


def tag(x, y, label, color):
    ax.add_patch(FancyBboxPatch(
        (x, y), 9, 9, boxstyle="round,pad=0.3,rounding_size=1.0",
        facecolor=color, edgecolor="none", zorder=5))
    ax.text(x + 4.5, y + 4.5, label, ha="center", va="center", rotation=90,
            fontsize=11, fontweight="bold", color="white", zorder=6)


# ---------- band 1: data ----------
box(58, 104, 64, 8, "公共数据集  images/",
    ["Mip-NeRF360 · NeRF On-The-Go"], GRAY_FILL, GRAY_EDGE)
box(12, 104, 36, 8, "VGGT 权重",
    ["model.pt（本地加载）"], GRAY_FILL, GRAY_EDGE)

# ---------- band 2: geometry bundle ----------
box(46, 86, 58, 12, "run_vggt_depth.py",
    ["每图回归 位姿 + 相对深度", "输出 vggt/ 目录"], BLUE_FILL, BLUE_EDGE)
box(120, 86, 48, 12, "vggt_to_colmap.py",
    ["深度反投影 → 稀疏点云", "写 COLMAP sparse/0"], BLUE_FILL, BLUE_EDGE)
box(46, 75, 58, 8, "深度监督", ["vggt/*_depth.npy"], GREEN_FILL, GREEN_EDGE)
box(120, 75, 48, 8, "初始化", ["sparse/0 点云"], GREEN_FILL, GREEN_EDGE)

# ---------- band 3: gated training ----------
ax.add_patch(FancyBboxPatch(
    (8, 30), 164, 42, boxstyle="round,pad=0.4,rounding_size=1.5",
    facecolor=BAND_FILL, edgecolor=BAND_EDGE, linewidth=1.4, zorder=0))
ax.text(90, 69.2, "train_depr.py —— 门控训练循环（本文核心）",
        ha="center", va="center", fontsize=13, fontweight="bold",
        color=ORANGE_DARK, zorder=1)

box(14, 48, 26, 10, "渲染 + 残差", ["render()", "|I − GT|"], GRAY_FILL, GRAY_EDGE)
box(48, 40, 44, 26, "DynamicsGate 动态熵门控",
    ["32×32 区域 · 100 步残差轨迹",
     "熵判不稳定度（优化轨迹自证）",
     "三态：PHOTO / GEO / PRIOR",
     "自适应先验强度 w ∈ {0.2…1.0}"],
    ORANGE_STRONG, ORANGE_DARK, title_fs=12)
box(100, 54, 32, 10, "光度路线", ["L1 + D-SSIM", "区域权重 w_p"],
    ORANGE_FILL, ORANGE_EDGE)
box(100, 40, 32, 12, "几何路线", ["|d_render − d_vggt|", "门控权重 w_g",
                                "λ_depth"], ORANGE_FILL, ORANGE_EDGE)
box(48, 31, 44, 6.5, "可选：对象先验",
    ["SAM3D PLY 注入 · prior mask 冻结增密"], ORANGE_FILL, ORANGE_EDGE,
    dashed=True, fs=9, title_fs=10)
box(140, 38, 28, 12, "损失与更新", ["Σ loss → backward", "densify · prune"],
    GRAY_FILL, GRAY_EDGE)
box(12, 67, 32, 7, "对照开关  --gate",
    ["none · uniform · entropy · fixed_w"], "#FFF8EE", ORANGE_DARK,
    dashed=True, fs=9.5, title_fs=10)

box(72, 21, 36, 8, "训练模型",
    ["output/<scene>_<gate>"], GREEN_FILL, GREEN_EDGE)

# ---------- band 4: evaluation ----------
box(60, 11, 60, 9, "render.py", ["渲染测试视图（上游脚本适配）"],
    GRAY_FILL, GRAY_EDGE)
box(128, 21, 40, 6, "SAM2 权重 + 提示点", None, GRAY_FILL, GRAY_EDGE,
    fs=9.5, title_fs=9.5)
box(128, 11, 40, 9, "make_sam2_masks.py", ["提示点 → 遮挡区掩码"],
    GRAY_FILL, GRAY_EDGE)
box(66, 1, 84, 8, "eval_masked.py（本文）",
    ["全图 PSNR · masked PSNR/SSIM"], ORANGE_FILL, ORANGE_EDGE)
box(154, 1, 24, 8, "metrics.csv", ["→ 论文主表"], GREEN_FILL, GREEN_EDGE,
    fs=9, title_fs=10)

# ---------- arrows ----------
arrow((90, 104), (75, 98))
arrow((30, 104), (58, 98))
arrow((104, 92), (120, 92))
arrow((75, 86), (75, 83))
arrow((144, 86), (144, 83))
elbow([(120, 79), (8, 79), (8, 53), (14, 53)])
ax.text(4.6, 66, "images + sparse/0（Scene 初始化）", rotation=90,
        ha="center", va="center", fontsize=8.5, color="#555", zorder=2)
arrow((75, 75), (116, 64), dashed=True)
arrow((44, 70), (54, 66), dashed=True, color=ORANGE_DARK)
arrow((40, 53), (48, 53))
arrow((92, 59), (100, 59))
arrow((92, 46), (100, 46))
arrow((132, 59), (140, 49))
arrow((132, 46), (140, 44))
arrow((48, 34), (30, 48), dashed=True, color=ORANGE_DARK)
elbow([(154, 38), (154, 29.5), (90, 29.5), (90, 29)])
arrow((90, 21), (90, 20))
arrow((148, 21), (148, 20))
arrow((90, 11), (90, 9))
arrow((148, 11), (148, 9))
arrow((150, 5), (154, 5))

# ---------- stage tags ----------
tag(0.5, 105, "① 数据", "#8C8C8C")
tag(0.5, 86, "② 几何束 · 无 COLMAP", BLUE_EDGE)
tag(0.5, 47, "③ 门控训练 · 本文核心", ORANGE_DARK)
tag(0.5, 13, "④ 遮挡评估", GREEN_EDGE)

# ---------- legend ----------
legend = [
    (ORANGE_STRONG, ORANGE_DARK, "本文贡献"),
    (BLUE_FILL, BLUE_EDGE, "本文脚本 + 上游模型"),
    (GRAY_FILL, GRAY_EDGE, "现成组件"),
    (GREEN_FILL, GREEN_EDGE, "中间 / 最终产物"),
]
for i, (fc, ec, label) in enumerate(legend):
    yy = 113.5 - i * 3.1
    ax.add_patch(FancyBboxPatch(
        (132, yy), 2.2, 2.2, boxstyle="round,pad=0.15,rounding_size=0.6",
        facecolor=fc, edgecolor=ec, linewidth=1.2, zorder=5))
    ax.text(136.5, yy + 1.1, label, ha="left", va="center", fontsize=9,
            color="#333", zorder=5)

ax.text(4, 4.2, "render.py：训练后渲染测试视图，需按上游 gaussian-splatting 脚本适配；仓库内待接线。",
        fontsize=7.8, color="#777", va="bottom")
ax.text(4, 2.4, "H1：uniform vs none 证先验污染；H2：entropy vs uniform 证按需路由；H3：fixed_w 消融证自适应强度。",
        fontsize=7.8, color="#777", va="bottom")

fig.savefig("docs/figures/pipeline.png", dpi=220, bbox_inches="tight",
            facecolor="white")
fig.savefig("docs/figures/pipeline.pdf", bbox_inches="tight",
            facecolor="white")
print("saved docs/figures/pipeline.png / pipeline.pdf")
