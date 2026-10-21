# DEPR-GS

Dynamics-Entropy Prior Routing for occlusion-robust 3D Gaussian Splatting.
对应论文方案文档：`../outputs/APF-GS_v2_论文初稿.md`。

## 结构

```text
depr-gs/
  depr/                  # 本项目自有代码（numpy 门控、指标、I/O）
  scripts/               # 数据与推理脚本
  configs/               # 实验配置
  third_party/           # 上游仓库（setup.ps1 拉取，git 忽略）
  train_depr.py          # 训练主循环（含门控注入）
  tests/                 # 离线自测
```

## 快速开始（有网环境）

```powershell
.\setup.ps1                 # 克隆上游 + 建 conda 环境 + 编译 rasterizer
python scripts/run_vggt_depth.py --images data/bicycle/images --out data/bicycle/vggt
python scripts/vggt_to_colmap.py --vggt data/bicycle/vggt --out data/bicycle/sparse/0
python train_depr.py -s data/bicycle --gate entropy --depth-path data/bicycle/vggt
python scripts/make_sam2_masks.py -s data/bicycle
python scripts/eval_masked.py -m output/bicycle --mask masks/bicycle
```

## 上游依赖与许可证

| 仓库 | 用途 | 许可证 |
|---|---|---|
| graphdeco-inria/gaussian-splatting | 基座与 rasterizer | INRIA License |
| facebookresearch/vggt-omega | 位姿/深度/置信 | 见仓库 LICENSE（通常 CC-BY-NC） |
| facebookresearch/sam-3d-objects | 对象先验 | SAM License |
| facebookresearch/sam2 | 可见区/实例掩码 | Apache-2.0 |

上游权重多为 HuggingFace gated，需提前申请；SAM 3D 被卡时用 TRELLIS 或 Hunyuan3D 兜底。
