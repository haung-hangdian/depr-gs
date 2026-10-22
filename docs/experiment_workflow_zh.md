# DEPR-GS 实验流程手册

> 适用环境：Linux（AutoDL 推荐 PyTorch 2.x + CUDA 12 镜像），单卡 24GB
> 状态：研究原型。本文档中的命令与仓库当前代码一一对应，标注"未接线"的步骤表示代码尚未实现，请勿先跑。

## 0. 总览

主证据链只有四步，全部使用公共数据集：

1. VGGT-Omega 出位姿 + 深度 + 置信度（几何束）
2. 几何束转 COLMAP `sparse/0`，作为 3DGS 的初始化
3. `train_depr.py` 在熵门控（或对照变体）下训练
4. 遮挡区掩码评估（masked PSNR / SSIM）

目标数据集：Mip-NeRF360（bicycle、garden、room 起步）、NeRF On-The-Go（高/低遮挡分档）。

## 1. 环境安装

```bash
cd ~
git clone https://github.com/<你的账号>/depr-gs.git
cd depr-gs

# AutoDL：先开学术加速，解决 GitHub 访问
source /etc/network_turbo

# 自检工具链（git/python/pip/gcc/torch/nvcc），有缺失会直接列出
bash setup.sh --self-test

# 先看脚本将要执行什么（不执行任何安装）
bash setup.sh --dry-run

# 正式安装（pip 步骤会自动剥掉代理，防加速干扰 PyPI）
bash setup.sh
```

常见问题：

- 报 `setuptools>=68 ... from versions: none`：pip 源被加速代理干扰。本版脚本已默认使用清华源并剥代理；若还出现，`export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple` 后重跑。
- 报 `No module named 'torch'`（构建隔离）：本版脚本已统一加 `--no-build-isolation`，不该再出现；若手动 pip 时遇到，同样加 `--no-build-isolation`。
- `nvcc` 找不到：镜像的 CUDA 在 `/usr/local/cuda`，脚本会自动加 PATH；`conda install -y -c nvidia cuda-nvcc=12.1` 可兜底。
- `sam-3d-objects` / `sam2` 克隆失败：它们被标为可选，失败只会告警，不阻塞主流程。

## 2. 权重准备（本地 checkpoint，运行时不再访问 HuggingFace）

把权重文件放到 `checkpoints/`，推理时用 `--ckpt` 直接加载：

```bash
mkdir -p checkpoints

# 情况 A：原版 VGGT（你已拿到的 model.pt / vggt_model.bin）
# 放到 checkpoints/ 下，运行时加 --model vggt

# 情况 B：VGGT-Omega（可选，1B-512）
# 文件例如 vggt_omega_1b_512.pt，放到 checkpoints/ 下
```

如果还没有权重、需要一次性下载（gated 仓库需先申请并登录）：

```bash
export HF_TOKEN=<你的 HuggingFace token>
huggingface-cli login --token $HF_TOKEN
# 原版 VGGT：
hf download facebook/VGGT-1B model.pt --local-dir checkpoints
# VGGT-Omega（可选）：
hf download facebook/VGGT-Omega vggt_omega_1b_512.pt --local-dir checkpoints
```

SAM2 掩码评估用（官方直链，无需申请）：

```bash
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt \
     -O checkpoints/sam2.1_hiera_large.pt
```

SAM 3D（对象先验，可选，非主证据链）：

```bash
hf download facebook/sam-3d-objects --include "checkpoints/*"
```

## 3. 数据准备

下载地址（On-The-Go 原项目页已失效，改用 HuggingFace 镜像）：

- Mip-NeRF360：<https://jonbarron.info/mipnerf360/>，官方直链见下方命令
- NeRF On-The-Go：HuggingFace 镜像 `onground/robustvggt` 的 `onthego/` 子集

下载命令（AutoDL 上先 `source /etc/network_turbo`）：

```bash
cd data
# Mip-NeRF360 全部 9 场景，约 7 GB
curl -O http://storage.googleapis.com/gresearch/refraw360/360_v2.zip
unzip -q 360_v2.zip
mv 360_v2 mipnerf360   # 解压得到 360_v2/ 目录，重命名对齐下方布局

# NeRF On-The-Go（12 场景原图，国内走 hf-mirror 加速）
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download onground/robustvggt --repo-type dataset \
  --include "onthego/*" --local-dir on_the_go
# 场景目录形如 on_the_go/onthego/on-the-go/<scene>/images/
```

两个数据集共用同一套目录约定（与官方 3DGS 数据读取一致）。`bicycle` 只是
Mip-NeRF360 里用作示例的一个场景，其余场景完全同构：

```text
data/
  mipnerf360/                  # Mip-NeRF360，9 个场景
    bicycle/                   # 示例场景
      images/                  # 该场景全部图片，jpg 或 png
      vggt/                    # run_vggt_depth.py 输出（深度 + 位姿）
      sparse/0/                # vggt_to_colmap.py 输出（COLMAP 二进制模型）
      masks/                   # SAM2 掩码输出
    garden/
    room/
    ...                        # stump/kitchen/bonsai/counter/flowers/treehill
  on_the_go/                   # NeRF On-The-Go，12 个场景
    mountain/
      images/
      vggt/
      sparse/0/
      masks/
    fountain/
    ...                        # corner/patio/spot/patio-high 等，见下载后目录
```

镜像下载的 On-The-Go 目录会多两级前缀（`onthego/on-the-go/<scene>/images/`），
直接把该 `images/` 路径传给 `--images` 即可，无需移动文件。

稀疏视图协议（可选，做"few-view 抗遮挡"实验时用）：

```bash
python scripts/make_sparse_split.py --images data/mipnerf360/bicycle/images \
  --num 12 --out-list data/mipnerf360/bicycle/sparse_train.txt
```

## 4. 几何束：VGGT -> COLMAP

```bash
# 原版 VGGT（已有权重放 checkpoints/model.pt）：
python scripts/run_vggt_depth.py \
  --images data/mipnerf360/bicycle/images --out data/mipnerf360/bicycle/vggt \
  --model vggt --ckpt checkpoints/model.pt

# 备选 VGGT-Omega（每张图额外输出 {name}_conf.npy）：
python scripts/run_vggt_depth.py \
  --images data/mipnerf360/bicycle/images --out data/mipnerf360/bicycle/vggt \
  --ckpt checkpoints/vggt_omega_1b_512.pt --res 512

# 注意：原版 VGGT 与 VGGT-Omega 输出的都是尺度无关的相对深度。
# 初始化点云与深度监督都来自同一份深度，尺度天然一致，可放心使用；
# 不要把这份深度与其他来源的度量深度混用。

# 位姿 + 深度反投影出点云，写 COLMAP sparse/0（无 COLMAP 也能训练 3DGS）
python scripts/vggt_to_colmap.py \
  --vggt data/mipnerf360/bicycle/vggt --out data/mipnerf360/bicycle/sparse/0 \
  --max-points 200000 --sample-step 4
```

跑完自检：

```bash
ls data/mipnerf360/bicycle/vggt/*_depth.npy | wc -l   # 应等于图片数
python -c "import json; print(len(json.load(open('data/mipnerf360/bicycle/vggt/cameras.json'))))"
ls data/mipnerf360/bicycle/sparse/0                   # cameras.bin images.bin points3D.bin
```

## 5. 训练：五条对照曲线

先跑离线单元测试确认门控模块正常（无需 GPU）：

```bash
python tests/test_gate.py
```

Pilot（每变体 7k 迭代，约 10 分钟/场景）：

```bash
python train_depr.py -s data/mipnerf360/bicycle -m output/bicycle_pilot_none \
  --gate none --iterations 7000 --quiet

python train_depr.py -s data/mipnerf360/bicycle -m output/bicycle_pilot_entropy \
  --gate entropy --depth-path data/mipnerf360/bicycle/vggt --iterations 7000 --quiet
```

正式训练（30k 迭代，约 40 分钟/场景）：

```bash
# 基线：纯光度 3DGS
python train_depr.py -s data/mipnerf360/bicycle -m output/bicycle_none --gate none

# 对照：均匀深度先验
python train_depr.py -s data/mipnerf360/bicycle -m output/bicycle_uniform \
  --gate uniform --depth-path data/mipnerf360/bicycle/vggt

# ours：熵门控路由
python train_depr.py -s data/mipnerf360/bicycle -m output/bicycle_entropy \
  --gate entropy --depth-path data/mipnerf360/bicycle/vggt

# 消融：固定先验权重（关闭自适应 w）
python train_depr.py -s data/mipnerf360/bicycle -m output/bicycle_fixedw \
  --gate entropy --depth-path data/mipnerf360/bicycle/vggt --no-adaptive-w --w-fixed 0.5
```

训练完成输出在 `output/<模型名>/`。每 1000 迭代进度条会显示 EMA loss 与门控耗时。

## 6. 遮挡区评估（SAM2 掩码 + masked PSNR）

先准备提示点文件 `data/mipnerf360/bicycle/masks/prompts.json`，每张测试图给一个遮挡物点：

```json
{
  "frame_0000.png": [{"label": "occluder", "points": [[512, 300]]}]
}
```

```bash
python scripts/make_sam2_masks.py -s data/mipnerf360/bicycle/images \
  --prompts data/mipnerf360/bicycle/masks/prompts.json \
  --out data/mipnerf360/bicycle/masks \
  --checkpoint checkpoints/sam2.1_hiera_large.pt \
  --model-config third_party/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml

# renders 是测试视图渲染目录（见第 8 节：需先用上游 render.py 适配后渲染）
# gt 指向 images/ 即可：测试图是其中按 8:1 抽出的子集，按文件名自动匹配
python scripts/eval_masked.py \
  --renders output/bicycle_entropy/test/ours_30000 \
  --gt data/mipnerf360/bicycle/images --masks data/mipnerf360/bicycle/masks \
  --out-csv output/bicycle_entropy/metrics.csv
```

指标：`psnr`（全图）、`masked_psnr` / `masked_ssim`（仅遮挡区，主指标）。

## 7. 消融矩阵与命令对应

| 论文变体 | 命令行 | 状态 |
|---|---|---|
| 基线 3DGS | `--gate none` | 可跑 |
| 均匀先验 | `--gate uniform --depth-path ...` | 可跑 |
| ours（熵门控） | `--gate entropy --depth-path ...` | 可跑 |
| evidence_mean | `--gate mean_grad --depth-path ...` | 未接线：与 entropy 同证据，待实现 |
| evidence_conf | `--gate depth_conf --depth-path ...` | 未接线：同上 |
| fixed_w | `--gate entropy --depth-path ... --no-adaptive-w --w-fixed 0.5` | 可跑 |
| global_ratio | （全局粒度门控） | 未实现，暂不跑 |

对象先验（SAM 3D）：

```bash
python scripts/run_sam3d_prior.py --image data/mipnerf360/bicycle/images/frame_0000.png \
  --mask data/mipnerf360/bicycle/masks/frame_0000_occluder.png \
  --out-ply output/prior_0000.ply --config <pipeline.yaml> --seed 42
```

注意：脚本内位姿对齐仍是 TODO，当前只验证推理通路，PLY 暂不参与训练。

## 8. 已知缺口（先别在这些点上浪费时间）

- `train_depr.py` 的 `--test-iterations` / `--save-iterations` 已定义但**未在循环中接线**：目前不保存测试渲染和中间 checkpoint。第 6 节的 `eval_masked.py` 需要渲染目录，跑评估前要补上"训练后渲染测试视图"的代码（参考上游 `gaussian-splatting` 的 `render.py`）。
- `--gate mean_grad` 与 `--gate depth_conf` 目前使用与 entropy 相同的残差证据，三者在数学上等价；实现证据切换前，这两行消融没有信息量。
- SAM 3D 输出与几何束的对齐（相似变换）未实现。
- VGGT-Omega / VGGT 权重为 gated，申请未通过时全链路会被阻塞；脚本已支持本地 checkpoint（`--ckpt`），拿到文件后无需运行时联网。

## 9. 建议执行顺序（先跑哪个）

1. `bash setup.sh --self-test`，确认 torch + nvcc 都在
2. `bash setup.sh`，以 `--dry-run` 先过一遍再实装
3. `python tests/test_gate.py`，门控模块离线自检
4. bicycle 上跑 VGGT 深度 + COLMAP 转换，肉眼/脚本核对相机数和点数
5. 7k pilot：`none` vs `entropy`，比训练 loss 轨迹
6. 30k 正式五条曲线（第 5 节命令）
7. 补 test 渲染接线 -> SAM2 掩码 -> masked PSNR
8. garden、room、On-The-Go 场景复制同一协议

## 10. 常见运行错误

- `Could not find a version that satisfies the requirement setuptools>=68`：镜像/代理问题，换 `PIP_INDEX_URL` 并去掉 `http_proxy` 环境变量。
- `ModuleNotFoundError: No module named 'torch'`（PEP 517 构建隔离）：任何 `pip install -e` 都加 `--no-build-isolation`。
- `nvcc fatal / No CUDA toolchain`：确认 `nvcc --version`；没有就 `conda install -y -c nvidia cuda-nvcc=12.1`。
- simple-knn editable 安装成功后 `import` 仍报 `No module named`：其 hatchling 后端不会编译 CUDA 扩展，改用 `cd third_party/gaussian-splatting/submodules/simple-knn && python setup.py build_ext --inplace`，`train_depr.py` 会自动把该目录加入 `sys.path`。
- OOM：`--resolution 2` 半分辨率跑 pilot；正式表只标全分辨率场景。
- HF `403 / gated repo`：权重申请未通过或 `HF_TOKEN` 未导出。
