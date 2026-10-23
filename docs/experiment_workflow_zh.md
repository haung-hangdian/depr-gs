# DEPR-GS 实验流程手册

> 适用环境：Linux（AutoDL 推荐 PyTorch 2.x + CUDA 12 镜像），单卡 24GB
> 状态：研究原型。本文档中的命令与仓库当前代码一一对应，标注"未接线"的步骤表示代码尚未实现，请勿先跑。

## 0. 总览

这篇工作要回答的问题只有一个：**深度先验对 3D 重建是一把双刃剑，如何在
需要它的区域用它，在不该用的区域停用？**

三个待验证的主张（论文里的 claim）：

- H1：均匀注入深度先验能改善遮挡区，但会污染可见区（全图质量不升反降）。
- H2：用优化轨迹上的残差熵做门控，可以把先验只路由到不确定区域（遮挡区），
  实现"该用才用"。
- H3：先验强度的自适应选择优于固定强度。

主证据链只有四步，全部使用公共数据集：

1. VGGT 出位姿 + 深度（几何束）
2. 几何束转 COLMAP `sparse/0`，作为 3DGS 的初始化，同一份深度留作训练监督
3. `train_depr.py` 在熵门控（或对照变体）下训练
4. 遮挡区掩码评估（masked PSNR / SSIM）

四步与主张的对应：第 1-2 步保证所有变体使用同一几何束，排除"初始化不同"的
混淆变量；第 3 步中 `none` vs `uniform` 对应 H1，`uniform` vs `entropy` 对应
H2，`entropy` vs `fixed_w` 对应 H3；第 4 步把训练过程的定性趋势变成定量主指标。

目标数据集分工：Mip-NeRF360 是标准稠密场景，证明"可见区不退化"；NeRF On-The-Go
是真实手持遮挡场景，证明"遮挡区真提升"，并按高/低遮挡分档。

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

**作用**：把 PyTorch、两个 CUDA 扩展（`diff-gaussian-rasterization`、`simple-knn`）、
SAM2 一次性装好，保证 `train_depr.py` 能编译运行。`--self-test` 先查工具链，
`--dry-run` 预览将要执行的安装，正式安装幂等、可重复跑。

**为了证明什么**：这一步不直接产出论文数据，但它保证五条训练曲线在同一环境、
同一版本下比较；环境不一致会让"门控带来的增益"被版本噪声淹没，对照就失去意义。

**应得到什么结果**：`rasterizer OK`、`simple-knn OK`、`gate tests passed` 三行
全部出现。任何一行缺失，后面的指标都不可信，先解决环境再往下走。

AutoDL 数据盘持久化：`setup.sh` 检测到 `/root/autodl-tmp` 时，会自动把
`data/`、`checkpoints/`、`output/`、`logs/` 软链到数据盘（分别对应
`/root/autodl-tmp/data`、`.../checkpoints`、`.../depr-runs/output`、
`.../depr-runs/logs`）。换容器后只需重新 `git clone` 仓库并跑一次 `bash
setup.sh`，之前下载的数据、权重和全部训练结果都在，无需重新下载。

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

**作用**：VGGT 权重在第 4 步生成几何束（位姿 + 相对深度），这份深度同时是
训练监督的来源；SAM2 权重在第 6 步把遮挡物从测试图里抠出来，形成评估掩码。

**为了证明什么**：用预训练 VGGT 取代 COLMAP 做 SfM，是方法"无 COLMAP、开箱
即用"卖点的事实基础；SAM2 掩码是主指标 masked PSNR 的标尺，掩码质量直接决定
主表的可信度。

**应得到什么结果**：`run_vggt_depth.py` 能加载本地 checkpoint 并为每张图输出
`{name}_depth.npy`；SAM2 能从单点提示分割出完整遮挡物轮廓（人工抽查 5-10 张）。

## 3. 数据准备

下载地址（On-The-Go 曾流传的旧链接已失效，以下为当前有效渠道）：

- Mip-NeRF360：<https://jonbarron.info/mipnerf360/>，官方直链见下方命令
- NeRF On-The-Go 官方：项目页 <https://rwn17.github.io/nerf-on-the-go/>，
  仓库 <https://github.com/cvg/nerf-on-the-go>；数据集由官方脚本
  `bash ./scripts/download_on-the-go.sh` 从 Google Drive 拉取（含全部 12 场景）
- NeRF On-The-Go 镜像（国内推荐）：HuggingFace `onground/robustvggt` 的
  `onthego/` 子集；另有一个去畸变版 `jkulhanek/nerfonthego-undistorted`
  （WildGaussians 预处理，6 个定量场景，图像已去畸变、可直接喂 3DGS）

下载命令（AutoDL 上先 `source /etc/network_turbo`）：

```bash
cd data
# Mip-NeRF360（官方 zip 含 COLMAP 稀疏模型，约 12 GB 压缩 / 12.7 GB 解压）
curl -O http://storage.googleapis.com/gresearch/refraw360/360_v2.zip
mkdir -p mipnerf360
unzip -q 360_v2.zip -d mipnerf360   # 场景直接落在 mipnerf360/<scene>/

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
  mipnerf360/                  # Mip-NeRF360；官方 zip 为 7 个场景（flowers/treehill 为 txt 占位，无图）
    bicycle/                   # 示例场景
      images/                  # 该场景全部图片，jpg 或 png
      sparse/0/                # 官方 zip 自带 COLMAP；VGGT 管线可覆盖重写
      vggt/                    # run_vggt_depth.py 输出（深度 + 位姿）
      masks/                   # SAM2 掩码输出
    garden/
    room/
    ...                        # stump/kitchen/bonsai/counter
  on-the-go/                   # NeRF On-The-Go，12 个场景
    mountain/
      images/
      transforms.json split.json   # 官方 zip 自带 COLMAP 导出的位姿
      vggt/
      sparse/0/
      masks/
    fountain/
    ...                        # corner/patio/spot/patio_high/arcdetriomphe/
                               # drone/statue/train/train_station/tree
```

当前 AutoDL 容器里数据已就位：`~/depr-gs/data -> /root/autodl-tmp/data`，
Mip-NeRF360 在 `data/mipnerf360/`（7 场景），On-The-Go 在 `data/on-the-go/`。
注意场景名是 `patio_high`（下划线），不是 `patio-high`。镜像渠道的目录会多
两级前缀（`onthego/on-the-go/<scene>/images/`），把 `images/` 路径传给
`--images` 即可。

On-The-Go 官方遮挡分档（用于 `configs/otg.yaml`）：低遮挡 mountain/fountain，
中遮挡 corner/patio，高遮挡 spot/patio_high；其余 6 个场景为定性展示场景。

稀疏视图协议（可选，做"few-view 抗遮挡"实验时用）：

```bash
python scripts/make_sparse_split.py --images data/mipnerf360/bicycle/images \
  --num 12 --out-list data/mipnerf360/bicycle/sparse_train.txt
```

**作用**：下载并组织两个公共数据集的图像，形成标准 3DGS 目录（`images/` +
稀疏模型 + 深度 + 掩码）。

**为了证明什么**：只用公共数据集、不用自制数据，保证审稿人和读者能 1:1 复现。
两个数据集角色不同：Mip-NeRF360 证明"先验不伤可见区"（标准稠密场景的内部
对照），On-The-Go 证明"遮挡区真的变好"（真实手持遮挡的外部效度）。

**应得到什么结果**：Mip-NeRF360 九个场景齐全，每场景 `images/` 图集完整
（bicycle 约 200 张）；On-The-Go 十二个场景按下载目录核对高/低遮挡两档，每
场景都能定位到 `images/`。可选 few-view 协议下，`sparse_train.txt` 恰好 12 张，
用于"少视图时先验更有价值"的扩展实验。

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

On-The-Go 场景完全同构，例如 mountain：

```bash
python scripts/run_vggt_depth.py \
  --images data/on-the-go/mountain/images --out data/on-the-go/mountain/vggt \
  --ckpt checkpoints/vggt_omega_1b_512.pt --res 512
python scripts/vggt_to_colmap.py \
  --vggt data/on-the-go/mountain/vggt --out data/on-the-go/mountain/sparse/0
```

**作用**：VGGT 直接回归每张图的相机位姿与相对深度；`vggt_to_colmap.py` 把深度
反投影成稀疏点云并写成 COLMAP `sparse/0`，供 3DGS 初始化；同一份深度在训练中
作为几何监督。

**为了证明什么**：一是证明整条管线不依赖 COLMAP/SfM；二是把"初始化和监督
同源同尺度"固定下来，消除"深度尺度不匹配导致训练塌掉"这个混淆变量，让后续
对照的差异只来自门控策略。

**应得到什么结果**：depth 文件数与图片数一致；`cameras.json` 长度等于图片数；
`sparse/0` 下三个二进制文件齐全且点数 > 0（建议 1e5 量级）。点数接近 0 说明
反投影参数有误，训练会退化，必须先修。

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

**作用**：五条曲线构成"一个主对照 + 两个消融"：

- `none`：纯光度 3DGS，作为基线和下界；
- `uniform`：全域均匀深度先验，回答"先验整体上是否有益"（H1 的实证）；
- `entropy`：ours，回答"熵门控路由是否优于均匀注入"（H2）；
- `fixed_w`：关闭自适应先验强度，回答"自适应 w 是否必要"（H3）。

Pilot（7k 迭代）只是冒烟：确认 loss 正常下降、无 NaN、显存与门控耗时在预期内。

**为了证明什么**：训练视角的 loss 是第一层证据。H1 的预期表现是 `uniform` 的
收敛值不一定比 `none` 好（先验把可见区带偏）；H2 的预期表现是 `entropy` 收敛值
不高于 `none` 且不高于 `uniform`（先验只用在需要处）。但训练 loss 只反映训练
视角，真正结论必须等第 6 节的测试视图指标，两层证据合起来才是完整论证。

**应得到什么结果**：三个 run 的 EMA loss 轨迹（每 1000 迭代打印）；`entropy`
的门控耗时在毫秒级，说明路由本身开销可忽略。另外建议补一个最小可视化（导出
`gate.state_map()` 与 `gate.masks()` 为 npy/PNG，当前代码未接线），人工核对
"prior 区域"是否落在遮挡物附近——这是方法行为正确性的定性证据。

训练完成输出在 `output/<模型名>/`。每 1000 迭代进度条会显示 EMA loss 与门控耗时。

分辨率说明：`-r` 默认 `-1`，与官方 3DGS 一致，图像宽度超过 1600px 会自动缩到
1600（本数据集的 bicycle 原图是 4946×3286，全分辨率会把 194 张图全留在显存
导致 OOM，别用 `-r 1`）。显存仍紧张时用 `-r 2` / `-r 4` 半分辨率跑 pilot。

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

**作用**：把训练好的模型在测试视角渲染出来，用 SAM2 掩码圈出遮挡区，分别计算
全图 psnr 与遮挡区 masked_psnr / masked_ssim。这是论文的主表数据。

**为了证明什么**：masked 指标直接度量"遮挡区重建质量"（核心主张）；全图 psnr
度量"先验是否污染了可见区"（核心主张的另一半）。两个数必须同表出现，只报
其中一个就是片面证据。

**应得到什么结果（判读）**：

- 成立模式：masked_psnr `entropy > uniform >= none`，同时全图 psnr `entropy`
  与 `none` 无显著差异（不退化）。
- 遮挡区涨了但全图大跌：门控漏放，先验仍污染可见区，回去检查 prior mask 的
  覆盖范围。
- `uniform` 全图也不差、`entropy` 没有增量：说明均匀先验已经足够，门控没有
  价值，需要换证据（如 VGGT-Omega 的置信度）或收紧门控阈值。

报告时只写实测数字，不预设提升幅度；最终以三个场景取平均为准。

## 7. 消融矩阵与命令对应

| 论文变体 | 证明什么 | 命令行 | 状态 |
|---|---|---|---|
| 基线 3DGS | 下界：没有先验时遮挡区能做到什么程度 | `--gate none` | 可跑 |
| 均匀先验 | H1：全域先验是否整体有益、是否污染可见区 | `--gate uniform --depth-path ...` | 可跑 |
| ours（熵门控） | H2：区域路由是否优于均匀注入 | `--gate entropy --depth-path ...` | 可跑 |
| evidence_mean | 证据形式消融：均值 vs 熵 | `--gate mean_grad --depth-path ...` | 未接线 |
| evidence_conf | 证据形式消融：置信度 vs 熵 | `--gate depth_conf --depth-path ...` | 未接线 |
| fixed_w | H3：自适应强度是否有增益 | `--gate entropy --depth-path ... --no-adaptive-w --w-fixed 0.5` | 可跑 |
| global_ratio | 门控粒度消融：全局 vs 区域 | （全局粒度门控） | 未实现 |

消融主表模板（每个可跑变体 × 每个场景一行，最终按场景平均）：

| 变体 | 全图 PSNR | masked PSNR | masked SSIM | Δmasked（对基线） |
|---|---|---|---|---|
| none | | | | 0 |
| uniform | | | | |
| ours（entropy） | | | | |
| fixed_w | | | | |

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

1. `bash setup.sh --self-test`，确认 torch + nvcc 都在 —— 环境门槛
2. `bash setup.sh`，以 `--dry-run` 先过一遍再实装 —— 同上
3. `python tests/test_gate.py`，门控模块离线自检 —— 门控三态状态机可离线验证
4. bicycle 上跑 VGGT 深度 + COLMAP 转换，核对相机数和点数 —— 几何束成立
5. 7k pilot：`none` vs `entropy`，比训练 loss 轨迹 —— 先看 H1/H2 的第一层信号
6. 30k 正式五条曲线（第 5 节命令） —— 产出对照 loss
7. 补 test 渲染接线 -> SAM2 掩码 -> masked PSNR —— 产出主表
8. garden、room、On-The-Go 场景复制同一协议 —— 主表平均与分档结论

## 10. 常见运行错误

- `Could not find a version that satisfies the requirement setuptools>=68`：镜像/代理问题，换 `PIP_INDEX_URL` 并去掉 `http_proxy` 环境变量。
- `ModuleNotFoundError: No module named 'torch'`（PEP 517 构建隔离）：任何 `pip install -e` 都加 `--no-build-isolation`。
- `nvcc fatal / No CUDA toolchain`：确认 `nvcc --version`；没有就 `conda install -y -c nvidia cuda-nvcc=12.1`。
- simple-knn editable 安装成功后 `import` 仍报 `No module named`：其 hatchling 后端不会编译 CUDA 扩展，改用 `cd third_party/gaussian-splatting/submodules/simple-knn && python setup.py build_ext --inplace`，`train_depr.py` 会自动把该目录加入 `sys.path`。
- OOM：`--resolution 2` 半分辨率跑 pilot；正式表只标全分辨率场景。
- HF `403 / gated repo`：权重申请未通过或 `HF_TOKEN` 未导出。

## 11. 结果判读：从指标到论文主张

论文最终靠三张表/图支撑三个主张，每个主张必须能指向其中一张：

1. **主表（定量核心）**：none / uniform / ours / fixed_w × Mip-NeRF360 三场景 +
   On-The-Go 高/低遮挡档，列全图 PSNR、masked PSNR/SSIM。H2 由 `ours > uniform`
   且全图 PSNR 不降证明。
2. **先验污染证据表**：none vs uniform 的全图 PSNR 差，以及 uniform 在掩码外
   （可见区）的 PSNR 下降，证明 H1，即"问题确实存在"，同时回应审稿人
   "为什么不直接到处加深度"的质疑。
3. **门控行为图（定性）**：几张测试图的 prior mask 叠加在输入图上，展示路由
   集中在遮挡物周围；配合训练期 `state_map` 时序变化说明门控是动态的。当前
   代码未导出这些图，属于待接线项。

写论文时逐条核对：没有表支撑的主张删掉或降级为 discussion，不写空头结论。
