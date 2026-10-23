# APF-GS v2 两周冲刺计划（初稿 + 实验数据 + 代码）

> 目标：14 天内产出"带全部实验数据的论文初稿 + 可复现代码"。前提假设：1 张 24GB 显卡（4090/A10 及以上，有 A100 更快）；HF 能拿到 VGGT-Ω 权重；网络可访问 GitHub/HF。每天有明确交付物与止损点。

## 0. 总策略：先跑通最小证据链，再补全

最小证据链只有三件事，任何论文故事都建立在这上面：

1. 3DGS 基线 vs 3DGS+VGGT-Ω 初始化（证明几何束可用）；
2. 均匀深度先验 vs **熵门控深度先验**（证明门控有用，这是 DEPR 立身之本）；
3. **可见区掩码 PSNR**（证明门控保护了保真区）。

对象先验（SAM 3D）是锦上添花：第 8-10 天做，超时立即降级为"定性图 + 子集表"，不影响主故事。**第 6 天是决策点**：如果门控在 2 个 pilot 场景上打不过均匀先验，转预案 B（见第 9 节），不硬写。

## 1. 底层代码与技术栈（照此实施）

| 组件 | 仓库 | 许可证 | 用途 |
|---|---|---|---|
| 基座 | `graphdeco-inria/gaussian-splatting` | INRIA | fork 成 `depr-gs/`，所有改动集中于此 |
| 几何 | `facebookresearch/vggt-omega` | CC-BY-NC（核对） | 位姿 + 深度 + depth_conf |
| 对象先验 | `facebookresearch/sam-3d-objects` | SAM License | 第 8-10 天，出 PLY |
| 掩码 | `facebookresearch/sam2` | Apache-2.0 | 可见区掩码指标 + 实例 mask |
| 兜底几何 | `facebookresearch/vggt`（原版） | 同上 | VGGT-Ω 权重申请被卡时的替代 |
| 兜底先验 | TRELLIS / Hunyuan3D（任一开源权重） | 各仓库 | SAM 3D 被卡时替代 |

环境：单 conda env，Python 3.10，PyTorch 2.x + CUDA 12，先编译 3DGS 的 `diff-gaussian-rasterization`。VGGT-Ω 与 SAM 3D 都是 torch 模型，同环境可跑，不需要双环境。

## 2. 仓库改造（文件级）

```text
depr-gs/
  scene/dataset_readers.py   # 扩展：读 depths/*.npy 与 depth_conf
  scene/gaussian_model.py    # 不动（先验 PLY 用 create_from_pcd 载入）
  utils/gate.py              # 新：EMA 迹、熵、三态路由、w 网格
  utils/metrics.py           # 新：masked PSNR、门控开销计时
  train.py                   # 主改：损失路由 + 门控刷新 + 先验区保护
  scripts/
    run_vggt_depth.py        # VGGT-Ω 推理 -> 位姿/深度/置信
    vggt_to_colmap.py        # 位姿写 COLMAP sparse/0 格式
    make_sam2_masks.py       # SAM2 生成可见区/实例掩码
    run_sam3d_prior.py       # SAM 3D -> 对齐 PLY（第 8 天起）
  configs/{mip360.yaml, otg.yaml, ablations.yaml}
```

**train.py 的五个注入点**：
1. 损失计算处：把 L_photo 乘以区域权重 mask；
2. 深度损失：rendered depth（rasterizer 可输出 depth 图）与 VGGT-Ω depth 的 L1，按 depth_conf 加权；
3. 门控：每 100 迭代用残差迹刷新 32×32 区域的三态权重（实现上用 **L1 残差 |C−I| 作为 g(t)** 的代理——L1 下残差方向即优化步方向，省掉每窗口一次反向，论文里写"gradient/residual dynamics"即可）；
4. 先验区保护：route=prior 的区域跳过 densify 与 opacity reset；
5. 先验 PLY 载入：`GaussianModel.create_from_pcd` 导入 SAM 3D 高斯，用深度束做相似变换对齐。

## 3. 数据与协议

- **开发集（前 5 天）**：Mip-NeRF360 的 bicycle、garden；NeRF On-The-Go 的 1 个高遮挡 + 1 个低遮挡场景。
- **正式集（第 6-10 天）**：Mip360 全 9 场景或至少 5 场景；On-The-Go 8-12 场景按遮挡分档。
- 位姿：VGGT-Ω 输出转 COLMAP `sparse/0`，走官方数据读取，保证与 3DGS 基线同协议。
- 稀疏协议：Mip360 抽 8-12 视图脚本开源、固定 seed。
- 可见区掩码：SAM2 在训练视图上零样本分割 occluder/目标，测试视图用同一区域投影做 masked PSNR；流程写进补充材料。

## 4. GPU 时间预算

- 30k 迭代约 40min/场景（4090）；pilot 用 7k（约 10min）。
- 必须跑：3 变体（3DGS、+init、ours）× 7 场景 ≈ 14 GPU 时；消融 6 变体 × 2 场景 ≈ 8 GPU 时；合计约 22 GPU 时，按每天 20h 排任务队列，2-3 天跑完。
- 显卡不足 16GB：pilot 半分辨率，最终表只在 3 场景跑完整版并注明。

## 5. 14 天日程

**第 1 天｜环境与数据**：克隆 5 个仓库；编译 rasterizer；下载开发集 4 场景；提交 VGGT-Ω 与 SAM 3D 的 HF 权重申请（第一时间）。
**第 2 天｜基线**：3DGS 官方跑通 bicycle；`run_vggt_depth.py` + `vggt_to_colmap.py` 出位姿/深度；可视化检查初始化质量。
**第 3 天｜+init 与均匀先验**：+VGGT init 变体；深度读取进 train.py；均匀深度先验（消融 1 的 none 档）。
**第 4 天｜门控核心**：`utils/gate.py` 全实现；bicycle 上 pilot；调迹窗口与分位数。
**第 5 天｜快消融**：7k 迭代跑消融 1/2/6（pilot 2 场景）；晚上开正式集全量队列。
**第 6 天｜决策点**：门控 > 均匀先验？是→继续；否→转预案 B。同时生成 On-The-Go 的 SAM2 掩码与 masked PSNR 脚本。
**第 7 天｜基线补齐**：收集 FSGS/Sparse-GS/OracleGS/RogSplat 论文数字；OracleGS 官方代码 timebox 4 小时试跑（跑不通就用论文数字+协议一致说明）。
**第 8 天｜对象先验（冲刺）**：`run_sam3d_prior.py`，2 个 On-The-Go 场景；身份约束；超 4 小时卡住即降级定性。
**第 9 天｜正式集跑完**：3 变体 × 正式集收数；masked PSNR 全套。
**第 10 天｜全消融**：6 组消融在 2 场景完整版；对象先验子集表（如有）。
**第 11 天｜图**：门控三色图、遮挡补全前后、与 OracleGS 对比图；转视频的定性材料。
**第 12 天｜写稿**：填 Table 1-4、Method、Related Work、Intro；初稿 v0.5。
**第 13 天｜自查**：防御表逐格核对；协议一致性检查；可复现脚本打包。
**第 14 天｜缓冲**：重跑异常场景、润色、输出终稿。

## 6. 每日交付物

每天结束提交：当日新增代码 + 实验日志（命令、种子、耗时）+ 当日表格 CSV。第 6/10/14 天各做一次全量自查（证据链是否闭合）。

## 7. 实验运行清单（命令级）

```bash
# 几何束
python scripts/run_vggt_depth.py --images data/mipnerf360/bicycle/images --out data/mipnerf360/bicycle/vggt
python scripts/vggt_to_colmap.py --vggt data/mipnerf360/bicycle/vggt --out data/mipnerf360/bicycle/sparse/0
# 基线
python train_depr.py -s data/mipnerf360/bicycle --gate none                          # 基线
python train_depr.py -s data/mipnerf360/bicycle --gate uniform \
  --depth-path data/mipnerf360/bicycle/vggt                                          # 均匀先验
python train_depr.py -s data/mipnerf360/bicycle --gate entropy \
  --depth-path data/mipnerf360/bicycle/vggt --gate-grid 32 --gate-window 100        # ours
# 掩码与评测
python scripts/make_sam2_masks.py -s data/mipnerf360/bicycle/images
python scripts/eval_masked.py --renders output/bicycle_entropy/test/ours_30000/renders \
  --gt data/mipnerf360/bicycle/images --masks data/mipnerf360/bicycle/masks
```

## 8. 论文与代码同步

初稿各表与 `runs/` 目录一一对应；`configs/ablations.yaml` 直接生成论文 Table 5 的每一行。图表色板沿用已生成的方法图（photo=蓝 / geo=青 / prior=紫）。

## 9. 风险与预案

- **预案 B（第 6 天门控失败）**：转写"可见区保真保护的评测协议 + 识别接地先验融合"——把 Table 4 的掩码指标当作主要贡献，门控降为辅助。文章仍有价值，但定位调整。
- VGGT-Ω 权重被卡：改用原版 VGGT；SAM 3D 被卡：改用 TRELLIS/Hunyuan3D，或第 3 天就并行申请。
- OracleGS 无法复现：论文数字 + 协议一致说明，防御表照填。
- 场景内无清晰可识别物体：On-The-Go 场景人工挑目标；实在不行对象先验只在定性图出现。

## 10. 成功标准（第 14 天）

初稿含：完整 Method + 4 张主表（数字齐全）+ 6 组消融 + 防御表 + 定性图；`depr-gs/` 一键复现（README 三行命令）；公开数据集 100%。缺一不可交付。
