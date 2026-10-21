# APF-GS v2：全公共数据集版（取代自建数据路线）

> 触发原因：自建合成套件与指代基准的工程成本高、且说服力需自证。改为全部采用公共数据集 + 公共模型 + 标准指标，与 RogSplat / OracleGS 在同一批基准上直接对比。v1 文档中的 Synthetic-Occl 与 Ref-Complete 降级为"后续工作/可选扩展"。

## 1. 重定位后的创新点

**核心问题不变**：遮挡/稀疏视角重建中，先验与观测是异质信息源，直接融合互相污染；现有方法的融合信号来自外部 oracle 或启发式，而非优化过程自身。我们把重建建模为优化环内的多源信息路由问题，路由证据从优化轨迹的动力学熵中读取。

**三条贡献（全部可用公共数据验证）**

1. **动力学熵路由（Dynamics-Entropy Prior Routing, DEPR）**：第一个以"优化轨迹自身的梯度/残差熵"为门控证据的方法，在光度观测、几何先验、对象级生成先验三路之间做逐区域路由。对比：OracleGS 用外部 MVS oracle 的 2D 注意力不确定性，UGOD 用视角相关不确定度，DOC-GS 用启发式 dropout——我们的证据是优化过程内生的。
2. **识别接地的对象先验融合**：用 SAM 3D（公共模型）的逐物体形状/纹理/布局作为第三路信息，并施加身份与布局约束。对比：OracleGS/FlowObject 用的是扩散视图合成先验，没有对象身份。
3. **可见区保真保护 + 遮挡鲁棒 NVS**：门控保证可见区屏蔽先验流，在公共遮挡基准（NeRF On-The-Go 等）与稀疏视角基准（Mip-NeRF360 稀疏协议）上达到 SOTA，并以"可见区掩码 PSNR"（用公共 SAM2 mask 计算，可复现）证明无污染。

**删掉的部分**：Referring 3D Completion 任务与基准、补全幻觉率基准、Synthetic-Occl、Ref-Complete。审计闭环保留为机制消融（用公共 VLM 探测渲染结果），不再承担"新指标"的贡献职责。

## 2. 公共数据集与协议

| 数据集 | 角色 | 协议 | 基线已有数字 |
|---|---|---|---|
| **NeRF On-The-Go**（CVPR 2024，12 个真实场景，不同遮挡率） | 主遮挡基准 | 用提供的含遮挡训练视图训练，保留视图算 PSNR/SSIM/LPIPS；按遮挡率分档报告 | RogSplat、WildGaussians、NeRF On-The-Go 官方 |
| **Mip-NeRF360**（full + 稀疏 8-12 视图两种协议） | 主标准基准 | 官方协议；稀疏版用统一抽视策略并开源脚本 | 3DGS、FSGS、Sparse-GS、OracleGS |
| **Tanks&Temples / Deep Blending** | 鲁棒性 | 官方协议 | 3DGS 等 |
| **RobustNeRF 4 场景**（可选） | 干扰物 | 官方协议 | RobustNeRF、部分 GS 工作 |
| **OccluScene3D(-E)**（ICCV 2025，若已公开） | 遮挡补充 | 官方协议 | OccluGaussian |
| **DTU**（可选） | 几何对照 | 官方协议 | OracleGS 等 |

关键点：On-The-Go 的训练视图里有遮挡物、保留视图能看到被遮挡内容，所以**标准保留视图 PSNR 天然度量了补全质量**——这就是 RogSplat 的评测逻辑，审稿人完全接受，不需要任何自建 GT。

## 3. 对象先验在公共数据上的落地

- 目标物体：On-The-Go 场景中的桌椅、汽车、自行车、家具等；Mip-NeRF360 场景中的家具/杂物。
- mask 来源：SAM2 / Grounding-DINO（均为公共模型，零样本），逐场景生成实例 mask 文件并随代码发布。
- SAM 3D 输入：crop + mask + VGGT-Ω 点图；先验对齐到 VGGT-Ω 世界系（可见表面 Chamfer 监督）。
- 身份/布局约束实现不变（先验只作用在实例 mask 投影内；位姿与深度束对齐）。

## 4. 指标

- 标准：保留视图 PSNR / SSIM / LPIPS（On-The-Go 按低/中/高遮挡分档）。
- **可见区掩码 PSNR**：训练视图可见、测试视图同区域，用 SAM2 mask 计算——证明"先验没有污染可见区"，是 DEPR 的核心证据指标，完全可复现。
- 可选：遮挡物区域 LPIPS（测试视图可见部分）；训练时长；门控开销 ms/iter。
- 消融用同套指标；人类偏好研究用 On-The-Go 的 10 组场景（可选，非必需）。

## 5. 基线

3DGS、3DGS+VGGT-Ω 初始化、FSGS、Sparse-GS、WildGaussians、RogSplat、OracleGS、GaussFiller、RobustNeRF。优先跑通 3DGS/OracleGS/RogSplat 三个（RogSplat 与 OracleGS 有官方代码），其余用论文数字并注明协议一致。

## 6. 消融（全部在 Mip-NeRF360 + On-The-Go 上跑）

1. 门控证据：entropy / mean-gradient / depth_conf / uniform（无门控）——DEPR 的立身之本。
2. 路由粒度：global / region / pixel。
3. 先验组合：none / 几何only / 对象only / 两者（证明三源路由必要）。
4. 身份/布局约束：full / no-layout / no-identity。
5. 审计闭环：on / off。
6. 自适应 w* vs 固定 w。

## 7. 与 OracleGS 的逐点差异（写进 Related Work 的防御表）

| 维度 | OracleGS | 我们 |
|---|---|---|
| 门控证据 | 外部 MVS oracle 的生成视图注意力不确定性 | 优化轨迹自身的梯度/残差熵（内生） |
| 先验来源 | 扩散视图合成 | SAM 3D 对象级（形状/纹理/布局，有身份） |
| 路由目标 | 生成先验加权 | 三路：光度 / 几何 / 对象先验 |
| 遮挡基准 | LLFF/Mip360/DTU 为主 | 加 On-The-Go 分档、RobustNeRF |
| 保真证据 | 整体指标 | 可见区掩码 PSNR 专项指标 |

## 8. 执行计划（约 6-8 周，无数据工程）

- 第 1 周：3DGS 官方复现 Mip360 + On-The-Go；VGGT-Ω 初始化与深度接入。
- 第 2-3 周：SAM 3D 对象先验 + 对齐 + 均匀融合基线（即消融 1 的 none）。
- 第 3-5 周：DEPR 门控（迹/熵/三态/自适应 w*）+ 消融 1/2/6。
- 第 5-6 周：身份约束 + 审计闭环消融；可见区掩码 PSNR 全套。
- 第 6-8 周：全基线复现/对齐、On-The-Go 分档表、定性图、论文撰写。

## 9. 风险

- 新颖性风险：OracleGS 很近，差异化必须写实（第 7 节表）；若审稿人仍认为增量不足，备选升级点是"门控证据的可解释性分析 + 可见区保护指标"做深。
- On-The-Go 场景物体识别依赖 SAM2/grounding 质量；备选是在 Mip360 上先证明门控有效（那里不需要对象先验也能跑几何+光度两路）。
- OccluScene3D 若未公开，以 On-The-Go + RobustNeRF 覆盖即可。

## 10. 与 v1 的关系

- v1 的 Referring 3D Completion、幻觉率基准、Synthetic-Occl、Ref-Complete 全部转入"未来工作/扩展"；方法图、论文框架的方法与 Intro 基本不动，实验章节按本文档重写。
- 若 v2 论文被接收，v1 的基准可作为下一篇的独立贡献。
