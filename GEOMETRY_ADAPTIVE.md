# Geometry-adaptive Octree-GS

本实现使用**渲染深度**。不读取外部深度图，不需要重编 CUDA 扩展。
默认不开启，用 `--geometry_adaptive` 开启，便于与原始 Octree-GS 比较。

## 使用

在原有 `python train.py ...` 命令后增加：

```bash
--geometry_adaptive \
--geometry_warmup 1000 \
--geometry_interval 500 \
--geometry_views 4 \
--geometry_max_anchors 16384 \
--geometry_min_offsets 2
```

例如（替换数据和输出路径，保留你实验所需的其他 Octree 参数）：

```bash
conda activate octree-gs
cd /mnt/nvme1n1/zhutianchi/ACDoctree/Octree-GS
python train.py -s /path/to/data -m /path/to/new_output \
  --appearance_dim 0 --geometry_adaptive
```

配置保存在 `cfg_args`，`render.py` 通过 `Scene` 自动读取几何缓存。
现有 shell 启动器若没有转发额外参数，请直接使用 `python train.py`。

## 信号与控制

1. **深度**：当前五输出光栅器返回的 `linear_depth` 是
   `sum(transmittance * alpha * camera_z)`，先除以 alpha 得到正向轴深度。
   若安装的是原始二输出光栅器，则以 `(camera_z, 1, 0)` 作为颜色做一次额外
   无梯度渲染，分别取得加权深度和覆盖率。正常训练的 RGB 渲染不增加此额外 pass。
2. **采样与分层**：依据相机矩阵和 FoV 反投影相邻像素。剔除低覆盖率、无效、
   超出图像以及被前景遮挡的 Anchor 观测。深度排序后在相对间距超过阈值处断开，
   曲率拟合仅使用与 Anchor 深度最近的一个连续深度层，且要求点位于局部半径内。
3. **真实曲率**：用深度置信度和距离加权 PCA 建立局部坐标，拟合六参数二次曲面，
   做三轮 Huber IRLS。使用包含一阶导数的完整形算子计算两个主曲率。
   `radius * sqrt(k1²+k2²)` 在已缓存且可靠的 Anchor 上按 Q95 归一化。
   半径为 `geometry_radius_factor * voxel_size / fork**level`。
4. **不足时的近似**：邻域不足或拟合退化时单独记录 `curvature_proxy`。
   它是同层有效邻域上的相对深度拉普拉斯，仅以 0.25 权重参与数量需求，
   不作为真实主曲率，也不产生不可靠的法向约束。
5. **不连续度**：相邻有效像素的相对深度差取 Q90；同时计算深度排序后的最大间距、
   两层样本平衡性和 sigmoid 分层分数。至少各有三个点才接受两层证据。
   跳变阈值/Q95 单独归一化，与分层分数加权后按有效视角置信度和支持比例融合。
6. **数量**：`0.7 * curvature + 0.3 * photo_residual` 决定
   `[geometry_min_offsets, n_offsets]` 的离散预算。光度残差也按多视角支持度聚合。
   依据 MLP 输出的 opacity 排序保留预算内的 offset，再应用原来的正 opacity 门控。
   因此预算是数量上限，不保证总有该数量的正 opacity Gaussian。
   **edge 不增加数量**；`n_offsets` 仍是每个 Anchor 的最大容量。
7. **空间约束**：依据曲率/法向限制切向范围和法向厚度；高 edge 区域还投影到
   各支持视角所选深度层的 slab 交集。若多视角约束互相冲突，则拒绝违规 Gaussian。
   约束对存活 offset 保持可微。生长候选也使用相同空间约束。

两个信号独立缓存并送入控制逻辑，未合并为单一“几何复杂度”。本版本使用可解释的
离散数量规则，不新增数量预测 MLP。颜色、opacity 和协方差仍由原网络预测；
几何约束直接作用于 Gaussian 中心。没有实现自动拆分前景/背景 Anchor 的新算法。

## 更新与可靠性

- 默认预热 1000 步，每 500 步更新；在当前迭代 RGB 前更新，不对统计反向传播。
- 统计渲染暂时关闭几何门控，避免自己的数量裁减直接让几何估计失去样本。
- 每次轮转选择最多 4 个不同训练图像和 16384 个 Anchor，按 256 个 Anchor 分块。
  同一图像的多个分辨率副本只算一个视角。不可见 Anchor 不产生有效观测。
- 至少两个有效视角才启用生成约束；未知/新生/低置信度 Anchor 保留原容量和 offset。
- 未轮到更新的 Anchor 保留缓存。生长补零，裁剪同步筛选。曲率、proxy、残差和跳变
  的归一化统计互相独立。Q95 基于当前已观测的可靠缓存，并非每次全场景扫描。
- 控制台和 TensorBoard 的 `geometry/*` 记录更新数量、可靠数量、有效拟合数量及分数均值。

常用阈值默认值：alpha 0.7、相对跳变 0.02、分层间距 0.04、边缘启用 0.3。
可通过同名 `--geometry_*` 参数调整。它们是初始实验值，未做数据集质量调优。

## 保存与恢复

PLY 字段保持原布局，几何参数与缓存写入旁边的 `point_cloud.geometry.pt`。
请将该文件与 PLY/MLP 一起保存。加载时验证 Anchor 位置、顺序和 offset 容量。
旧模型在不开启几何功能时正常加载；开启功能却缺失缓存时明确报错，避免悄悄改变结果。

训练 checkpoint 使用版本 2 字典，保存真实的 anchor features、MLP、Adam 状态、
训练统计及几何缓存。原代码中的 `capture/restore` 引用了不存在的 `_local`、`denom`，
且元组长度不匹配，因此一并修正。旧格式训练元组不自动转换；旧 PLY/MLP 格式仍可加载。
这不承诺跨恢复逐步数值完全一致（原训练循环没有保存相机采样 RNG 状态）。
同时将 PLY 读取的旧 `np.int` 改为 `int`，适配当前 NumPy。

## 验证与局限

```bash
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2 python -m unittest discover -s tests -v
```

测试覆盖平面/球面、尺度一致性、退化拟合、前后景分层、噪声点与无效深度、遮挡、
数量与 edge 解耦、可微约束、原版/扩展光栅器深度、实际 CUDA 渲染和反向传播、
真实 Anchor 生长/裁剪、PLY sidecar 与训练 checkpoint 恢复。

渲染深度是多 Gaussian 的期望深度，不是独立测量真值。低质量初始重建、薄结构、
透明区域和陡坡可能导致偏差；sorted-depth 分层是启发式，不能保证复杂交叠曲面被完美分开。
有限视角/Anchor 更新预算使缓存可能滞后。尚未进行完整数据集训练、PSNR 对比或速度基准。
