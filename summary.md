# MetaFi Project Summary

## 1. 项目目的

本项目的目标是基于现成的 MM-Fi WiFi CSI 数据集，实现一个面向人体姿态估计（Human Pose Estimation, HPE）的复现系统。  
输入是单帧 CSI 数据，核心使用三根接收天线采集到的幅度信息，输入张量形状为 `[3, 114, 10]`；输出是与 COCO 顺序对齐的 17 个二维人体关键点坐标，输出张量形状为 `[17, 2]`。

从任务本质上看，本项目希望学习一个从无线信号到人体结构的映射关系。它不是传统的视觉姿态估计，而是利用人体动作对无线传播路径、相位和幅度扰动的影响，从 CSI 中恢复人体骨架信息。

---

## 2. 项目设计

### 2.1 数据设计

项目使用 MM-Fi 数据集，原始数据按 `Axx/Syy` 组织：

- `A01-A27` 表示 27 类动作
- `S01-S40` 表示 40 个样本
- 每 10 个样本构成一个环境：
  - `S01-S10` 对应 `env1`
  - `S11-S20` 对应 `env2`
  - `S21-S30` 对应 `env3`
  - `S31-S40` 对应 `env4`

标签数据位于 `rgb/framexxx.npy`，每帧是 `17 x 2` 的关键点坐标。  
CSI 数据位于 `wifi-csi/framexxx.mat`，每帧是 `3 x 114 x 10` 的 CSI 数据。

当前项目采用的划分策略不是 frame 级随机划分，而是先在 sample-sequence 级别划分，再展开到 frame 级。  
具体规则是：对每个 `(action, environment)` 组内的 10 个样本按照 `6:2:2` 分成 train / val / test，然后再把每个样本中的 297 帧全部展开。这保证了：

- 训练、验证、测试之间不会发生同一样本序列的 frame 泄漏
- 三个子集都混合了所有环境
- 数据划分和论文复现目标保持一致

### 2.2 模型设计

当前模型由三个核心部分组成：

1. `SharedCNN`
2. `TransformerDecoderModule`
3. `WPFormer`

#### Shared CNN

`SharedCNN` 负责对三根天线的 CSI 幅度分支做共享权重的特征提取。

- 输入形状：`[B, 3, 114, 10]`
- 先按天线维切分为 3 个分支，每个分支为 `[B, 1, 114, 10]`
- 每个分支先双线性插值到 `[B, 1, 136, 32]`
- 三个分支通过同一个 CNN backbone 实例
- 单分支输出为 `[B, 512, 17, 4]`
- 最后沿最后一维拼接，得到 `[B, 512, 17, 12]`

该 backbone 基于 ResNet 风格的 `BasicBlock` 结构，通道按 `64 -> 128 -> 256 -> 512` 逐步增加，同时通过 stride 做空间下采样。

#### Transformer + Decoder

`TransformerDecoderModule` 接收 `SharedCNN` 输出的 `[B, 512, 17, 12]` 特征，完成全局关系建模和关键点回归。

- 先将后两个维度展平，变成 `[B, 512, 204]`
- 加入一个可学习的位置嵌入 `pos_embed`
- 使用 1 层自注意力模块，包含 3 个 head
- 与标准 Transformer 不同，这里不是把 head 结果简单拼接，而是先对各 head 的注意力矩阵做平均，再作用到 value
- 注意力残差后接 `InstanceNorm1d`
- 将输出重排回 `[B, 512, 17, 12]`
- 通过一个卷积式 decoder：
  - `3x3 conv: 512 -> 32`
  - `1x1 conv: 32 -> 2`
- 得到 `[B, 2, 17, 12]`
- 最后沿最后一个维度取平均，转置为 `[B, 17, 2]`

#### WPFormer 封装

`WPFormer` 是当前端到端模型封装，前向流程很直接：

- 输入 CSI 幅度
- 进入 `SharedCNN`
- 输出进入 `TransformerDecoderModule`
- 得到最终关键点预测

---

## 3. 项目设计的原理与合理性

### 3.1 为什么使用 Shared CNN

三根接收天线采集到的是同一人体动作在不同接收通道上的响应。  
这些分支既存在差异，也存在很强的结构一致性。

使用共享权重 CNN 有两个直接好处：

- 降低参数量，避免为三路输入重复学习几乎相同的局部模式
- 强制模型在三路天线中学习一致的低层空间特征表示

这对于样本规模有限的数据集是合理的，因为它在模型表达能力和泛化能力之间做了更稳妥的平衡。

### 3.2 为什么要把 CSI resize 后再编码

原始 CSI 尺寸是 `[114, 10]`，时间维较小，直接送入深层卷积网络时空间尺度不足。  
将其插值到 `[136, 32]` 的做法，本质上是把 CSI 映射到一个更适合 CNN 层级提取的二维特征平面。

这样做的合理性在于：

- 保留了原始二维结构
- 为连续卷积和下采样提供足够空间
- 与后续输出映射到 `17 x 4` 的尺度变化更自然

### 3.3 为什么在 CNN 后接 Transformer

CNN 更擅长提取局部模式，但人体姿态估计需要建模远距离关键部位之间的依赖关系，例如肩部与髋部、手腕与肘部之间的整体结构约束。  
Transformer 的自注意力正适合完成全局关系整合。

本项目的设计是先让 CNN 提取局部稳定特征，再让 Transformer 在高层语义空间中做全局信息交互，这个组合比单独依赖 CNN 或单独依赖 Transformer 更合理。

### 3.4 为什么使用 averaged-head attention

当前实现不是标准多头注意力拼接，而是先对多个 head 的注意力矩阵做平均。  
这样做的意义在于：

- 保留多头感受不同关系模式的能力
- 同时抑制不同 head 输出差异过大带来的不稳定性
- 更贴近目标论文结构

从复现角度看，这种实现优先保证论文对齐，而不是追求通用 Transformer 模板化写法，是合理的工程决策。

### 3.5 为什么损失函数直接用 MSE

当前训练损失直接对 `[B, 17, 2]` 坐标做 MSE，不额外加入姿态邻接矩阵等结构约束。  
这样做的原因有两个：

- 论文明确指出直接用坐标 MSE 是其训练设定
- 当前任务首先追求稳定复现，而不是额外堆叠先验约束

这是一个很务实的选择。先保证基线复现成立，再决定是否做后续增强，比一开始就引入复杂正则更稳。

### 3.6 为什么评估指标使用 torso-normalized PCK

不同样本中的人体尺度不同，直接比较像素距离会导致评估不公平。  
因此本项目使用以躯干长度为归一化参考的 PCK 指标：

- 右肩与左髋之间的欧氏距离作为 torso length
- 计算预测点与真值点的距离是否小于归一化阈值
- 统计 `PCK@10` 到 `PCK@50`

这样可以让评估更关注姿态结构恢复质量，而不是受人体绝对尺寸影响。

### 3.7 为什么引入 HDF5 离线数据打包

当前项目后期遇到的主要瓶颈不是模型计算，而是数据读取：

- 原始训练流程需要频繁随机读取大量 `.npy` 和 `.mat` 小文件
- `scipy.io.loadmat` 对 `.mat` 的解析开销很重
- GPU 速度远快于 CPU + 磁盘准备速度，导致 GPU 饥饿

为解决这一问题，项目新增了离线 HDF5 打包流程：

- 先一次性遍历原始数据集
- 将 `keypoints`、`csi_amplitude`、清洗后的 `csi_phase`、`csi_phase_cos` 和元信息写入单个 `.h5`
- 在离线打包阶段完成相位非有限值修复、子载波维度解卷绕、线性趋势/均值偏置去除以及 `cos(phase)` 计算
- 同时固化 train / val / test 的 frame 索引

这样训练时只需要顺序或近顺序访问一个大文件，显著减少小文件随机 I/O 和 `loadmat` 解析负担。  
这属于非常典型且合理的数据工程优化，尤其适合当前这种“原始数据小文件极多，但训练会重复读取很多轮”的场景。

---

## 4. 项目的实现

### 4.1 已完成的模块

当前项目已经完成以下模块：

- `dataloader.py`
  - 原始数据目录扫描
  - sample-level `6:2:2` 划分
  - frame 级展开
  - HDF5 数据集构建
  - HDF5 训练数据读取
- `scripts/build_h5_dataset.py`
  - 将原始 MM-Fi 数据集离线打包为单个 HDF5 文件
- `models/shared_cnn.py`
  - Shared CNN 编码器
- `models/transformer_decoder.py`
  - 自注意力与姿态解码器
- `models/wpformer.py`
  - 端到端模型封装
- `training/objectives.py`
  - MSE loss
  - torso-normalized PCK 指标
- `training/config.py`
  - 训练超参数
  - SGDM optimizer
  - Lambda 学习率衰减
- `training/trainer.py`
  - 训练与验证循环
  - best checkpoint 保存
  - loss 曲线绘制
  - `tqdm` 批级进度条
- `train.py`
  - 训练入口
  - 构建 dataloader、模型和 trainer

### 4.2 当前训练流程

当前推荐训练流程如下：

1. 在 Linux 服务器上使用原始数据生成 HDF5：

```bash
python scripts/build_h5_dataset.py \
  --dataset-root /data/WiFiPose/dataset/dataset \
  --output-path /data/WiFiPose/dataset/mmfi_pose.h5
```

2. 使用 HDF5 启动训练：

```bash
python train.py \
  --dataset-root /data/WiFiPose/dataset/mmfi_pose.h5 \
  --device cuda \
  --batch-size 32 \
  --num-epochs 50 \
  --num-workers 4 \
  --output-dir outputs/run_h5
```

训练过程中当前会完成：

- 每个 epoch 的 train / val 前向与反向传播
- MSE loss 计算
- `PCK@10` 到 `PCK@50` 统计
- batch 级 `tqdm` 速度显示
- 每个 epoch 的 loss 曲线更新
- 根据验证集 `PCK@50` 保存最佳 checkpoint

### 4.3 当前验证状态

当前仓库已经包含较完整的测试覆盖，测试内容包括：

- Shared CNN 输出形状
- Transformer Decoder 输出形状
- WPFormer 端到端前向
- MSE 与 PCK 计算逻辑
- Trainer 的训练、验证、checkpoint 与 loss 曲线
- 训练入口参数解析
- HDF5 数据打包与 HDF5 dataloader 读取

当前本地测试已经可以通过，说明从数据、模型到训练入口的主链路是贯通的。

### 4.4 当前项目所处阶段

从工程进度看，当前项目已经完成了一个可训练、可验证、可扩展的复现基线，主要包括：

- 数据预处理链路
- 模型主体链路
- 训练与评估链路
- 数据吞吐优化链路

当前更偏向“第一阶段完整复现系统已经成形”，后续工作的重点将不再是从零搭框架，而是：

- 在 Linux 服务器上开展正式训练
- 观察 loss 和 PCK 曲线
- 调整 batch size / num_workers 等吞吐参数
- 分析模型精度与论文结果之间的差距
- 决定是否继续做归一化、增强策略或结构改进

---

## 5. 总结

当前项目的核心价值在于：它已经不只是一个模型定义文件集合，而是一套完整的 WiFi 姿态估计复现管线。

从方法上看，项目采用了“共享卷积提取局部特征 + Transformer 整合全局依赖 + 直接回归关键点坐标”的设计，这一结构与任务目标高度匹配；从工程上看，项目又通过 HDF5 离线打包解决了原始 `.mat` 数据读取过慢的问题，使训练系统更适合在 Linux GPU 服务器上长期稳定运行。

因此，当前项目是一个结构清晰、目标明确、实现闭环基本完整的论文复现工程，已经具备正式训练和进一步优化的基础。
