# ABCD 影像空间、缺失处理与归一化说明

本文只说明当前 ABCD severity3 主实验，不涉及 ADNI。对应数据为 3,000 名
参与者、`I/G/C/B` 四模态和约 15% 整模态缺失，正式 manifest 的 SHA-256 为：

`d02be48acbe655052b201ba7fbd061d8e6219ac8ffa2d3e3c40000c494de82af`

## 1. 导师问“MR image template 到什么空间”时怎么回答

最准确的回答是：

> 当前模型没有读取配准到 MNI 的三维或四维 MRI 图像。我们读取的是 ABCD
> 官方管线已经提取好的区域级 tabulated imaging derivatives。fMRI 保留在
> 个体空间并配准到该受试者的 T1，之后投影到个体 FreeSurfer 皮层表面；
> Desikan 和 Gordon atlas 标签被映射到个体空间，再计算每个 ROI 的平均值。
> T1 被刚体配准到 ABCD 管线的内部标准参考脑，DTI 配准到同一受试者的 T1。
> 因而跨受试者一致的是 ROI 的解剖/功能标签，不是一个共同 MNI 体素网格。

不能简单回答“都在 MNI 空间”，也不能回答“模型使用 fsaverage 图像”。当前
输入已经是每人一行、每个 ROI 一列的数值表，进入 CERD 时不再保留体素坐标。

### 1.1 T1 结构像

ABCD 官方管线先进行梯度非线性和强度不均匀校正。T1 图像随后通过**刚体配
准**与一个内部平均参考脑对齐。该参考脑为 1 mm 各向同性、近似沿 AC--PC
轴定向，由 500 名成人 T1 图像构建；它不是论文中应随意写成的 MNI152。

皮层重建和分区在每名受试者的 FreeSurfer 表面上完成。Desikan 标签通过基于
皮层折叠模式的表面非线性注册映射到个体表面。我们使用的
`mr_y_smri__t1__gm__dsk` 不是 T1 体素图，而是 68 个左右半球 Desikan 区域
以及 3 个汇总项的灰质 T1-weighted intensity 平均值，共 71 维。

所以 T1 可以概括为：**ABCD 内部 1 mm 标准方向用于预处理和统一显示；最终
模型输入是在个体 FreeSurfer/Desikan ROI 上提取的区域平均强度。**

### 1.2 静息态和任务态 fMRI

ABCD 管线对 fMRI 完成头动、B0 畸变和梯度非线性校正，并将 fMRI 与该受试
者的 T1 配准。官方方法明确说明，处理后的 fMRI 仍在 **individual native
space**，分辨率为 2.4 mm 各向同性，并没有把我们的 ROI 特征先变成 MNI
体素图。

随后，皮层灰质的 fMRI 时间序列从体素空间采样到每名受试者自己的
FreeSurfer 表面：

- task-fMRI 使用个体 Desikan ROI，GLM beta 是 ROI 内平均任务效应；
- rs-fMRI 的 Desikan ROI 同样位于个体表面；
- Gordon 功能分区从 atlas space 映射到 individual subject space 后，再提取
  ROI 时间序列和网络相关；
- `aseg` 皮下核团标签来自个体 FreeSurfer 分割，并重采样到 fMRI 空间。

因此，SST、n-back、MID 的 68 维 beta，以及 rs-fMRI 的网络连接和时序方差，
都应称为**个体空间的 atlas-constrained ROI summaries**，而不是 MNI-space
voxel maps。

### 1.3 DTI

扩散数据先完成运动、涡流、B0 和梯度非线性校正。b=0 图像通过互信息配准
到个体 T1；dMRI 以 1.7 mm 各向同性分辨率重采样，并采用相对于已标准定向
T1 的固定旋转和平移，从而使扫描方向一致。

我们使用的 `mr_y_dti__is__fa__wm__dsk` 是 inner-shell DTI 的 FA。FA 被采样
到个体 FreeSurfer 皮层表面邻近白质，并按 Desikan 区域求平均，共 71 维。
这里没有使用 AtlasTrack tract-level 特征，也没有向模型输入 DTI 体素图。

### 1.4 四类影像子数据的空间总结

| 当前输入 | 原始处理空间/配准 | 最终进入模型的形式 |
|---|---|---|
| T1 intensity | T1 刚体对齐 ABCD 内部 1 mm、近 AC--PC 参考脑；个体 FreeSurfer 表面 | 个体 Desikan ROI 平均值，71 维 |
| Task-fMRI | 2.4 mm 个体 fMRI 空间，配准个体 T1；采样到个体表面 | SST/n-back/MID 的 Desikan ROI beta，各 68 维 |
| Resting-state fMRI | 2.4 mm 个体 fMRI 空间，配准个体 T1；Gordon/Desikan/aseg 映射到个体 | 网络相关、网络--皮下相关和 ROI 时序方差，439 维 |
| DTI FA | 1.7 mm，b=0 配准个体 T1；采样到个体表面邻近白质 | Desikan white-matter ROI 平均 FA，71 维 |

结论是：**影像模态的共同“空间”主要由相同的 ROI 标签建立，而不是通过把
所有人的图像非线性变换到同一个 MNI 体素模板来建立。**

## 2. 模型实际使用的 ABCD 四模态

| 模态 | 内容 | 物化维数 | 最终输入维数 |
|---|---|---:|---:|
| I | rs-fMRI + SST/n-back/MID task-fMRI + T1 intensity + DTI FA | 785 | 785 |
| G | QC 和 LD pruning 后的直接 SNP 加性剂量 | 1,172 | 1,172 |
| C | NIH Toolbox、WISC、Little Man、RAVLT、睡眠、身体活动 | 90 | 90 |
| B | 非 ADHD 行为、家庭、社区、社会经济和物理环境变量 | 221 | 219 |

四个模态不会先拼成一个长向量再共同归一化。每个模态单独处理，而且每个特
征列各有自己的训练中位数、均值和标准差。

## 3. 缺失与 normalization 的完整顺序

### 3.1 先划分，再 mask，再拟合统计量

ABCD 按遗传家系划分为训练 2,134、验证 438、测试 428，家系不跨集合。固定
缺失表的 `apply_before_preprocessing=true`，所以顺序为：

1. 冻结 train/validation/test；
2. 应用固定整模态 mask；
3. 只从训练集中该模态仍真实可见的参与者估计统计量；
4. 将训练得到的处理规则固定应用到验证和测试。

验证集、测试集以及训练集中被 mask 的整模态均不参与 feature filtering、
median、mean 或 standard deviation 的估计。

### 3.2 每一个特征列如何处理

对于每个模态 (m) 和每个特征 (j)：

1. 只用该模态可见的训练样本计算缺失率；缺失率大于 80% 则删除；
2. 用训练观测值的中位数 \(\widetilde{x}_{mj}\) 填补单元格缺失；
3. 删除中位数填补后训练样本方差不大于 \(10^{-8}\) 的列；
4. 在填补后的训练列上计算均值 \(\mu_{mj}\) 和标准差 \(\sigma_{mj}\)；
5. 对所有集合应用

   \[
   z_{imj}=\frac{x^*_{imj}-\mu_{mj}}{\sigma_{mj}}.
   \]

这里 (x^*) 是中位数填补后的值。实现使用
`sklearn.preprocessing.StandardScaler`，方差分母为 (n)，即 `ddof=0`。
验证和测试只调用 transform，不重新 fit，也不把超出训练范围的值截断。

### 3.3 当前数据的实际统计

| 模态 | 用于拟合的训练人数 | 内部单元格缺失率 | 单列最大缺失率 | 最终维数 |
|---|---:|---:|---:|---:|
| I | 2,021 | 7.70% | 18.60% | 785 |
| G | 1,951 | 0.41% | 2.00% | 1,172 |
| C | 2,072 | 3.02% | 32.05% | 90 |
| B | 2,079 | 1.12% | 29.39% | 219 |

B 中有一列因训练缺失率超过 80% 删除，另一列因近零方差删除。最终 2,266
个输入特征的训练列标准差均为 1，各列均值绝对值最大值小于
(7.2\times10^{-8})，说明实际张量确实完成了逐列 z-score。

## 4. 每种模态的 normalization 到底表示什么

### 4.1 影像 I

785 个影像 ROI 特征分别 z-score，不是整个 MRI 模态共用一个均值和标准差。
例如，某个 DTI FA 的 `z=1` 表示它比该 FA 列的训练均值高一个训练标准差；
它不会与 task-fMRI beta 的原始物理单位直接比较。

ABCD 官方 task-fMRI beta 的单位为 percent signal change。rs-fMRI 网络相关在
官方 ROI 汇总前包含 Fisher z 变换。T1 intensity、FA、beta 和相关系数原始
量纲不同，逐列 z-score 正是为了避免其中某类变量仅因数值范围较大而支配线
性投影。

### 4.2 遗传 G

1,172 列分别对应一个 SNP。处理顺序是训练集 SNP missingness/MAF QC、训练
集 LD pruning、逐 SNP 中位数填补、逐 SNP z-score。原始剂量为 0/1/2，但
进入网络后是相对该 SNP 训练分布的标准差单位。

没有 PCA、PRS 或祖源主成分，也没有把 1,172 个 SNP 合成一个遗传风险分数。

### 4.3 认知/健康 C 与行为/环境 B

原始列可能是标准分、反应时、次数、量表均值、连续暴露或 0/1 指示变量。它
们仍然逐列中位数填补和 z-score。这里不是对同一受试者的一整行做 normalization，
也不是把不同问卷项目强行共用一个 scaler。

## 5. “一个影像子类型缺失”如何处理

当前 `I` 是一个组合影像模态。只要参与者有至少一类所选影像记录，影像表中
就有该参与者：

- 例如有 T1、但没有 task-fMRI 时，`I` 仍标为可见；task-fMRI 对应空单元格
  用训练中位数填补；
- 固定 mask 若隐藏整个 `I`，则全部 785 个影像特征都视为不可用，不做一整
  行中位数替代，而由 CERD 根据其他模态生成影像 token。

这一区别很重要：前者是**模态内部的部分缺失**，后者是**整个影像模态缺
失**。当前模型没有把 rs-fMRI、task-fMRI、T1 和 DTI 各自作为四个独立模态。

## 6. `-2` 是否会作为异常 MRI 数值进入模型

不会。为了把不同模态整理成等长数组，加载器将整模态缺失行临时写为 `-2`，
同时令 `observed_mask=False`。编码阶段只把 `observed=True` 的数据送入对应
patch encoder；缺失模态先得到全零 token 缓冲区，再由条件生成器补全。

因此 `-2` 只是程序内部哨兵，不是 z-score 后的 MRI 极端值。

## 7. 可以直接向导师复述的版本

> 我们使用的是 ABCD 官方 ROI-level tabulated imaging derivatives，而不是
> MNI 空间体素图。T1 在官方管线中刚体对齐到一个 1 mm、近 AC--PC 的内部
> 平均参考脑；fMRI 保留在 2.4 mm 个体空间并配准到个体 T1；DTI 以 1.7 mm
> 重采样并配准个体 T1。Desikan 和 Gordon 分区被映射到每个受试者自己的
> FreeSurfer 表面/个体空间，再提取 ROI 平均值。因此模型跨人的空间对应来自
> 相同 ROI 标签，而不是共同 MNI 体素坐标。下游对 785 个影像特征逐列使用
> 训练中位数填补和训练 z-score，验证、测试和被 mask 的模态不参与统计量
> 估计。

## 8. 数据与代码依据

当前九张影像源表及其 JSON data dictionary 位于 ABCD tabulated release：

- `mr_y_rsfmri__corr__gpnet`；
- `mr_y_rsfmri__corr__gpnet__aseg`；
- `mr_y_rsfmri__var__dsk`、`mr_y_rsfmri__var__aseg`；
- `mr_y_tfmri__sst__csvcg__dsk`；
- `mr_y_tfmri__nback__2bv0b__dsk`；
- `mr_y_tfmri__mid__arvn__dsk`；
- `mr_y_smri__t1__gm__dsk`；
- `mr_y_dti__is__fa__wm__dsk`。

本仓库对应实现：

- 特征组装：`build_abcd_adhd_current3_snp_v2.py`；
- mask、训练中位数和 z-score：`multimodal_data/abcd.py::load_abcd_data`；
- 缺失值不进入编码器：`MoE/baseline_runner.py::encode_batch`；
- 正式结果协议：`results/abcd_severity3_n3000_missing15_protocol_v2.json`。

空间处理依据为 ABCD 官方影像文档和 Hagler et al., *NeuroImage* 202 (2019)
116091：

- <https://docs.abcdstudy.org/latest/documentation/imaging/index.html>
- <https://doi.org/10.1016/j.neuroimage.2019.116091>
