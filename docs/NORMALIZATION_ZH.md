# CERD 数据归一化与缺失值处理说明（当前冻结版本）

本文记录当前主实验**实际执行的数值处理**，重点回答：不同模态的量纲如何
统一、缺失值如何处理、统计量由哪些受试者估计，以及数据归一化与模型内部
`LayerNorm`、可靠性权重归一化有什么区别。本文是代码行为说明，不把上游影像
派生流程和下游神经网络归一化混为同一步。

本文绑定以下正式实验：

- ABCD：`abcd-severity3-n3000-missing15-e4k2-formal-v2`；
- ABCD manifest SHA-256：
  `d02be48acbe655052b201ba7fbd061d8e6219ac8ffa2d3e3c40000c494de82af`；
- ADNI：`adni-matched-updated-cerd-v4-three-seed-mean`；
- 模态顺序均为 `I/G/C/B`，但两个数据集的字母含义不同。

## 1. 先给出结论

| 数据集 | 模态 | 单元格缺失填补 | 编码器前的尺度变换 |
|---|---|---|---|
| ABCD | I：影像 | 训练集逐列中位数 | 训练集逐列 z-score |
| ABCD | G：SNP | 训练集逐 SNP 中位数 | 训练集逐 SNP z-score |
| ABCD | C：认知/健康 | 训练集逐列中位数 | 训练集逐列 z-score |
| ABCD | B：行为/环境 | 训练集逐列中位数 | 训练集逐列 z-score |
| ADNI | I：FreeSurfer MRI | 训练集逐列众数（冻结的历史设置） | 训练集逐列 z-score |
| ADNI | G：SNP | 训练集逐 SNP 众数 | 训练集逐 SNP min-max 到 `[-1,1]` |
| ADNI | C：临床/认知 | 训练集逐列均值 | 当前加载器不再做额外尺度变换 |
| ADNI | B：生物标志物 | 训练集逐列均值 | 当前加载器不再做额外尺度变换 |

最关键的一点是：**ABCD 不是先把四个模态拼在一起再做一个 scaler，也不是
每个模态只计算一个均值和标准差。它为每个模态的每一个特征列分别计算中位
数、均值和标准差。** 因此，相关系数、任务态 beta、FA、SNP 剂量和量表分数
虽然原始尺度不同，进入各自模态编码器前都按本列训练分布换算成标准差单位。

## 2. 三类容易混淆的“归一化”

当前系统存在三个不同层次的归一化，作用对象并不相同。

1. **数据层的 feature scaling**：沿受试者维度，对每一个原始特征列进行；
   ABCD 使用 z-score。这一步固定训练集统计量，验证集和测试集只能应用，不能
   重新拟合。
2. **网络层的 LayerNorm**：沿单个受试者、单个 token 的隐藏维度进行；它稳
   定 Transformer 和生成器中的隐藏表示，不能代替数据层的训练集 z-score。
3. **决策层的权重归一化**：用 softmax 把当前样本的有效专家权重或预测分支
   权重变成和为 1 的权重。这一步归一化的是路由权重，不是原始生物变量。

本文后续分别解释这三层。

## 3. ABCD：精确的数据处理顺序

### 3.1 统计量的拟合集合

ABCD 有 3,000 名参与者，家系隔离划分为训练 2,134、验证 438、测试 428。
冻结的 15% 整模态缺失表设置了
`missingness.apply_before_preprocessing=true`。因此，对模态 (m)，用于估计
统计量的集合是：

\[
T_m=\{i: i\text{ 属于训练集，原始表中有模态 }m，且冻结 mask 保留 }m\}.
\]

验证集、测试集，以及训练集中被 mask 隐藏的整模态，都不进入列筛选、中位
数、均值或方差估计。训练时额外的 0.25 模态 dropout 发生在标准化和编码之
后，仅用于构造训练视图，也不会重新拟合任何数据统计量。

### 3.2 列筛选、填补和 z-score

每个模态独立执行以下步骤。

1. 将所选列转为数值；无法转为数值的单元格记为缺失。
2. 在 (T_m) 中计算每列缺失率，删除缺失率大于 80% 的列。
3. 对剩余特征 (j)，计算训练集观测值的中位数
   \(\widetilde{x}_{mj}\)，用它填补**可用模态内部**的单元格缺失。
4. 在填补后的训练矩阵上计算列方差，删除样本方差不大于
   \(10^{-8}\) 的近常数列。
5. 在填补后的训练矩阵上拟合 `sklearn.preprocessing.StandardScaler`。
   对特征 (j)：

   \[
   \mu_{mj}=\frac{1}{|T_m|}\sum_{i\in T_m}x^*_{imj},\qquad
   \sigma_{mj}=\sqrt{\frac{1}{|T_m|}\sum_{i\in T_m}
   (x^*_{imj}-\mu_{mj})^2},
   \]

   \[
   z_{imj}=\frac{x^*_{imj}-\mu_{mj}}{\sigma_{mj}}.
   \]

   这里 (x^*) 表示完成训练中位数填补后的值；`StandardScaler` 的方差分母
   是 (n)，即 `ddof=0`。
6. 固定训练得到的列集合、\(\widetilde{x}\)、\(\mu\) 和 \(\sigma\)，原样
   应用于验证集和测试集。验证/测试值不会裁剪到训练范围内。

中位数填补发生在 z-score 之前，所以缺失单元格进入模型的值是
\((\widetilde{x}_{mj}-\mu_{mj})/\sigma_{mj}\)，不一定恰好为 0。

### 3.3 当前物化数据的实际审计值

下面的“拟合人数”是 (T_m) 的大小；“内部缺失率”只统计这些可用训练模态
中的单元格缺失，不包含整模态 mask。

| 模态 | 物化维数 | 最终维数 | 拟合人数 | 内部单元格缺失率 | 单列最大缺失率 |
|---|---:|---:|---:|---:|---:|
| I：影像 | 785 | 785 | 2,021 | 7.70% | 18.60% |
| G：SNP | 1,172 | 1,172 | 1,951 | 0.41% | 2.00% |
| C：认知/健康 | 90 | 90 | 2,072 | 3.02% | 32.05% |
| B：行为/环境 | 221 | 219 | 2,079 | 1.12% | 29.39% |

B 模态删除了两个训练集不合格特征：

- `ab_p_demo__child__time_001__01`：训练可见数据缺失率超过 80%；
- `mh_p_cbcl__rule_006`：中位数填补后的训练方差不大于 `1e-8`。

对最终保留的 2,266 个 ABCD 特征重新计算可得：每个训练列的标准差均为
1.000000，各列均值绝对值的最大值小于 (7.2\times10^{-8})（浮点误差量
级）。这说明实际张量确实执行了逐列标准化，而不是只在配置中声明。

## 4. ABCD 四个模态分别发生了什么

### 4.1 I：影像，785 维

输入是 ABCD 已发布的区域/网络级 tabulated derivatives，而不是在本仓库中
重新处理原始 BOLD 或 DICOM：

- 静息态 fMRI 439 维：Gordon 网络间连接、网络—皮下结构连接、Desikan
  皮层和皮下结构的时序方差；
- 任务态 fMRI 204 维：SST correct-stop vs correct-go、n-back 2-back vs
  0-back、MID reward anticipation vs neutral，各 68 个区域 beta；
- T1 71 维：Desikan 区域 T1-weighted gray-matter intensity；
- DTI 71 维：Desikan 对应白质区域的 fractional anisotropy（FA）。

785 列分别做中位数填补和 z-score。模型没有把“相关系数”“beta”“T1 强
度”“FA”混合计算一个总均值。例如某个 FA 特征的 1.0 表示它比该 FA 列的
训练均值高一个训练标准差；它不与某个 task-fMRI beta 的原始物理单位直接
比较。本仓库的这一步也不额外执行 ComBat、站点残差化或 ICV 校正；如果源
表已经做过上游处理，那属于 ABCD 派生表生成流程，不属于本加载器的 z-score。

### 4.2 G：遗传，1,172 维

遗传模态是 14 个预指定基因窗口中保留的直接加性 SNP 剂量。训练集先完成
SNP 缺失率不超过 2%、MAF 不低于 1% 和
`--indep-pairwise 50 5 0.8` 的 LD pruning；之后加载器对每个 SNP 单独执行：

1. 用训练可见基因模态中的该 SNP 中位数填补；
2. 计算该 SNP 的训练均值和标准差；
3. 把 0/1/2 剂量转换成相对训练分布的标准差单位。

因此，网络实际读取的是 1,172 个标准化 SNP，而不是原始 0/1/2，也不是把
全部 SNP 压缩为一个分数。这里没有 PCA、PRS 或祖源主成分。某个低频等位
基因携带状态在 z-score 后可能具有较大的绝对值，这是逐 SNP 方差标准化的
自然结果；MAF 下限和训练方差过滤共同限制了近零方差位点。

### 4.3 C：认知与健康，90 维

包括 NIH Toolbox、WISC、Little Man、RAVLT、睡眠和身体活动变量。原始列
可能是分数、反应时、次数或量表均值，每列都独立中位数填补并做 z-score。
标准化不在同一个受试者内部做，也不要求不同测验使用共同量纲。年龄校正分
数若已由源表提供，作为一个独立特征继续按训练分布标准化。

### 4.4 B：行为与环境，221 维物化、219 维输入

包括去除 ADHD/注意条目后的 CBCL 维度、BIS/BAS、UPPS、家庭冲突、父母
监护、社区安全、人口学和居住/产前暴露。连续变量、等级变量和 0/1 指示变
量都按列处理。以训练患病率为 (p) 的二元列为例，z-score 后的两个值为

\[
z(0)=\frac{-p}{\sqrt{p(1-p)}},\qquad
z(1)=\frac{1-p}{\sqrt{p(1-p)}}.
\]

所以进入模型后它不再保持 0/1，但语义仍是同一个指示变量。极少见指示变量
可能产生较大标准化幅度，因此近常数列过滤是必要步骤。

## 5. 整模态缺失与单元格填补不是一回事

- **单元格缺失**：受试者的该模态总体可用，只是个别变量为空。它使用训练
  中位数填补，随后作为“观测模态”进入编码器。
- **整模态缺失**：原始表无该模态，或冻结 keep mask 将它隐藏。它不会用
  一整行中位数伪装成观测数据，也不参与该样本的模态编码。

加载器为张量对齐临时把整模态缺失行写为 `-2`，同时将布尔
`observed_mask` 设为 `False`。`MoE/baseline_runner.py::encode_batch` 只把
`observed=True` 的行送入对应编码器；缺失行的 token 缓冲区直接初始化为
0，之后才由 CERD 的条件生成器根据其他可用模态生成。因此：

> `-2` 是数据容器中的缺失哨兵，不是标准化后的真实数值，也不会进入线性
> patch projection。

ABCD 冻结 mask 在训练/验证/测试中分别保留约 85% 完整样本，缺 1–2 个模
态为主、少量缺 3 个，从不人为隐藏全部四个模态。所有方法读取同一个
manifest 和同一 observed mask。

## 6. ADNI：当前冻结实现的精确行为

ADNI 使用已有合并表和其记录的模态可用性，没有套用 ABCD 的 15% 人工
mask。它与 ABCD 的输入归一化并不完全相同。

### 6.1 I：187 维 FreeSurfer MRI

从 `UCSFFSX7_06Jan2026.csv` 中选取以 `ST` 开头且以 `CV`、`TA` 或 `SV`
结尾的区域特征，并按 `update_stamp` 为每位参与者保留最新记录。冻结命令使
用 `--adni_image_imputation legacy_mode --initial_filling mean`。虽然参数名中
出现 `mean`，该历史分支的实际代码是：

1. 每个 MRI 特征用训练集第一众数填补；
2. 在填补后的训练 MRI 矩阵上拟合逐列 `StandardScaler`；
3. 固定应用到验证和测试。

结果回执中的 `resolved_statistic` 明确记录为 `mode`。因此不能把当前 ADNI
MRI 主实验描述成“均值填补”；它是训练集众数填补后 z-score。

### 6.2 G：387,639 维 SNP

每个 SNP 按块处理以控制内存，但统计仍是逐 SNP 的：

1. 在训练受试者中分别统计 0、1、2 的次数，用出现最多的剂量填补；相同次
   数时 `argmax` 选择数值较小的剂量；
2. 记训练最小值为 (a_j)、最大值为 (b_j)，执行

   \[
   g'_{ij}=2\frac{g_{ij}-a_j}{b_j-a_j}-1.
   \]

3. 若训练跨度为 0，代码把分母替换为 1；该训练常数值因而映射到 -1。

训练数据的非常数 SNP 被映射到 `[-1,1]`。验证或测试若出现训练中未见的剂
量，结果可能超出 `[-1,1]`，因为当前实现不裁剪。这仍不使用验证/测试统计
量重新定标。

### 6.3 C：4,189 维临床/认知；B：151 维生物标志物

两类表都使用训练集逐列均值填补，全缺失列回退为 0。当前加载器在填补后不
再执行 z-score 或 min-max。因此，能从代码严格确认的是“训练均值填补，保
持合并表中的数值尺度”；不能额外声称当前仓库对这两类输入重新做了标准化。
若将来为它们新增 z-score，应当定义为新的数据协议，并为 CERD 和所有
baseline 重新训练，不能把新预处理替换进现有结果。

## 7. 模型内部如何处理剩余的尺度差异

### 7.1 独立模态编码器

四个原始向量不会先直接拼接。每个模态有独立调用的 patch tokenizer：特征
按固定顺序分成连续块，末端不足处补 0，再投影为相同宽度的 token。当前
CERD 还使用 rank-4 patch-specific residual adapter。四个模态只有在进入共同
隐藏空间后才交互，因此 ADNI 临床/生物标志物的原始尺度不会与 MRI/SNP 在
原始特征坐标上直接相加。

### 7.2 LayerNorm

Transformer 块在注意力和 MoE-FFN 前使用 `LayerNorm`；条件生成器分别对
query、key/value 和 FFN 输入使用 `LayerNorm`；token pooler 和模态可靠性
网络也包含 `LayerNorm`。对单个 token (h\in\mathbb{R}^d)：

\[
\operatorname{LN}(h)=\gamma\odot
\frac{h-\operatorname{mean}(h)}{\sqrt{\operatorname{var}(h)+\epsilon}}+\beta.
\]

它沿隐藏维度 (d) 计算，而不是沿训练受试者计算。它不会读取验证/测试总体
统计，也不会改变数据层 scaler。

### 7.3 MoE 路由与预测分支

- MoE router 对选出的 top-k 专家 logits 做 softmax，因此同一个 token 的
  top-k 专家权重之和为 1。
- 每个模态的原始可靠性是独立的可信度，不要求四个可靠性分数本身相加为 1。
  当前设置中，观测模态可靠性为 1，生成模态可靠性由 sigmoid 给出，不可用
  模态为 0。
- 最终预测使用 joint、四个 modality-centered 和六个 pair-centered 分支。
  分支的可靠性、置信度和可学习先验组成 branch log-score，再对当前有效分支
  做 softmax。由此得到的最终 branch weights **严格和为 1**。
- 对外报告的模态贡献把 joint 和 pair 分支质量分摊回四个模态，屏蔽不可用
  模态后再次除以总和，所以每位受试者的最终模态贡献之和也为 1；汇总图用
  百分比表示时四项之和为 100%。

因此，“原始模态可靠性不和为 1”与“最终融合权重已经归一化”并不矛盾：前
者是四个独立质量系数，后者才是用于凸组合的概率单纯形权重。

## 8. 导师问答速查

| 问题 | 当前实验的准确回答 |
|---|---|
| 不同模态的尺度差这么大，是否直接拼接？ | ABCD 不直接拼原始值；四个模态逐特征 z-score 后分别编码。ADNI 也分别编码，但 C/B 保留上游合并表尺度。 |
| z-score 是每个模态一个均值/方差吗？ | 不是；每一列各有自己的训练中位数、均值和标准差。 |
| 验证集和测试集是否参与 scaler？ | 不参与；只应用训练集拟合的统计量。 |
| ABCD 被人工 mask 的值是否参与统计？ | 不参与；冻结 mask 在列筛选和统计拟合之前应用。 |
| 单元格填补后算观测模态吗？ | 算，因为该模态总体可用；整模态缺失则不算。 |
| `-2` 会不会被网络当成极端 z-score？ | 不会；只有 `observed=True` 的行进入编码器。 |
| ABCD SNP 是否降维？ | 没有；QC/LD 后的 1,172 个 SNP 分别中位数填补和 z-score。 |
| ABCD SNP 是否为 PRS 或祖源 PC？ | 都不是；是直接加性剂量。 |
| LayerNorm 是否等于数据 z-score？ | 不等于；前者按单样本 token 隐藏维归一化，后者按训练参与者逐特征定标。 |
| 可靠性为什么不直接和为 1？ | 原始可靠性表示各模态独立质量；真正参与概率融合的 branch weights 经过 softmax，严格和为 1。 |

## 9. 对应代码位置

- ABCD 列筛选、填补、z-score 和整模态 mask：
  `multimodal_data/abcd.py::load_abcd_data`；
- ABCD 特征构建和 manifest：`build_abcd_adhd_current3_snp_v2.py`；
- ADNI 四模态处理：`MoE/data.py::load_and_preprocess_adni`；
- ADNI MRI 历史填补解析：`MoE/data.py::adni_image_fill_values`；
- 缺失行不进入编码器：`MoE/baseline_runner.py::encode_batch`；
- patch tokenizer、LayerNorm、生成可靠性和分支 softmax：`MoE/models.py`；
- 冻结 ABCD 协议：
  `results/abcd_severity3_n3000_missing15_protocol_v2.json`。

ABCD manifest 可用以下命令做只读结构检查：

```bash
python validate_abcd_manifest.py \
  --dataset_manifest /path/to/manifest.json \
  --modality IGCB
```

正式发布前再运行 `python scripts/validate_release.py`，检查结果聚合、实验协议
和公开回执之间的一致性。
