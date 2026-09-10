# ABCD ADHD 三分类：数据、模态、预处理、CERD 与实验全流程

> 版本：`presentation3-snp-v4`，2026-09-09 复核；这是主 matched 实验
>
> 结果口径：seeds 31/32/33 三个独立模型指标的算术平均 ± 样本标准差
>
> 本版结果：Accuracy 59.24%，Macro-F1 53.62%，Macro-AUROC 74.43%。结构 MRI
> 采用真实区域体积和厚度的独立 v5 表征实验见
> [`abcd_adhd_presentation3_feature_refinement_v5.md`](../results/abcd_adhd_presentation3_feature_refinement_v5.md)。

这份文档对应本轮重新构建并完成正式测试的 ABCD 实验。它回答三个最容易混淆的问题：三类标签在临床上分别是什么，四个模型模态到底包含什么，以及每一步数据处理和实验汇总是怎样完成的。这里记录的是实际运行配置，不使用 CatBoost、预测后偏置、概率 ensemble、祖源主成分或外部 ADHD PRS。

## 1. 任务定义

任务是在 ABCD 基线访视 `ses-00A` 上，根据家长版 K-SADS ADHD 诊断和症状条目构造三个具有直接临床含义的类别：

| 类别 | 名称 | 可操作定义 | 样本数 |
|---|---|---|---:|
| 0 | 低症状非 ADHD 对照 | 当前、既往、部分缓解和未特定 ADHD 四个诊断字段均不为阳性；当前注意缺陷和多动/冲动两个症状域分别不超过 3 项 | 1,270 |
| 1 | 注意缺陷为主的完整 ADHD | 当前或既往完整 ADHD 阳性；所选诊断时期的 9 项注意缺陷症状不少于 6 项，同时 9 项多动/冲动症状少于 6 项 | 635 |
| 2 | 含多动/冲动成分的完整 ADHD | 当前或既往完整 ADHD 阳性；所选诊断时期的 9 项多动/冲动症状不少于 6 项，包括多动/冲动为主型和混合型 | 963 |

这不是“0、1、≥2 个诊断域”的人为负担分级。Class 1 和 Class 2 都是完整 ADHD 病例，两者的临床区别是是否达到多动/冲动维度的症状阈值。Class 0 是严格低症状对照，而不是把所有非 ADHD 儿童都直接作为对照。

### 1.1 当前与既往诊断时期如何选择

- 如果 `present full ADHD` 为阳性，使用当前 18 个症状条目。
- 如果当前完整 ADHD 不阳性、但 `past full ADHD` 为阳性，使用既往 18 个症状条目。
- 用于确定 ADHD 表型的所选 18 个症状必须全部是明确的 0/1；不能用跳题编码代替症状阴性。
- 症状阈值为每个九项域至少 6 项，适用于 ABCD 基线儿童年龄段。
- 仅有部分缓解或未特定 ADHD 的参与者不进入两个完整 ADHD 类，也不进入对照类。

K-SADS 中 `888` 表示由问卷分支导致的跳题，`555` 表示未施测。四个诊断状态字段允许 `0/1/888`，因为阴性的上游分支可以使后续诊断字段合理跳过；但是一旦某人被选入当前或既往完整 ADHD 病例，其对应的 18 个症状条目必须全部为显式 `0/1`。对照的跳题条目不计为阳性，并且还需同时满足四个诊断字段均不阳性及两个当前症状域各不超过 3 项。

### 1.2 为什么把多动/冲动为主型与混合型合为 Class 2

严格病例中，注意缺陷为主型 635 人，多动/冲动为主型只有 177 人，混合型 786 人。如果把三种 presentation 各自设为一类，就会丢掉低症状对照且产生很小的 177 人类别；如果单独保留多动/冲动为主型，也会显著放大训练不稳定性。当前合并保留了“是否存在达到诊断阈值的多动/冲动成分”这一清晰临床边界，同时利用全部 1,602 个症状记录完整且满足 presentation 阈值的完整 ADHD 病例。

标签审计还得到：原始基线 11,867 人，低症状对照候选池 9,593 人，完整 ADHD 1,618 人；其中 16 人因所选时期症状不是完整 0/1 而排除，另有 4 人虽然诊断阳性但两个九项域均未达到 6 项阈值。对照从候选池用固定 SHA-256 顺序抽取 1,270 人，即 Class 1 的两倍，抽样种子为 2026。

## 2. 队列与数据划分

最终队列共 2,868 人。先确定标签队列，再按 ABCD 遗传家庭 ID 做分层分组划分：

| 划分 | Class 0 | Class 1 | Class 2 | 合计 |
|---|---:|---:|---:|---:|
| 训练集 | 913 | 442 | 686 | 2,041 |
| 验证集 | 170 | 94 | 150 | 414 |
| 测试集 | 187 | 99 | 127 | 413 |
| 全队列 | 1,270 | 635 | 963 | 2,868 |

训练/验证、训练/测试、验证/测试的家庭 ID 交集均为 0。同一家庭成员不会跨集合。多数类基准 Accuracy 为 44.28%，因此三分类的难度不能拿二分类的 50% 随机水平作参照。

## 3. 四个输入模态总览

| 代码 | 模态 | 原始维数 | 模型有效维数 | 队列中有该模态的人数 | 主要内容 |
|---|---|---:|---:|---:|---|
| I | Imaging | 785 | 785 | 2,843 | rs-fMRI、SST/n-back/MID task-fMRI、T1 灰质信号强度、DTI FA |
| G | Genetics | 1,178 | 1,178 | 2,808 | 14 个预设 ADHD 相关基因窗口内的加性 SNP dosage |
| C | Cognition & Health | 90 | 90 | 2,868 | NIH Toolbox、WISC、LMT、RAVLT、睡眠和体力活动 |
| B | Behavior & Environment | 221 | 218 | 2,868 | 非 ADHD 行为维度、冲动/奖赏、家庭和社区、社会经济与环境暴露 |

`I/G/C/B` 是模型的四个模态，不代表四个单独文件必须对每个人都完整。表内“有该模态”表示至少存在一个可用源字段；模态内部的空单元格随后由训练集统计量处理。B 的三个字段在训练集列过滤后被移除，所以实际进入模型的是 218 维。

## 4. Imaging 模态：具体用了什么影像

本实验没有把 rs-fMRI、task-fMRI、T1 和 DTI 当作四个模型模态，而是把它们组成一个 785 维 Imaging 模态。这样整个任务仍保持与 CERD 设计一致的四模态输入，同时影像内部覆盖功能连接、任务激活、区域 T1 信号和白质微结构。

### 4.1 rs-fMRI：439 维

| 表型 | 维数 | 实际数值含义 |
|---|---:|---|
| Gordon network-to-network correlation | 91 | Gordon 功能网络两两相关的唯一项 |
| Gordon network-to-subcortical correlation | 247 | Gordon 皮层网络与皮层下分区之间的相关性 |
| Desikan cortical temporal variance | 68 | Desikan 皮层区的静息态时间序列方差 |
| Subcortical temporal variance | 33 | 皮层下分区的静息态时间序列方差 |

这里的连接特征是 ABCD 发布的区域/网络级衍生表型，不是在本项目中从原始 NIfTI 重新计算的。我们保留其数值定义和分区顺序，再做训练集标准化。

### 4.2 task-fMRI：204 维

三个任务各取 68 个 Desikan 皮层区的 all-run beta contrast，不使用 t statistic：

- SST：正确停止相对正确 go，反映反应抑制相关激活；
- n-back：2-back 相对 0-back，反映工作记忆负荷相关激活；
- MID：奖赏预期相对中性条件，反映奖赏预期相关激活。

每个任务贡献 68 维，总计 204 维。选择的不是 task accuracy 或问卷分数，而是任务态区域 beta 表型。

### 4.3 T1 信号强度与 DTI：142 维

- T1：71 个 Desikan 皮层灰质区域信号强度特征；
- DTI：71 个与 Desikan 区域对应的白质 fractional anisotropy（FA）特征。

这里的 `mr_y_smri__t1__gm__dsk` 是区域 T1 灰质信号强度，并不是 cortical volume 或 thickness；FA 表征扩散张量沿主方向的一致程度，是白质微结构的区域级指标。本文模型使用的是 ABCD 已发布的区域级衍生量；原始扫描的畸变校正、配准和分割属于 ABCD 上游影像流程，不能表述为本代码重新实施的步骤。

总维数核对为 `91 + 247 + 68 + 33 + 3×68 + 71 + 71 = 785`。

## 5. Genetics 模态：SNP 是怎样选择和处理的

### 5.1 当前版本明确不包含什么

当前 G 模态只包含 ABCD 内部提供的 SNP 基因型 dosage：

- 不使用外部 GWAS summary statistics；
- 不计算 ADHD PRS；
- 不用同一 ABCD 队列拟合“内部 PRS”；
- 不加入 32 个 genetic ancestry principal components；
- 不把基因型交给 CatBoost 或其他额外预测器。

因此本实验不会依赖任何外部遗传文件。PRS 若没有独立外部效应权重，就会变成当前队列内的特征筛选/加权问题，不能当成标准外部 PRS；本版不走这条路径。

### 5.2 候选基因窗口

使用 hg19 基因坐标上下游各扩展 20 kb，在以下 14 个预先指定窗口内提取 SNP：

| 功能系统 | 基因 | 进入模型的 SNP 数 |
|---|---|---:|
| 多巴胺受体 | DRD2, DRD3, DRD4, DRD5 | 62, 73, 30, 17 |
| 儿茶酚胺合成/转运 | DBH, SLC6A3, SLC6A2 | 104, 96, 98 |
| 去甲肾上腺素受体 | ADRA2A | 30 |
| 5-HT 受体/转运与单胺代谢 | HTR2A, HTR1B, SLC6A4, MAOA | 123, 38, 50, 28 |
| 抑制性神经传递 | GABRB3 | 377 |
| 神经元分选与突触相关 | SORCS3 | 52 |

以上计数合计 1,178。这个设计的目标是在不把整个芯片噪声全部输入模型的情况下，保留与单胺信号、抑制性神经传递和神经元功能相关的遗传变异。它是候选窗口 SNP 表征，不等于全基因组多基因风险，也不能把单个窗口的模型贡献解释为某个基因的因果效应。

### 5.3 PLINK 处理顺序

原始候选窗口中有 2,313 个变异。只用 1,997 名训练集中有基因型的参与者完成变异过滤和 LD pruning：

1. `--keep` 限制到训练参与者；
2. `--geno 0.02`，去除训练集中变异缺失率大于 2% 的 SNP；
3. `--maf 0.01`，去除训练集中 minor allele frequency 小于 1% 的 SNP；
4. `--indep-pairwise 50 5 0.8`，以 50-SNP 窗口、5-SNP 步长和 $r^2=0.8$ 做 LD pruning；
5. 对保留变异执行 `--recode A`，得到 0/1/2 型加性等位基因 dosage；
6. 将 1,178 个 dosage 列映射回全部队列，随后用训练集进行缺失填补和标准化。

本轮没有额外执行 HWE、个体基因型缺失率或杂合度过滤，因此不能在论文或答辩中声称做过这些步骤。队列中 2,808/2,868 人有至少一个基因型记录，覆盖率 97.91%。

## 6. Cognition & Health 模态：90 维

这个模态保留与 ADHD 的认知功能和健康背景相关、但不直接定义 K-SADS 标签的测量：

| 来源 | 维数 | 内容 |
|---|---:|---|
| NIH Toolbox | 10 | 晶体/流体/总认知复合分数，以及认知灵活性、抑制控制/注意、工作记忆、情景记忆、词汇、加工速度和阅读等年龄校正分数 |
| WISC-V Matrix Reasoning | 2 | 矩阵推理 raw score 与 scaled score |
| Little Man Task | 25 | 不同刺激类型的正确数和反应时，以及总体正确、错误、超时、效率等指标 |
| RAVLT | 24 | Trial 1–5 学习、短延迟和长延迟回忆，以及 intrusion/repetition 计数 |
| Sleep Disturbance Scale | 26 | 唤醒、睡眠启动/维持、过度嗜睡、睡眠多汗、睡眠呼吸和睡眠-觉醒转换相关条目 |
| Physical activity | 3 | 每周至少 60 分钟活动天数、力量活动天数、学校体育天数 |

总计 `10 + 2 + 25 + 24 + 26 + 3 = 90`。保留多个任务内部指标是为了避免只用一个总分抹掉注意、抑制、工作记忆、学习和反应时的差异；模型的 patch encoder 再学习其局部组合。

## 7. Behavior & Environment 模态：221 原始维，218 有效维

### 7.1 行为与家庭背景：206 维

| 来源 | 维数 | 内容 |
|---|---:|---|
| CBCL | 122 | 焦虑、抑郁、攻击、对立、品行、社会、躯体、思维、退缩和压力等非 ADHD 条目与部分 syndrome/DSM-oriented T-score |
| BIS/BAS | 24 | 行为抑制、drive、fun seeking、reward responsiveness 的条目和汇总分 |
| UPPS | 25 | negative/positive urgency、缺乏计划、缺乏坚持、sensation seeking 的条目和汇总分 |
| Family Environment Scale | 10 | 家庭冲突条目及均值 |
| Parental Monitoring | 6 | 青少年报告的父母监护条目及均值 |
| Neighborhood Safety | 4 | 家长报告的社区安全条目及均值 |
| Demographic context | 15 | 家庭收入、照护者与伴侣教育/收入、婚姻和家庭困难等背景字段 |

CBCL 的 ADHD、attention problems 和 sluggish cognitive tempo 条目/分数没有进入预测模态，宽泛的 total/broadband 分数也未纳入。K-SADS 的诊断和症状字段只用于生成标签，四个预测模态与标签字段的列名交集为 0。

### 7.2 居住与孕期环境：15 维

- Area Deprivation Index national percentile：1 维；
- Child Opportunity Index：总分、教育、健康/环境和社会/经济四个 national z-score；
- census-tract lead-risk index：1 维；
- 24-hour mean environmental noise：1 维；
- 2016 annual PM2.5 和 NO2：各 1 维；
- National Air Toxics respiratory-hazard index：1 维；
- 到主要道路距离：1 维；
- census-tract park-land proportion：1 维；
- prenatal residential NO2、O3、PM2.5：3 维。

这组变量覆盖社会经济机会、邻里安全、家庭环境、空气污染、噪声、铅风险和绿地等外部暴露，不要求每个变量都只针对 ADHD。其作用是为模型提供儿童神经发育与行为表现所处的环境背景，而不是用大量行政字段堆高维数。

### 7.3 为什么最终是 218 维

原始 B 表有 221 列。仅依据训练集的缺失率与方差规则，以下三列未进入模型：`mh_p_cbcl__rule_005`、`mh_p_cbcl__rule_006` 和 `ab_p_demo__child__time_001__01`。因此 checkpoint 记录的 B 输入维数为 218；其他三个模态的原始维数与有效维数相同。

## 8. 缺失数据与标准化

### 8.1 先确定整模态 observed mask

固定随机表的种子为 2026。每个划分分别保持约 85% 完整样本，即约 15% 样本为不完整多模态输入。mask 在任何列过滤、插补和标准化之前应用，因此被隐藏的整模态不会参与该模态训练统计量估计。

缺失模式以丢失一个模态为主、丢失两个为辅、丢失三个极少；永远不允许四个模态同时缺失。最终每个划分的可观测模态个数如下：

| 划分 | 仅 1 个可见 | 2 个可见 | 3 个可见 | 4 个完整 | 完整比例 |
|---|---:|---:|---:|---:|---:|
| 训练 | 13 | 59 | 234 | 1,735 | 85.01% |
| 验证 | 3 | 10 | 49 | 352 | 85.02% |
| 测试 | 2 | 13 | 47 | 351 | 84.99% |

固定 missingness manifest 的 SHA-256 为 `c2a956bc60094786cefbe973d42de707c3592c33e2bfa43b435381c3ebfe94b2`。CERD 和所有 baseline 读取同一个 manifest，因此它们看到的是完全相同的参与者、标签、划分和缺失模式。

### 8.2 模态内单元格怎样处理

对每个模态分别执行，并且只从该模态在训练集中仍为 observed 的参与者估计参数：

1. 删除训练集中缺失比例大于 80% 的列；
2. 对保留列，用训练集中位数填补数值缺失；若训练列全空则回退为 0；
3. 用完成训练中位数填补后的训练矩阵拟合 z-score 标准化器；
4. 将同一个中位数和标准化器应用到验证集与测试集；
5. 整模态缺失仍由 observed mask 表示，不会被中位数填补伪装成真实观测。

这一区分很重要：median imputation 处理模态内部缺值，CERD 的 conditional generator 处理整模态不可用，两者不是同一件事。

## 9. CERD 实际运行方法

### 9.1 从向量到模态 token

每个模态独立分成 8 个固定顺序的 feature patches，线性投影到 128 维 token。若原始维数不能被 8 整除，先在尾部补零。每个 patch 还带 rank-4 的低秩残差适配器，使同一模态内不同 feature block 可以保留局部差异。四个模态都产生 `8 × 128` token 序列。

### 9.2 条件生成缺失模态

对每个目标模态设置一个专属生成器。其 8 个可学习目标 query 通过两层、四头 cross-attention 从当前参与者的所有已观测模态 token 中取条件信息，随后经 FFN 和 sigmoid output gate 得到目标 token。输出门控制各通道响应强度。

训练时，对完整参与者随机选择可观测目标模态进行重建，每个样本最多覆盖四个目标。重建项为投影后 pooled token 的 cosine distance。本配置中分类路径使用生成 token 的前向数值，但分类梯度不直接更新 generator；生成器主要由重建目标学习，避免分类器把生成器退化成任意标签编码器。

### 9.3 真实/生成来源标记

每个模态 token 在融合之前加入 modality-specific provenance embedding，区分它是 observed 还是 generated。这个标记先于全局 self-attention 和 expert routing，使相同位置的真实证据与补全证据在后续融合中保持可区分。

### 9.4 sparse MoE 融合

四个模态的 token 进入一层全局 Transformer fusion block：

- hidden dimension 128；
- self-attention heads 4；
- 16 个 FFN experts；
- 1 个 router；
- 每个 token 选择 top-4 experts；
- dropout 0.35；
- router load-balancing loss 权重 0.01。

MoE 位于 Transformer 的 FFN 位置，不是训练多个完整网络后再投票。每个 token 只激活 16 个专家中的 4 个，路由依据融合后的 token 内容、模态位置和来源标记形成。

### 9.5 多粒度决策分解

融合后的每个模态位置通过 attention pooling 得到 128 维表示。分类器一共有 11 条 decision branches：

- 1 个 joint branch，读取四个模态位置拼接；
- 4 个 modality-centered branches，每个读取一个模态位置；
- 6 个 pair-centered branches，对每一对模态使用 `[f_i, f_j, f_i⊙f_j, |f_i-f_j|]`。

这里的 modality-centered 和 pair-centered 指 readout anchor，不应说成未经交互的纯单模态/纯双模态分类器。因为 branch 读取的是全局融合后的 token，它已经含有其他位置带来的上下文；这一设计分解的是决策视角，而不是强行假设神经影像、基因、认知和环境彼此独立。

### 9.6 可靠性与最终概率

真实观测位置的 reliability 固定为 1。生成位置的 reliability 由该位置融合特征经过小型 scorer，再与该模态的 learned generated bias 相加并通过 sigmoid 得到。branch quality 定义为：

- joint branch：当前可用模态 reliability 的均值；
- modality-centered branch：对应模态 reliability；
- pair-centered branch：两个模态 reliability 乘积的平方根。

每条 branch 还根据其 softmax 概率的归一化熵得到 prediction confidence。`log(quality) + log(confidence) + branch prior` 经 softmax 后成为 11 条 branch 的归一化权重，最终概率是 branch 概率的加权和。因而 reliability 不是四个数独立相加，最终归一化发生在有效 decision branches 上。

### 9.7 训练目标

本次 CERD 的实际目标由以下部分构成：

- 主输出的 inverse-frequency class-weighted cross-entropy；
- 重建损失，权重 0.25；
- 11 条 branch 的辅助分类损失，权重 0.10；
- sparse router balancing loss，权重 0.01；
- 训练期 observed-modality dropout 后的分类损失，权重 0.10；
- 完整视图向减少模态视图的 self-distillation，温度 2、权重 0.15；
- more-view 相对 fewer-view 的 per-sample CE 非劣排序约束，权重 0.10。

训练时 modality dropout probability 为 0.25，并保证每个样本至少留下一个真实可见模态。它只产生训练辅助视图；验证和测试始终使用固定 15% missingness manifest。Class weight 只从训练集 `913/442/686` 的频率按 inverse-frequency 计算；所有方法验证/测试都直接用原始 softmax argmax，不做阈值移动或 class offset。

## 10. 训练与评估协议

- Seeds：31、32、33；三个 seed 分别初始化、训练、选 epoch 和测试。
- 优化器：AdamW，learning rate `1e-4`，weight decay `0.01`。
- Batch size：64；gradient clipping：5。
- 最多 50 epochs；前 5 epochs 为 warm-up data order；至少训练 15 epochs。
- 验证 Macro-F1 选择每个 seed 的最佳 checkpoint；patience 为 10。
- 不做 checkpoint soup；每个 seed 只使用自己的 rank-1 validation checkpoint。
- 正式测试前严格重放 checkpoint 的验证指标；21/21 个 CERD/baseline checkpoint 均通过，再各执行一次 test evaluation。
- 汇总值为三个 seed 的指标算术平均，标准差为三个值的 sample standard deviation；绝不是把三个模型概率先平均后计算指标。

例如 CERD 三个测试 Accuracy 为 57.3850%、59.8063%、60.5327%，所以报告值是

`(57.3850 + 59.8063 + 60.5327) / 3 = 59.2413%`。

## 11. 正式实验结果

### 11.1 验证集

| Method | Accuracy (%) | Macro-F1 (%) | Macro-AUROC (%) |
|---|---:|---:|---:|
| CERD | 60.47 ± 0.74 | **56.27 ± 0.92** | **75.29 ± 0.18** |
| Flex-MoE | 60.39 ± 1.47 | 53.35 ± 0.86 | 74.15 ± 0.73 |
| I2MoE | **60.55 ± 1.64** | 55.42 ± 0.94 | 74.66 ± 1.04 |
| MoE++ | 59.98 ± 1.70 | 53.04 ± 0.42 | 74.00 ± 0.98 |
| AnyMod | 57.49 ± 2.66 | 52.78 ± 1.26 | 70.63 ± 2.56 |
| AGDiC | 56.60 ± 2.65 | 51.55 ± 2.29 | 70.99 ± 1.55 |
| ACADiff | 57.65 ± 1.75 | 50.95 ± 1.48 | 69.03 ± 0.66 |

### 11.2 测试集

| Method | Accuracy (%) | Macro-F1 (%) | Macro-AUROC (%) |
|---|---:|---:|---:|
| CERD | **59.24 ± 1.65** | **53.62 ± 1.40** | **74.43 ± 1.03** |
| Flex-MoE | 57.71 ± 1.14 | 50.07 ± 1.03 | 72.58 ± 1.25 |
| I2MoE | 55.29 ± 1.33 | 49.12 ± 1.20 | 73.38 ± 1.05 |
| MoE++ | 57.06 ± 0.98 | 48.54 ± 0.78 | 73.44 ± 1.65 |
| AnyMod | 54.48 ± 4.44 | 48.46 ± 4.23 | 69.60 ± 3.16 |
| AGDiC | 54.80 ± 0.92 | 48.58 ± 0.99 | 70.70 ± 0.75 |
| ACADiff | 54.32 ± 1.22 | 47.00 ± 0.71 | 68.31 ± 0.63 |

CERD 在测试集三个主指标上均取得最高三种子均值。相对每个指标最强的 baseline，Accuracy 提高 1.53 个百分点，Macro-F1 提高 3.55 个百分点，Macro-AUROC 提高 0.99 个百分点。最明显的差异出现在 Macro-F1，说明收益不只是来自数量最多的低症状对照，而是更均衡地覆盖三个 clinical presentation 类。这里是三种子数值比较；尚未执行参与者级配对显著性检验，因此只表述为 numerical advantage。

### 11.3 CERD 每个 seed 的测试结果

| Seed | Best epoch | Accuracy (%) | Macro-F1 (%) | Macro-AUROC (%) |
|---:|---:|---:|---:|---:|
| 31 | 8 | 57.38 | 52.06 | 73.37 |
| 32 | 10 | 59.81 | 54.05 | 74.49 |
| 33 | 8 | 60.53 | 54.75 | 75.44 |
| Mean ± SD | — | 59.24 ± 1.65 | 53.62 ± 1.40 | 74.43 ± 1.03 |

### 11.4 三类中哪一类最难

CERD 的三种子测试集 class-wise 均值为：

| 类别 | Precision (%) | Recall (%) | F1 (%) |
|---|---:|---:|---:|
| 低症状对照 | 69.79 | 77.01 | 73.18 |
| 注意缺陷为主 ADHD | 33.07 | 25.93 | 29.05 |
| 含多动/冲动成分 ADHD | 58.28 | 59.06 | 58.63 |

Class 1 是主要难点。它与 Class 2 共享“完整 ADHD”诊断，与 Class 0 的差异又主要集中在注意缺陷而非明显多动/冲动，因此它处于两个更容易识别的表型之间。当前 Macro-F1 的提升说明 CERD 对这种跨模态、非单一生物标志物决定的中间表型更有帮助，但 Class 1 recall 仍是后续模型改进的主要空间。

## 12. 导师常问问题的简明回答

### Q1：ABCD 三类到底是什么？

不是病程轻中重，也不是诊断域数量。三类是低症状非 ADHD 对照、完整 ADHD 注意缺陷为主型、完整 ADHD 中存在达到阈值的多动/冲动成分。第三类包含多动/冲动为主型和混合型。

### Q2：为什么 ABCD 准确率比 ADNI 低？

ABCD 当前任务区分的是儿童 ADHD presentation，特别是两个都已达到完整 ADHD 诊断的亚型；它们共享大量症状、遗传和神经发育背景。ADNI 的认知正常、MCI、dementia 在年龄相关脑结构、认知和生物标志物上通常有更强的总体分离。两个数据集的类别定义、疾病阶段和可分性不同，Accuracy 不能直接横向解释为模型在 ABCD 上失效。

### Q3：为什么遗传模态没有想象中那么强？

当前输入是 14 个候选窗口内 1,178 个常见 SNP dosage，而 ADHD 是高度多基因且表型异质的神经发育障碍。单个常见变异的边际效应通常有限，候选窗口不能覆盖全部遗传结构；因此不能预期 G 单模态像直接症状量表一样产生很高分类准确率。G 的合理角色是与影像、认知和环境提供互补信息。若未来使用 PRS，应采用独立、祖源匹配的外部 GWAS 权重并单独报告；本版按要求不引入外部数据。

### Q4：T1、DTI 和 fMRI 分别代表什么？

本 v4 版 T1 项是 Desikan 区域灰质信号强度，不是体积或皮层厚度；DTI FA 是区域对应的白质微结构指标；rs-fMRI 是无显式任务时网络连接和时间方差；task-fMRI 是抑制控制、工作记忆负荷和奖赏预期三个任务的区域激活 contrast。v5 已移除这些信号强度项，改用实际皮层/皮下体积和皮层厚度。四类影像合在 I 模态中，并不是只用了某一种 fMRI。

### Q5：为什么没有把所有行为量表都放进去？

当前保留非 ADHD 共病/行为、冲动与奖赏、家庭和环境背景，同时不把直接定义 ADHD 的 K-SADS 条目、CBCL ADHD/attention 分数和用药变量作为预测输入。这样 B 仍有 218 个有效维度，并没有缩成很小的数据集，但每组特征都有明确构念。

### Q6：v4 的 59.24% 是不是三个模型 ensemble？

不是。它是三个独立 seed 的 Accuracy 算术平均。Macro-F1 和 AUC 也分别对三个 seed 的对应指标取算术平均。任何结果表都必须保持这个 aggregation rule。

## 13. 可复核边界

公开仓库只保存方法代码、聚合指标和不含参与者身份的处理说明。ABCD 原始表、PLINK 个体基因型、家庭 ID、逐参与者预测 CSV 和模型 checkpoint 不上传。机器可读结果见 [`../results/abcd_adhd_presentation3_snp_missing15_v4.json`](../results/abcd_adhd_presentation3_snp_missing15_v4.json)，简表见 [`../results/abcd_adhd_presentation3_snp_missing15_v4.md`](../results/abcd_adhd_presentation3_snp_missing15_v4.md)。
