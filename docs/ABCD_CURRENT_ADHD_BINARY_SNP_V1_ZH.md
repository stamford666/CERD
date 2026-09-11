# ABCD 当前 ADHD 二分类：标签、四模态、预处理与实验口径

> 当前主任务：严格低症状非 ADHD 对照 vs 当前完整 ADHD。
>
> 所有结果均为 seeds 31/32/33 三个独立模型的指标算术平均 ± 样本标准差，不是概率 ensemble。

## 1. 标签到底是什么

标签来自 ABCD 基线访视 `ses-00A` 的家长版 K-SADS ADHD 诊断字段和 18 个当前症状条目。

| 类别 | 临床含义 | 可执行定义 | 人数 |
|---|---|---|---:|
| 0 | 严格低症状非 ADHD 对照 | 当前、既往、部分缓解、未特定 ADHD 均不阳性；当前注意缺陷和多动/冲动两个九项域各不超过 3 项 | 1,272 |
| 1 | 当前完整 ADHD | 当前完整 ADHD 阳性；18 个当前症状均为显式 0/1；至少一个九项域达到 6 项 | 848 |

既往 ADHD 但当前不满足者、仅部分缓解者、未特定 ADHD、只有症状而没有当前完整诊断者全部排除，不硬塞进对照或病例。原始基线有 11,867 人；严格对照候选 9,593 人，当前完整 ADHD 诊断 863 人，其中 15 人因症状不是完整显式 0/1 而排除。对照在分家系之前按固定 SHA-256 顺序、种子 2026 抽取为病例的 1.5 倍。这样避免海量容易对照把 Accuracy 人为抬高。

家庭互斥划分为训练 1,520、验证 297、测试 303；相应对照/病例为 917/603、177/120、178/125。三个划分之间遗传家庭 ID 交集均为 0。

## 2. 四个模型模态总览

| 代码 | 模态 | 进入模型的维数 | 具体表示 |
|---|---|---:|---|
| I | Imaging | 785 | rs-fMRI、SST/n-back/MID task-fMRI、T1 区域灰质体积、DTI FA 的拼接 |
| G | Genetics | 1,150 | 14 个预设基因窗口中经训练集 QC 和 LD pruning 后的 0/1/2 SNP dosage |
| C | Cognition/health | 90 | 认知任务、睡眠和体力活动 |
| B | Behavior/environment | 219（221 原始列） | 非 ADHD 行为、冲动/奖赏、家庭/社区、社会经济、孕期与居住暴露 |

四类数据先各自形成一个长向量，再由各自的 patch encoder 转为 16 个 128 维 token。影像内部的四种成像并不是 CERD 的四个模型模态。

## 3. Imaging：785 维怎样得到

这些是 ABCD 已发布的区域/网络级 tabulated derivatives，不是本项目从 NIfTI 重新做预处理。

- rs-fMRI 439 维：91 个 Gordon 网络间唯一连接、247 个 Gordon 网络到皮层下连接、68 个 Desikan 皮层时间序列方差、33 个皮层下时间序列方差。
- task-fMRI 204 维：三个任务各 68 个 Desikan 区域 all-run beta contrast。SST 为 correct-stop vs correct-go；n-back 为 2-back vs 0-back；MID 为 reward anticipation vs neutral。使用 beta，不使用 t statistic。
- T1 sMRI 71 维：Desikan 区域灰质体积表型。
- DTI 71 维：对应区域的 fractional anisotropy，表示扩散方向一致性的白质微结构指标。

维数核对：`439 + 3×68 + 71 + 71 = 785`。队列中 2,101/2,120 人至少有一个影像字段。

## 4. Genetics：1,150 个 SNP 怎样得到

遗传模态只使用 ABCD 内部微阵列基因型；没有 ADHD PRS、没有祖源 PC、没有 CatBoost、没有外部 GWAS 权重。候选窗口为 hg19 基因坐标上下游各 20 kb，包含：DRD2/3/4/5、DBH、SLC6A3、SLC6A2、ADRA2A、MAOA、HTR2A、HTR1B、SLC6A4、GABRB3、SORCS3。

只在训练参与者上依次执行：变异缺失率 ≤2%；MAF ≥1%；`--indep-pairwise 50 5 0.8`；对保留变异 `--recode A` 得到 0/1/2 加性 dosage。2,313 个输入候选变异最终保留 1,150 个，其中各窗口计数为 ADRA2A 31、DBH 100、DRD2 61、DRD3 69、DRD4 31、DRD5 15、GABRB3 365、HTR1B 38、HTR2A 120、MAOA 30、SLC6A2 95、SLC6A3 97、SLC6A4 45、SORCS3 53。

SNP 列不做 PCA。模态内部缺失 dosage 用训练中位数填补，再按训练统计量标准化。2,078/2,120 人具有基因型记录。候选基因窗口是预测表示，不应解释成基因因果效应。

## 5. Cognition/health：90 维具体内容

- NIH Toolbox 10 维：晶体/流体/总认知及认知灵活性、抑制控制/注意、工作记忆、情景记忆、词汇、加工速度、阅读等年龄校正分数。
- WISC-V Matrix Reasoning 2 维：raw score 与 scaled score。
- Little Man Task 25 维：不同刺激条件正确数、反应时、错误、超时和效率。
- RAVLT 24 维：Trial 1–5、短/长延迟回忆、intrusion 与 repetition。
- Sleep Disturbance Scale 26 维：睡眠启动/维持、过度嗜睡、呼吸、觉醒、出汗和睡眠-觉醒转换。
- Physical activity 3 维：每周至少 60 分钟活动天数、力量活动天数、体育课天数。

这些变量与 ADHD 相关但不直接复用 K-SADS 标签字段。

## 6. Behavior/environment：具体包含什么

行为和家庭背景包括：非 ADHD 的 CBCL 焦虑、抑郁、攻击、对立、品行、社会、躯体、思维、退缩与压力条目/量表；BIS/BAS 行为抑制、drive、fun seeking、reward responsiveness；UPPS 的 urgency、计划、坚持、sensation seeking；家庭冲突、父母监护、社区安全；家庭收入、照护者教育/收入、婚姻和家庭困难。

环境暴露包括：Area Deprivation Index；Child Opportunity Index 总分以及教育、健康/环境、社会/经济分量；铅风险；24 小时平均噪声；年均 PM2.5 和 NO2；空气毒物呼吸危害；到主要道路距离；公园用地比例；孕期居住地 NO2、O3、PM2.5。

CBCL ADHD、attention problems、sluggish cognitive tempo 及宽泛 total/broadband 分数被排除，避免把标签近似复制到输入。K-SADS 诊断/症状列与四个预测模态列的交集为 0。B 表原始 221 列，经训练集缺失率/方差筛选后模型有效维数为 219。

## 7. 缺失和模态内缺值不是一回事

固定种子 2026，在每个划分内令约 15% 参与者缺至少一个整模态，以缺一个为主、缺两个为辅、缺三个极少，永远不允许四个都缺。最终可见模态数分布为：训练 `10/47/171/1292`、验证 `1/9/35/252`、测试 `1/7/37/258`，顺序对应可见 1/2/3/4 个模态。

mask 在列筛选、填补和标准化之前生效。每个可观测模态内部，删除训练缺失率 >80% 或近零方差列，用训练中位数填补并按训练统计量 z-score；验证和测试不重新估计任何统计量。整模态缺失一直由 observed mask 表示，只由 CERD 条件生成器处理。

## 8. CERD 与正式评估设置

ABCD 使用每模态 16 个 token、宽度 128、四头一层融合、8 个专家 top-2 路由、dropout 0.30、rank-4 patch adapter。每个缺失目标由同一参与者的可见 token 条件生成；observed/generated provenance 在 self-attention 与路由前加入。最终使用 1 个全局、4 个模态锚定和 6 个模态对锚定分支的归一化概率混合。

训练最多 50 epochs，AdamW，学习率 1e-4，weight decay 0.01，batch 64。每个 seed 在验证集选 checkpoint；二分类决策阈值也只在验证集按 Macro-F1 确定，然后冻结。正式程序必须先精确回放验证指标，再允许一次测试评估。

CERD 三种子正式结果为 Accuracy 78.44±0.76、Macro-F1 77.81±0.35、Macro-AUROC 85.12±0.05。全体测试上 AGDiC-inspired 的 Accuracy/Macro-F1 更高，Flex-MoE 的 AUC 更高，因此不能写 CERD 全指标第一；在 45 个实际缺失测试参与者中，CERD 与最佳 Accuracy 持平，并给出最高 Macro-F1 和 AUC。完整数值见 [`../results/abcd_current_binary_matched_v1.md`](../results/abcd_current_binary_matched_v1.md)。

## 9. 模态贡献的当前口径

旧的 branch allocation 是预测混合权重的描述量，不等同于删掉输入后的因果贡献。当前忠实度实验对原本完整的同一参与者严格移除一个模态、标记为 unavailable，并关闭该模态补全：

1. 模型输出相关性：统计移除后预测类别发生变化的比例，在每个 seed 内对四模态归一化到 100%；不使用标签。
2. 结果下降占比：四个移除实验的正 Accuracy 下降在每个 seed 内归一化到 100%；使用标签。
3. 两者独立计算后比较。ADNI Pearson r=0.979；ABCD r=0.947，且 ABCD 四模态排序完全一致（Spearman rho=1.0）。

ABCD 的相关性/下降占比分别为：Imaging 10.63/10.23，Genetics 23.58/20.12，Cognition/health 27.89/23.51，Behavior/environment 37.90/46.14。该结果说明模型输出变化和真实性能下降指向相同的模态顺序，而不是用路由权重替代贡献结论。
