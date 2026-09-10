# ABCD clinical-course 三分类：四个模态与预处理明细

> 当前实验版本：`abcd_adhd_course3_snp_v3_random_missing15`
>
> 适用标签：低症状且无 ADHD／既往 ADHD 或部分缓解／当前完整 ADHD
>
> 本文只描述本次实际进入模型的数据。ABCD 原始影像和问卷由 ABCD 项目采集并处理；本项目读取其发布的 Tabulated 派生表，不把上游处理写成我们重新完成的步骤。

## 1. 一张表看懂四个模态

| 代码 | 用人话说是什么 | 本次实际内容 | 构建维数 | 进入模型的数值 |
|---|---|---|---:|---|
| I | 一个人的大脑影像摘要 | 静息态 fMRI、SST/n-back/MID 任务态 fMRI、T1 结构像灰质信号、DTI 白质 FA 拼成一个影像模态 | 785 | 网络相关、区域 BOLD 波动、任务 contrast 的区域 beta、区域 T1 平均强度、区域 FA |
| G | 一个人在候选 ADHD 相关基因附近携带哪些等位基因 | 14 个预设基因窗口内、经过训练集 QC 和 LD pruning 的直接 SNP dosage；无 PRS、无祖源 PC | 1,183 | 每个 SNP 的 0/1/2 加性等位基因计数 |
| C | 一个人在认知任务、睡眠和运动方面的表现 | NIH Toolbox、WISC、Little Man、RAVLT、家长睡眠障碍量表、儿童体力活动 | 90 | 年龄校正认知分、正确数、反应时、记忆错误、睡眠条目和活动天数 |
| B | 一个人的非 ADHD 行为特点及其家庭、社会和居住环境 | 非 ADHD CBCL、BIS/BAS、UPPS、家庭冲突、父母监护、社区安全、家庭经济与居住地环境暴露 | 221（训练预处理后 219） | 条目分/量表分、家庭经济困难、收入教育、社区及地址链接的环境指标 |

需要特别注意：

- 四种影像被合并成模型的一个 **I 模态**，不是四个独立模型模态。
- 当前 **C 模态没有身高、体重或腰围**。这些人体测量出现在早期、已废弃的数据版本中，不能用于描述本次实验。
- 当前 T1 表是 **灰质区域平均 T1-weighted intensity**，不是皮层体积或厚度。
- K-SADS ADHD 诊断/症状、ADHD 药物和 CBCL attention/ADHD 字段都没有进入预测模态。

## 2. I：影像模态，785 维

### 2.1 影像组成

| 影像来源 | ABCD Tabulated 表 | 本次取出的量 | 维数 | 它描述什么 |
|---|---|---|---:|---|
| rs-fMRI 网络内/网络间连接 | `mr_y_rsfmri__corr__gpnet` | Gordon 网络之间的平均相关；对称矩阵只保留一个三角，含网络内项 | 91 | 静息状态下不同功能网络活动是否同步 |
| rs-fMRI 网络—皮层下连接 | `mr_y_rsfmri__corr__gpnet__aseg` | Gordon 网络与 FreeSurfer aseg 皮层下区域的平均相关 | 247 | 皮层功能网络与丘脑、纹状体等深部结构的功能耦合 |
| rs-fMRI 皮层波动 | `mr_y_rsfmri__var__dsk` | 68 个 Desikan 皮层区的 BOLD 时间方差 | 68 | 每个皮层区域自身的静息态波动幅度 |
| rs-fMRI 皮层下波动 | `mr_y_rsfmri__var__aseg` | 33 个皮层下/深部区域的 BOLD 时间方差 | 33 | 每个皮层下区域自身的静息态波动幅度 |
| SST task-fMRI | `mr_y_tfmri__sst__csvcg__dsk` | correct stop − correct go 的 all-run 区域 beta | 68 | 成功停止反应相对于正常按键时的脑激活差异，主要对应反应抑制 |
| Emotional n-back task-fMRI | `mr_y_tfmri__nback__2bv0b__dsk` | 2-back − 0-back 的 all-run 区域 beta | 68 | 高工作记忆负荷相对于低负荷时的脑激活差异 |
| MID task-fMRI | `mr_y_tfmri__mid__arvn__dsk` | anticipated reward − neutral 的 all-run 区域 beta | 68 | 期待可能获得奖励相对于中性提示时的脑激活差异 |
| T1 sMRI | `mr_y_smri__t1__gm__dsk` | 68 个左右半球 Desikan 区域的灰质平均 T1 强度，加左右半球及全脑 3 个汇总 | 71 | T1 加权结构像中灰质区域的平均信号强度；不是体积 |
| dMRI/DTI | `mr_y_dti__is__fa__wm__dsk` | inner-shell DTI 的 68 个 Desikan 对应白质区域平均 FA，加左右半球及全脑 3 个汇总 | 71 | 水分子扩散方向一致程度的区域摘要，反映白质微结构 |
| **合计** | 9 张表 | `439 rs-fMRI + 204 task-fMRI + 71 T1 + 71 DTI` | **785** | — |

### 2.2 task-fMRI 从扫描到本项目输入经历了什么

我们没有把四维 task-fMRI NIfTI 直接输入网络，也没有在本项目中重新拟合 GLM。实际数据链是：

| 阶段 | 谁完成 | 发生了什么 | 这一阶段的输出 |
|---|---|---|---|
| 1. 任务扫描 | ABCD | 儿童在扫描仪内完成 SST、Emotional n-back 和 MID；扫描得到随时间变化的 BOLD 序列 | 每个任务、每个 run 的 task-fMRI 时间序列 |
| 2. 影像预处理 | ABCD 上游影像流程 | 对扫描实施运动、几何畸变和空间配准等标准处理，使功能像能够与个体结构像及脑区分区对齐 | 可用于统计建模的预处理 BOLD |
| 3. 条件建模 | ABCD 上游影像流程 | 根据任务事件时间拟合条件相关 BOLD 反应，并计算预先定义的条件 contrast | SST 的 correct-stop vs correct-go、n-back 的 2-back vs 0-back、MID 的 anticipated-reward vs neutral |
| 4. 区域汇总 | ABCD 上游影像流程 | 按 Desikan atlas 的左右半球 68 个皮层 ROI，汇总每个 contrast 的平均 beta weight | 每人每个任务 68 个 beta，单位为 percent signal change |
| 5. 本项目取数 | 本项目 | 读取基线 `ses-00A` 的 all-run Tabulated 表，只保留字段名以 `_beta` 结尾的 68 列 | 三个任务共 204 维 |
| 6. 本项目预处理 | 本项目 | 与其他影像特征外连接；只用训练集完成列筛选、中位数填补和 z-score | 可送入 CERD 的数值矩阵及 observed mask |

“all-run beta”表示 ABCD 已经把该任务可用 run 的信息合并成一个区域 contrast 估计。它不是准确率，也不是反应时；例如 SST 的某个数值表示某个 Desikan 区域在“成功停止”相对于“正确 go”时的平均 BOLD 百分信号变化。原始任务表还提供 beta 的标准误字段，本项目不使用这些 `_sem` 字段，也不额外读取同一 contrast 的 t statistic 表，避免重复表达同一个效应。

### 2.3 rs-fMRI、sMRI 和 dMRI 的取数方式

- **rs-fMRI：** ABCD 上游对静息态 BOLD 完成标准预处理与低运动帧控制，然后发布网络/区域级相关和时间方差。本项目对 Gordon 网络相关矩阵删除重复的镜像项，再与网络—皮层下连接和区域方差拼接。
- **sMRI：** ABCD 已将 T1 结构像与 Desikan 皮层分区对应。我们读取每个灰质 ROI 的平均 T1-weighted intensity，以及左右半球/全脑汇总。这里不能写成“我们用 FreeSurfer 计算了皮层体积”，因为本次用的既不是体积表，也不是我们从原始像重新处理所得。
- **dMRI：** ABCD 上游由扩散扫描拟合 DTI，并在 inner shell 的 Desikan 对应白质区域发布平均 fractional anisotropy。FA 越高通常表示扩散方向更一致，但在本任务中它只是预测特征，不能直接解释为白质更好或更差。

## 3. G：SNP 遗传模态，1,183 维

G 模态不是基因表达，而是芯片 SNP 基因型经 QC 后导出的加性 dosage。一个 SNP 的数值通常为 0、1 或 2，表示个体携带 PLINK A1 等位基因的份数；缺测随后只用训练集的中位数填补。

### 3.1 候选窗口和保留维数

按 hg19 坐标取 14 个预先指定基因的基因区间，并向上下游各扩展 20 kb：

| 生物学方向 | 基因 | QC 与 LD pruning 后的 SNP 数 |
|---|---|---:|
| 多巴胺受体 | `DRD2`, `DRD3`, `DRD4`, `DRD5` | 63, 75, 29, 16 |
| 儿茶酚胺合成/转运 | `DBH`, `SLC6A3`, `SLC6A2` | 105, 100, 99 |
| 去甲肾上腺素受体 | `ADRA2A` | 31 |
| 5-HT 受体、转运和单胺代谢 | `HTR2A`, `HTR1B`, `SLC6A4`, `MAOA` | 122, 39, 46, 30 |
| GABA-A 受体 | `GABRB3` | 376 |
| 神经元分选/突触相关 | `SORCS3` | 52 |
| **合计** | 14 个窗口 | **1,183** |

这些窗口不等于覆盖了 ADHD 的全部遗传度，也不能把某个窗口的预测贡献解释成该基因的因果效应。

### 3.2 QC、LD pruning 与明确排除项

| 步骤 | 范围 | 具体规则 | 目的 |
|---|---|---|---|
| 样本限定 | 固定训练集内有芯片基因型的 2,470 人 | PLINK `--keep` | 不让验证/测试参与任何变异选择 |
| 变异缺失率 | 训练样本 | `--geno 0.02` | 删除训练集中缺失率超过 2% 的 SNP |
| 次要等位基因频率 | 训练样本 | `--maf 0.01` | 删除训练集中 MAF 小于 1% 的 SNP |
| LD pruning | 训练样本 | `--indep-pairwise 50 5 0.8` | 在 50-SNP 滑窗中每次移动 5 个 SNP，减少高度相关的冗余变异 |
| 加性编码 | 全队列，仅保留训练集选出的 SNP | PLINK `--recode A` | 导出 0/1/2 A1 allele count |

候选窗口原有 2,313 个 SNP，最终保留 1,183 个。这里的“降维”只指 QC 和 LD pruning；**没有 PCA**。模型看到全部 1,183 个保留 dosage。

本次没有 ADHD PRS、外部 GWAS 权重或 genetic ancestry principal components（检测到的 32 个祖源 PC 全部排除），也没有按标签筛 SNP。当前代码没有做 HWE、个体杂合度或样本缺失率 QC，因此文稿不能声称做过。

## 4. C：认知、睡眠与体力活动，90 维

| 来源 | 谁完成/报告 | 实际输入 | 维数 | 用人话解释 |
|---|---|---|---:|---|
| NIH Toolbox | 儿童完成 | 3 个年龄校正复合分，以及 DCCS、Flanker、List Sorting、Picture Sequence、Picture Vocabulary、Pattern Comparison、Oral Reading 的年龄校正分 | 10 | 总体认知、认知灵活性、抑制控制与注意、工作记忆、情景记忆、词汇、加工速度和阅读能力 |
| WISC-V Matrix Reasoning | 儿童完成 | raw score 与 scaled score | 2 | 看图形规律并选择缺失图形，反映非言语推理 |
| Little Man Task | 儿童完成 | 8 类视角/方向条件的正确数和正确反应时，加总正确率、正确/错误/超时、错误比例、反应时和效率 | 25 | 视空间转换、速度—准确性权衡和执行控制 |
| RAVLT | 儿童完成 | List B、学习 Trial 1–5、短延迟 Trial 6、长延迟 Trial 7；每阶段记录正确、侵入词和重复词 | 24 | 言语学习、即时/延迟记忆以及记忆错误模式 |
| Sleep Disturbance Scale | 家长报告 | 26 个睡眠条目 | 26 | 睡眠时长/入睡时间、入睡与维持困难、夜醒、日间嗜睡、梦魇/夜惊、夜间呼吸、打鼾、出汗、磨牙、肢体抽动等 |
| Physical Activity | 儿童报告 | 过去 7 天每天至少活动 60 分钟的天数、力量训练天数、通常每周体育课天数 | 3 | 日常活动量和学校体育参与 |
| **合计** | — | — | **90** | — |

### 4.1 睡眠 26 项具体覆盖什么

| 睡眠领域 | 条目数 | 具体例子 |
|---|---:|---|
| 睡眠觉醒异常 | 3 | 梦游、夜间惊叫/意识混乱、次日不记得的噩梦 |
| 入睡和睡眠维持 | 7 | 每晚睡眠时长、入睡需要多久、不愿上床、难入睡、入睡焦虑、夜醒超过两次、醒后难再睡 |
| 过度嗜睡 | 5 | 早晨难唤醒、醒来仍疲劳、醒来不能动、白天困倦、不合时宜地突然睡着 |
| 睡眠多汗 | 2 | 入睡时或整夜出汗过多 |
| 睡眠呼吸 | 3 | 夜间呼吸困难、喘气/呼吸暂停、打鼾 |
| 睡眠—觉醒转换 | 6 | 入睡时惊跳、摇摆/撞头、入睡幻景、夜间腿部抽动、说梦话、磨牙 |

C 是当前认知/健康功能测量，不是“所有身体指标”模态。本次没有身高、体重、BMI、腰围或青春期发育量表。

## 5. B：非 ADHD 行为与环境，221 维

### 5.1 行为量表

| 来源 | 报告者 | 实际内容 | 维数 |
|---|---|---|---:|
| CBCL | 家长 | 焦虑/抑郁、攻击、对立、品行、规则破坏、社会问题、躯体不适、思维问题、退缩/抑郁、压力和强迫等非 ADHD 条目及安全的 syndrome/DSM-oriented T-score | 122 |
| BIS/BAS | 儿童 | BAS drive、fun seeking、reward responsiveness 和 BIS 行为抑制；保留条目和各分量表总分 | 24 |
| UPPS | 儿童 | negative urgency、positive urgency、缺乏计划、缺乏坚持、sensation seeking；保留条目和各分量表总分 | 25 |

CBCL 中直接描述 ADHD、attention problems 或 sluggish cognitive tempo 的字段及相应汇总分均被排除。B 仍包含与 ADHD 共病常见的焦虑、对立、冲动和奖赏敏感性，但不复制 K-SADS 标签。

### 5.2 家庭与社区环境

| 来源 | 报告者 | 逐项测量什么 | 维数 |
|---|---|---|---:|
| Family Environment Scale：Conflict | 家长 | 家庭争吵、公开生气、扔东西、发脾气、互相批评、肢体冲突、冲突后和解、家庭成员竞争等 9 项及均值 | 10 |
| Parental Monitoring | 儿童 | 父母是否知道孩子在哪里、和谁在一起；独自在家时能否联系父母；是否沟通第二天计划；每周共同晚餐次数，以及均值 | 6 |
| Neighborhood Safety & Crime | 家长 | 昼夜在社区步行是否安全、社区暴力是否成问题、是否免受犯罪威胁，以及均值 | 4 |

### 5.3 家庭结构与社会经济背景

| 类别 | 具体字段 | 维数 |
|---|---|---:|
| 居住安排 | 孩子是否还在另一家庭长期居住、每周在另一家庭停留多少小时 | 2 |
| 教育 | 主要照护者最高学历、伴侣最高学历 | 2 |
| 过去一年经济困难 | 无力购买食物、电话停机、付不起房租/房贷、被驱逐、能源服务被切断、因费用放弃就医、因费用放弃牙科 | 7 |
| 收入 | 家庭总收入、照护者本人收入、伴侣收入 | 3 |
| 家庭结构 | 主要照护者婚姻/伴侣状态 | 1 |
| **合计** | — | **15** |

### 5.4 地址链接的环境因素

以下 15 个变量按参与者基线主要居住地址 `addr1` 链接到地区或环境模型，不是儿童佩戴设备直接测得的个人剂量。

| 环境变量 | 实际含义 | 数据形式 |
|---|---|---|
| Area Deprivation Index | 居住地区社会经济剥夺程度 | 全国百分位；数值越高通常表示地区剥夺越高 |
| Child Opportunity Index：overall | 儿童在居住地区总体可获得的机会水平 | 全国 z-score |
| COI：education | 学校和教育资源机会 | 全国 z-score |
| COI：health/environment | 健康与环境条件机会 | 全国 z-score |
| COI：social/economic | 社会与经济资源机会 | 全国 z-score |
| Lead-risk index | census tract 层面的铅暴露风险代理 | 风险指数，不是儿童血铅浓度 |
| Environmental noise | 主要地址周围的 24 小时平均总声级 | 地址链接的平均声级 |
| PM2.5 | 2016 年居住地年均细颗粒物 | 地址链接的年均浓度 |
| NO2 | 2016 年居住地年均二氧化氮 | 地址链接的年均浓度 |
| NATA respiratory hazard | National Air Toxics Assessment 的呼吸危害 | 地区呼吸危害指数 |
| Road proximity | 主要地址到主要道路的距离 | 米；反映交通接近度的代理 |
| Park-land proportion | census tract 中公园用地占比 | 比例；反映附近绿色空间 |
| Prenatal NO2 | 孕期居住地址的 NO2 暴露估计 | 地址链接的孕期平均值 |
| Prenatal O3 | 孕期居住地址的臭氧暴露估计 | 地址链接的孕期平均值 |
| Prenatal PM2.5 | 孕期居住地址的 PM2.5 暴露估计 | 地址链接的孕期平均值 |

这些值是环境暴露的代理指标。横断面分类可以说明它们提供预测信息，但不能据此声称某一种污染物导致了个体 ADHD。

## 6. 四个模态如何变成最终模型输入

| 顺序 | 操作 | 使用哪些数据 | 目的 |
|---|---|---|---|
| 1 | 固定 clinical-course 标签与纳入队列 | 基线家长版 K-SADS | 先定义临床任务，再准备预测特征 |
| 2 | family-disjoint 划分 | 遗传家庭 ID | 同一家系不跨训练、验证、测试 |
| 3 | 训练集内遗传 QC/LD pruning | 仅训练集基因型 | 验证/测试不参与 SNP 选择 |
| 4 | 物化 I/G/C/B 四张参与者级表 | 基线 `ses-00A` Tabulated 数据 | 固定来源和列定义 |
| 5 | 应用固定整模态缺失 mask | 每个 split 内约 15% 的人；主要缺 1–2 个模态、少量缺 3 个、绝不四个全缺 | 所有方法看到同一缺失背景 |
| 6 | 删除高缺失或零方差列 | 仅训练集中仍可见的模态值；缺失率上限 80%，方差下限 `1e-8` | 删除无法稳定学习的列 |
| 7 | 逐列中位数填补和 z-score | 统计量只由训练集中真实可见值拟合 | 不让验证/测试信息进入预处理 |
| 8 | 固定变换验证和测试 | 复用训练列集合、中位数、均值和标准差 | 保持评估独立 |
| 9 | 同时传递数值和 observed mask | 四个模态 | 填补值不会被模型误认为真实观测模态 |

数据构建维数为 `I=785, G=1183, C=90, B=221`。在当前固定训练集上，B 有 2 列因训练可见值的缺失/方差规则被移除，因此训练程序实际报告 `B=219`；其他三个模态维数不变。

## 7. 答辩时可以直接使用的简短版本

> 我们把 ABCD 基线数据整理成四个模型模态。影像模态把 rs-fMRI、三个 task-fMRI contrast、T1 灰质区域平均强度和 DTI 白质 FA 拼在一起。task-fMRI 不是原始图像：ABCD 已经完成 BOLD 预处理、任务条件建模和 Desikan ROI 汇总，我们读取 SST 的 correct-stop vs correct-go、n-back 的 2-back vs 0-back、MID 的 anticipated-reward vs neutral 三组 all-run beta，每组 68 维。遗传模态是 14 个预设 ADHD 相关基因窗口中的直接 SNP dosage，只在训练集做缺失率、MAF 和 LD pruning，2313 个候选 SNP 最终保留 1183 个；没有 PRS，也没有祖源 PC。第三个模态是认知、睡眠和体力活动，包括 NIH Toolbox、WISC、Little Man、RAVLT、睡眠障碍条目和活动天数，不含身高体重。第四个模态包括非 ADHD 行为量表、家庭冲突、父母监护、社区安全、家庭经济状况，以及 ADI、COI、铅风险、噪声、空气污染、道路、公园和孕期污染暴露等地址链接环境指标。所有填补和标准化参数都只由训练集拟合。

## 8. 数据依据

- [ABCD Task-fMRI documentation](https://docs.abcdstudy.org/latest/documentation/imaging/type_tfmri.html)
- [ABCD Resting-state fMRI documentation](https://docs.abcdstudy.org/v/6_0_0/documentation/imaging/type_rsfmri.html)
- [ABCD Genetics documentation](https://docs.abcdstudy.org/v/6_0_0/documentation/non_imaging/gn.html)
- [ABCD Neurocognition documentation](https://docs.abcdstudy.org/v/6_0_2/documentation/non_imaging/nc.html)
- [ABCD Mental Health documentation](https://docs.abcdstudy.org/v/6_0_0/documentation/non_imaging/mh.html)
- [ABCD Physical Health documentation](https://docs.abcdstudy.org/v/6_0_2/documentation/non_imaging/ph.html)
- [ABCD External Linked Data documentation](https://docs.abcdstudy.org/latest/documentation/external_linked/external-linked-data.html)
- [Hagler et al., 2019, ABCD image processing and analysis methods](https://doi.org/10.1016/j.neuroimage.2019.116091)

本仓库不发布 ABCD 受控 parquet、PLINK 文件、参与者 ID、家庭 ID 或逐参与者预测；文档中的表名、字段定义、维数和聚合计数用于复现审计。
