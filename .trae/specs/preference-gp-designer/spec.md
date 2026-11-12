# 偏好高斯过程 Designer（Preference GP Designer）- 产品需求文档

## Overview
- **Summary**: 在 Vizier 中新增一个仅依赖成对偏好（胜负）比较、无需数值目标的 Designer。它通过带高斯噪声的 probit 偏好高斯过程（Laplace 近似）从 Trial 成对胜负中学习潜效用函数与核超参数，支持对新参数对预测胜出概率与效用差的后验不确定性，并以 Thompson 采样给出去重的批量建议。
- **Purpose**: 服务于只能获得成对偏好反馈（人类偏好、A/B 成对实验、排序评价）而无法获得数值目标的黑箱优化场景。
- **Target Users**: Vizier 算法使用者、偏好学习/贝叶斯优化研究人员、基准实验开发者。

## Goals
- G1: 从带高斯噪声的成对胜负数据学习潜效用 GP 后验，并自动学习核参数（信号方差、连续/类别 ARD 长度尺度）与比较噪声。
- G2: 无需任何数值目标指标即可运行：`ProblemStatement` 允许不含 `MetricInformation`。
- G3: 对任意新参数对输出胜出概率，以及效用差的后验均值与标准差；对单点输出潜效用后验均值/标准差。
- G4: `suggest(count)` 支持批量建议，批量内参数互不重复且合法。
- G5: 对重复参数、矛盾比较、增量反馈三类数据情形保持预测有效、不崩溃。
- G6: 纯增量交付，现有数值研究、既有 Designer 与测试保持兼容。

## Non-Goals
- NG1: 不支持平局（tie）标签，仅支持二元胜负。
- NG2: 不支持条件（conditional）搜索空间，与 `VizierGPBandit` 的限制一致。
- NG3: 不做多任务/多目标偏好学习，不做偏好的主动学习配对选择（pair selection）策略。
- NG4: 不实现 `PartiallySerializableDesigner` 的 dump/load（与 `VizierGPBandit` 同为普通 `Designer`）。
- NG5: 不修改任何现有 Designer、实验器、基准 runner 的行为；不引入新的第三方依赖。

## Background & Context
- 仓库为 Google Vizier OSS。现有 `VizierGPBandit`（[gp_bandit.py](file:///Users/xavskye/workspace/Projects/Phecda/SWEMarkup/data/runs/cb385d1c-154e-4da3-93e8-7c4a2cc45a08/repo/vizier/_src/algorithms/designers/gp_bandit.py)）面向数值指标，依赖 `TrialToModelInputConverter` 特征转换、TFP-JAX 的 Matérn ARD 核（`FeatureScaledWithCategorical`）与 jaxopt 优化器。
- 偏好学习标准模型（Chu & Ghahramani 2005 路线；Rasmussen & Williams 第 3 章二分类 GP）：潜函数 f ~ GP，观测为成对符号偏好，probit 似然 `P(i≻j|f)=Φ((f_i-f_j)/s)`；高斯评估噪声 ε~N(0,σ_n²) 作用于两次评估时差，给出 `s=√(2σ_n²)`；后验用 Laplace 近似（牛顿迭代 + Cholesky），核超参数用 Laplace 边际似然做类型-II 极大似然。
- 已验证：环境内 jax 0.10.x + tfp-nightly[jax] 可用；`tfd.Normal.log_cdf` 在 ±30 量级稳定；proto 构建与现有测试（quasi_random_test）可运行。
- 设计决策（用户授权由实现方决定）：
  1. 数据接入双轨：显式 `report_preference(winner, loser)` + 标准 `update()` 解析 Trial metadata（namespace `preference_gp`，键 `winner_id`/`loser_id`）。
  2. 支持连续 + 离散/类别混合 flat 空间。
  3. 批量建议采用后验 Thompson 采样 + 候选池去重。

## Functional Requirements
- **FR-1**: 提供不可变数据类 `PreferenceComparison(winner, loser)`，二者为 `TrialSuggestion`/`Trial` 的参数赋值。
- **FR-2**: `PreferenceGPDesigner.report_preference(winner, loser)` 追加一条胜负比较；同参数重复比较、双向矛盾比较均可追加。
- **FR-3**: `update(completed, all_active)` 满足 `Designer` 契约：对含 `preference_gp` 元数据（`winner_id`/`loser_id`）的 completed trial，依据已累积 trial 存储解析为比较；不含该元数据的普通数值 trial 被安全忽略，不抛异常。
- **FR-4**: 特征转换复用 `TrialToModelInputConverter.from_problem(scale=True, max_discrete_indices=0)`；核为 Matérn 5/2 + `FeatureScaledWithCategorical` 的 ARD 核，覆盖连续与类别维度。
- **FR-5**: 似然为 probit：`P(winner≻loser)=Φ((f_w-f_l)/√(2σ_n²))`，其中 σ_n² 为可学习的高斯评估噪声方差。
- **FR-6**: Laplace 后验：固定次数牛顿迭代求潜函数后验众数，经 `I+W^{1/2}TKT'W^{1/2}` 的 Cholesky 稳定求解；Cholesky 失败时按递增 jitter 回退。
- **FR-7**: 核超参数（信号方差、各维长度尺度、噪声方差）通过最大化 Laplace 对数边际似然学习，使用多重启 LBFGS（jaxopt），在无约束 log 空间优化。
- **FR-8**: `predict_utility(trials)` 实现 `Predictor` 风格预测，返回 `vza.Prediction(mean, stddev)`；`predict(trials, rng=None, num_samples=None)` 为同义接口。
- **FR-9**: `predict_win_probability(trials_a, trials_b)` 返回逐对胜出概率数组；`predict_utility_difference(trials_a, trials_b)` 返回效用差的后验均值与标准差。
- **FR-10**: `suggest(count)`：数据不足时返回准随机种子建议；否则在准随机候选池上拟合后验，以 Thompson 采样逐次取 argmax 并排除已观测/已选重复点，凑满 count；所有建议参数合法。
- **FR-11**: 内部按特征行去重维护唯一参数点索引，比较以唯一点索引对存储；增量数据按比较计数使拟合并发缓存失效，无新数据时复用缓存。
- **FR-12**: 新类经 `vizier/algorithms/designers/__init__.py` 导出为公共 API `PreferenceGPDesigner`，并提供 `from_problem(problem, seed=None, **kwargs)` 工厂。

## Non-Functional Requirements
- **NFR-1（数值稳定性）**: 全部预测输出为有限值；probit 对数似然使用稳定 `log_cdf`/`log_pdf`；置信区间极端值不产生 NaN/Inf；矛盾与重复数据下 Cholesky 保持正定。
- **NFR-2（性能）**: 核心拟合/预测函数 JIT 编译；百级唯一参数点、百级比较规模下单次 `suggest` 秒级完成（测试机 CPU）。
- **NFR-3（工程一致性）**: 遵循仓库 License 头、`attrs`/`equinox` 风格、float64、absl.testing 测试约定；数学核心与 Designer 分层为两个模块。
- **NFR-4（依赖约束）**: 仅使用 requirements-jax.txt 中已有依赖（jax、jaxopt、tfp-jax、equinox、flax、chex、jaxtyping）。

## Constraints
- **Technical**: flat 非条件搜索空间；JAX float64；TFP-JAX 核与分布；原型/服务无关的纯 Python 实现。
- **Business**: 不破坏任何现有测试与公共 API 行为；交付物位于 `vizier/_src/algorithms/designers/preference_gp/`。
- **Dependencies**: `vizier.pyvizier.converters`、`vizier._src.jax.types`、`vizier._src.algorithms.designers.quasi_random`、jaxopt/tfp-jax。

## Assumptions
- A1: 偏好仅二元胜负；平局不在本期范围。
- A2: 元数据路径中 `winner_id`/`loser_id` 引用同一 designer 通过 `update()` 已见的 trial `id`；无法解析的 id 抛 `ValueError` 并给出可诊断信息。
- A3: 效用只可辨识到相对差值（GP 零均值先验），不承诺绝对标度。
- A4: 候选池由准随机序列生成，规模可配置，覆盖空间均匀性由 Halton 序列保证。

## Acceptance Criteria

### AC-1: 公共 API 与基本契约
- **Type**: `rule`
- **Given**: 安装好 vizier 的环境
- **When**: 执行 `from vizier.algorithms.designers import PreferenceGPDesigner` 并以无指标 `ProblemStatement` 实例化
- **Then**: 导入与实例化成功；类暴露 `report_preference/update/suggest/predict_utility/predict_win_probability/predict_utility_difference/from_problem`
- **Pass Condition**: 导入与方法存在性断言全部通过
- **Evidence**: 对应测试用例与运行输出

### AC-2: 种子期与空数据行为
- **Type**: `rule`
- **Given**: 未上报任何比较的 designer
- **When**: 调用 `suggest(count=3)` 与 `predict_utility`/`predict_win_probability`
- **Then**: 返回 3 个合法且互不重复的参数建议；预测返回有限值（先验均值 0、胜出概率 0.5）
- **Pass Condition**: 建议数量、合法性、有限值断言通过
- **Evidence**: designer 测试输出

### AC-3: 胜出概率的方向性与对称性
- **Type**: `rule`
- **Given**: 在 1D 合成偏好数据（由已知效用函数加高斯噪声生成胜负）上完成拟合
- **When**: 对高效用点 a 与低效用点 b 计算双向胜出概率
- **Then**: `P(a≻b) > 0.5`，且 `P(a≻b)+P(b≻a)=1`（容差 1e-6），所有概率位于 (0,1)
- **Pass Condition**: 断言全部成立
- **Evidence**: 核心模块测试输出

### AC-4: 后验不确定性收缩
- **Type**: `rule`
- **Given**: 同一合成数据集
- **When**: 比较观测密集点的后验效用标准差与先验标准差
- **Then**: 后验标准差显著小于先验；所有 mean/stddev 有限无 NaN
- **Pass Condition**: 收缩断言与有限性断言通过
- **Evidence**: 核心模块测试输出

### AC-5: 重复参数鲁棒性
- **Type**: `rule`
- **Given**: 含大量重复参数赋值与重复比较对的数据集
- **When**: 拟合并预测
- **Then**: 内部唯一参数点数等于去重后数量（不随重复对膨胀）；拟合与预测无异常、输出有限
- **Pass Condition**: 唯一点数断言与无异常运行通过
- **Evidence**: 核心模块测试输出

### AC-6: 矛盾比较鲁棒性
- **Type**: `rule`
- **Given**: 同一参数对被上报双向胜负（各若干次）
- **When**: 拟合后预测该参数对胜出概率
- **Then**: 不抛异常且输出有限；等量双向证据下胜出概率趋近 0.5（容差 0.1）
- **Pass Condition**: 有限性与概率区间断言通过
- **Evidence**: 核心模块测试输出

### AC-7: 增量反馈等价性与缓存
- **Type**: `rule`
- **Given**: 同一批比较分别 (a) 一次性上报、(b) 分三次增量上报
- **When**: 两组各自拟合后在相同测试点预测
- **Then**: 胜出概率与效用后验数值一致（容差 1e-6）；无新数据的重复拟合命中缓存（ incorporated 计数不变）
- **Pass Condition**: 数值等价断言与缓存计数断言通过
- **Evidence**: designer 测试输出

### AC-8: update 元数据路径与普通 trial 兼容
- **Type**: `rule`
- **Given**: (a) 带 `preference_gp:winner_id/loser_id` 元数据的 completed trials；(b) 仅含数值 metric 的普通 completed trials
- **When**: 分别调用 `update()`
- **Then**: (a) 正确解析并累积比较；(b) 不抛异常、不产生偏好数据；无法解析的 id 抛 `ValueError`
- **Pass Condition**: 三类断言通过
- **Evidence**: designer 测试输出

### AC-9: 批量建议合法性与去重
- **Type**: `rule`
- **Given**: 已拟合足够偏好数据的混合空间 designer
- **When**: 调用 `suggest(count=5)`
- **Then**: 返回 5 个建议，全部通过搜索空间 `assert_contains`，参数赋值两两不同，且不与已观测参数点重复
- **Pass Condition**: 数量、合法性、去重断言通过
- **Evidence**: designer 测试输出

### AC-10: 混合空间端到端
- **Type**: `rule`
- **Given**: 含 DOUBLE/INTEGER/CATEGORICAL 参数的 flat 空间
- **When**: 执行 suggest→偏好上报→update→predict 全流程多轮
- **Then**: 全程无异常，预测形状正确且有限
- **Pass Condition**: 端到端测试通过
- **Evidence**: designer 测试输出

### AC-11: 现有数值研究兼容
- **Type**: `rule`
- **Given**: 未改动既有算法代码（仅新增文件与一行导出）
- **When**: 运行 `quasi_random_test.py`、`gp_bandit_test.py` 等既有设计师测试
- **Then**: 既有测试全部通过
- **Pass Condition**: 退出码 0，无失败用例
- **Evidence**: pytest 运行日志

### AC-12: 超参数学习有效性
- **Type**: `rule`
- **Given**: 合成偏好数据集与一组固定初始超参数
- **When**: 运行多重启 LBFGS 最大化 Laplace 边际似然
- **Then**: 优化后负对数边际似然不高于初始值；学得长度尺度与噪声均为正且有限
- **Pass Condition**: 似然下降与参数正定性断言通过
- **Evidence**: 核心模块测试输出

### AC-13: 端到端偏好优化效果
- **Type**: `rubric`
- **Dimension**: 偏好优化有效性
- **Scale**: 1-5
- **Anchors**: 1 = 建议点效用与随机无差异或无法运行；3 = 建议点平均效用稳定优于随机基线但优势微弱；5 = 多轮后建议集中于高效用区域，Top 建议效用显著优于随机基线（检验通过）
- **Pass Threshold**: >= 4
- **Evidence**: 合成 oracle 循环实验测试（含统计对比输出）

### AC-14: 工程质量
- **Type**: `rubric`
- **Dimension**: 模块化、可读性与测试覆盖
- **Scale**: 1-5
- **Anchors**: 1 = 数学与流程耦合、无文档字符串、无边界测试；3 = 分层基本清晰、核心路径有测试；5 = 数学核心与 Designer 完全解耦、公共 API 均有 docstring 与示例、边界（空/重复/矛盾/增量）测试齐备
- **Pass Threshold**: >= 4
- **Evidence**: 代码结构与测试清单评审

## Open Questions
- 无遗留开放问题。三项设计选择（双轨数据接入、混合空间、Thompson 批量）已由用户授权实现方决策，记录于 Background & Context。
