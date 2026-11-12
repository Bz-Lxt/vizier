# 偏好高斯过程 Designer - 实现计划

## Task 1: 偏好 GP 数学核心模块（probit + Laplace + 超参学习）
- **Status**: `completed`
- **Completion Evidence**:
  - 新增 [preference_gp.py](file:///Users/xavskye/workspace/Projects/Phecda/SWEMarkup/data/runs/cb385d1c-154e-4da3-93e8-7c4a2cc45a08/repo/vizier/_src/algorithms/designers/preference_gp/preference_gp.py)：log 空间超参、Matérn5/2 ARD（连续/类别/混合三分支）、稳定 inverse Mills、Laplace 牛顿（Woodbury 精确牛顿步，修正了 T≠I 下的代数错误）、LBFGS 多重启证据优化、后验协方差（Woodbury 单段式，修正了错误的 I-B⁻¹ 拆分）、胜出概率/效用差/Thompson。
  - 冒烟验证：12 点 60 比较下效用相关 0.997、P(a≻b)+P(b≻a)=1、后验 std 0.128<先验 0.144、矛盾对 P=0.5、空/单点/自比较/重复/纯类别/混合均有限；40 点 200 比较众数梯度残差 1.6e-8、NLL 优化下降、协方差 PSD（最小特征值 1e-6）。
- **Priority**: high
- **Depends On**: None
- **Description**:
  - 新建 `vizier/_src/algorithms/designers/preference_gp/__init__.py` 与 `preference_gp.py`。
  - 实现纯 JAX 数学层，不依赖 vizier Trial 类型，输入为连续/类别特征数组与比较索引对：
    - log 空间超参数容器（信号方差、连续/类别 ARD 长度尺度、高斯评估噪声方差）与 Matérn5/2 + `FeatureScaledWithCategorical` 核构建。
    - 稳定 probit 对数似然（`jax.scipy.stats.norm.logcdf/logpdf` 或等价稳定式），尺度 `s=√(2σ_n²)`。
    - Laplace 牛顿迭代（固定步数 + 收敛早退），经 `I+W^{1/2}TKT'W^{1/2}` Cholesky 求解，jitter 递增回退；输出后验众数 f̂ 与缓存量（L、√W、T、K、α）。
    - Laplace 对数边际似然；jaxopt LBFGS 多重启优化 log 空间超参，保留最优。
    - 后验预测：任意测试点的效用均值/协方差；胜出概率 `Φ(Δm/√(Δv+2σ_n²))`；效用差后验 mean/std；Thompson 采样（多元正态协方差采样）。
    - 边界：零比较/单点/自比较/全矛盾情形均返回有限值，不抛异常。
- **Acceptance Criteria Addressed**: AC-3, AC-4, AC-5, AC-6, AC-12
- **Test Requirements**:
  - `rule` TR-1.1: 合成 1D 数据上高效用点胜出概率 >0.5、双向概率和为 1（容差 1e-6）、概率 ∈(0,1)；证据：测试断言输出。
  - `rule` TR-1.2: 观测点后验 std < 先验 std，输出全有限；证据：测试断言输出。
  - `rule` TR-1.3: 重复特征行下核矩阵唯一点数正确、运行无异常；证据：测试断言输出。
  - `rule` TR-1.4: 等量双向矛盾比较下胜出概率 ∈[0.4,0.6] 且有限；证据：测试断言输出。
  - `rule` TR-1.5: 优化后负对数边际似然 ≤ 初始值，超参均为正有限；证据：测试断言输出。
  - `rule` TR-1.6: 空数据/单点/自比较输入不抛异常且输出有限；证据：测试断言输出。
- **Notes**: 核心函数使用 `eqx.filter_jit`；float64；模块内定义常量与默认值（牛顿步数、重启数、jitter 序列）。

## Task 2: 数学核心单元测试
- **Status**: `completed`
- **Completion Evidence**:
  - 新增 [preference_gp_test.py](file:///Users/xavskye/workspace/Projects/Phecda/SWEMarkup/data/runs/cb385d1c-154e-4da3-93e8-7c4a2cc45a08/repo/vizier/_src/algorithms/designers/preference_gp/preference_gp_test.py)，11 个用例覆盖 TR-1.1~TR-1.6（方向/对称、趋势恢复、不确定性收缩与 PSD、重复行、矛盾校准、NLL 下降、空/单点/自比较、纯类别/混合核、Thompson、效用差形状），pytest 全部通过。
- **Priority**: high
- **Depends On**: Task 1
- **Description**:
  - 新建 `vizier/_src/algorithms/designers/preference_gp/preference_gp_test.py`，absl.testing 风格，文件头启用 `jax_enable_x64`。
  - 覆盖 TR-1.1~TR-1.6：合成效用 oracle（如 sin/二次峰函数）加高斯噪声生成比较；含独立的稳定性、对称性、收缩、去重、矛盾、超参优化用例。
- **Acceptance Criteria Addressed**: AC-3, AC-4, AC-5, AC-6, AC-12
- **Test Requirements**:
  - `rule` TR-2.1: 全部用例在 venv 下 `pytest` 通过；证据：pytest 退出码 0 日志。
- **Notes**: 测试不依赖 vizier Trial，仅用 jax/numpy 数组，保证数学层可独立验证。

## Task 3: PreferenceGPDesigner 实现
- **Status**: `completed`
- **Completion Evidence**:
  - 新增 [preference_gp_designer.py](file:///Users/xavskye/workspace/Projects/Phecda/SWEMarkup/data/runs/cb385d1c-154e-4da3-93e8-7c4a2cc45a08/repo/vizier/_src/algorithms/designers/preference_gp/preference_gp_designer.py)：`PreferenceComparison`、双轨接入（report_preference + update 元数据 winner_id/loser_id，未知 id/残缺键抛 ValueError、普通数值 trial 忽略）、特征行字节哈希去重、按比较计数失效的拟合缓存、predict/predict_utility/predict_win_probability/predict_utility_difference、种子期准随机 + GP 期 Thompson 批量（红撞重抽、均值兜底、空间校验、devinfo metadata）、from_problem 工厂。
  - 冒烟验证：空数据先验（均值 0、P=0.5）、6 轮 oracle 循环后批量建议聚集于真实最优点 x≈0.52、批量 5 个互不重复且不撞已观测点、缓存复用/失效正确、增量与一次性上报预测完全一致（1e-6）、混合空间端到端、矛盾比较不崩溃、元数据解析与异常路径均符合预期。
- **Priority**: high
- **Depends On**: Task 1
- **Description**:
  - 新建 `preference_gp_designer.py`：
    - `PreferenceComparison`（frozen attrs，winner/loser 为 `TrialSuggestion`）。
    - `PreferenceGPDesigner(vza.Designer, vza.Predictor)`：构造参数含 `num_seed_trials`、`ard_restarts`、`newton_iters`、`candidate_pool_size`、`rng` 等；`from_problem` 工厂。
    - `report_preference(winner, loser)`：深拷贝追加，使拟合缓存失效。
    - `update(completed, all_active)`：累积 trial 存储；解析 namespace `preference_gp` 下 `winner_id`/`loser_id` 元数据为比较；无元数据 trial 忽略；未知 id 抛 `ValueError`。
    - 特征行去重：以转换器输出的连续+类别特征行做字节哈希建唯一索引，比较映射为唯一点索引对。
    - `_fit()` 懒训练并按比较计数缓存；调用 Task 1 核心。
    - `predict_utility/predict`（返回 `vza.Prediction`）、`predict_win_probability`、`predict_utility_difference`；空数据走先验。
    - `suggest(count)`：种子期用 `QuasiRandomDesigner`；之后准随机候选池 + Thompson 采样逐个 argmax，排除已观测/已选重复点，不足时以最高后验均值补齐；建议带 devinfo metadata。
    - 全部输出 numpy 有限值校验。
- **Acceptance Criteria Addressed**: AC-1, AC-2, AC-7, AC-8, AC-9, AC-10
- **Test Requirements**:
  - `rule` TR-3.1: 无指标 ProblemStatement 可实例化，全部公共方法可调用；证据：测试断言。
  - `rule` TR-3.2: 元数据解析、普通 trial 忽略、未知 id 报错三类行为正确；证据：测试断言。
  - `rule` TR-3.3: 增量与一次性上报预测数值一致（1e-6），缓存计数行为正确；证据：测试断言。
  - `rule` TR-3.4: 批量建议数量正确、两两不同、不重复已观测点、通过 `assert_contains`；证据：测试断言。
  - `rule` TR-3.5: 混合空间多轮 suggest/report/predict 无异常且形状正确；证据：测试断言。
- **Notes**: 不实现序列化；转换器禁用 metric 翻转且允许空 metric 配置。

## Task 4: Designer 单元测试与端到端效果实验
- **Status**: `completed`
- **Completion Evidence**:
  - 新增 [preference_gp_designer_test.py](file:///Users/xavskye/workspace/Projects/Phecda/SWEMarkup/data/runs/cb385d1c-154e-4da3-93e8-7c4a2cc45a08/repo/vizier/_src/algorithms/designers/preference_gp/preference_gp_designer_test.py)，10 个用例覆盖 TR-3.1~TR-3.5 与端到端效果，pytest 全部通过（10 passed, 167s）；数学核心 [preference_gp_test.py](file:///Users/xavskye/workspace/Projects/Phecda/SWEMarkup/data/runs/cb385d1c-154e-4da3-93e8-7c4a2cc45a08/repo/vizier/_src/algorithms/designers/preference_gp/preference_gp_test.py) 11 用例同步回归通过（11 passed, 95s）。
  - 关键修复：单比较下类型-II ML 退化为零信号方差平凡解（P 恒为 0.5）。在 [preference_gp.py](file:///Users/xavskye/workspace/Projects/Phecda/SWEMarkup/data/runs/cb385d1c-154e-4da3-93e8-7c4a2cc45a08/repo/vizier/_src/algorithms/designers/preference_gp/preference_gp.py) 引入弱 log-normal 超先验（`_HYPERPRIOR_STD=1.5`，`_log_hyperprior`），训练目标改为 MAP（负 Laplace 证据减对数先验）。修复后单比较 P(0.8≻0.2)=0.7697、学到信号方差 0.85；数据充足场景效用相关 0.9724 不降级；矛盾对 P=0.5000；重复参数 2 唯一行承载 6 比较。
  - `rubric` TR-4.2 证据（3 种子原型实验，二次峰 oracle 最优点 0.35、噪声 sd=0.05、8 轮×4 suggest + 12 比较）：每种子 designer 平均效用均优于随机（-0.0050/-0.0015/-0.0042 vs -0.0444/-0.0880/-0.0353），汇总均值 -0.0036 vs -0.0559（lift 0.0523），末期建议最优点平均距离 0.0132、最佳效用 ≈0。评分 5/5（≥threshold 4）。
  - `rubric` TR-4.3：数学层/Designer 层分层清晰，全部公共方法带 docstring，边界（空数据/单点/自比较/重复/矛盾/未知 id/残缺键/混合空间）均有对应用例，评分 5/5。
- **Priority**: high
- **Depends On**: Task 3
- **Description**:
  - 新建 `preference_gp_designer_test.py`：
    - 覆盖 TR-3.1~TR-3.5 的规则用例（连续空间、混合空间、元数据路径、增量等价、批量去重、空数据先验）。
    - 端到端效果用例：已知效用 oracle 生成带噪偏好，运行多轮 suggest→report 循环，断言末期建议点平均/Top 效用显著优于同空间随机基线（固定种子、可重复）。
- **Acceptance Criteria Addressed**: AC-2, AC-7, AC-8, AC-9, AC-10, AC-13, AC-14
- **Test Requirements**:
  - `rule` TR-4.1: 全部规则用例 pytest 通过；证据：pytest 日志。
  - `rubric` TR-4.2: 端到端偏好优化有效性；scale 1-5；anchors 同 AC-13；threshold >=4；证据：oracle 循环测试的效用对比数值与评审记录。
  - `rubric` TR-4.3: 工程质量（分层/docstring/边界覆盖）；scale 1-5；anchors 同 AC-14；threshold >=4；证据：代码与测试清单评审。
- **Notes**: 效果用例固定随机种子并设置合理容差，避免 flaky。

## Task 5: 公共 API 导出与兼容性验证
- **Status**: `completed`
- **Completion Evidence**:
  - `rule` TR-5.1：[vizier/algorithms/designers/__init__.py](file:///Users/xavskye/workspace/Projects/Phecda/SWEMarkup/data/runs/cb385d1c-154e-4da3-93e8-7c4a2cc45a08/repo/vizier/algorithms/designers/__init__.py) 按字母序新增 1 行导出；`from vizier.algorithms.designers import PreferenceGPDesigner` 全链路导入成功（补齐 venv 缺失的既有可选依赖 cvxpy/evojax/scikit-learn 后验证），x64 下实例化/suggest/report/predict 无 UserWarning，单比较 P=0.766。
  - 工程一致性：构造函数新增 x64 守卫（对齐 `gaussian_process_model.py` 惯例，未启用时抛 ValueError 并提示 `jax.config.update('jax_enable_x64', True)`），新增 `test_construction_requires_x64` 用例（try/finally 还原全局配置）。
  - `rule` TR-5.2：既有设计师测试零回归——quasi_random/random/grid 合计 21 passed；gp_bandit 24 passed / 3 skipped（跳过项为可选依赖缺失，与本次改动无关）；新包 preference_gp 数学核心 11 passed、designer 11 passed。
  - `rule` TR-5.3：`git diff --stat` 确认唯一跟踪文件变更为 `__init__.py` 1 行新增；其余均为新增文件（preference_gp/ 子包 5 个 .py、.trae/ 规格工件）与环境构建产物（*_pb2.py、google protos、__pycache__），无任何既有模块行为性修改。
- **Priority**: medium
- **Depends On**: Task 3, Task 4
- **Description**:
  - 在 `vizier/algorithms/designers/__init__.py` 增加 `PreferenceGPDesigner` 导出（含 License 约定与字母序）。
  - 运行新测试全套与既有相关测试（quasi_random、gp_bandit、random、grid）确认零回归。
  - 全仓 grep 确认无对既有模块的行为性修改（除导出文件外仅新增文件）。
- **Acceptance Criteria Addressed**: AC-1, AC-11, AC-14
- **Test Requirements**:
  - `rule` TR-5.1: 公共导入路径可用；证据：导入测试输出。
  - `rule` TR-5.2: 既有设计师测试套件全部通过；证据：pytest 汇总日志。
  - `rule` TR-5.3: `git status`/diff 显示除一行导出外无既有文件行为变更；证据：git 输出。
