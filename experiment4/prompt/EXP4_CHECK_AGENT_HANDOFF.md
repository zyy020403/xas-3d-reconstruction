# EXP4_CHECK_AGENT_HANDOFF.md
# 脚本身份核对 · Check Agent 交接文档

> **撰写者**：DiffCSP-Exp4-Main-Agent 4
> **接收者**：Check Agent（一次性核对任务）
> **日期**：2026-04-26
> **范围**：仅核对 5 个直接复用脚本的"文件名声称的身份"是否与"实际代码内容"一致
> **不做**：不写任何代码、不重新审计仓库、不评论数据集决策、不质疑物理学选择

---

## 0. 你（Check Agent）的工作概述

**一句话目标**：用户在 Exp2 仓库里有多个版本号相似的脚本（如 `_v1` `_v2` `_L6` `_step4c` `_v6`），他坦白其中部分文件**内容曾被原地替换但文件名没改**。MA4 不能信任 MA3 handoff §7 列的"文件→身份"映射，必须在动 Phase 5/5b/6 之前对 5 个关键文件**逐字段核对**。

**输出**：一份核对报告（格式见 §6），逐项给 PASS / FAIL / SUSPICIOUS 判定 + 简短证据。

**不要做的事**（避免重复 Step 2/3 多 agent 内耗）：
- ❌ 不要"顺便"审计其他文件
- ❌ 不要重新讨论方案、修订 PROPOSAL、提优化建议
- ❌ 不要写代码、不要建议改动方案
- ❌ 不要做超过 §5 列出的检查项之外的事
- ❌ 不要因为单个 PASS 就乐观推断其他文件也 PASS（每个文件独立验证）
- ❌ 单个文件检查不超过 ~10 行 grep/inspect（精确检查，不通读全文）

**输出格式**：直接填 §6 模板，**不展开**为长篇报告。MA4 看你的判定矩阵就够。

---

## 1. 背景压缩版

DiffCSP-Experiment4 是 Exp2 的扩展（Fe→88 元素，11K→75K 样本）。目前 Step 1/2/2.5 已完成，Step 3 走到 Phase 0/3/4 也已完成。下一步 Sub-Agent 4 要做 Phase 5/5b/6（修改 / 重写 / 测试 3 个核心 Step 3 脚本）。

**你不需要理解全部 Exp4**。你只需要知道：
1. **Exp2 的"最终有效版本"是 Step4d**：L=6 虚拟晶格、`frac -= np.round(frac)` min-image 折叠、坐标系 [-0.5, 0.5]、forward() **无** `% 1.`、feat_dim 当时是 73
2. **Exp2 中间走过的 3 个失败版本**：
   - **Step4**：L=12，forward() 有 `% 1.` bug（双峰分布）
   - **Step4b**：Dataset 加 `% 1.0` 折叠到 [0,1]，但先验仍不匹配
   - **Step4c**：去掉 `% 1.`，但 L=12 让原子只占 box 3% 体积，先验空间太大
3. **Exp4 的核心改动只有两条**：feat_dim 73 → 74（feff 多一维 `has_pre_edge`），Dataset 重写支持全元素

完整四版本演化在 EXPERIMENT2_FINAL_REPORT.md §2.3。其他细节查 EXP4_MAINAGENT4_HANDOFF.md。**你不需要为了这次检查通读两份文档**——本文档已把判定所需的事实压缩在 §5。

---

## 2. 核心待验证假设（MA3 handoff §7 推断的，可能错）

| 文件（Exp2 仓库 step3/ 或 step2/） | MA3 §7 标注的身份 | 如果错了的后果 |
|---|---|---|
| `step3/diffusion_w_type_xas.py`（无后缀） | Step4c/4d 共用扩散逻辑（**EXP2 权威**） | Phase 5 改错对象，训练直接 NaN 或学到错误分布 |
| `step3/xas_local_datamodule.py`（无后缀） | Step4d datamodule（**EXP2 权威**） | Phase 5b 重写蓝本是错的，可能引入失败版本的 bug |
| `step3/xas_local_dataset_L6.py` | Step4d L=6 最终版（**EXP2 权威**） | Sub-Agent 3 已据此 fork 出 v2 dataset，逻辑可能继承错版 |
| `step2/spectrum_encoder.py` | EXP2 权威 + Sub-Agent 3 已改 5 处 73→74 | feff 输入维度若没真改成 74 → 训练第一步 RuntimeError shape mismatch |
| **服务器** `/home/tcat/diffcsp_exp4/code/step3/xas_local_dataset_v2.py` | Sub-Agent 3 新建，12 字段 schema | 12 字段缺漏 / frac sentinel 缺失 / 防御 raise 漏写 → Phase 6 forward test 阶段才暴露 |

5 个文件**逐文件独立验证**，不要一通过就推断后续也通过。

---

## 3. 你将收到的文件（用户会粘贴文本进对话）

### 3.1 必收（用户从本地 + 服务器 cat 给你）

| # | 文件路径 | 来源 | 用途 |
|---|---|---|---|
| F1 | `experiment2/step3/diffusion_w_type_xas.py` | 本地 Windows | §5.1 检查 |
| F2 | `experiment2/step3/xas_local_datamodule.py` | 本地 Windows | §5.2 检查 |
| F3 | `experiment2/step3/xas_local_dataset_L6.py` | 本地 Windows | §5.3 检查 |
| F4 | 服务器 `/home/tcat/diffcsp_exp4/code/step2/spectrum_encoder.py` | 服务器（Sub-Agent 3 已改）| §5.4 检查 |
| F5 | 服务器 `/home/tcat/diffcsp_exp4/code/step3/xas_local_dataset_v2.py` | 服务器（Sub-Agent 3 新建）| §5.5 检查 |

**重点**：F4 必须取**服务器**版本（被 Sub-Agent 3 修改过的），不是本地原版。本地原版仍是 73，会误判。

### 3.2 可选（仅当 §5 任一项判 SUSPICIOUS / FAIL 才请求用户提供）

- `experiment2/step3/diffusion_w_type_xas_v1.py`（声称 Step4 buggy）
- `experiment2/step3/diffusion_w_type_xas_v2.py`（声称 Step4c）
- `experiment2/step3/xas_local_dataset_step4c.py`（声称 Step4c）
- `experiment2/step3/xas_local_dataset_v6.py`（声称 Step4b 死路）

这 4 个文件**默认不取**——它们是声称的"废弃文件"，只在主验证暴露异常时用作交叉对照（如果 `_L6.py` 实际是 Step4 buggy，那真正的 Step4d 代码可能藏在 `_v6.py` 或别处）。

### 3.3 参考资料（你已有的）

- 本文档（你正在读）
- EXPERIMENT2_FINAL_REPORT.md（§2.3 是关键，四版本演化）
- EXP4_PROPOSAL_v2.md（§1.3 不可变量列表）
- EXP4_MAINAGENT4_HANDOFF.md（§7 文件归属表 + §5.4 Phase 3 schema 详情）

如果你还没收到上面 3 份参考，请用户先发给你再开工。

---

## 4. 通用检查协议

每文件检查时按下面步骤走：

1. **接收文件全文**（用户 cat / type 粘贴）
2. **不要通读**——只 grep 关键 marker（每文件不超过 6-8 个 marker）
3. **逐 marker 判定**：found / not found / unexpected
4. **整体判定**：PASS（全部 marker 符合预期）/ SUSPICIOUS（1-2 项不符或位置异常）/ FAIL（关键 marker 反例出现）
5. **填 §6 模板**

**判定原则**：
- "缺一个无害 marker" → SUSPICIOUS（不是 FAIL，但报告里说明）
- "出现一个反 marker"（如 `% 1.` 出现在声称无 `% 1.` 的文件里）→ **FAIL**
- "找到但行号 / 上下文与声称不符" → SUSPICIOUS（可能是 in-place 替换了内容）

---

## 5. 逐文件检查清单

### §5.1 — F1: `step3/diffusion_w_type_xas.py`（无后缀）

**声称身份**：Step4c/4d 共用扩散逻辑，EXP2 最终权威版（Step4d 训练用的就是这个）。Phase 5 Sub-Agent 4 要在它上面改 4 项。

**MUST FIND**（缺即 SUSPICIOUS / 反向出现即 FAIL）：

| # | Marker | 期望 | 反向 = FAIL |
|---|---|---|---|
| M1 | `cost_lattice` | 出现且赋值 0 或被乘 0 / 或在 loss 计算中显式 disable | 赋值非 0 |
| M2 | forward() 内 `% 1` 或 `% 1.` 或 `torch.fmod(..., 1)` | **零次** | 任何一次出现 = FAIL |
| M3 | `feat_dim` 默认值（约 line 108 附近） | `feat_dim=73` | `feat_dim` 出现但值是 12 / 200 / 其他奇怪数 = SUSPICIOUS |
| M4 | `import pytorch_lightning` | 出现 | 仅 `import lightning`（无 `pytorch_`）= SUSPICIOUS |
| M5 | `XASLocalStructureDataset` 或 `xas_local_dataset_L6` | 引用其中之一 | 引用 `xas_local_dataset_step4c` / `xas_local_dataset_v6` = SUSPICIOUS |
| M6 | `TypeClassifier` 类定义 | **零次**（Exp3 已证伪） | 任何一次出现 = FAIL |

**判定矩阵**：6/6 符合 → PASS；4-5/6 → SUSPICIOUS；M2/M6 反向 = FAIL（即使其他 4 个都对）。

---

### §5.2 — F2: `step3/xas_local_datamodule.py`（无后缀）

**声称身份**：Step4d datamodule，与 `xas_local_dataset_L6.py` 配套使用。Phase 5b Sub-Agent 4 要重写为 v2 版本。

**MUST FIND**：

| # | Marker | 期望 |
|---|---|---|
| M1 | 类名包含 `XasLocalDataModule` 或 `XASLocalDataModule` | 出现 |
| M2 | `import pytorch_lightning as pl` 或 `from pytorch_lightning ...` | 出现 |
| M3 | dataset import：`from xas_local_dataset_L6` 或 `from xas_local_dataset` | 出现其中之一 |
| M4 | `train_dataloader` / `val_dataloader` / `test_dataloader` 方法 | 至少 train + val 存在 |
| M5 | `setup()` 方法 | 存在（不论签名是 PL 1.x 还是 2.x，Phase 5b 会兼容） |

**反 markers（出现即 FAIL 或 SUSPICIOUS）**：

- 引用 `xas_local_dataset_step4c` 或 `xas_local_dataset_v6` → SUSPICIOUS（说明这不是 Step4d 时代版本）
- 引用 `holdout_*` 路径 → SUSPICIOUS（holdout 不应进 datamodule）

**判定矩阵**：5/5 符合 + 无反 marker → PASS；任一缺失 → SUSPICIOUS；任一反 marker → SUSPICIOUS（不到 FAIL 程度，但要标记）。

---

### §5.3 — F3: `step3/xas_local_dataset_L6.py`

**声称身份**：Step4d (L=6) 最终有效版本，是 Sub-Agent 3 fork 出 `xas_local_dataset_v2.py` 的源头。

**MUST FIND**（最关键的 3 个 marker：L=6 / min-image / 无 `%1`）：

| # | Marker | 期望 | 反向 = FAIL |
|---|---|---|---|
| M1 | `L = 6` 或 `L_VIRTUAL = 6` 或类似常量赋值（值 = 6 或 6.0） | 出现 | 出现 `L = 12` 或 L 值非 6 = **FAIL**（这就是 Step4/4c 的失败版本）|
| M2 | `frac -= np.round(frac)` 或 `frac = frac - np.round(frac)`（min-image 折叠） | 出现至少一次 | 完全找不到 = **FAIL**（这是 Step4d 区分于其他版本的核心改动）|
| M3 | `% 1` 或 `frac %= 1` 或 `frac = frac % 1.0`（应用于分数坐标） | **零次** | 出现 = **FAIL**（这是 Step4 / 4b 的 bug）|
| M4 | `N_NEIGHBORS` 或类似常量 = 20 | 出现，值 20 | 值非 20 = SUSPICIOUS |
| M5 | `SpacegroupAnalyzer` import 或调用 | 出现 | 缺失 = SUSPICIOUS |
| M6 | feat_dim 73（如果出现于 dataset 内）/ 或 feff 维度 73 | 73（Exp4 改 74 在 v2 文件，不在原 L6） | 已经是 74 = SUSPICIOUS（说明被 Exp4 改过原文件，违反"原文件保留"约定） |

**判定矩阵**：M1+M2+M3 三项都对 = 至少 PASS-LITE（核心确认）；其余项不对 = SUSPICIOUS 但不阻塞（次要属性）。M1/M2/M3 任一反向 = **FAIL**。

**特别强调**：M1/M2/M3 是 Step4d 的"基因"，错任一项就说明这个文件不是 Step4d。

---

### §5.4 — F4: 服务器 `/home/tcat/diffcsp_exp4/code/step2/spectrum_encoder.py`

**声称身份**：EXP2 权威 + Sub-Agent 3 已改 5 处 73→74。

**MUST FIND**（5 处 73→74 修改，按 MA3 拍板的 §5.4 行号表，可能因换行略有偏移）：

| # | 位置（约） | 期望内容 | 检查方式 |
|---|---|---|---|
| M1 | line 2 附近（docstring header） | 出现 "74" 或不再出现独立的 "73"（除非用于其他无关用途） | grep `74\|73` line 1-5 |
| M2 | line 28 附近（docstring forward 描述） | 出现 "74" | grep line 25-30 |
| M3 | line 37 附近（docstring `__init__` 参数） | 出现 "74" | grep line 35-40 |
| M4 | line 41 附近（**code**：`def __init__(..., feat_dim=74, ...)`） | `feat_dim=74` 默认值 | grep `def __init__` 后第一个 feat_dim |
| M5 | line 80 附近（docstring forward Args） | 出现 "74" | grep line 78-82 |

**关键**：M4 是**唯一影响数值**的修改（其他 4 个是 docstring）。M4 错 = **FAIL**（训练第一步 RuntimeError）。

**Sanity markers**（确认这是 spectrum_encoder.py 而非别的文件）：
- 出现 xmu / chi1 / feff 三路分支
- 出现维度 150（xmu）/ 200（chi1）/ 256（latent，可能在某处）

**判定矩阵**：M4 = 74 → 至少 PASS-LITE；M1-M3-M5 docstring 也都 = 74 → PASS；M4 = 73 → **FAIL**；M4 = 74 但 M1-M3-M5 仍是 73 → SUSPICIOUS（功能 OK 但 docstring 没改全，未来 debug 陷阱）。

**遗漏 marker 提醒**：grep 完成后补一条 `grep -n "73" spectrum_encoder.py | head`，看还有几个 "73" 残留。**任何还在的 "73"** 都需要 check agent 给出"它在哪行、是不是无关上下文（如其他维度 / 注释中提到 Exp2 历史）"。

---

### §5.5 — F5: 服务器 `/home/tcat/diffcsp_exp4/code/step3/xas_local_dataset_v2.py`

**声称身份**：Sub-Agent 3 新建文件，按 MA3 §5.4 Phase 3 决策实现 12 字段 schema + 双 raise 防御。

**MUST FIND**（按 MA4 handoff §5.4 Phase 3 全清单）：

| # | Marker | 期望 |
|---|---|---|
| M1 | 类名 `XasLocalDatasetV2` | 出现 |
| M2 | `L_VIRTUAL = 6.0` 常量（顶层）| 出现 |
| M3 | `N_NEIGHBORS = 20` 常量 | 出现 |
| M4 | `CUTOFF_R = 10.0` 常量 | 出现 |
| M5 | `SYMPREC = 0.1` 常量 | 出现 |
| M6 | DATA_DIR 从环境变量读：`os.environ.get("EXP4_DATA_DIR", ...)` | 出现 |
| M7 | `frac -= np.round(frac)`（min-image 折叠）| 出现 |
| M8 | frac sentinel：`raise RuntimeError(...)` 在 frac 越界检查附近，epsilon=1e-6 或类似 | 出现 |
| M9 | 显式 `np.argsort(dists)[:20]` 或 `np.argsort(...)[:N_NEIGHBORS]` | 出现 |
| M10 | center_idx 用 `next(... for ... in ...)` 形式（StopIteration 兜底） | 出现 |
| M11 | sklearn `catch_warnings()` 上下文管理器（用于 RobustScaler unpickle） | 出现 |
| M12 | `__init__` 末尾 5 样本对齐验证（init defensive 防御 raise） | 出现 |
| M13 | return dict 包含全部 12 字段：`xmu, chi1, feff, frac_coords, atom_types, sample_name, mp_id, center_element, eval_cutoff, eval_cutoff_fallback, n_center_sites, site_equivalence_tag` | **12 字段全在** |

**反 markers（出现即 FAIL）**：

| # | 反 marker | 原因 |
|---|---|---|
| AM1 | `% 1` 或 `frac %= 1` | Step4 bug |
| AM2 | `TypeClassifier` | Exp3 已证伪 |
| AM3 | 引用 `incompat_pool.csv` | Exp4 全程封存 |
| AM4 | `holdout_*` 在 dataset 内被默认 load | holdout 留 Step 5 |
| AM5 | padding 处理 < 20 邻居（应 raise 而非 pad） | MA3 决策 6 推断 |

**判定矩阵**：M1-M13 全 PASS + 0 反 marker → PASS；M1/M7/M8/M13 是核心，缺任一 = **FAIL**；M2-M6/M9-M12 缺一两个 = SUSPICIOUS。任何 AM 出现 = **FAIL**。

**特别提示**：return dict 12 字段是 MA3 与 Sub-Agent 3 反复讨论得到的权威 schema。少一个字段会让 Step 5 评估缺信息。多一个字段（如 shell_info 嵌套）违反 MA3 明确的 "shell_info 不进嵌套字段，只透传 3 个标量" 决策。

---

## 6. 输出报告模板（你只填这个）

```
# EXP4 关键脚本身份核对报告
# 由 Check Agent 填写

## 总体判定
- 5 个核心文件中 PASS: ___ / SUSPICIOUS: ___ / FAIL: ___
- 是否阻塞 Phase 5/5b/6 启动: [是 / 否]
- 推荐下一步: [继续推进 / 取 Tier B 文件交叉对照 / 终止并联系 MA4]

## F1: step3/diffusion_w_type_xas.py（无后缀）
判定: [PASS / SUSPICIOUS / FAIL]
markers:
  M1 cost_lattice=0:        [✓ / ✗] 证据: ___
  M2 forward 无 %1:          [✓ / ✗] 证据: ___
  M3 feat_dim=73 默认值:     [✓ / ✗] 行号: ___
  M4 pytorch_lightning import:[✓ / ✗] 行号: ___
  M5 xas_local_dataset_L6 引用:[✓ / ✗] 行号: ___
  M6 无 TypeClassifier:      [✓ / ✗]
备注: ___

## F2: step3/xas_local_datamodule.py（无后缀）
判定: [PASS / SUSPICIOUS / FAIL]
markers:
  M1 类名 XasLocalDataModule: [✓ / ✗]
  M2 pytorch_lightning import:[✓ / ✗]
  M3 dataset import:         [✓ / ✗] 实际引用了哪个 dataset 文件: ___
  M4 train+val dataloader:   [✓ / ✗]
  M5 setup() 方法:           [✓ / ✗] 签名: ___
反 markers:
  AM 引用 _step4c / _v6:     [是 / 否]
  AM 引用 holdout_*:         [是 / 否]
备注: ___

## F3: step3/xas_local_dataset_L6.py
判定: [PASS / PASS-LITE / SUSPICIOUS / FAIL]
markers:
  M1 L = 6:                  [✓ / ✗] 实际值: ___
  M2 frac -= np.round(frac):  [✓ / ✗] 行号: ___
  M3 forward 无 %1:          [✓ / ✗]
  M4 N_NEIGHBORS = 20:       [✓ / ✗]
  M5 SpacegroupAnalyzer:     [✓ / ✗]
  M6 feff 维度 = 73:         [✓ / ✗] 实际值: ___
备注: ___

## F4: 服务器 step2/spectrum_encoder.py（Sub-Agent 3 已改）
判定: [PASS / PASS-LITE / SUSPICIOUS / FAIL]
markers:
  M1 line 2 docstring 74:    [✓ / ✗]
  M2 line 28 docstring 74:   [✓ / ✗]
  M3 line 37 docstring 74:   [✓ / ✗]
  M4 line 41 code feat_dim=74:[✓ / ✗] ★ 关键
  M5 line 80 docstring 74:   [✓ / ✗]
全文 grep "73" 残留:
  共 ___ 处，分别在: ___（每处简述上下文）
备注: ___

## F5: 服务器 step3/xas_local_dataset_v2.py（Sub-Agent 3 新建）
判定: [PASS / SUSPICIOUS / FAIL]
markers:
  M1  类名 XasLocalDatasetV2:        [✓ / ✗]
  M2  L_VIRTUAL = 6.0:               [✓ / ✗]
  M3  N_NEIGHBORS = 20:              [✓ / ✗]
  M4  CUTOFF_R = 10.0:               [✓ / ✗]
  M5  SYMPREC = 0.1:                 [✓ / ✗]
  M6  EXP4_DATA_DIR 环境变量:         [✓ / ✗]
  M7  frac -= np.round(frac):        [✓ / ✗]
  M8  frac sentinel raise:           [✓ / ✗]
  M9  显式 np.argsort:                [✓ / ✗]
  M10 next() center_idx:             [✓ / ✗]
  M11 catch_warnings sklearn:        [✓ / ✗]
  M12 __init__ 5-sample 对齐:        [✓ / ✗]
  M13 return dict 12 字段:           [✓ / ✗] 实际字段数: ___
       缺失字段: ___
       多出字段: ___
反 markers:
  AM1 % 1:           [是 / 否]
  AM2 TypeClassifier:[是 / 否]
  AM3 incompat:      [是 / 否]
  AM4 holdout 默认 load:[是 / 否]
  AM5 padding 处理:  [是 / 否]
备注: ___

## 异常 / 需要 MA4 注意的事项
（如果有任何 SUSPICIOUS 或 FAIL，简短说明影响）
___

## 你（Check Agent）的上下文消耗估算
约 ___ %
```

---

## 7. 你的工作哲学（继承自之前 Sub-Agent）

1. **诚实 > 流畅**：找不到 marker 就说找不到，不要强行解读。SUSPICIOUS 比强行 PASS 更有用。
2. **不要替 MA4 做决定**：你只判定，不建议改方案。
3. **不展开**：每个判定写 1-2 行证据就够，不写散文。
4. **闸门**：你预计上下文超过 50% 立刻停下来交报告，剩下未检查的项目标 "TIME_OUT" 让 MA4 决定要不要再开一个 check agent。

---

## 8. 接下来发生什么

1. 你（Check Agent）收到本文档 + §3.1 的 5 份文件
2. 你按 §5 走 5 文件检查
3. 你填 §6 模板交回给用户
4. 用户把你的报告转给 MA4
5. MA4 根据报告决定是否继续 Phase 5/5b/6（如果有 FAIL，先解决；全 PASS / 1-2 个 SUSPICIOUS 可接受 → 推进）

**你的工作到此结束**。不要主动请求新文件、不要建议下一步。

---

*MA4 撰写,2026-04-26,本文档为一次性核对任务,不接力*
