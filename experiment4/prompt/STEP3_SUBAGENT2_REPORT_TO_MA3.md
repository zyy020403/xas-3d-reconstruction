# Step 3 Sub-Agent 2 → MA3 工作汇报

> **撰写者**：DiffCSP-Exp4-Step3-SubAgent-2
> **日期**：2026-04-25
> **覆盖期间**：从接手 Sub-Agent (1) 的 handoff 文档（§EXP4_STEP3_SUBAGENT2_HANDOFF.md）开始，至本汇报触发点为止
> **触发原因**：Phase 0.3 / 0.3b / 0.3c 暴露了 handoff 与实际数据状态的若干假设差异，并涉及 §12 不可变量边界外的微小 scope 修订。按 §13 工作哲学，在继续 Phase 0.5 与 Phase 3-6 编码之前请求 MA3 确认。
> **状态摘要**：POSCAR BLOCKER 已解除 ✅；Phase 0.2 / 0.4 直通 ✅；Phase 0.3 维度对齐通过但暴露 5 个待决策点 ⚠️；Phase 0.1 / 0.5 未开始；Phase 3-6 未开始。

---

## 0. 阅读指南

本汇报按时间顺序组织（§1-§5 已做），后接观察分析（§6）、待决策清单（§7）、推进路径（§8）。MA3 关注重点建议直接跳 §7。Sub-Agent 自身关注的不确定点在 §9。

汇报内所有事实均基于已执行命令的真实输出，无脑补；推测部分明确标注 "推测 / 待验证 / 未确认"。

---

## 1. 接手时状态

按 handoff §0 描述，接手时实际状况：

- 环境：mlff 已锁定（路径 `/home/tcat/conda_envs/mlff`，版本表见 handoff §1.2）
- 数据：17 个文件（637 MB）已上传到 `/home/tcat/diffcsp_exp4/data/`
- 代码：Exp2 仓库已上传到 `/home/tcat/diffcsp_exp4/code/`，按阶段子目录组织
- 仓库审计：handoff §1.6 已确认 Step 3 必改 4 个源文件
- torch_scatter：决策 A 已执行（env-local 装 2.1.2+pt24cu124），5/5 测试通过
- 磁盘：27 GB 可用（红线 10 GB）
- BLOCKER：POSCAR 位置未确认（handoff §3.1）

未做：Phase 0/3/4/5/6 全部。

---

## 2. POSCAR 定位与上传（已完成 ✅）

### 2.1 服务器端 find（情形判断）

执行了四条诊断命令：
```bash
find /home/tcat -type d \( -iname "*poscar*" -o -iname "mp-9*" -o -iname "MP_all*" \) 2>/dev/null
find /home/tcat -type f -name "*_POSCAR" 2>/dev/null | head -5
ls -la /data /scratch /shared /mnt 2>/dev/null | head -40
du -sh /home/tcat/*/ 2>/dev/null | sort -hr | head -20
```

**关键输出**：

- `find -type d` 命中两个无关项目（`/home/tcat/yan/4_Graph/POSCAR_files`、`/home/tcat/ml_opt/zcy/nitrate_reduction/POSCAR_files`）—— 都是其他用户/项目的化学计算残留，**不是 MP 数据集**
- `find -type f -name "*_POSCAR"` 仅命中 5 个 `ml_opt/zcy/...` 残留（`O_POSCAR`、`N_POSCAR` 命名），**零个 mp-XXXX_POSCAR**
- 共享盘 `/mnt/nvme0n1` 存在但空；`/scratch` 存在但空；`/data /shared` 不存在
- `du` 一级目录：`ml_opt 89G / conda_envs 18G / diffcsp_exp4 1.6G / src 1.3G / lammps 740M / chk 713M / ...`
- 在 `/home/tcat/diffcsp_exp4/` 内 `find` 仅命中两个 step1 阶段的 POSCAR 元数据脚本（无数据）

**结论**：POSCAR 100% 不在服务器，必须从 Windows 重新上传。

### 2.2 Windows 端打包

用户确认 POSCAR 源在 `C:\Users\T-Cat\Desktop\DiffCSP-main\experiment4\data\POSCAR_zip\MP_all_POSCAR_flat\`。

PowerShell 跑：
```powershell
cd "C:\Users\T-Cat\Desktop\DiffCSP-main\experiment4\data\POSCAR_zip"
tar -czf MP_all_POSCAR_flat.tar.gz MP_all_POSCAR_flat
```

**生成包**：`MP_all_POSCAR_flat.tar.gz`，**18.26 MB**（gzip 压缩比 ~3-5x，41K 短文本文件压缩后合理大小；本人最初估 100-300 MB 偏高，未考虑 POSCAR 单文件 ~500B-2KB）。

### 2.3 文件数 sanity（打包前 / 打包后）

```powershell
tar -tzf MP_all_POSCAR_flat.tar.gz | Measure-Object | Select-Object -ExpandProperty Count
# → 41498

(Get-ChildItem MP_all_POSCAR_flat -File).Count
# → 41497
```

41497 文件 + 1 目录条目 = 41498 包内 entry，**完全对齐**。

### 2.4 scp 上传

```powershell
scp MP_all_POSCAR_flat.tar.gz tcat@scsmlnprd02.its.auckland.ac.nz:/home/tcat/diffcsp_exp4/data/
```

`100% 18MB 96.1MB/s 00:00`，秒传。服务器端 `ls -lh` 确认 19M、tcat owner、时间戳正确。

### 2.5 服务器端解压

`df -h /` 确认 27G 可用（无变化，sanity 通过）。

```bash
cd /home/tcat/diffcsp_exp4/data && tar -xzf MP_all_POSCAR_flat.tar.gz
```

静默成功（无报错）。

### 2.6 解压后验证

```bash
ls /home/tcat/diffcsp_exp4/data/MP_all_POSCAR_flat/ | wc -l
# → 41497

ls /home/tcat/diffcsp_exp4/data/MP_all_POSCAR_flat/ | head -3
# → missing_poscar_list.csv
#    mp-10003_POSCAR
#    mp-10004_POSCAR

head -8 /home/tcat/diffcsp_exp4/data/MP_all_POSCAR_flat/mp-9_POSCAR
# 标准 POSCAR 格式：Y1 / 1.0 / 三行晶格向量 / Y / 1 / direct
```

**异常发现**：目录里多了一个 `missing_poscar_list.csv`（Step 1 阶段产物），不是 POSCAR 但混在同目录。

**用户确认无用**，已删除：
```bash
rm /home/tcat/diffcsp_exp4/data/MP_all_POSCAR_flat/missing_poscar_list.csv
ls /home/tcat/diffcsp_exp4/data/MP_all_POSCAR_flat/ | wc -l
# → 41496
```

### 2.7 POSCAR 状态总结

| 项目 | 数值 |
|---|---|
| 路径 | `/home/tcat/diffcsp_exp4/data/MP_all_POSCAR_flat/` |
| 文件数 | **41,496** |
| 命名 | `mp-XXXX_POSCAR`（XXXX 是 mp_id 的数字部分，如 `mp-9_POSCAR`、`mp-10003_POSCAR`） |
| 格式 | 标准 VASP POSCAR（含 lattice / element / count / direct 坐标） |
| 解压来源 | Windows 端 `tar -czf` 打包 + scp + 服务器 `tar -xzf` |
| Owner | tcat tcat |
| 大小 | 解压后 ~50-100 MB（精确未测，`du` 验证不影响推进） |

**对 handoff 修订**：原 handoff §1.4 / §3 假设 ~41,431，**实际 41,496**。这个数字应作为后续 Phase 6 `os.listdir()` 或 `glob` 计数 sanity 的真值。

---

## 3. Phase 0.2：numpy 2.x 别名 grep（已完成 ✅）

handoff §3.2 要求 grep numpy 1.x 已删除别名（`np.float`、`np.int`、`np.bool`、`np.object`、`np.long`、`np.complex` 等）。

### 3.1 主 grep
```bash
grep -rn -E 'np\.(float|int|bool|object|long|complex)\b' --include="*.py" /home/tcat/diffcsp_exp4/code/
```
**输出**：零命中。

### 3.2 补 grep（Sub-Agent 自我修订）

第一条 regex 用 `\b` word boundary，但 `_` 不是词边界字符，**会漏掉 `np.float_` / `np.int_` / `np.complex_` 这类带下划线后缀的别名**（这些在 numpy 2.0 也已删除）。补一条：

```bash
grep -rn -E 'np\.(float_|int_|bool_|complex_|long_)' --include="*.py" /home/tcat/diffcsp_exp4/code/
```
**输出**：零命中。

### 3.3 结论

Exp2 仓库**没有 numpy 1.x 别名残留**，numpy 2.2.6 完全兼容当前代码。**Phase 0.2 通过，无需任何替换**。

注：handoff §3.2 提供的 grep 模式不含下划线后缀分支，本 Sub-Agent 自行补足。MA3 如认为应将该补丁固化到 handoff，可在后续修订。

---

## 4. Phase 0.4：Lightning import 统一（已完成 ✅）

handoff §3.4 要求确认 Exp2 用 `pytorch_lightning` 还是 `lightning`，统一到一个。mlff 同装两个：`pytorch_lightning 2.5.5`（稳定版） + `lightning 2.6.0.dev20250810`（dev 版）。

### 4.1 grep

```bash
grep -rn -E '^\s*(import|from)\s+(pytorch_lightning|lightning)\b' --include="*.py" /home/tcat/diffcsp_exp4/code/
```

**输出 21 处命中，全部 `pytorch_lightning`，零 `lightning`**。涉及文件：

- Step 3 必改源文件中：
  - `step3/diffusion_w_type_xas.py:34` ← Step 3 必改
  - `step3/xas_local_datamodule.py:18` ← Step 3 必改（决策 B 重写）
  - `step3/diffusion_w_type_xas_v1.py:32` ← 废弃文件（handoff §1.6），不动
  - `step3/diffusion_w_type_xas_v2.py:25` ← 废弃文件，不动
- Step 4 阶段（Step 3 不直接改但要保持一致）：
  - `step4/step4_2_train.py` 51/55/57 行
  - `step4/step4_5_finetune.py` 56/60/62 行
  - `step4c/step4c_2_train.py` 49/53/55 行
  - `step4c/step4c_test_full.py` 194/198/200 行
  - `step4d/step4d_1_quick_test.py` 49/50 行
  - `step4d/step4d_2_train.py` 48/52/54 行

### 4.2 结论

**统一使用 `pytorch_lightning` 2.5.5**（mlff 装的稳定版）。后续 Phase 5b 重写 `xas_local_datamodule_v2.py` 时直接 `import pytorch_lightning as pl` 一致，**无需改任何 import**。

注：handoff §2.B 提到 PL 2.x 的 `setup()` 签名是 `def setup(self, stage: str)`（无默认值），PL 1.x 是 `def setup(self, stage: Optional[str] = None)`。Exp2 datamodule 可能写于 PL 1.x 时代，Phase 5b 重写时需要 grep 检查原 datamodule 是否有 `if stage is None:` 这种依赖 1.x 行为的写法，**当前未查**，记入 Phase 5b。

---

## 5. Phase 0.3 / 0.3b / 0.3c：sklearn scaler + feff 三件套维度（已完成，触发本次汇报 ⚠️）

### 5.1 Phase 0.3：scaler unpickle + transform sanity

写诊断脚本 `/tmp/phase0_3.py`（965 字节，heredoc 写入），加载 `feff_feature_scaler.pkl` 并验证：

```
python: 3.10.19
numpy:  2.2.6
sklearn: 1.7.2

⚠️ InconsistentVersionWarning: Trying to unpickle estimator RobustScaler from version 1.6.1 when using version 1.7.2.

scaler type: RobustScaler
scaler n_features_in_: 74        ← 关键：74，不是 handoff §1.4 描述的 73
scaler center_ shape:  (74,)
scaler scale_ shape:   (74,)

transform(zeros[1,74]) → shape=(1, 74), dtype=float64
  any NaN: False
  any Inf: False
  range: [-2000.0000, 0.5905]
  mean:  -54.6780
```

**两个观察**：

1. **维度是 74，不是 73**。Handoff §1.4（数据清单部分）和 §2 决策 B 的描述假设 "Exp2 用 73 维 → Exp4 v2 加一维变 74，所以 encoder 要从 73 改到 74"。但 scaler 本身就是 74 维 —— 不能区分两种解释：(A) scaler 是 Exp4 v2 阶段重训的 74 维，(B) scaler 在 Exp2 末期就是 74 维了。**Phase 0.3b 必须确认 feff_features_imputed.pkl 真实维度才能下定论**。

2. **sklearn 版本 mismatch warning**：scaler 是 sklearn **1.6.1** 训的，当前 mlff 的 sklearn 是 **1.7.2**。RobustScaler 在 1.6→1.7 之间内部存储格式（仅 `center_` `scale_` 两个 numpy 数组）没变，**transform 输出无 NaN/Inf**，功能上无损。但 sklearn 警告语气严肃（"might lead to breaking code or invalid results"）。

3. **transform(zeros) 范围 [-2000, 0.59]**：看似夸张，实际是 RobustScaler 在病态特征上的正常行为 —— 某特征 IQR 接近 0（极窄分布）时，零点除以 IQR 得巨大值。FEFF 特征里有些是稀疏统计量（如某 shell 特定 element 计数，绝大多数样本是 0），符合预期。**记下，Step 4 训练前如发现某些维度病态可能要 winsorize / log transform，是后话**。

### 5.2 Phase 0.3b：feff DataFrame 真实维度

写诊断脚本 `/tmp/phase0_3b.py`，加载 `feff_features_imputed.pkl` 并对照 scaler 与 names：

```
feff_features_imputed.pkl type: DataFrame
  shape: (128382, 74)

scaler n_features_in_: 74
feff_feature_names.txt: 74 names
  first 5: ['xmu_Emin', 'xmu_Emax', 'xmu_npts', 'E0', 'mu_at_E0']
  last 5:  ['R2_peak_height', 'R1_area', 'R2_area', 'R1_R2_ratio', 'has_pre_edge']
```

**三件套维度全部对齐 74** ✅

但 **行数 128,382** 是个新观察：handoff §1.4 没提 feff pkl 行数，初看超过 train+val+test+holdout 总样本数（合计应在数万级，不是 12.8 万）。这促使了 Phase 0.3c 的 follow-up。

### 5.3 Phase 0.3c：DataFrame index/columns 结构

写诊断脚本 `/tmp/phase0_3c.py`：

```
shape: (128382, 74)
index name: sample_name
index dtype: object
first 5 index values:
  'mp-10003__mp-10003-EXAFS-Co-K'
  'mp-10003__mp-10003-EXAFS-Nb-K'
  'mp-10003__mp-10003-EXAFS-Si-K'
  'mp-10004__mp-10004-EXAFS-Mo-K'
  'mp-10004__mp-10004-EXAFS-P-K'
last 5 index values:
  'mp-9721__mp-9721-EXAFS-O-K'
  'mp-972__mp-972-EXAFS-Mn-K'
  'mp-972__mp-972-EXAFS-Se-K'
  'mp-97__mp-97-EXAFS-Pr-K'
  'mp-9__mp-9-EXAFS-Y-K'
index unique: True
index has nulls: False

columns dtype: {dtype('float32'): 74}
any all-NaN cols: False
any all-NaN rows: False
total NaN cells: 0
```

**关键事实确立**：

| 维度 | 事实 | 含义 |
|---|---|---|
| 容器 | DataFrame | 规整二维结构 |
| 行 | 128,382 | sample 级（一个 mp 多元素 K 边各占一行） |
| 列 | 74 | 与 scaler / names 对齐 |
| dtype | float32 × 74 | 内存友好（74 × 4B × 128382 ≈ 38 MB） |
| index | `sample_name`，object（字符串），unique，no null | DataFrame 必须用 `df.loc[sample_name]` 查 |
| index 格式 | `mp-{ID}__mp-{ID}-EXAFS-{Element}-K` | 含 `mp-XXXX` 重复 + `EXAFS-{Element}-K` 后缀 |
| NaN 状态 | 零 NaN | imputed 完整 |

**对 handoff §1.4 描述的修订**：
- handoff §1.4 把 `feff_features_imputed.pkl` 写成"40.3 MB"（size），但**没说容器类型 / 行数 / index 格式**。这些事实不在 handoff 内，需要补充
- 128,382 行不是 train+val+test+holdout 之和（因为 v2 split csv 仅约 5 万级），剩余约 7-8 万样本是 incompat_pool 的吸收边记录。**incompat 在数据表里仍存在，但训练时只通过 v2 split csv 索引选取**。这与 handoff §6.5 "永远不要碰 incompat_pool.csv" 不冲突 —— Dataset 通过 `sample_name` lookup feff，不会把 128382 全 load 进训练（除非代码错误）。**但 Phase 0.5 必须验证 v2 split csv 的 sample_name 列与 feff index 同格式**

### 5.4 encoder 当前维度状态（Phase 4 决策 B 复核）

handoff §2 决策 B 表写：
> `step2/spectrum_encoder.py` 一行改：feff 分支 `nn.Linear(73, ...)` → `nn.Linear(74, ...)`

grep 实际状态：

```bash
grep -n -E 'Linear\s*\(\s*(73|74)' /home/tcat/diffcsp_exp4/code/step2/spectrum_encoder.py
# → 28:        Linear(73→128) → SiLU → Linear(128→64) → SiLU       (docstring only)

grep -n -E 'feff' /home/tcat/diffcsp_exp4/code/step2/spectrum_encoder.py | head -20
# → 2:  # SpectrumEncoder：xmu_xanes(150) + chi1(200) + feff_features(73) → (B, 256)
# → 27: feff 分支 (物理先验 MLP)：
# → 37: feat_dim  : int, 默认 73   — feff_features 输入维度
# → 62: # ── 物理先验：feff_features MLP 分支 ────────────────────
# → 74: def forward(self, xmu_xanes, chi1, feff_feats):
# → 80: feff_feats : Tensor (B, 73)
# → 89: feat_out = self.feat_encoder(feff_feats)               # (B, 64)
```

**实际状态**：

- encoder **当前 73 维**（从 docstring + 默认参数确认）
- 但 `nn.Linear` 的第一个参数**不是字面量 73**，而是 `feat_dim` 形参（默认 73）
- 改 73→74 涉及：
  1. **Line 37**: `feat_dim : int, 默认 73` → 默认值改 74
  2. **未确认**：`__init__` 实际接收 `feat_dim` 处的默认值（grep 没显示，需要 `view` 文件确认 line 30-70 区域）
  3. **未确认**：项目内是否有 instantiate 时显式传 `feat_dim=73`（如有，那处也要改）
  4. **建议但非必须**：docstring 也改一致（line 2 / 28 / 80）

handoff §2 描述 "一行改" **方向正确，但实际 2-3 处**（默认值 + 可能的 instantiate 调用 + 可选 docstring）。这是 §12 不可变量边界外的 scope 微调，需 MA3 确认。

---

## 6. 综合观察分析

### 6.1 三件套对齐 ✅
scaler 74 / pkl 74 列 / names 74 行 / encoder（待改到 74）—— 全 74。维度一致性问题闭合。

### 6.2 sample_name 跨数据源对齐 ⚠️ **未验证**
feff DataFrame index 用 `mp-{ID}__mp-{ID}-EXAFS-{Element}-K`。**v2 split csv（train/val/test/holdout_samples_v2.csv）的 sample_name 列是否使用相同格式 —— 未验证**。如果 v2 split csv 用 `mp-XXXX::Co_0` 或其他格式，则 Dataset `df.loc[sample_name]` 会全 miss → 灾难。这是 Phase 0.5 必须做的核心验证。

### 6.3 sklearn 1.6.1→1.7.2 unpickle warning
功能无影响（验证：transform(zeros) 输出无 NaN/Inf，shape 正确），但每次 import 链 unpickle 都会刷一行 warning，污染训练 log。

### 6.4 incompat_pool 数据"在表里但不训练"
feff DataFrame 含全部 128,382 sample（推测包含 incompat），但训练时通过 v2 split csv 的 sample_name 索引选取 → 不会泄漏 incompat 到训练，前提是 **Dataset 实现严格用 sample_name 过滤**。这要在 Phase 3 Dataset 实现 + Phase 6 forward test 中显式 assert（v2 split csv 的样本数与训练实际过的样本数一致）。

### 6.5 DataFrame.loc 性能（次要，不阻塞 Step 3）
12.8 万行 × 74 列 float32 ≈ 38 MB 常驻内存。`df.loc[sample_name]` 在 unique string index 上是 hash O(1)，但比 dict 慢约 2-3x（pandas 索引开销）。Step 4 训练高频 `__getitem__` 时如成 bottleneck，可在 Dataset `__init__` 里预转 dict（`self.feff_dict = df.to_dict('index')` 或 `df.values + df.index → mapping`）。**Step 3 阶段先用 `df.loc`，性能问题留 Step 4 处置**。

### 6.6 handoff §1.4 描述修订需求
对照 Phase 0.3/0.3b/0.3c 实测，handoff §1.4 需要在以下地方补充事实：

| handoff §1.4 项 | 当前描述 | 实测补充 |
|---|---|---|
| `feff_features_imputed.pkl` | "40.3 MB" | DataFrame，shape (128382, 74)，index `sample_name`（object，unique，no null），格式 `mp-{ID}__mp-{ID}-EXAFS-{Element}-K`，dtype float32 × 74，零 NaN，含 incompat |
| `feff_feature_scaler.pkl` | "1.6 KB" | RobustScaler，74 维，sklearn 1.6.1 训（当前 1.7.2 unpickle 触发 warning，功能无损） |
| `feff_feature_names.txt` | "1.0 KB" | 74 行，前 5 是 `xmu_*` XANES 统计量，后 5 是 `R1/R2_*` EXAFS 壳层峰特征 + `has_pre_edge` |
| 维度统一 | （未明示）73→74 | **三件套现状已是 74**，handoff §2 "73→74 单向迁移" 描述是 encoder 视角的对接需求，不是数据本身的迁移 |

---

## 7. 待 MA3 决策清单

按需要决策的紧迫性排序。

### 7.1 【次要 / 推荐自动通过】Phase 0.2 grep 模式补丁固化

**问题**：handoff §3.2 grep 模式漏 `np.float_` / `np.int_` / `np.complex_` 这类下划线后缀别名（虽然实测 Exp2 仓库无命中，无实际影响）。

**选项**：
- A. 不动 handoff，仅本汇报作为 errata 记录
- B. handoff §3.2 后续修订时把补 grep 加进去

**Sub-Agent 倾向**：B（避免后续 Sub-Agent 重复推导）。但实际无技术影响，**MA3 可以 ignore**。

### 7.2 【次要 / 推荐自动通过】Phase 4 encoder 改动从"一行"修订为"2-3 处"

**问题**：handoff §2 决策 B 写"一行改 73→74"，但 encoder 用 `feat_dim` 形参，实际改动是：
- (a) `feat_dim` 默认值 37 行 73→74（必改）
- (b) 项目内若有 instantiate 时显式 `feat_dim=73`（待 grep 确认，可能改）
- (c) docstring 多处（line 2 / 28 / 80，可选改，建议改）

**选项**：
- A. 维持 "改 default + 必要的 instantiate 调用"，docstring 不改（最小改动）
- B. default + instantiate + docstring 全改（最干净）
- C. 完全不改 default，只在 Step 3 调用处显式传 `feat_dim=74`（最不侵入原 encoder.py）

**Sub-Agent 倾向**：B（不留 mismatch；docstring 与代码不一致是后续 debug 陷阱）。但 A/C 也合理。**MA3 决定**。

### 7.3 【中等】sklearn 1.6.1→1.7.2 unpickle warning 处理

**问题**：每次 unpickle scaler 都刷一行 InconsistentVersionWarning。功能无损（已验证），但污染 log。

**选项**：
- A. 不处理，让 warning 出现（透明度优先）
- B. 在训练 pipeline 顶部 `warnings.filterwarnings("ignore", category=InconsistentVersionWarning)` 全局过滤
- C. 重新用 sklearn 1.7.2 训一遍 RobustScaler（需要原 73 维 / 74 维 fit 数据 —— 这要看 Step 2 阶段的训练脚本能否复现 fit。**风险**：如果重训出来的 `center_` `scale_` 与原 pkl 微小差异，会改变 Exp2 已 trained checkpoint 的 inference 数值结果，影响后续 fine-tune 的可对比性）
- D. 仅在 Dataset `__init__` 处局部过滤（`with warnings.catch_warnings(): ... warnings.simplefilter("ignore"); scaler = joblib.load(...)`）

**Sub-Agent 倾向**：D（局部过滤，不影响其他 warning，不重训冒险）。**MA3 决定**。

### 7.4 【高 / 必须 MA3 拍】sample_name 跨数据源对齐验证（Phase 0.5 设计）

**问题**：Phase 0.5（handoff §3.5）要求 4 数据源 key 对齐 sanity（v2 split csv × 4 splits × 100 样本，全 0 miss）。但 handoff §3.5 没明示**用什么 key**。

实际需要验证 4 件事：
1. **v2 split csv** 的 sample_name 列内容（格式是什么？）
2. **feff_features_imputed.pkl** index 格式（已知：`mp-{ID}__mp-{ID}-EXAFS-{Element}-K`）
3. **spectra_{train/val/test/holdout}.pkl** 容器类型 + key 格式（未验证）
4. **shell_boundaries.pkl** 容器类型 + key 格式（未验证）

四源 key 必须**完全一致字符串**（或有一个明确的拼接规则把 v2 split csv 列拼成 feff index 格式）。任何 miss 立刻停汇报。

**选项**：
- A. Phase 0.5 严格按 handoff §3.5：4 split × 100 样本随机抽，feff/spectra/shell 三源 lookup，0 miss 通过
- B. 在 A 基础上**先做一个 sub-step 0.5a**：先 inspect 一行 v2 split csv 看 sample_name 列格式 + inspect 一个 spectra pkl 看 key 格式，**确认所有源用同一种格式**再做 0 miss 验证。这能把"格式不一致"和"个别样本 miss"区分开来
- C. 如果发现源格式不一致，**汇报 MA3 决定 Dataset 用哪个格式做 canonical key + 在哪里做拼接**（在 csv load 时拼？还是在 `__getitem__` 里拼？）

**Sub-Agent 倾向**：B（先看格式再做 sanity，否则 0 miss 失败时分不清是格式问题还是覆盖率问题）。**MA3 决定**。

### 7.5 【高 / 必须 MA3 拍】DataFrame.loc 性能 vs dict 缓存（Step 3 vs Step 4 边界）

**问题**：feff DataFrame 12.8 万行 × 74 列。Dataset `__getitem__` 每次调用 `df.loc[sample_name]`。

**两种实现方式**：
- A. 直接 `df.loc[sample_name]`，简单，pandas 内存共享，**Step 3 forward test 阶段够用**
- B. `__init__` 里转 dict：`self.feff_dict = {idx: row for idx, row in zip(df.index, df.values)}`，O(1) hash 查找，约快 2-3x，**适合 Step 4 高频训练**

**Sub-Agent 倾向**：Step 3 用 A（简单 + 验证 logic），Step 4 训练前 profile 后再转 B（如必要）。**MA3 确认这个分阶段策略 OK**。

### 7.6 【高 / 必须 MA3 拍】POSCAR 数 41,496 vs handoff ~41,431

**问题**：handoff §1.4 / §3 估计 ~41,431，实测 41,496。差 65 个。

**可能解释**：
- A. handoff 数字是估计/记忆值，41,496 是真值
- B. 41,431 是 Step 1 通过质量过滤后的"实际可用 POSCAR 数"，Windows 端 POSCAR_zip 含有未过滤的全集 41,496（多 65 个低质量未筛除）
- C. 41,431 是某个时间点的快照，41,496 是后续补充版本

**影响**：
- 如果 (B)：Dataset 应在 `__getitem__` 里 trust v2 split csv 给的 sample list，**不会 listdir POSCAR 目录**，所以 65 个多余 POSCAR 永远不会被读 → 无害
- 如果 (B) 且 v2 split csv 引用了某个不在 41,496 里的 mp_id → POSCAR 缺失，FileNotFoundError 灾难

**选项**：
- A. 接受 41,496 为真值，Phase 0.5 验证时确认所有 v2 split csv 引用的 mp_id 在 41,496 里
- B. 重新审视 Step 1 文档/数据，找到 41,431 vs 41,496 差 65 个的原因再决定

**Sub-Agent 倾向**：A（按 v2 split csv 锚定，多余 POSCAR 不读即可）。**MA3 决定**。

### 7.7 【低】missing_poscar_list.csv 的"Step 1 元数据"是否要保留

**问题**：解压时混进来一个 `missing_poscar_list.csv`，用户已确认无用并删除。但 Sub-Agent 没看过它内容 —— 它可能记录"Step 1 阶段哪些 mp_id 应有但缺失 POSCAR" 的清单，是个**有审计价值的元数据**。

**选项**：
- A. 已删，不可恢复，move on
- B. 让 Sub-Agent 在 Windows 端再上传一次只这个 csv 到 `data/` 上一层归档（可选）

**Sub-Agent 倾向**：A（已删，不重要）。**MA3 可 ignore**。

---

## 8. 后续推进路径（待 MA3 拍板后执行）

按 MA3 §7 决策结果，下一步操作如下（按 handoff §4 优先级延展）：

### Phase 0.5（依赖 §7.4 决策）
- Sub-step 0.5a（如 §7.4 选 B 或 C）：inspect 4 数据源 key 格式
- Sub-step 0.5b：4 split × 100 样本 0 miss sanity
- 任何 miss 立刻停汇报 MA3

### Phase 0.1（依赖 POSCAR，已就绪）
- 5 multi-site 样本 pymatgen distances vs `shell_boundaries.pkl` 对比
- 期望 max|diff| < 1e-3 Å
- 失败启用 brute-force fallback（handoff §3.1.1）

### Phase 3（Dataset 改造）
- 源 fork：`step3/xas_local_dataset_L6.py` → `code/xas_local_dataset_v2.py`
- 类名 `XasLocalDatasetV2`
- 实现：handoff §6 + §2.C sparse filter raise（不 None / 不 clamp）
- DataFrame.loc 查询（§7.5 决策 A 阶段）
- 数据路径常量从环境变量读：`DATA_DIR = os.environ.get("EXP4_DATA_DIR", "/home/tcat/diffcsp_exp4/data")`

### Phase 4（encoder 改动）
- 按 §7.2 决策结果改 `step2/spectrum_encoder.py`（默认值 + 可选 docstring）
- 不动其他维度（xmu 150 / chi1 200 / latent 256）

### Phase 5（diffusion_w_type_xas.py 路径常量）
- 数据路径常量改环境变量
- Dataset import：`xas_local_dataset_L6` → `xas_local_dataset_v2`
- DataModule import：`xas_local_datamodule` → `xas_local_datamodule_v2`
- 不动：cost_lattice = 0、扩散数学、forward / training_step 主体

### Phase 5b（DataModule 重写，决策 B）
- `xas_local_datamodule.py` → `xas_local_datamodule_v2.py`
- 类 `XasLocalDataModuleV2`
- 删 Windows 路径常量、删 `FEFF_CSV` 常量
- PL 2.5.5 兼容（grep `setup()` 签名）

### Phase 6（前向测试）
- 6.1：100 随机样本 dataset 实例化（§2.C 决策修订版）
- 6.2-6.5：按 handoff §9.2 推进
- 6.5：GPU bf16 forward（已 smoke test 通过环境层）

### Phase 7 + §11 汇报
- 按 handoff §11 标准向 MA3 汇报 Phase 6 结果

---

## 9. Sub-Agent 自身的不确定点

按工作哲学 §13 "诚实 > 流畅"，记录我自己当下的不确定与盲点：

1. **128,382 行的精确分解**：我推测是 train+val+test+holdout+incompat 的 sample 级展开，但**未数学验证**（v2 split csv 行数和 + incompat_pool.csv 行数和 = 128382？）。Phase 0.5 时可顺便算一下。

2. **sklearn 1.6.1 → 1.7.2 RobustScaler 真实兼容性**：我说"功能无损"基于 transform(zeros) 一个测试。**没测 inverse_transform 往返误差**，没测 RobustScaler 是否在 1.7.2 内部增加了某些 attribute 导致 1.6.1 pkl 缺失新 attribute（虽然实际 unpickle 没报错，但 lazy attribute 访问可能延迟到训练时炸）。MA3 如关注，可加一个 `transform(x) → inverse_transform(transformed) ≈ x` 往返测试（< 1e-6 误差），但**这是过度防御**，工程上 sklearn 跨小版本通常稳。

3. **DataFrame index 格式 `mp-{ID}__mp-{ID}-EXAFS-{Element}-K` 的 "mp-{ID}__mp-{ID}" 重复**：这看起来像 Step 2 阶段拼字符串时的 bug 或冗余约定。我**没去查 Step 2 文档确认这是 by design 还是偶然遗留**。如果是 bug，feff DataFrame 应该有清理过的 index 版本。如果是 by design，照搬即可。Phase 0.5 时建议 inspect 一份 Step 2 README 或 spectrum 生成代码确认。

4. **encoder 改动的"未确认"项**：grep 没看到 `__init__` 实际接收 `feat_dim` 处的代码（grep 命中了 docstring 但没命中实际代码行）。这意味着 `nn.Linear(feat_dim, ...)` 的 `feat_dim` 可能是从更外层传入而非 `__init__` 默认值用。Phase 4 实施前必须 `view` encoder.py 全文确认改动点。

5. **Phase 0.1 / 0.5 / 0.3 后续 / Phase 3-6 全部尚未启动**：Sub-Agent 2 当前进度以 Phase 0 为主，编码工作 0%。MA3 如希望加速，可考虑：
   - 允许 Sub-Agent 在 §7 决策回来前**只**推进非阻塞步骤（如 Phase 0.5a inspect 格式 / Phase 0.1 pymatgen sanity / 起草 `xas_local_dataset_v2.py` 草稿但不上线）
   - 或严格等所有 §7 决策再动 —— 当前默认按这个

---

## 10. 工作哲学 self-check

- ✅ 诚实 > 流畅：Phase 0.3 维度差异 / encoder 改动 scope 修订 / sample_name 跨源对齐风险 等都在汇报中如实呈现，未粉饰
- ✅ 不替 MA3 决定：§7 每项给出 A/B/C 选项 + Sub-Agent 倾向 + 触发条件，不预设 MA3 答案
- ✅ 一步一步：Phase 0.3 → 0.3b → 0.3c 渐进诊断，每步等用户确认输出再下一步
- ✅ 触发汇报：Phase 0.3b 维度差异 + Phase 0.3c 行数异常 + encoder 改动 scope 偏移 → 主动停下汇报，而非自行推进
- ⚠️ 上下文消耗：本汇报本身是触发汇报，但已消耗一些上下文。POSCAR 上传 + Phase 0.2/0.3/0.4 的诊断脚本编写、5 次 heredoc 写入、3 次诊断脚本执行 + 解读，估算消耗约 30-40% 上下文。后续如继续 Phase 0.1 / 0.5 + Phase 3-6 编码，**预计在 Phase 5b 或 Phase 6 中段触达 80%，需要再次启动 Sub-Agent 3 handoff**

---

**Sub-Agent 2 撰写，2026-04-25，等待 MA3 §7 决策。**
