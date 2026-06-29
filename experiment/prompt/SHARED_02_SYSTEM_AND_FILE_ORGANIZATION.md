# SHARED_02_SYSTEM_AND_FILE_ORGANIZATION.md
# 系统配置与文件组织规范 — 所有 Sub-Agent 必读

> **本文档版本**: v1.0  
> **维护者**: Main Agent

---

## 1. 硬件配置

| 项目 | 规格 |
|------|------|
| GPU（主力训练） | NVIDIA RTX A4000，显存 16 GB |
| GPU（集成，不用于训练） | Intel UHD 集成显卡 |
| 操作系统 | Windows（路径使用反斜杠 `\`） |

### GPU 使用注意
- 所有 PyTorch 脚本默认使用 `cuda:0`（RTX A4000）
- 批量大小（batch size）需根据 16 GB 显存限制设计，如无特殊说明，数据加载时使用 `pin_memory=True`
- 集成显卡不参与计算，无需在脚本中处理多 GPU 逻辑

---

## 2. 项目根目录

```
DiffCSP 代码库根目录:
  C:\Users\T-Cat\Desktop\DiffCSP-main\

所有新增脚本和输出的根目录:
  C:\Users\T-Cat\Desktop\DiffCSP-main\experiment\
```

**⚠️ 严禁修改以下原始目录内的任何文件**：
```
C:\Users\T-Cat\Desktop\DiffCSP-main\diffcsp\
C:\Users\T-Cat\Desktop\DiffCSP-main\conf\
C:\Users\T-Cat\Desktop\DiffCSP-main\data\
```
如需引用原始代码，使用 `import` 或 `sys.path.insert`，不要直接编辑原文件。

---

## 3. 脚本命名规范

所有脚本文件名格式：
```
step{X}.{Y}_{描述}.py
```

示例：
```
step1.1_scan_and_dedup.py
step1.2_parse_spectra_poscar.py
step1.3_extract_features.py
step1.4_split_holdout.py
step2.1_spectrum_encoder.py
step2.2_multisite_aggregator.py
...
```

**规则**：
- 大步骤用整数（step1, step2 ...）
- 子步骤用小数点（step1.1, step1.2 ...）
- 文件名全部小写，空格用下划线替代
- 脚本开头注释必须说明：该脚本的 Step 编号、输入来源、输出目标

---

## 4. 文件存储结构（experiment/ 目录）

```
C:\Users\T-Cat\Desktop\DiffCSP-main\experiment\
│
├── step1\                          # Step 1 所有输出
│   ├── data_inventory.csv          # 数据清单（所有位点文件夹列表）
│   ├── dedup_report.txt            # 去重报告
│   ├── parsed_spectra\             # 解析后的谱（numpy/hdf5格式）
│   ├── feature_tables\             # 提取的特征表
│   ├── holdout_1000_ids.txt        # 1000个保留集 mp_id（Step 1.4 生成）
│   ├── train_ids.txt               # 训练集 mp_id 列表
│   ├── val_ids.txt                 # 验证集 mp_id 列表
│   └── test_ids.txt                # 测试集 mp_id 列表
│
├── step2\                          # Step 2 所有输出
│   ├── encoder_checkpoints\        # 谱编码器权重
│   └── embedding_cache\            # 预计算的谱 embedding（可选）
│
├── step3\                          # Step 3 所有输出
│   ├── modified_diffcsp\           # 改造后的模块（复制自原代码）
│   └── configs\                    # 训练配置文件
│
├── step4\                          # Step 4 所有输出
│   ├── training_logs\              # 训练日志
│   ├── checkpoints\                # 模型权重
│   └── val_metrics.csv             # 验证集指标
│
└── step5\                          # Step 5 所有输出（最终评估）
    └── holdout_eval_results.csv    # 保留集评估结果
```

---

## 5. 脚本开头模板

每个脚本**必须**以以下格式开头：

```python
# =============================================================================
# 脚本编号: step{X}.{Y}
# 脚本名称: step{X}.{Y}_{描述}.py
# 输入:
#   - {输入文件/目录路径}
# 输出:
#   - {输出文件/目录路径}
# 说明:
#   {简短说明本脚本做什么}
# =============================================================================

import os
import sys

# 项目根目录
PROJECT_ROOT = r"C:\Users\T-Cat\Desktop\DiffCSP-main"
EXPERIMENT_DIR = os.path.join(PROJECT_ROOT, "experiment")
STEP_DIR = os.path.join(EXPERIMENT_DIR, "step{X}")
os.makedirs(STEP_DIR, exist_ok=True)

# 数据根目录
SITE_DATASET_DIR = r"C:\Users\T-Cat\Desktop\XAS-FeO\site_dataset"
IONIC_DATASET_DIR = r"C:\Users\T-Cat\Desktop\XAS-FeO\test_missing_keep3_packed_A"
```

---

## 6. 数据路径常量（所有脚本统一引用）

```python
# 原始数据
SITE_DATASET_DIR   = r"C:\Users\T-Cat\Desktop\XAS-FeO\site_dataset"
IONIC_DATASET_DIR  = r"C:\Users\T-Cat\Desktop\XAS-FeO\test_missing_keep3_packed_A"

# 参考表格
BOND_CONSTRAINT_CSV  = r"C:\Users\T-Cat\Desktop\XAS-FeO\all_center_neighbors_summary.csv"
FEATURE_REF_CSV      = r"C:\Users\T-Cat\Desktop\XAS-FeO\feff_features_all_stable_v2.csv"
FEATURE_SITE_CSV     = r"C:\Users\T-Cat\Desktop\XAS-FeO\feff_features_all_site_v2.csv"
FEATURE_IONIC_CSV    = r"C:\Users\T-Cat\Desktop\XAS-FeO\feff_features_all_ionic_v3.csv"

# DiffCSP
DIFFCSP_ROOT       = r"C:\Users\T-Cat\Desktop\DiffCSP-main"
EXPERIMENT_DIR     = r"C:\Users\T-Cat\Desktop\DiffCSP-main\experiment"
```

---

## 7. 依赖环境参考

DiffCSP 原始依赖（从 repo 结构推断），各 Step Agent 实现时可能需要：
- Python 3.8+
- PyTorch（CUDA 支持）
- PyTorch Lightning（pl_modules 结构）
- pymatgen（POSCAR 解析）
- numpy, pandas, scipy
- hydra-core（conf/ 配置体系）
- ase（可选，晶体结构处理）

如需新增依赖，在脚本开头注释中说明：`# 新增依赖: {包名}`

---

## 8. Sub-Agent 工作流程规范

每个 Step Agent 完成任务后，必须输出一份**总结报告**，格式如下：

```markdown
## Step {X}.{Y} 执行报告

### 执行状态
[成功 / 部分成功 / 失败]

### 完成的 Actions
1. {action 1}
2. {action 2}

### 输出文件
- {文件路径}: {文件说明}

### 发现的问题 / 需要注意的事项
- {问题描述}

### 对后续步骤的建议
- {建议}
```

此报告由用户转交 Main Agent，Main Agent 据此决定是否继续下一步。
