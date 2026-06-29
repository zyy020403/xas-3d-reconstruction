# EXP4_PREPARATION_AGENT_HANDOFF.md
# 服务器资产完整性盘点 · Preparation Agent 交接文档

> **撰写者**：DiffCSP-Exp4-Main-Agent 4
> **接收者**：Preparation Agent（一次性盘点 + 上传 + 验证任务）
> **日期**：2026-04-26
> **触发原因**：Sub-Agent 4 在 Phase 6.4 暴露 `ModuleNotFoundError: No module named 'diffcsp'`。根因是 MA1/2/3 全程的 scp 命令 (`scp -r experiment2\* ...`) 假设 Exp2 = experiment2/ 子目录全部内容,但 Windows 实际结构是 `DiffCSP-main/{diffcsp/, experiment2/, ...}`,**diffcsp 框架包是 experiment2/ 的姊妹目录**,从未上传。Sub-Agent 1/2/3 没踩坑是因为他们没 import `diffcsp.*`,Sub-Agent 4 第一次 CSPDiffusion 实例化才触发。
> **本任务范围**：在恢复 Sub-Agent 4 前,**全面盘点 Windows + 服务器**两端,识别所有"在 Windows 端存在但服务器缺失"的代码资产,补传后逐项验证。不限于 diffcsp/。
> **本任务不做**：不写 Python 代码、不修改任何已有文件、不跑 forward_test.py、不动 Sub-Agent 4 任何已交付资产、不改 config 内容、不装 pip 包

---

## §0 工作概述

**一句话目标**：把 Windows `DiffCSP-main\` 下**所有 Sub-Agent 4-续 在 Phase 6.4/6.5 训练时可能 import 到**的 Python 代码资产,补传到 `/home/tcat/diffcsp_exp4/code/`,并逐项验证 import 链可达。

**非目标**：
- ❌ 不验证代码内容正确性（那是 Check Agent / Sub-Agent 4 的事）
- ❌ 不评估方案、不提优化建议
- ❌ 不上传数据（数据在 `/home/tcat/diffcsp_exp4/data/` 已就绪,Sub-Agent 4 已确认）
- ❌ 不上传 checkpoints / logs / 实验产物（那些是训练后才有的）
- ❌ 不修改任何文件,只**新增**

**输出**：一份盘点报告（§7 模板）,逐项 PASS / MISSING / SUSPICIOUS。

**单窗口预算**：MA4 估算 30-50K token（盘点 + scp 指令 + import 验证清单）。**60% 上下文闸门**触发即停汇报。

---

## §1 你必须读的文档

| # | 文档 | 必读? | 重点 |
|---|------|-------|------|
| 1 | **本文档**(你在读) | ✅ | 全文 |
| 2 | EXP4_MAINAGENT4_HANDOFF.md | ✅ | §3 服务器环境、§4.2 已上传数据列表（**这些不动**）、§7 文件归属表 |
| 3 | EXP4_FILE_INVENTORY.md | ✅ | §1-§5 完整本地 + 服务器目录树（参考 Sub-Agent 4 之前盘点的"应该有什么"）|
| 4 | exp2tree.txt | ✅ | Windows experiment2/ 子目录树（已知）|
| 5 | Sub-Agent 4 中途停汇报全文 | ✅ | 暴露的 diffcsp 缺失证据（Check 1-5 输出）|
| 6 | EXP4_STEP3_SUBAGENT4_HANDOFF.md | 可选 | 仅参考 §5 文件归属表 + Sub-Agent 4 已交付资产清单 |

**你不要读**：EXP4_PROGRESS_LOG / EXPERIMENT2_FINAL_REPORT / EXP4_PROPOSAL_v2 / Check Agent 报告 / 各文件具体内容。这些与盘点任务无关,读了浪费 token。

---

## §2 Sub-Agent 4 暴露的盲区（你工作的起点）

### §2.1 直接证据

```
=== Check 1: pip-installed diffcsp in mlff env? ===
ModuleNotFoundError: No module named 'diffcsp'

=== Check 2: filesystem search for diffcsp directory ===
（零命中,/home/tcat 全盘下不存在 diffcsp/ 目录）

=== Check 4: code/ 下到底有啥 ===
check.py  prompt  step1  step2  step3  step4  step4b  step4c  step4d  step5  step6
（仅 experiment2 子目录,无 diffcsp 框架包,无 run.py,无 setup.py 等根级文件）
```

### §2.2 根因

旧 `step4d_2_train.py:14` 的 `PROJECT_ROOT = r"C:\Users\T-Cat\Desktop\DiffCSP-main"` + `sys.path.insert(0, PROJECT_ROOT)` 证明：

```
Windows 实际结构（推断）:
  C:\Users\T-Cat\Desktop\DiffCSP-main\
    ├── diffcsp\              ← 框架包(姊妹目录,服务器缺失)
    ├── experiment2\          ← 已上传到 /home/tcat/diffcsp_exp4/code/ 内容
    ├── experiment4\          ← 已上传(数据)
    ├── (可能还有 run.py / setup.py / pyproject.toml / conf/ / scripts/)
```

MA3 SUBAGENT_HANDOFF §4.2 那条 `scp -r C:\...\experiment2\* ...` 只取了 experiment2 的**内容**,**没取**它的兄弟目录。

### §2.3 推断必须存在的 diffcsp 子结构

旧 `step4d_2_train.py` 的 sys.path 引用 + 标准 DiffCSP 库的开源结构,推断 `diffcsp/` 至少包含：

```
diffcsp/
├── __init__.py
├── common/
│   ├── __init__.py
│   ├── utils.py
│   ├── data_utils.py
│   └── ...
├── pl_modules/
│   ├── __init__.py
│   ├── cspnet.py
│   ├── diff_utils.py
│   ├── gnn.py
│   ├── model.py(or diffusion.py)
│   └── ...
├── pl_data/
│   └── ...
└── ...
```

**但这只是推断**。实际结构以 Windows `DiffCSP-main\diffcsp\` 真实内容为准。这正是你（Preparation Agent）要先去查的。

### §2.4 可能的其他盲区（你要识别）

| 类别 | 假设 | 验证方式 |
|---|---|---|
| `DiffCSP-main\` 根目录是否还有 `run.py` / `setup.py` / `pyproject.toml` / `requirements.txt` | 标准 DiffCSP 库通常有 | Windows 端 `dir /B` 根目录 |
| 是否还有顶层 `conf/`（hydra root config）| 有 `conf_xas/` 在 step3 内,但根 conf 可能也存在 | Windows 端检查 |
| 是否需要 `diffcsp` 包 `pip install -e .` 安装 | 如果有 `setup.py`,可能需要 | §6 决策点 |
| Exp2 `experiment2/` 下是否还有未传的根级文件（比如 `requirements.txt`、`README` 之类的元数据）| 已传过 step3/* 等子目录,根级文件可能漏 | Windows 端 `dir experiment2 /B` 对照服务器 |

---

## §3 你（Preparation Agent）的工作流（4 阶段）

| 阶段 | 任务 | 用户交互 |
|---|---|---|
| §4 盘点 Windows 端 | 让用户跑 PowerShell 命令查根目录 + diffcsp/ + experiment2/ 真实结构 | 用户粘贴 dir 输出 |
| §5 盘点服务器端 | 让用户跑服务器命令查 /home/tcat/diffcsp_exp4/code/ 现状 | 用户粘贴 ls 输出 |
| §6 生成上传清单 + scp 命令 | 你对照两端差异,列出**所有**应补传文件;给用户一次性 scp 命令 | 用户跑 scp |
| §7 上传后验证 | import 链可达性测试（Python 层面）+ 树结构对照 | 用户跑 python + ls,粘贴输出 |

**每一阶段做完才进下一阶段**。不要并行让用户跑多组命令。

---

## §4 阶段 A：Windows 端盘点

### §4.1 给用户的命令

让用户**逐条**跑下面 PowerShell 命令,每条粘贴输出回来:

```powershell
# A1: DiffCSP-main 根目录所有内容(目录+文件)
cd C:\Users\T-Cat\Desktop\DiffCSP-main
Get-ChildItem -Force | Select-Object Mode, Name | Format-Table -AutoSize
```

```powershell
# A2: diffcsp/ 包结构(2 级深度,只看 .py 文件 + 目录)
Get-ChildItem -Path C:\Users\T-Cat\Desktop\DiffCSP-main\diffcsp -Recurse -Depth 2 |
    Where-Object { $_.PSIsContainer -or $_.Extension -eq ".py" } |
    Select-Object FullName |
    Format-Table -AutoSize
```

```powershell
# A3: 根目录是否有顶层 conf/ 或 configs/
Test-Path C:\Users\T-Cat\Desktop\DiffCSP-main\conf
Test-Path C:\Users\T-Cat\Desktop\DiffCSP-main\configs
Test-Path C:\Users\T-Cat\Desktop\DiffCSP-main\config
```

```powershell
# A4: 根目录是否有标准 Python 项目元数据
Test-Path C:\Users\T-Cat\Desktop\DiffCSP-main\setup.py
Test-Path C:\Users\T-Cat\Desktop\DiffCSP-main\pyproject.toml
Test-Path C:\Users\T-Cat\Desktop\DiffCSP-main\requirements.txt
Test-Path C:\Users\T-Cat\Desktop\DiffCSP-main\README.md
Test-Path C:\Users\T-Cat\Desktop\DiffCSP-main\run.py
```

```powershell
# A5: experiment2/ 根级文件(看是否有未上传的非子目录文件)
Get-ChildItem -Path C:\Users\T-Cat\Desktop\DiffCSP-main\experiment2 -File |
    Select-Object Name |
    Format-Table -AutoSize
```

### §4.2 你解读 A 阶段输出的判定

| A1 输出 | 解读 |
|---|---|
| 看到 `diffcsp` 目录存在 | ✓ 上传源就绪 |
| 看到 `experiment2` 目录 | ✓ 已上传过 |
| 看到 `experiment4` 目录 | ✓ 数据来源（已上传） |
| 看到顶层 `.py` 文件（如 `run.py`、`hydra_main.py`）| 标记为可能需上传 |
| 看到 `setup.py` / `pyproject.toml` | 标记为决策点（§6.3 决定是否 pip install -e .） |
| 看到 `*.txt` / `*.yaml` / `*.cfg` | 个案判断（如 requirements.txt 通常不传到服务器） |

| A2 输出 | 解读 |
|---|---|
| `diffcsp/__init__.py` 存在 | ✓ 是合法 Python 包 |
| `diffcsp/pl_modules/cspnet.py` / `diff_utils.py` | ✓（必须上传） |
| `diffcsp/common/*.py` | ✓（必须上传） |
| 还有其他子目录（如 `pl_data/`、`scripts/`、`utils/`、`script/`） | 全部上传（Sub-Agent 不知道哪些被 import,全传安全） |

| A3 输出 | 解读 |
|---|---|
| 顶层 conf/ 存在 | 上传到 `code/` 根级（与 step3/conf_xas/ 区分） |
| 全 False | OK,只用 step3/conf_xas/ |

| A4 输出 | 解读 |
|---|---|
| `setup.py` 存在 → §6.3 决策点 | 是否需要 `pip install -e .`,可能影响 import 路径 |
| `run.py` 存在 | 上传（DiffCSP 标准训练入口可能依赖它） |
| 其他元数据 | 个案 |

| A5 输出 | 解读 |
|---|---|
| 仅 `EXP2_PROPOSAL_FINAL.md` 等 .md 文件 | 不需上传（文档） |
| 有 `.py` 根级文件 | **必须**上传（可能被 import） |

---

## §5 阶段 B：服务器端盘点

### §5.1 给用户的命令

让用户在服务器上跑（**已 ssh + conda activate mlff**）:

```bash
# B1: code/ 根级当前内容
ls -la /home/tcat/diffcsp_exp4/code/
```

```bash
# B2: 是否有任何 diffcsp 痕迹
find /home/tcat/diffcsp_exp4 -name "diffcsp" -type d 2>/dev/null
find /home/tcat/diffcsp_exp4 -name "diffcsp" -type f 2>/dev/null
find /home/tcat/diffcsp_exp4 -name "__init__.py" 2>/dev/null | head -20
```

```bash
# B3: mlff env 是否曾装过 diffcsp（pip）
pip show diffcsp 2>&1 | head -5
python -c "import diffcsp" 2>&1 | head -3
```

```bash
# B4: experiment2 顶层 .py 是否传过
ls /home/tcat/diffcsp_exp4/code/*.py 2>&1 | head
```

```bash
# B5: Sub-Agent 4 已交付资产 sanity（不动,只验证存在）
ls -la /home/tcat/diffcsp_exp4/code/step3/diffusion_w_type_xas.py \
       /home/tcat/diffcsp_exp4/code/step3/xas_local_dataset_v2.py \
       /home/tcat/diffcsp_exp4/code/step3/xas_local_datamodule_v2.py \
       /home/tcat/diffcsp_exp4/code/step3/forward_test.py \
       /home/tcat/diffcsp_exp4/code/step2/spectrum_encoder.py \
       /home/tcat/diffcsp_exp4/code/step3/conf_xas/model/diffusion_xas.yaml 2>&1
```

### §5.2 你解读 B 阶段输出

| B1 期望 | 解读 |
|---|---|
| `check.py prompt step1...step6` | Sub-Agent 4 报告的现状,符合预期 |
| 看到 `diffcsp/` | 已上传过(罕见,通常是 None) |
| 看到额外目录 | 标记调查 |

| B2 期望 | 解读 |
|---|---|
| 全空(diffcsp 目录/文件零命中) | 确认 Sub-Agent 4 诊断,需要补传 |
| 任何命中 | **意外**,停下来给 MA4 报告 |

| B3 期望 | 解读 |
|---|---|
| `pip show diffcsp` 报错"WARNING: Package(s) not found" + `import diffcsp` ModuleNotFoundError | 确认 env 也没装 |
| 有任何输出（即 pip 装过） | **意外**,停下报告（可能存在 stale install） |

| B5 期望 | 解读 |
|---|---|
| 6 个文件全 ls 通过 | Sub-Agent 4 资产完好 |
| 任一文件缺 | **严重**,停下报告 MA4（这是 Sub-Agent 4 交付物,缺失即数据丢失）|

---

## §6 阶段 C：生成上传清单 + scp 指令

### §6.1 你的逻辑

对照 A 阶段（Windows 端有什么）和 B 阶段（服务器缺什么）,对每条差异给一个动作：

```
diff = (A 端有 / B 端没有)

对每条 diff:
  if 是 *.py 或 *.yaml 或 *.json 或 *.toml 或目录:
    → 加入上传清单
  if 是 *.md / *.txt(README) / *.png:
    → 不传(文档/产物)
  if 是 .git / __pycache__ / *.pyc:
    → 不传
  if 模糊:
    → 标记 SUSPICIOUS,问 MA4
```

### §6.2 推荐 scp 指令模板（你给用户填空）

基于 A2 输出的真实 diffcsp 结构,你给出**一条** scp 指令（最简洁）：

```powershell
# 在 Windows PowerShell 跑(只输一次密码)
scp -r "C:\Users\T-Cat\Desktop\DiffCSP-main\diffcsp" `
       tcat@scsmlnprd02.its.auckland.ac.nz:/home/tcat/diffcsp_exp4/code/
```

如果 A1/A4 显示根目录还有 `run.py` / 顶层 `conf/` / 其他必传文件,在第二条指令补上：

```powershell
# 仅当 A 阶段确认存在时才需要
scp "C:\Users\T-Cat\Desktop\DiffCSP-main\run.py" `
    tcat@scsmlnprd02.its.auckland.ac.nz:/home/tcat/diffcsp_exp4/code/

scp -r "C:\Users\T-Cat\Desktop\DiffCSP-main\conf" `
       tcat@scsmlnprd02.its.auckland.ac.nz:/home/tcat/diffcsp_exp4/code/
```

**关键原则**：不要拍脑袋让用户传他没确认过的文件。**只传 A 阶段证实存在的**。

### §6.3 决策点：是否 pip install -e . diffcsp 包

如果 A4 输出显示 `setup.py` 存在,出现一个分叉：

| 选项 | 操作 | 优劣 |
|---|---|---|
| **方案 X：纯文件上传 + sys.path** | scp diffcsp/ 到 code/,代码靠 `sys.path.insert(0, "/home/tcat/diffcsp_exp4/code")` 找到 | 简单,无副作用,但 forward_test.py 当前未必有这行 sys.path |
| **方案 Y：pip install -e .** | scp diffcsp/ + setup.py 到 code/,服务器跑 `pip install -e /home/tcat/diffcsp_exp4/code/` | 标准做法,但需装新包,可能引入版本冲突 |

**你不替 MA4 选**。如果发现 setup.py 存在,把这两个选项原样写进 §7 报告,让 MA4 拍板。

如果 A4 输出 setup.py **不存在**：默认走方案 X（纯文件 + sys.path）。Sub-Agent 4 的 forward_test.py 由 MA4 在下一棒决定是否补 sys.path 行。

---

## §7 阶段 D：上传后验证 + 报告模板

### §7.1 验证命令（用户跑完 scp 后）

让用户跑：

```bash
# D1: 树结构对照(预期 4 个核心文件可见)
ls /home/tcat/diffcsp_exp4/code/diffcsp/__init__.py \
   /home/tcat/diffcsp_exp4/code/diffcsp/common/utils.py \
   /home/tcat/diffcsp_exp4/code/diffcsp/pl_modules/cspnet.py \
   /home/tcat/diffcsp_exp4/code/diffcsp/pl_modules/diff_utils.py 2>&1
```

```bash
# D2: diffcsp 包完整性(对照 Windows A2 输出的子目录列表)
find /home/tcat/diffcsp_exp4/code/diffcsp -type d | sort
find /home/tcat/diffcsp_exp4/code/diffcsp -name "*.py" | wc -l
```

```bash
# D3: import 可达性(关键)
cd /home/tcat/diffcsp_exp4/code
python -c "
import sys
sys.path.insert(0, '/home/tcat/diffcsp_exp4/code')
import diffcsp
print('diffcsp at:', diffcsp.__file__)
from diffcsp.common import utils
print('common.utils OK')
from diffcsp.pl_modules import cspnet
print('pl_modules.cspnet OK')
from diffcsp.pl_modules import diff_utils
print('pl_modules.diff_utils OK')
" 2>&1
```

```bash
# D4: Sub-Agent 4 资产仍然完好(再次 sanity,确认 scp 没污染)
ls -la /home/tcat/diffcsp_exp4/code/step3/diffusion_w_type_xas.py \
       /home/tcat/diffcsp_exp4/code/step3/xas_local_dataset_v2.py \
       /home/tcat/diffcsp_exp4/code/step3/xas_local_datamodule_v2.py \
       /home/tcat/diffcsp_exp4/code/step3/forward_test.py \
       /home/tcat/diffcsp_exp4/code/step2/spectrum_encoder.py 2>&1
```

### §7.2 验证判定

| D1 期望 | 解读 |
|---|---|
| 4 文件全 ls 通过 | diffcsp 上传成功 |
| 任一缺 | scp 失败 / Windows 实际结构与推断不符,停汇报 MA4 |

| D2 期望 | 解读 |
|---|---|
| 子目录列表与 Windows A2 输出对齐 | ✓ |
| .py 文件数与 A2 大致相同（±2 容差） | ✓ |
| 显著差异 | 部分文件遗漏,可能 .gitignore / 隐藏文件 |

| D3 期望 | 解读 |
|---|---|
| 4 个 print 全打印,无 ImportError | **diffcsp import 链可达,Sub-Agent 4-续 可继续** |
| 任一 ImportError | 包结构问题,停汇报 MA4 |

| D4 期望 | 解读 |
|---|---|
| 5 文件全 ls 通过 | Sub-Agent 4 资产未受 scp 污染 |
| 任一缺 | **严重**,scp 误覆盖,停汇报（极不可能,但要查） |

---

## §8 输出报告模板

```
# Preparation Agent 服务器资产盘点报告

## 总体判定
- diffcsp 包上传: [SUCCESS / PARTIAL / FAILED]
- import 链可达: [PASS / FAIL]
- Sub-Agent 4 资产完好: [PASS / FAIL]
- 是否阻塞 Sub-Agent 4-续启动: [否 / 是]

## §4 阶段 A:Windows 端盘点
A1 DiffCSP-main 根目录:
   主要内容: ___（diffcsp/ experiment2/ experiment4/ 等）
   是否有 run.py / setup.py: ___
   其他根级 .py: ___

A2 diffcsp/ 包结构:
   子目录列表: ___
   .py 总数: ___
   关键文件全在: [是/否] (__init__.py, common/utils.py, pl_modules/cspnet.py, pl_modules/diff_utils.py)

A3 顶层 conf/: ___
A4 setup.py / pyproject.toml: ___
A5 experiment2 根级 .py: ___

## §5 阶段 B:服务器端盘点
B1 code/ 现状: [与 Sub-Agent 4 报告一致 / 偏差: ___]
B2 diffcsp 痕迹: [零命中 / 意外命中: ___]
B3 mlff env diffcsp: [未装(预期) / 已装(意外)]
B4 experiment2 根级 .py: ___
B5 Sub-Agent 4 资产: [6/6 完好 / 缺失: ___]

## §6 阶段 C:上传清单
执行的 scp 命令(实际跑的):
1. scp -r "...\diffcsp" tcat@.../code/    [运行结果: ___]
2. (可选) scp run.py / conf/ ___          [运行结果: ___]

§6.3 setup.py 决策:
- setup.py 存在: [是/否]
- 推荐方案: [X 纯 sys.path / Y pip install -e . / N/A]
- 留给 MA4 决定: [是/否]

## §7 阶段 D:上传后验证
D1 4 关键文件 ls: [PASS / FAIL: ___]
D2 包完整性: 子目录数 ___, .py 文件数 ___
D3 import 链:
   import diffcsp: [PASS / FAIL: ___]
   from diffcsp.common import utils: [PASS / FAIL: ___]
   from diffcsp.pl_modules import cspnet: [PASS / FAIL: ___]
   from diffcsp.pl_modules import diff_utils: [PASS / FAIL: ___]
D4 Sub-Agent 4 资产: [5/5 完好 / 缺失: ___]

## 异常 / 需要 MA4 注意

(任何 SUSPICIOUS 或 FAIL,简短说明)

## 给 Sub-Agent 4-续 的注意事项
(基于本次盘点发现的,影响下一棒的事项,如 sys.path 是否要加在 forward_test.py 顶部)

## 你(Preparation Agent)的上下文消耗
约 ___ %
```

---

## §9 工作哲学（继承 Sub-Agent 1/2/3/4 + Check Agent）

1. **诚实 > 流畅**：A/B 阶段输出与你预期不符就停下报告,不强行解读
2. **不替 MA4 决定**：尤其 §6.3 setup.py 决策点,原样陈述给 MA4
3. **不动已交付资产**：Sub-Agent 4 的 5 个文件碰一下就是错
4. **不并行**：A → B → C → D 严格顺序,每阶段完成才进下一阶段
5. **不创新**：你不会"顺便"上传你觉得"也许有用"的文件。不在 A 阶段确认存在的,不传
6. **60% 上下文闸门**：本任务比 Sub-Agent 4 简单,buffer 留充足

---

## §10 关键禁令（重申）

| 禁令 | 原因 |
|---|---|
| ❌ 不修改任何已上传文件 | 不是你的工作 |
| ❌ 不动 Sub-Agent 4 的 5 个交付资产 | Check Agent + Sub-Agent 4 已验证,你碰一下就是污染 |
| ❌ 不上传 .git / __pycache__ / *.pyc | 无用产物,污染服务器 |
| ❌ 不装 pip 包(除非 §6.3 选了方案 Y 且 MA4 同意) | 可能引入版本冲突,影响已稳定的 mlff env |
| ❌ 不读 dataset_v2 / forward_test.py 内部代码 | 与盘点任务无关 |
| ❌ 不跑 forward_test.py | 那是 Sub-Agent 4-续的事 |
| ❌ 不接触 data/ 目录 | 数据已就绪,Sub-Agent 4 已确认 |
| ❌ 不接触 incompat_pool.csv / holdout | 同 Sub-Agent 4 §11.2 / §11.7 |

---

## §11 接下来发生什么

1. 你（Preparation Agent）按 §4-§7 走 4 阶段
2. 每阶段让用户跑命令,粘贴输出
3. 你按 §8 模板填报告
4. 用户把报告转给 MA4
5. MA4 决定：
   - **D3 import 链全 PASS** → MA4 写 Sub-Agent 4-续 极简 handoff,启动 forward_test.py
   - **任一 FAIL** → MA4 解读后给选项

**你的工作到此结束**,不要主动跳到 Sub-Agent 4-续的工作。

---

*MA4 撰写,2026-04-26,本文档为一次性盘点任务,不接力*
