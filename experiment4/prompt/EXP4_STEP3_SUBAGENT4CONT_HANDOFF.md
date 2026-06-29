# EXP4_STEP3_SUBAGENT4CONT_HANDOFF.md
# Sub-Agent 4-续 极简交接文档

> **撰写者**：DiffCSP-Exp4-Main-Agent 4
> **接收者**：DiffCSP-Exp4-Step3-SubAgent-4-Continued
> **日期**：2026-04-26
> **接力关系**：Sub-Agent 4 因 70% 上下文闸门停 + diffcsp 缺失阻断 Phase 6.4;Preparation Agent 已补传 diffcsp/ + conf/,异常已盘清。本窗口接 Phase 6.4 + 6.5 + 完成汇报。
> **MA4 已锁定决策**：方案 **T**(.env 文件)+ chdir 副作用选 **a**(接受 chdir,代码用绝对路径)

---

## §0 你做什么

**核心一句话**:跑一次 `forward_test.py`,看 Phase 6.4 / 6.5 是否 PASS,按 §6 模板汇报。

**前置已完成(由 MA4 + Preparation Agent + Sub-Agent 4 接力做完)**:

- ✅ diffcsp/ 框架包已传到 `/home/tcat/diffcsp_exp4/code/diffcsp/`
- ✅ conf/ 已传到 `/home/tcat/diffcsp_exp4/code/conf/`
- ✅ Phase 5 改动 1 (`diffusion_w_type_xas.py:108` 73→74) 已 PASS
- ✅ Phase 5 改动 2 (`conf_xas/model/diffusion_xas.yaml:18` 73→74) 已 PASS
- ✅ Phase 5b `xas_local_datamodule_v2.py` 已 PASS(Sub-Agent 4 交付)
- ✅ Phase 6.1 / 6.2 / 6.3 已 PASS(Sub-Agent 4 跑过,日志在 `logs/step3_forward_test_log.txt`)

**只剩 Phase 6.4 (CPU forward+backward) + Phase 6.5 (GPU bf16) 没跑过**——上次 6.4 import 阶段炸 ModuleNotFoundError,没真正进 forward。现在 diffcsp 包到位 + .env 配好,**这次有真信号**。

**单窗口预算**:MA4 估算 30-50K token(顺利)/ 80-120K token(6.4 或 6.5 需 debug)。**60% 上下文闸门**触发即停汇报。

---

## §1 你必须读的文档(精简到 3 份)

| # | 文档 | 必读? | 重点 |
|---|------|-------|------|
| 1 | **本文档**(你在读) | ✅ | 全文 |
| 2 | EXP4_STEP3_SUBAGENT4_HANDOFF.md(原 Sub-Agent 4 完整 handoff) | ✅ | §8.1 修订 4 条(尤其 model 类名 grep)、§8.2 五子 phase 期望表、§9 停汇报触发条件 |
| 3 | Sub-Agent 4 中途停汇报全文(Sub-Agent 4 留下的) | ✅ | "已完成"列表(知道哪些不用重做)、"我没做的事"(继承约束) |

**你不要读**(节省 token):
- ❌ EXP4_PROGRESS_LOG.md / EXPERIMENT2_FINAL_REPORT.md / EXP4_PROPOSAL_v2.md
- ❌ EXP4_MAINAGENT4_HANDOFF.md(本文档已透传 §6 锁定决策的精华)
- ❌ Check Agent 报告 / Preparation Agent 报告(已被本文档浓缩)

---

## §2 启动序列(精确到命令,你按顺序跑,每条粘贴输出)

### §2.1 Step 1:创建 .env 文件(MA4 决策方案 T)

```bash
ssh tcat@scsmlnprd02.its.auckland.ac.nz
conda activate mlff
which python   # 期望 /home/tcat/conda_envs/mlff/bin/python

cd /home/tcat/diffcsp_exp4/code
cat > .env <<'EOF'
export PROJECT_ROOT=/home/tcat/diffcsp_exp4/code
export HYDRA_JOBS=/home/tcat/diffcsp_exp4/logs/hydra
export WABDB_DIR=/home/tcat/diffcsp_exp4/logs/wandb
EOF

# 创建对应的目录(防 hydra/wandb 找不到目录报错)
mkdir -p /home/tcat/diffcsp_exp4/logs/hydra
mkdir -p /home/tcat/diffcsp_exp4/logs/wandb

# 验证 .env 写入成功
cat /home/tcat/diffcsp_exp4/code/.env
ls -la /home/tcat/diffcsp_exp4/logs/
```

**期望输出**:
- `cat .env` 显示 3 行 export
- `ls logs/` 看到 hydra/ 和 wandb/ 两个新目录

**注意**:`WABDB_DIR` 故意写成上游 typo(不是 WANDB),与 `.env.template` 一致。Preparation Agent 报告事实 2 已确认。

### §2.2 Step 2:验证 dotenv 加载链可达

```bash
cd /home/tcat/diffcsp_exp4/code
python -c "
import sys
sys.path.insert(0, '/home/tcat/diffcsp_exp4/code')
print('--- before any diffcsp import ---')
import os
print('cwd:', os.getcwd())
print('PROJECT_ROOT in env:', os.environ.get('PROJECT_ROOT', '<NOT SET>'))

print('--- importing diffcsp.common.utils ---')
from diffcsp.common import utils
print('cwd after import:', os.getcwd())
print('PROJECT_ROOT after import:', os.environ.get('PROJECT_ROOT', '<NOT SET>'))

print('--- importing pl_modules ---')
from diffcsp.pl_modules import cspnet
from diffcsp.pl_modules import diff_utils
print('all transitive imports OK')
"
```

**期望输出**:
- `cwd` 在 import 之后切到 `/home/tcat/diffcsp_exp4/code`(这是 chdir 副作用,**MA4 决策选 a 接受**)
- `PROJECT_ROOT after import` 是 `/home/tcat/diffcsp_exp4/code`
- 全部 import 通过,没有 traceback
- 出现 "all transitive imports OK"

**任一异常立刻停汇报 MA4**(按 §5 模板):
- 仍然 ModuleNotFoundError → diffcsp 上传不完整,Preparation Agent 漏了文件
- `assert PROJECT_ROOT.exists()` AssertionError → .env 没加载或路径错
- 其他 traceback → 框架结构问题

### §2.3 Step 3:跑 forward_test.py

**⚠️ 注意**:Sub-Agent 4 留下的 forward_test.py 在 `/home/tcat/diffcsp_exp4/code/step3/forward_test.py`。它**可能没有 sys.path.insert**(因为之前 import 在阶段就炸了)。先 grep 确认:

```bash
head -20 /home/tcat/diffcsp_exp4/code/step3/forward_test.py
grep -n "sys.path" /home/tcat/diffcsp_exp4/code/step3/forward_test.py
```

**判定**:
- 如果 grep 命中 `sys.path.insert(0, ...)` 指向 code/ 根目录 → 直接进 §2.4
- 如果未命中 → 不要改 forward_test.py(保持 Sub-Agent 4 交付物完整),改用 §2.4 的 PYTHONPATH 启动方式
- 如果命中但路径错(如指向 Windows 路径) → 停汇报 MA4

### §2.4 Step 4:跑 forward_test 主命令

```bash
cd /home/tcat/diffcsp_exp4/code/step3
PYTHONPATH=/home/tcat/diffcsp_exp4/code python forward_test.py 2>&1 | tee /home/tcat/diffcsp_exp4/logs/step3_forward_test_console.log
```

**说明 PYTHONPATH 写法**:
- 即使 §2.3 grep 显示 forward_test.py 已有 sys.path.insert,加 PYTHONPATH 是**双保险**(无副作用)
- `tee` 会同时写到屏幕 + 文件,你和 MA4 都能看完整 log
- 不要加 `&`(不要后台跑,你需要看到实时输出)

### §2.5 期望结果(直接对照 §3 期望表)

跑完后,看输出最后 30 行:

| 看到 | 解读 |
|---|---|
| 6.1 / 6.2 / 6.3 重新 PASS(因为是从头跑,1-3 子 phase 也会再过一遍) | ✓(已经验证过,这次复跑是健康的) |
| 6.4 PASS + 6.5 PASS | **Step 4 启动闸门 CLEAR**,按 §6 完成汇报模板写给 MA4 |
| 任一 6.4/6.5 FAIL(traceback / NaN / loss 越界) | 按 §4 决策树处理 |
| 6.1/6.2/6.3 之前 PASS 现在 FAIL | **意外**(可能 .env 引入副作用,或路径解析变了) → 按 §4 处理 |

---

## §3 五子 phase 期望(从原 SUBAGENT4_HANDOFF §8.2 摘抄,不变)

| 子 Phase | PASS 条件 |
|---|---|
| 6.1 | 100 个 random sample 全部 frac ∈ [-0.5, 0.5];零 frac sentinel 触发;12 字段集合严格等于预期 |
| 6.2 | bs=4 collate 不报错;tensor 字段 batch dim 正确;str 字段是 list of 4 |
| 6.3 | SpectrumEncoder forward 输出 (4, 256);no NaN;mean ∈ [-5, 5];std ∈ [0.01, 5](Sub-Agent 4 已松到 0.01) |
| **6.4** | CPU model.training_step(batch) loss ∈ [2, 6];backward 后 grad_norm ∈ (0, 1e4);no NaN grad |
| **6.5** | GPU bf16 loss 与 CPU 同范围 ±10%;grad_norm 正常;no NaN grad |

**全 PASS 才进 Step 4**。

---

## §4 6.4 / 6.5 失败决策树(预判已知风险类型)

预判来自 Sub-Agent 4 中途停汇报的"FAIL 路径"分析(它已经想过了,你直接照办):

| 错误类型 | 你的处置 |
|---|---|
| `ModuleNotFoundError`(diffcsp 子模块缺) | **不自行 debug**,直接停汇报。Preparation Agent 漏了文件,需要重新盘点(让 MA4 决定补传哪些) |
| `AssertionError: PROJECT_ROOT does not exist` | .env 加载链断 — 检查 §2.1 cat .env 输出。仍异常 → 停汇报 |
| `omegaconf` / `hydra` 配置解析错(`${oc.env:...}` 之类) | 停汇报。不要改 yaml(MA4 锁了 line 18 改动,其他不动) |
| **6.4 loss = NaN/Inf** | 停汇报。**不要重试**(可能是 yaml `sigma_begin: 0.005` 太小、可能 feat_dim 改不全有 yaml 漏改、可能 dataset_v2 输出有 NaN feff) |
| **6.4 loss 越出 [2, 6] 但有限** | 看具体值: 范围 [1, 8] 内 → 报告但不停(可能 random init 漂移,不是 bug);超出 [1, 8] → 停汇报 |
| **6.4 grad_norm = 0**(梯度全零) | 停汇报。这意味着某个分支断梯度,可能 cost_lattice 改非 0 或 detach() 误用 |
| **6.5 GPU bf16 NaN grad** | 停汇报(MA4 决策点,如 fp16 mixed vs fp32 训练 vs warmup) |
| **6.5 GPU bf16 loss 与 CPU 漂移 > 20%** | 停汇报。bf16 数值稳定性问题,需 MA4 决策 |
| **6.5 GPU OOM** | 停汇报。bs=4 超 24 GB 不应该,可能 Phase 5b datamodule_v2 出错或 Phase 6.5 测试代码本身有 bug |

**你的工作不是 debug,是 PASS 就 PASS、FAIL 就汇报**。Sub-Agent 4 工作哲学第 1 条:诚实 > 流畅。

---

## §5 中途停汇报模板(任一 §4 失败用这个)

```
# Sub-Agent 4-续 中途停汇报

## §2.1 .env 创建
cat .env 输出: ___
mkdir logs/hydra logs/wandb 结果: ___

## §2.2 dotenv 加载链测试
cwd before: ___
cwd after import: ___
PROJECT_ROOT after: ___
all transitive imports: [PASS / FAIL]
若 FAIL,traceback: ___

## §2.3 forward_test.py sys.path 检查
sys.path 命中: [是 / 否]

## §2.4 forward_test.py 跑结果(粘 console log 末尾 30-50 行)

## 触发停因
[精确描述哪一步,期望什么,实际什么]

## 我观察的事实(不解读)
[traceback / loss 数值 / grad_norm / 任何客观信息]

## 我没做的事
[防御性陈述:没改 dataset_v2、没改 datamodule_v2、没改 diffusion 主体、
 没动 yaml 内容、没装新包、没接触 holdout]

## 给 MA4 的选项(不替决定,见 §4 决策树)
A. ___(代价 ___)
B. ___(代价 ___)
C. ___(代价 ___)

## 上下文消耗
约 __ %
```

---

## §6 全 PASS 完成汇报模板

```
# Sub-Agent 4-续 完成汇报(Phase 6.4 + 6.5 全 PASS,Step 4 启动闸门 CLEAR)

## §2.1 .env 创建
cat .env: ___
mkdir hydra/wandb 目录: SUCCESS

## §2.2 dotenv 加载链
cwd before / after: ___ / /home/tcat/diffcsp_exp4/code
PROJECT_ROOT: /home/tcat/diffcsp_exp4/code ✓
chdir 副作用: 已确认接受(MA4 决策 a)

## §2.4 forward_test.py 跑结果
- 6.1 100 random samples: PASS / 用时 ___ s
- 6.2 DataLoader collate (bs=4): PASS
- 6.3 SpectrumEncoder forward: shape (4, 256), mean ___, std ___, PASS
- 6.4 CPU forward+backward: loss ___, grad_norm ___, PASS
- 6.5 GPU bf16 forward+backward: loss ___, grad_norm ___, PASS
- 总耗时: ___ s

## Step 4 启动闸门
- 五子全 PASS: ✓
- 零 NaN/Inf: ✓
- 预期范围内: ✓
- diffcsp / conf / .env 全到位: ✓

## 我没做的事
- 没改 dataset_v2 / datamodule_v2 / diffusion / yaml / spectrum_encoder
- 没接触 incompat_pool / holdout
- 没装新 pip 包
- 没改不可变量

## 给 MA4 的开放问题
[有就列,没有就写"无"]

## 资产清单(供 MA4 写 Step 4 handoff 时引用)
/home/tcat/diffcsp_exp4/code/
├── .env                                   ← 本窗口新建
├── diffcsp/                              ← Preparation Agent 上传
├── conf/                                  ← Preparation Agent 上传(如有)
├── step2/spectrum_encoder.py             ← Sub-Agent 3 改完
├── step3/diffusion_w_type_xas.py         ← Sub-Agent 4 改完(.bak 备份)
├── step3/conf_xas/model/diffusion_xas.yaml ← Sub-Agent 4 改完(.bak 备份)
├── step3/xas_local_dataset_v2.py         ← Sub-Agent 3 交付
├── step3/xas_local_datamodule_v2.py      ← Sub-Agent 4 交付
└── step3/forward_test.py                 ← Sub-Agent 4 交付

/home/tcat/diffcsp_exp4/data/             ← 数据全在(Sub-Agent 1/2/3 上传)
/home/tcat/diffcsp_exp4/logs/
├── hydra/                                ← 本窗口新建(Step 4 用)
├── wandb/                                ← 本窗口新建(Step 4 用)
├── step3_forward_test_log.txt            ← Sub-Agent 4 之前写的中途产物
└── step3_forward_test_console.log        ← 本窗口产出

## 上下文消耗
约 __ %
```

---

## §7 工作哲学(继承 Sub-Agent 1/2/3/4 + Check + Preparation)

1. **诚实 > 流畅**:不确定就停下汇报,给 MA4 选项
2. **不创新**:你的工作就是 §2.1 → §2.2 → §2.3 → §2.4 这 4 步,跑通就 §6 汇报,跑不通就 §5 汇报
3. **不动 Sub-Agent 4 资产**:5 个文件碰一下就是污染
4. **60% 上下文闸门**:任务简单 buffer 留充足
5. **不主动跳到 Step 4**:即使 6.4/6.5 全 PASS,Step 4 训练交接由 MA4 写

---

## §8 关键禁令(重申)

| 禁令 | 原因 |
|---|---|
| ❌ 不修改 forward_test.py / datamodule_v2 / dataset_v2 / diffusion / yaml | 已被 Check / Sub-Agent 4 验证,改就是污染 |
| ❌ 不改 .env 内容(三行 export 写完就不改) | MA4 决策方案 T 锁定 |
| ❌ 不删 .bak 备份文件 | Sub-Agent 4 留的回滚锚点 |
| ❌ 不改 utils.py / chdir 行为 | MA4 决策选 a 接受副作用 |
| ❌ 不装新 pip 包 | mlff env 已稳,不污染 |
| ❌ 不接触 holdout / incompat_pool | 同所有前任 |
| ❌ 不在 6.4/6.5 出错时尝试 debug 超过 1 轮 | 立刻停汇报 |

---

*MA4 撰写,2026-04-26,等待用户 review 后投入新窗口*
