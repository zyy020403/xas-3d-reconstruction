# EXP4_STEP5_STEP5AGENT_HANDOFF.md
# DiffCSP-Experiment4 Step 5 评估 Step5Agent 交接文档

> **撰写者**: DiffCSP-Exp4-Main-Agent 5
> **接收者**: Step 5 评估 Sub-Agent(用户命名 = Step5Agent)
> **日期**: 2026-04-27
> **背景**: Step4Agent 已交付 `best-epoch366-val0.7300.ckpt`,训练正常早停退出。Step 5 启动闸门 CLEAR。
> **核心约束**: 第一棒**只跑 val + test**,holdout 段在脚本里就位但 disabled,等 MA5 phase 5b 解锁。
> **参考前棒**: Step4Agent 命名是用户的"按 step 命名"方式;接力链整体编号是 Sub-Agent 5(Step 4 阶段)→ Step5Agent(本阶段)。两套命名同一回事。

---

## §0 你是谁,你的工作边界

你是 DiffCSP-Exp4 接力链的 **Step5Agent**(前棒 = Step4Agent,后棒 = Step6Agent)。你的工作是评估 Step4Agent 训练出的 `best-epoch366-val0.7300.ckpt`,产出指标 + 数据表 + 简报,**不写 final report**(那归 Step6Agent)。

**你做什么**:
1. Phase 5.0 hard check(env / 文件 / ckpt 加载 sanity)
2. Phase 5.1 写 `step5_1_sample.py` —— 在 ckpt 上跑 reverse diffusion 采样,产出每个 split 的 `predictions_{split}.pt`
3. Phase 5.2 写 `step5_2_compute_metrics.py` —— 计算 RMSD / Type Accuracy / pred_in_cutoff / true_in_cutoff,**包括按 eval_cutoff 分层**
4. Phase 5.3 跑 val + test 两段,产出 `metrics_report_val_test.txt` + 数据表 CSV
5. Phase 5.4 写中期报告交回 MA5(包含 val/test 数字 + 与 Exp2 可比性判断)
6. **不跑 holdout** —— MA5 看完 5.4 报告决议后,在 phase 5b 单独指令再跑

**你不做什么**:
- 不动 dataset_v2 / datamodule_v2(Step4Agent Phase 4.6 修复版,继承使用)
- 不动 spectrum_encoder / diffusion / yaml / forward_test.py / .bak3
- 不动训练脚本 / smoke_test 脚本 / ckpt 文件
- 不写 final report(Step6Agent 任务)
- 不画图(Step6Agent 任务)
- 不做 multi-sample 平均(MA5 决策: 单 sample/sample,与 Exp2 可比)
- **不读 holdout 数据**(直到 MA5 phase 5b 显式解禁)
- 不深 debug(任何 phase FAIL ≤ 1 轮观察 + 候选解释,然后停)
- 不替 MA5 做"是不是该 fine-tune"等决策

**上下文闸门**: **70%**(同 Step4Agent)。到 70% 必须停,把"未做完事项 + 当前状态"交回 MA5。

---

## §1 必读文档清单

按读取顺序,**1-3 精读,4-7 速读**:

| # | 文档 | 必读? | 重点 |
|---|------|-------|------|
| 1 | **本文档** | ✅ 精 | 全文,尤其 §3-§7 |
| 2 | **EXP4_STEP4_SUBAGENT5_INTERIM_REPORT.md**(Step4Agent 中期 + 末段) | ✅ 精 | Phase 4.6 修复细节 + 训练曲线 + 末段 5 个开放问题 |
| 3 | **EXPERIMENT2_FINAL_REPORT.md** | ✅ 精 | §1 指标定义(RMSD / Type Acc / pred_in_cutoff)+ §2.4 holdout 数字 + §2.5 随机基线 |
| 4 | EXP4_STEP4_SUBAGENT5_HANDOFF.md(Step4Agent 自身 handoff) | 速 | §4 文件归属表(继承使用) |
| 5 | EXP4_PROPOSAL_v2.md | 速 | §1.3 不变量 + §6 预期指标 |
| 6 | EXP4_FILE_INVENTORY.md | 速 | shell_boundaries.pkl schema(Step 5 分层评估关键) |
| 7 | EXP4_MAINAGENT5_HANDOFF.md | 速 | 整体接力链状态 |

**Exp2 step5 / step4d 评估脚本参考**(找到后精读,作模板):
- `experiment2/step5/step5_1_sample.py` —— sample 脚本模板
- `experiment2/step5/step5_2_compute_metrics.py` —— metrics 脚本模板
- `experiment2/step4d/step4d_3_sample.py` / `step4d_4_compute_metrics.py` —— val/test 评估模板

如果服务器上 Exp2 fork 找不到这些文件,**停下来问用户路径**,不要凭空写。

---

## §2 当前项目状态(继承,不重新讨论)

**Step 0/1/2/2.5/3/4 全部完成**:
- Step 4 训练: 396 epoch 早停,best ckpt 在 epoch 366,val_loss=0.72998
- 75,637 v2 样本(60507/7624/4481/3025 train/val/test/holdout)
- Step 4 Phase 4.6 修复: dataset_v2 两处 raise → return None,datamodule_v2 加 None-filter collate(Exp2 silent-drop 行为对齐)
- 因此 evaluation 时**有效预测样本 ≤ 名义样本**,silent drop ~9 个/100k 样本(Step4Agent 报告 §8 O1)
- Step4Agent 中期报告 §8 O1 已确认 silent drop 数 < 0.05%,Step 5 不阻塞

**Step 5 关键不变量**:
- `cost_lattice = 0`(MA4 锁定)→ lattice 不参与 loss / metrics(只有 coord + type)
- L = 6 / N_NEIGHBORS = 20 / coord 系 [-0.5, 0.5] / 88 元素
- 评估 ckpt: **`best-epoch366-val0.7300.ckpt`**(MA5 决策,不用 last)
- 单 sample/sample(MA5 决策,不做 multi-sample 平均)
- holdout 第一棒**禁用**(MA5 决策)

**Step 4 修复后的 dataset_v2 行为(继承,不动)**:
- `__getitem__` 内 < 20 邻居 / frac 越界 → `return None`(silent)
- collate_fn 内 filter None,极端整批 None → 返回 None,DataLoader 自动跳过
- 你写 sample 脚本时**必须用同一个 datamodule_v2**,不能绕开

---

## §3 Step 5 流水线总览

```
best ckpt
   │
   ├──→ Phase 5.1: step5_1_sample.py
   │       ├── load datamodule_v2 (各 split DataLoader)
   │       ├── load CSPDiffusion from ckpt
   │       ├── for each batch: model.sample(spectrum, num_steps=1000)
   │       │     → 输出 pred_frac_coords, pred_atom_types
   │       └── 累积保存 → predictions_{split}.pt
   │
   └──→ Phase 5.2: step5_2_compute_metrics.py
           ├── load predictions_{split}.pt
           ├── load ground truth from datamodule(同 split)
           ├── load shell_boundaries.pkl(Step 2.5 分层信息)
           ├── for each sample:
           │     ├── 匈牙利匹配(min-image 距离)
           │     ├── 计算 RMSD / Type Acc / pred_in_cutoff / true_in_cutoff
           │     └── 按 eval_cutoff 分层(Exp2 baseline 是 4Å cap,Exp4 一致)
           └── aggregate → metrics_report_{split}.txt + per_sample_metrics_{split}.csv
```

---

## §4 文件归属总表(继承 Step 4,本阶段新增)

### 4.1 继承使用(本阶段不动)

| 文件 | 用法 |
|------|------|
| `step3/xas_local_dataset_v2.py` | Step4Agent Phase 4.6 修复版,silent drop |
| `step3/xas_local_datamodule_v2.py` | Step4Agent Phase 4.6 加 None-filter collate |
| `step3/diffusion_w_type_xas.py` | `from diffusion_w_type_xas import CSPDiffusion`(类名以实际为准,grep 验) |
| `step3/conf_xas/model/diffusion_xas.yaml` | hydra instantiate config(sample 脚本可能 reuse) |
| `step2/spectrum_encoder.py` | 间接被 diffusion 依赖,你不直接 import |
| `checkpoints/best-epoch366-val0.7300.ckpt` | **本阶段唯一 ckpt 输入** |
| `data/data_inventory_v2.csv` / `*_samples_v2.csv` | dataset_v2 内部 load |
| `data/spectra_val.pkl` / `spectra_test.pkl` | val + test 评估用 |
| `data/spectra_holdout.pkl` | **本阶段不读**(MA5 phase 5b 才解禁) |
| `data/shell_boundaries.pkl` | 分层评估关键(369.5 MB,含全 128k 样本) |

### 4.2 你新建(放在 `/home/tcat/diffcsp_exp4/code/step5/`)

| 文件 | 阶段 | 说明 |
|------|------|------|
| `step5_1_sample.py` | Phase 5.1 | sample 脚本,支持 `--split val/test/holdout`,默认遍历 `[val, test]` |
| `step5_2_compute_metrics.py` | Phase 5.2 | metrics 脚本,产出 .txt 报告 + per-sample .csv |
| `step5_README.md` | Phase 5.3 | 启动命令、log 位置、数字一览 |
| `predictions_val.pt` | Phase 5.3 | sample 产出 |
| `predictions_test.pt` | Phase 5.3 | sample 产出 |
| `metrics_report_val.txt` | Phase 5.3 | val 主报告(给人看) |
| `metrics_report_test.txt` | Phase 5.3 | test 主报告 |
| `per_sample_metrics_val.csv` | Phase 5.3 | val 逐样本数据表(Step6Agent 画图用) |
| `per_sample_metrics_test.csv` | Phase 5.3 | test 逐样本数据表 |

### 4.3 holdout 阶段产物(MA5 phase 5b 后才生)

| 文件 | 备注 |
|------|------|
| `predictions_holdout.pt` | phase 5b 才生 |
| `metrics_report_holdout.txt` | phase 5b 才生 |
| `per_sample_metrics_holdout.csv` | phase 5b 才生 |

---

## §5 Phase 子任务清单

### Phase 5.0:Hard check

```bash
# 5.0.1 disk + 文件存在
df -h ~                  # 期望 ≥ 30 GB(Step 4 训练完后应仍宽裕)
[ -f /home/tcat/diffcsp_exp4/checkpoints/best-epoch366-val0.7300.ckpt ] && echo "OK ckpt" || echo "MISS"
[ -f /home/tcat/diffcsp_exp4/data/shell_boundaries.pkl ] && echo "OK shell" || echo "MISS"
[ -f /home/tcat/diffcsp_exp4/data/spectra_val.pkl ] && echo "OK val" || echo "MISS"
[ -f /home/tcat/diffcsp_exp4/data/spectra_test.pkl ] && echo "OK test" || echo "MISS"
# spectra_holdout.pkl 也存在但本阶段不读

# 5.0.2 env
which python && python --version
python -c "import torch, pytorch_lightning as pl, scipy; print(torch.__version__, pl.__version__, scipy.__version__)"

# 5.0.3 ckpt 加载 sanity(关键,避免 Phase 5.1 跑一半才发现 hyperparam mismatch)
PYTHONPATH=/home/tcat/diffcsp_exp4/code:/home/tcat/diffcsp_exp4/code/step3:/home/tcat/diffcsp_exp4/code/step2 \
python -c "
import sys
sys.path.insert(0, '/home/tcat/diffcsp_exp4/code/step3')
sys.path.insert(0, '/home/tcat/diffcsp_exp4/code/step2')
import torch
ckpt = torch.load('/home/tcat/diffcsp_exp4/checkpoints/best-epoch366-val0.7300.ckpt', 
                  map_location='cpu', weights_only=False)
print('ckpt keys:', list(ckpt.keys())[:5])
print('hyper_parameters keys (sample):', list(ckpt.get('hyper_parameters', {}).keys())[:10])
print('state_dict size:', len(ckpt['state_dict']))
print('global_step:', ckpt.get('global_step'))
print('epoch:', ckpt.get('epoch'))
"
# 期望: epoch ≈ 366, state_dict size > 0, hyper_parameters 含 cost_coord / cost_type / cost_lattice 等
```

**任一异常处理**: disk / 文件 MISS / ckpt 加载报错 → 立刻停,贴输出给 MA5。

### Phase 5.1:`step5_1_sample.py`

#### 5.1.1 模板参考

服务器 Exp2 fork 应有 `step5_1_sample.py` 或 `step4d_3_sample.py`。**优先用 step5_1**(Exp2 step5 是 holdout 评估,与本任务最近)。`find / -name "step5_1_sample.py" 2>/dev/null` 可定位。

#### 5.1.2 必改清单

| Exp2 step5_1_sample.py | Exp4 step5_1_sample.py |
|---|---|
| `from xas_local_datamodule import XASDataModule` | `from xas_local_datamodule_v2 import XasLocalDataModuleV2` |
| `XASDataset` 直接 import | 不直接 import,由 datamodule 内部处理 |
| `from diffusion_w_type_xas import ...`(EXP2) | 同名 import(Exp4 已改完 line 108) |
| 路径常量(Win) | Linux 路径 `/home/tcat/diffcsp_exp4/...` |
| ckpt path 硬编码 | 用 argparse 或常量 `CKPT = "/home/tcat/diffcsp_exp4/checkpoints/best-epoch366-val0.7300.ckpt"` |
| `precision='bf16'` 推理 | **`fp32` 推理**(与训练一致,MA4 决策) |
| `gpus=1` (PL 1.x) | `accelerator="gpu", devices=1`(PL 2.5.5) |
| 单 split 硬写 | argparse `--split` 参数,默认值 `["val", "test"]`,**holdout 不在默认列表** |
| `num_samples=1` | **保持 1**(MA5 决策,与 Exp2 可比) |
| `num_steps=1000` | 看 Exp2 实际值,通常是 1000 反扩散步数 |
| 输出 .pt 文件名 | `predictions_{split}.pt` |

#### 5.1.3 Sample 脚本核心逻辑(伪码)

```python
import argparse, torch, pickle
from pathlib import Path
from torch.utils.data import DataLoader
from xas_local_datamodule_v2 import XasLocalDataModuleV2
from diffusion_w_type_xas import CSPDiffusion  # 类名以实际为准

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/home/tcat/diffcsp_exp4/checkpoints/best-epoch366-val0.7300.ckpt")
    ap.add_argument("--splits", nargs="+", default=["val", "test"])  # holdout 默认排除
    ap.add_argument("--out_dir", default="/home/tcat/diffcsp_exp4/code/step5")
    ap.add_argument("--num_steps", type=int, default=1000)  # 反扩散步数,与 Exp2 一致
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()
    
    # holdout 安全闸门
    if "holdout" in args.splits:
        raise RuntimeError(
            "holdout 评估需要 MA5 phase 5b 显式批准。"
            "Step5Agent 第一棒不跑 holdout。"
        )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # load model
    model = CSPDiffusion.load_from_checkpoint(args.ckpt, map_location=device)
    model.eval()
    model.to(device)
    
    # for each split
    dm = XasLocalDataModuleV2(...)  # 参数与训练时一致
    dm.setup("test")  # 或 "fit",看 datamodule_v2 实际签名
    
    for split in args.splits:
        loader = dm.{split}_dataloader()  # train_dataloader / val_dataloader / test_dataloader
        
        all_preds = []  # list of dicts
        with torch.no_grad():
            for batch in loader:
                if batch is None:
                    continue  # collate 整批 None,DataLoader 已跳过,但保险一下
                batch = batch.to(device)
                
                # 反扩散采样(关键 API 名称 grep diffusion_w_type_xas.py 确认)
                pred = model.sample(batch, num_steps=args.num_steps)
                # pred 应包含: frac_coords, atom_types, batch index
                
                all_preds.append({
                    "sample_names": batch.mp_id,  # 或 sample_name,看 dataset_v2 输出字段
                    "pred_frac_coords": pred["frac_coords"].cpu(),
                    "pred_atom_types": pred["atom_types"].cpu(),
                    "true_frac_coords": batch.frac_coords.cpu(),
                    "true_atom_types": batch.atom_types.cpu(),
                    "lengths": batch.lengths.cpu(),
                    "eval_cutoff": batch.eval_cutoff.cpu(),
                })
        
        out_path = Path(args.out_dir) / f"predictions_{split}.pt"
        torch.save(all_preds, out_path)
        print(f"[{split}] saved {len(all_preds)} batches → {out_path}")

if __name__ == "__main__":
    main()
```

**关键确认事项**(写代码前 grep `diffusion_w_type_xas.py` 找答案):
1. `model.sample(...)` 的真实方法名(可能是 `sample` / `predict` / `denoise` / `forward_sample`)
2. sample 输出结构(dict / tuple / 自定义对象)
3. 反扩散步数默认值(通常 1000,可能 model 内部已设)
4. `batch` object 的字段名(frac_coords / atom_types / lengths / mp_id / sample_name 等),与 dataset_v2 `__getitem__` 输出对齐

如果以上任一不确定,**停下来问 MA5**,不要猜行为。

#### 5.1.4 启动命令

```bash
cd /home/tcat/diffcsp_exp4/code/step5

# 选闲 GPU
GPU_ID=0
LOG=/home/tcat/diffcsp_exp4/logs/step5_sample_val_test.log

PYTHONPATH=/home/tcat/diffcsp_exp4/code:/home/tcat/diffcsp_exp4/code/step3:/home/tcat/diffcsp_exp4/code/step2 \
CUDA_VISIBLE_DEVICES=$GPU_ID \
python step5_1_sample.py --splits val test 2>&1 | tee $LOG
```

**预期 wall time**:
- val(7624 sample)+ test(4481 sample)= 12,105 sample
- 反扩散 1000 步 × bs=16 × ~0.05 s/step ≈ 50 s/batch × 758 batch ≈ **~10-12 小时**(GPU)
- 这是 MA5 估的量级,你 sample 完后实际 wall time 写报告里给我做下次估算 reference

如果 sample 跑了 1 小时还没第一个 batch 结果,**停下来汇报**(可能反扩散步数设错或 sample API 调用有问题)。

### Phase 5.2:`step5_2_compute_metrics.py`

#### 5.2.1 必算指标

| 指标 | 定义 | Exp2 baseline |
|---|---|---|
| **RMSD**(Å) | 匈牙利匹配后的平均原子位置距离(笛卡尔,使用 lengths × frac) | val 1.47 / holdout 1.47 |
| **Type Accuracy** | 匈牙利匹配后的元素类型一致比例 | val 0.249 / holdout 0.241 |
| **pred_in_cutoff**(/20) | 预测原子中,距 Fe 原点 ≤ eval_cutoff 的数量 | val 17.47 / holdout 17.52 |
| **true_in_cutoff**(/20) | 真实原子中,距 Fe 原点 ≤ eval_cutoff 的数量 | holdout 18.99 |
| **N 有效样本** | dataset_v2 silent drop 后实际预测的样本数 | 不直接对比 Exp2 |
| **N 名义样本** | split CSV 行数(60507/7624/4481/3025 中的对应) | val 7624 / test 4481 |

**所有指标都要报"有效/名义"两套数**(MA5 提醒,因 Phase 4.6 silent drop)。

#### 5.2.2 分层评估(Exp4 新增,Exp2 没做)

利用 `shell_boundaries.pkl`(Step 2.5 产出),按 `eval_cutoff` 范围分层:

```python
import pickle
shell = pickle.load(open("/home/tcat/diffcsp_exp4/data/shell_boundaries.pkl", "rb"))
# shell[sample_name] = {threshold, distances, species_Z, shell_starts, shell_ends, 
#                       shell_n_atoms, shell_of_atom, eval_cutoff, n_center_sites}
```

**分组建议**(用每样本的 `eval_cutoff` 字段,Step 2.5 算出的"含 d20 的最小壳层外缘"):
- Tier A: eval_cutoff ≤ 3.0 Å(密集壳层,较易)
- Tier B: 3.0 < eval_cutoff ≤ 4.0 Å(中等)
- Tier C: 4.0 < eval_cutoff ≤ 5.0 Å(稀疏)
- Tier D: eval_cutoff > 5.0 Å(极稀疏)

每层报上述 6 项指标。**这是 Exp4 vs Exp2 的关键 differentiator**,Exp2 没做分层,所以这部分数字无法对比,但可作为 Exp4 自身的诊断。

#### 5.2.3 关键算法注意

**1. 匈牙利匹配必须用 min-image 距离**(Exp2 final report 附录已写明)

```python
def min_image_dist(pred_frac, true_frac, lengths):
    # frac diff 折叠到 [-0.5, 0.5]
    diff = pred_frac - true_frac
    diff = diff - np.round(diff)
    # 转笛卡尔(此项目 lengths = [L, L, L] 即 [6, 6, 6])
    cart_diff = diff * lengths
    return np.linalg.norm(cart_diff, axis=-1)
```

然后用 `scipy.optimize.linear_sum_assignment` 求最优配对。

**2. eval_cutoff 是 per-sample 的**(不是固定 4.0 Å)

每个样本的 eval_cutoff 在 dataset_v2 输出 / batch 字段里有,数据类型 `torch.float32`。**不要用全局常量 4.0**,Exp2 是因为 Fe-only 数据稀疏度有限才用 cap=4.0,Exp4 88 元素分布更广,要用每样本实际值。

**3. true_in_cutoff 是 ground truth 的统计量,可以预算**

虽然 dataset_v2 silent drop 之后才进 sample,但 true_in_cutoff 计算只需要 ground truth + eval_cutoff,在 metrics 阶段算就行。

#### 5.2.4 启动命令

```bash
cd /home/tcat/diffcsp_exp4/code/step5

PYTHONPATH=/home/tcat/diffcsp_exp4/code:/home/tcat/diffcsp_exp4/code/step3:/home/tcat/diffcsp_exp4/code/step2 \
python step5_2_compute_metrics.py --split val 2>&1 | tee /home/tcat/diffcsp_exp4/logs/step5_metrics_val.log

PYTHONPATH=/home/tcat/diffcsp_exp4/code:/home/tcat/diffcsp_exp4/code/step3:/home/tcat/diffcsp_exp4/code/step2 \
python step5_2_compute_metrics.py --split test 2>&1 | tee /home/tcat/diffcsp_exp4/logs/step5_metrics_test.log
```

**预期 wall time**: 5-15 分钟/split(纯 CPU 算法,匈牙利匹配 N=20 极快)。

### Phase 5.3:跑 val + test

按 5.1.4 + 5.2.4 启动,完整流程预计 12-15 小时(主要 sample 阶段)。

**期间监控**(每 1-2 小时抽样查):
- `tail -f $LOG` 看进度(应有 batch 计数滚动)
- `nvidia-smi` 看 GPU 占用
- `df -h ~` 看磁盘(predictions_*.pt 文件大小约 N_sample × N_neighbors × 4 dim × 4 byte ~ 12k × 20 × 4 × 4 ≈ 4 MB,完全不大)

### Phase 5.4:中期报告 + 关窗口

写中期报告交回 MA5,结构:

```markdown
# Step5Agent Phase 5 Interim Report

## 5.0 Hard check
- 文件齐全 / ckpt 加载: PASS/FAIL
- ckpt epoch / global_step: <值>

## 5.1 Sample 脚本
- step5_1_sample.py 行数: <N>
- 与 Exp2 step5_1 关键差异 grep: <附在底部>
- API 关键确认: model.sample 实际名/输入输出/反扩散步数: <值>

## 5.3 Val + Test 评估结果

### Val(N 名义=7624,有效=<N>)
| 指标 | Exp4 (epoch 366 ckpt) | Exp2 baseline | Δ |
|---|---|---|---|
| RMSD (Å) | <X.XX> | 1.47 | <ΔX.XX> |
| Type Acc | <0.XXX> | 0.249 | <Δ0.XXX> |
| pred_in_cutoff | <XX.XX>/20 | 17.47/20 | <ΔX.XX> |
| true_in_cutoff | <XX.XX>/20 | — | — |

### Test(N 名义=4481,有效=<N>)
[同表]

### 分层(val,4 tiers)
| Tier | N | RMSD | Type Acc | pred_in_cutoff |
|---|---|---|---|---|
| A (≤3.0Å) | <n> | <x> | <x> | <x> |
| B (3-4Å) | <n> | <x> | <x> | <x> |
| C (4-5Å) | <n> | <x> | <x> | <x> |
| D (>5Å) | <n> | <x> | <x> | <x> |

## 5.4 与 Exp2 可比性判断(给 MA5 决策)

- 是否达到 Exp2 baseline(RMSD ≤ 1.6 Å,Type Acc ≥ 0.25)?
- 偏好/偏差: <你的客观观察,不替 MA5 决策>

## 给 MA5 / Step6Agent 的开放问题

- O1: <例如 Tier D 数字异常,需 ablation>
- O2: <例如 holdout 是否照常评 / 还是先 fine-tune>

## 上下文用量自估
- <%>

## 下一步
- 等 MA5 phase 5b 决议(holdout 评 or fine-tune)
```

写完关窗口,等 MA5 决议下一步。

---

## §6 红灯 / 绿灯

### 红灯(立刻停 + 汇报)

1. Phase 5.0 任一项 FAIL
2. ckpt 加载报 hyperparam mismatch / state_dict shape mismatch
3. Sample 跑 1 小时无第一个 batch 结果
4. RMSD > 3.0 Å(异常,可能 sample API 用错 / coord 系不对)
5. RMSD < 0.5 Å(异常,可能 metrics 算错 / 对了 ground truth 自己)
6. Type Acc > 0.6(异常,Exp2 是 0.249,Exp4 88 元素更难,应该差不多或略低)
7. pred_in_cutoff < 5(异常,可能模型没学到聚集行为)
8. 上下文 ≥ 70%

### 绿灯(可继续)

1. RMSD 在 1.2-2.0 Å(Exp2 baseline 1.47)
2. Type Acc 在 0.20-0.35(Exp2 0.249,88 元素难度大)
3. pred_in_cutoff 在 14-19(Exp2 17.47)
4. 有效样本数 ≥ 名义 99.9%(silent drop 应 ≤ 0.05%)
5. 分层 Tier A → D 单调劣化(密集层评估更准是物理常识)

**误判防御**:
- 如果绿灯但你直觉哪里不对,**写进报告**,不要替 MA5 决定 OK/NOT OK
- 如果红灯阈值附近(如 RMSD = 2.05 Å)**也写进报告**,让 MA5 判断

---

## §7 禁令清单

- ❌ 不动 dataset_v2 / datamodule_v2 / spectrum_encoder / diffusion / yaml / forward_test.py
- ❌ 不读 holdout(spectra_holdout.pkl / holdout_samples_v2.csv)直到 MA5 phase 5b 解禁
- ❌ 不读 incompat_pool.csv
- ❌ 不动 ckpt 文件(ckpt 是只读输入)
- ❌ 不画图 / 不写 final report(Step6Agent)
- ❌ 不做 multi-sample 平均
- ❌ 不调 num_steps(默认 1000,与 Exp2 一致;若 Exp2 实际不同,以 Exp2 为准)
- ❌ 不深 debug(任何 phase FAIL ≤ 1 轮)
- ❌ 不替 MA5 决定 fine-tune / 调架构 / 重训

---

## §8 PL 2.5.5 + ckpt 兼容已知坑

继承 Step4Agent 经验:

1. `LightningModule.load_from_checkpoint(ckpt_path, map_location=...)` PL 2.5.5 默认 `weights_only=False`,但 PyTorch 2.4 可能给 warning。如果报错,显式传 `strict=True`
2. ckpt 里的 `hyper_parameters` 字段 PL 2.5.5 应自动恢复,但若 model 实例化要求 hydra config,**可能需要手动重建**:
   ```python
   from omegaconf import OmegaConf
   cfg = OmegaConf.load("/home/tcat/diffcsp_exp4/code/step3/conf_xas/model/diffusion_xas.yaml")
   import hydra
   model = hydra.utils.instantiate(cfg.model)
   ckpt = torch.load(...)
   model.load_state_dict(ckpt["state_dict"])
   ```
3. `model.eval()` 必须显式调,否则 dropout / batch_norm 训练态下采样结果不稳定
4. `torch.no_grad()` context 必须包,否则会爆 GPU memory(反扩散 1000 步 × bs 16)

---

## §9 第一条回复建议格式

```
我已读完 MA5 给我的 Step5Agent handoff + 必读文档清单 §1 的全部 7 份(其中
Exp2 step5/step4d 评估脚本路径需用户确认)。

[简要复述: Step 4 训练完成,best ckpt epoch 366 val_loss=0.7300;
我的工作 = Phase 5.0 hard check → 写 sample 脚本 → 写 metrics 脚本 →
跑 val + test → 中期报告 + 关窗口。holdout 不做,等 MA5 phase 5b]

我注意到三个关键约束:
1. ckpt 加载必须 fp32 推理,与训练精度一致
2. silent drop 沿用 Step4Agent Phase 4.6 修复,所有指标报"有效/名义"两组
3. 分层评估按 eval_cutoff 4 tier,这是 Exp4 vs Exp2 的诊断 differentiator

开始执行前需要确认 2 件事:
1. Exp2 fork 中 step5_1_sample.py / step5_2_compute_metrics.py 在服务器
   哪个绝对路径?(我可以 find,但请给目录提示)
2. predictions_*.pt 输出 schema 用 list-of-dicts 还是单 dict-of-tensors?
   Exp2 用前者(便于 batch 累积)我倾向 follow,等 MA5 / 用户确认。
```

---

## §10 最后提醒

**接力链工作哲学**(继承所有前棒):

1. **诚实 > 流畅**: 任何观察与文档假设不一致,先承认,再说影响,再给 MA5 选项
2. **70% 闸门是硬线**
3. **不深 debug**
4. **状态锚定**: 报告里所有数字 / 路径 / log 行号给具体值,不写"大约"
5. **Step 5 是接力链倒数第二棒**: 你的报告质量决定 Step6Agent 能否一棒写完 final report,也决定 MA5 能否在 phase 5b 安全决议是否解禁 holdout

Step 5 数字一旦出来,Exp4 实验结论就基本定型。**这是你的 deliverable 价值最高的一棒**。

---

*MA5 撰写,2026-04-27,等用户 review 后转发到 Step5Agent 窗口启动*
