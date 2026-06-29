# MA4 → Sub-Agent 4-续 决策回复 + 安装指令

> **MA4 决策**：方案 **B**(分两步装),并加 sklearn/numpy/pymatgen 防升级守卫
> **chdir 副作用**:已锁定接受(选 a),与方案 B 正交,不重新讨论
> **预算**:你当前 ~25%,本轮安装 + 验证 + 跑 forward_test 估 ~30-40K token,不会触 60% 闸门

---

## §1 我对你 §2.2 汇报的认可

依赖盘点干净、6 缺失分级清晰、yaml 误报你抓到了、torch_sparse 风险等级你判对了、不替我决定 A/B/C/D 严格守约。**这是诚实 > 流畅工作哲学的标准执行。**

不需要再讨论方案矩阵,**直接进入安装序列**。

---

## §2 执行序列(逐步,每步粘输出)

### §2.X.0 先验证 yaml 假设(零副作用,你之前问过)

```bash
python -c "import yaml; print('yaml version:', yaml.__version__); print('yaml file:', yaml.__file__)"
```

**期望**:输出 yaml 版本号 + 路径在 mlff env 内。

**判定**:
- 输出版本号 → yaml 已装,确认你 §2.2 误报判定正确,继续 §2.X.1
- ImportError → yaml 也缺,在 §2.X.1 装包列表加 `pyyaml`,继续

### §2.X.1 装前快照:记录 mlff env 关键包版本(防升级守卫)

**这一步是 MA4 加的预防措施**。如果 §2.X.2/§2.X.3 装包过程中 pip resolve 把关键包升级,我们要能立刻发现并 rollback。

```bash
python -c "
import sklearn, numpy, scipy, pymatgen, torch, pytorch_lightning, torch_scatter
print('=== Pre-install version snapshot ===')
print(f'sklearn:           {sklearn.__version__}')
print(f'numpy:             {numpy.__version__}')
print(f'scipy:             {scipy.__version__}')
print(f'pymatgen:          {pymatgen.__version__}')
print(f'torch:             {torch.__version__}')
print(f'pytorch_lightning: {pytorch_lightning.__version__}')
print(f'torch_scatter:     {torch_scatter.__version__}')
" | tee /tmp/preinstall_versions.txt
```

**期望**(继承 Sub-Agent 1/2/3 已验证版本):
```
sklearn:           1.7.2
numpy:             2.2.6
scipy:             1.15.3
pymatgen:          2025.10.7
torch:             2.4.1+cu124
pytorch_lightning: 2.5.5
torch_scatter:     2.1.2+pt24cu124
```

**任一与上面期望不符立刻停汇报**(说明 mlff env 已经被外部污染,不是装包能修的)。

### §2.X.2 第 1 步:装纯 Python 包(einops + p_tqdm,几乎零子依赖)

```bash
pip install einops p_tqdm
```

**期望**:
- "Successfully installed einops-X.X.X p_tqdm-X.X.X"
- pip 不应该 uninstall 任何已有包(如果输出有 "Uninstalling sklearn-1.7.2" 之类,**立刻 Ctrl+C 停**汇报)

装完立刻验证关键包版本未变:

```bash
python -c "
import sklearn, numpy, scipy, pymatgen
print(f'sklearn:  {sklearn.__version__}  (期望 1.7.2)')
print(f'numpy:    {numpy.__version__}    (期望 2.2.6)')
print(f'scipy:    {scipy.__version__}    (期望 1.15.3)')
print(f'pymatgen: {pymatgen.__version__} (期望 2025.10.7)')
"
```

任一关键包版本变了 → **停汇报**(选项:rollback 装的包 vs 重启 env vs MA4 改方案)。

### §2.X.3 第 2 步:装中风险化学包(smact + matminer + pyxtal)

```bash
pip install smact matminer pyxtal 2>&1 | tee /tmp/install_chem.log
```

**关键监控**:盯着输出有没有出现:
- `Uninstalling scikit-learn-1.7.2:` ← **致命**(sklearn 被升级,RobustScaler pkl 可能炸)
- `Uninstalling numpy-2.2.6:` ← **致命**(numpy 升级可能再次撞 1.x→2.x 时已修过的别名问题反向)
- `Uninstalling pymatgen-2025.10.7:` ← **致命**(POSCAR 读取/SpacegroupAnalyzer 行为可能变)
- `Uninstalling torch:` 或 `Uninstalling torch_scatter:` ← **致命**(GPU 路径炸)

任何一条出现 → **立刻 Ctrl+C 中断 pip**,停汇报。pip resolve 会把已 download 的 wheel 缓存,但 install 阶段未完成的还能 rollback 干净。

如果 pip 自然走完,验证版本守卫:

```bash
python -c "
import sklearn, numpy, scipy, pymatgen, torch, pytorch_lightning, torch_scatter
print(f'sklearn:           {sklearn.__version__}  (期望 1.7.2,任何变化 = 致命)')
print(f'numpy:             {numpy.__version__}    (期望 2.2.6,任何变化 = 致命)')
print(f'scipy:             {scipy.__version__}    (期望 1.15.3,小版本变化 OK)')
print(f'pymatgen:          {pymatgen.__version__} (期望 2025.10.7,任何变化 = 致命)')
print(f'torch:             {torch.__version__}    (期望 2.4.1+cu124,任何变化 = 致命)')
print(f'pytorch_lightning: {pytorch_lightning.__version__} (期望 2.5.5,小版本变化 OK)')
print(f'torch_scatter:     {torch_scatter.__version__}    (期望 2.1.2+pt24cu124,任何变化 = 致命)')
"
```

**判定**:
- "致命"标记的包版本完全没变 → PASS,进 §2.X.4
- "致命"标记的包版本变了 → **停汇报**,粘贴 install_chem.log 给 MA4
- 仅 scipy/pl 小版本变化(如 1.15.3 → 1.15.4) → 报告但不停,继续

### §2.X.4 第 3 步:装 torch_sparse(必须用 PyG wheel index)

```bash
pip install torch_sparse -f https://data.pyg.org/whl/torch-2.4.1+cu124.html
```

**期望**:
- 装到 `torch_sparse-0.6.X+pt24cu124`(版本号约 0.6.18,与 torch_scatter 2.1.2+pt24cu124 同源)
- 不 uninstall torch / torch_scatter

如果 pip 报"找不到 wheel"或"装到 CPU-only 版本"(版本号无 `+pt24cu124` 后缀):
**停汇报**——可能 wheel index 路径变了或 torch_sparse 在该版本系列已停止发布。

装完验证:

```bash
python -c "
import torch_sparse
print(f'torch_sparse: {torch_sparse.__version__}')
# 简单 sanity:确认 torch_sparse 能用 cuda
import torch
if torch.cuda.is_available():
    from torch_sparse import SparseTensor
    print('SparseTensor import OK')
"
```

**期望**:版本号含 `+pt24cu124`,SparseTensor import 不报错。

### §2.X.5 第 4 步:**全量验证**——重跑 §2.2 import 链

回到原 SUBAGENT4CONT_HANDOFF §2.2 的 dotenv + transitive import 测试,确认装完所有包后整个 import 链通了:

```bash
cd /home/tcat/diffcsp_exp4/code
python -c "
import sys
sys.path.insert(0, '/home/tcat/diffcsp_exp4/code')
import os
print('--- before any diffcsp import ---')
print('cwd:', os.getcwd())

print('--- importing diffcsp.common.utils ---')
from diffcsp.common import utils
print('cwd after:', os.getcwd())
print('PROJECT_ROOT:', os.environ.get('PROJECT_ROOT'))

print('--- importing pl_modules ---')
from diffcsp.pl_modules import cspnet
from diffcsp.pl_modules import diff_utils
print('all transitive imports OK')
"
```

**期望**:
- cwd 切到 /home/tcat/diffcsp_exp4/code(chdir 副作用确认)
- PROJECT_ROOT = /home/tcat/diffcsp_exp4/code
- 全部 import 通过
- 出现 "all transitive imports OK"

**任一异常**:停汇报。可能是:
- 还有缺包(我们没盘点全 → 让 MA4 决定补装哪个)
- 装的某个包内部又拽了缺包 → 同上
- 版本冲突报错(如 sklearn ABI mismatch) → 严重,可能要 rollback

### §2.X.6 第 5 步:跑 forward_test.py(原 §2.4)

import 链全通后,直接进原 SUBAGENT4CONT_HANDOFF §2.4:

```bash
cd /home/tcat/diffcsp_exp4/code/step3
PYTHONPATH=/home/tcat/diffcsp_exp4/code python forward_test.py 2>&1 | tee /home/tcat/diffcsp_exp4/logs/step3_forward_test_console.log
```

**结果按原 §3 期望表 + §4 失败决策树处理**(本回复不重复)。

---

## §3 重要约束(继承 + 新加)

继承的:

- 60% 上下文闸门(你当前 25%,装完 + 重跑 forward_test 估到 ~50-55%,不会触发)
- 不动 dataset_v2 / datamodule_v2 / forward_test.py / yaml / spectrum_encoder
- 不接触 holdout / incompat_pool / .bak 备份

**MA4 新加**:

- ❌ 不要 `pip install --upgrade` 任何包(隐式触发 pip resolve)
- ❌ 不要尝试 conda install(mlff 是 conda env 但已稳定,只允许 pip 装新包)
- ❌ 不要在装包过程中"顺便 upgrade pip"(会拽大量子依赖)
- ✅ pip install 命令出意外,**Ctrl+C 中断**比"等它走完看损失"更安全
- ✅ 任何"致命"关键包版本被改 → rollback 所有本回合装的包,停汇报
  - rollback 命令(仅在确认致命变化后,MA4 同意才跑):
    ```bash
    pip uninstall -y einops p_tqdm smact matminer pyxtal torch_sparse
    # 注意:这只 uninstall 装的 6 个,被升级的关键包不会自动 downgrade
    # 致命情况需要 MA4 决定是否 reinstall mlff env
    ```

---

## §4 你接下来的输出

走完 §2.X.0 → §2.X.6,把每步的命令输出**完整粘贴**(prompt + 命令 + 结果)给我。

**两种结果路径**:

| 全程顺利(§2.X.6 forward_test.py 6.4 + 6.5 PASS) | 中途任一步出意外 |
|---|---|
| 直接按原 SUBAGENT4CONT_HANDOFF §6 完成汇报模板写给 MA4 | 按原 §5 中途停汇报模板写给 MA4 |

---

*MA4 撰写,2026-04-26,继续 Sub-Agent 4-续 同一窗口工作*
