# STEP3_HANDOFF.md
# Step3 Agent 交接文档：数据集实现 + 模型改造

> **你的角色**：Step3 Agent
> **你的任务**：
>   1. 实现 XASLocalStructureDataset 和 XASDataModule
>   2. 修改 diffusion_w_type.py（最小手术）
>   3. 新建 yaml 配置文件
>   4. 端到端前向测试（1个 batch，loss 不为 NaN）
> **前置文档**：先读 SHARED_00_v2.md
> **输出目录**：`C:\Users\T-Cat\Desktop\DiffCSP-main\experiment2\step3\`

---

## 所有前序步骤的关键参数（以此为准）

| 参数 | 值 | 来源 |
|------|----|------|
| 虚拟晶格边长 L | **12 Å** | Step2.5 实测 |
| 局部结构原子数 N | **20个邻居**（含 Fe 共 21 个节点，但 Fe 固定在原点，CSPNet 只处理 20 个邻居） | Step1.8 决策 |
| xmu.dat 能量列 | **data[:,0]**（绝对能量，7117-8638 eV） | Step2.2 实测修正 |
| xmu.dat μ(E) 列 | **data[:,3]** | Step2.2 实测 |
| chi1.dat k列 | data[:,0]，chi1 列 data[:,1] | 不变 |
| feff_features 路径 | `C:\Users\T-Cat\Desktop\DiffCSP-main\tesst_feff_features_all_full_v4.csv` | Step1.5 更新 |
| feff_features 数值列 | iloc[3:76]，共 73 列 | 不变 |
| batch_size | **16**（原胞最多 80 个原子的遗留设计，现固定 20 邻居，16 也够） | Main Agent 建议 |

---

## 你需要的文件（向用户索取）

```
1. 共享文档：
   SHARED_00_v2.md

2. Step1 输出（直接访问路径）：
   experiment2/step1/data_inventory.csv
   experiment2/step1/train_ids.txt / val_ids.txt / test_ids.txt
   experiment2/step1/feff_feature_scaler.pkl
   experiment2/step1/feff_feature_stats.csv

3. Step2 输出（直接访问路径）：
   experiment2/step2/spectrum_preprocessor.py   ← 预处理函数库
   experiment2/step2/spectrum_encoder.py        ← SpectrumEncoder 类

4. DiffCSP 原项目文件（需要用户发给你，用于改造）：
   diffcsp/pl_modules/diffusion_w_type.py       ← 主要修改目标
   diffcsp/pl_data/dataset.py                   ← 重写
   diffcsp/pl_data/datamodule.py                ← 小改
   conf/model/diffusion_w_type.yaml             ← 参考，新建 xas 版本
   conf/data/mp_20.yaml                         ← 参考，新建 xas 版本

5. Exp1 参考文件（直接访问路径）：
   experiment/step3/xas_dataset.py              ← 原胞转换 + clone 修复的参考
   experiment/step3/xas_datamodule.py           ← DataModule 结构参考
```

---

## 工作内容

### Step 3.1：实现 XASLocalStructureDataset

**文件名**：`xas_local_dataset.py`，存入 `experiment2/step3/`

这是改动最大的部分，完整逻辑如下：

#### 核心设计

```
对每个样本（一个化合物）：
  1. 读 POSCAR_supercell_fixed → 原胞转换（symprec=0.1）
  2. 在原胞中找 LVSI 对应的 Fe 位点（用 selected_site_map.csv 的 site_nn 信息）
  3. 取该 Fe 的最近 20 个邻居（考虑周期性边界，用 get_neighbors）
  4. 以 Fe 为原点，得到 20 个邻居的笛卡尔坐标（相对坐标）
  5. 转换为分数坐标：frac = cart / L（L=12），确保 ∈ (-0.5, 0.5)
  6. 构建输入 tensor：
       frac_coords  (20, 3)
       atom_types   (20,)   原子序数
       lengths      (3,)    [12, 12, 12]
       angles       (3,)    [90, 90, 90]
       num_atoms    int     20
  7. 读 chi1.dat → load_chi1() → (200,)
  8. 读 xmu.dat + E0 → load_xmu_xanes() → (150,)
  9. 查 feff_features 表 → load_feff_features() → (73,)
```

#### Fe 位点定位方式

```python
# LVSI 的 site_nn 是文件夹名中的序号（如 "01"），对应 FEFF 计算时的位点编号
# 在原胞中，需要找到所有 Fe 原子，取 site_nn 对应的那个
# 由于 FEFF site 编号从 01 起对应 POSCAR 中 Fe 的出现顺序，
# site_nn="01" → 取原胞中第 0 个 Fe（index 从 0 开始）
# site_nn="02" → 取原胞中第 1 个 Fe，依此类推

fe_indices = [i for i, site in enumerate(primitive)
              if site.specie.symbol == 'Fe']
site_idx = int(site_nn) - 1   # "01" → 0, "02" → 1
if site_idx >= len(fe_indices):
    site_idx = 0  # 容错：原胞转换后 Fe 数量可能减少，退回第一个
fe_index = fe_indices[site_idx]
```

#### 邻居截取方式

```python
# 用 pymatgen 的 get_neighbors，考虑周期性边界
# 截取半径设为 10Å（远大于 L/2=6Å，确保能取到 20 个邻居）
neighbors = primitive.get_neighbors(primitive[fe_index], r=10.0)
neighbors_sorted = sorted(neighbors, key=lambda x: x.nn_distance)[:20]

# 若不足 20 个，跳过此样本（Step1 已统计 1% 概率，可接受）
if len(neighbors_sorted) < 20:
    return None  # collate_fn 中过滤掉 None

# 邻居坐标：以 Fe 为原点的笛卡尔坐标
fe_cart = primitive[fe_index].coords
neighbor_carts = np.array([n.coords - fe_cart for n in neighbors_sorted])
# shape (20, 3)

# 转换为分数坐标（虚拟立方晶格 L=12）
L = 12.0
frac_coords = neighbor_carts / L
# 确保在 (-0.5, 0.5) 内（d20_99th=5.14Å，5.14/12=0.43，安全）
```

#### __getitem__ 完整返回

```python
return {
    # DiffCSP 原版字段（保留字段名，供 collate_fn 和 CSPNet 识别）
    'frac_coords': torch.tensor(frac_coords, dtype=torch.float32),   # (20, 3)
    'atom_types':  torch.tensor(atom_types,  dtype=torch.long),      # (20,)
    'lengths':     torch.tensor([L, L, L],   dtype=torch.float32),   # (3,)
    'angles':      torch.tensor([90.,90.,90.],dtype=torch.float32),  # (3,)
    'num_atoms':   20,

    # 新增字段（谱和特征）
    'xmu_xanes':      torch.tensor(xmu_feat, dtype=torch.float32),   # (150,)
    'chi1':           torch.tensor(chi1_feat,dtype=torch.float32),   # (200,)
    'feff_features':  torch.tensor(feats,    dtype=torch.float32),   # (73,)

    # 评估用元信息（不入模型）
    'mp_id':      mp_id,
    'eval_cutoff': float(min(neighbors_sorted[19].nn_distance, 4.0)),  # 动态评估截断
}
```

#### 关于 clone() bug（Exp1 经验）

```python
# Exp1 中发现一个 bug：pymatgen 返回的 embedding 共享内存，
# 直接转 tensor 会导致梯度计算异常
# 修复方式：对 frac_coords 和 atom_types 做 .copy()

frac_coords = neighbor_carts / L
frac_coords = frac_coords.copy()   # ← 必须
atom_types  = np.array([n.specie.Z for n in neighbors_sorted]).copy()  # ← 必须
```

---

### Step 3.2：实现 XASDataModule

**文件名**：`xas_local_datamodule.py`，存入 `experiment2/step3/`

参考 `experiment/step3/xas_datamodule.py` 的结构，改动点：

```python
class XASDataModule(pl.LightningDataModule):
    def __init__(self, data_root, step1_dir, batch_size=16, ...):
        # 读 train_ids.txt / val_ids.txt / test_ids.txt
        # 用 XASLocalStructureDataset 构建三个 split
        # num_workers=0（Windows）

    def collate_fn(self, batch):
        # 过滤 None（邻居不足 20 的样本）
        batch = [b for b in batch if b is not None]
        
        # 沿用原版 DiffCSP 的 collate 逻辑处理 frac_coords/atom_types/lengths/angles
        # 新增字段 stack：
        #   xmu_xanes     (B, 150)
        #   chi1          (B, 200)
        #   feff_features (B, 73)
        #   eval_cutoff   (B,)    ← 评估用，不入模型
```

---

### Step 3.3：修改 diffusion_w_type.py

**文件名**：在 `experiment2/step3/` 存放修改后的副本，命名 `diffusion_w_type_xas.py`。
**不直接修改原项目文件**，Step4 训练时用这个副本替换或 sys.path 优先加载。

#### 修改点1：import SpectrumEncoder

```python
# 文件顶部添加（在其他 import 之后）：
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                '..', '..', 'experiment2', 'step2'))
from spectrum_encoder import SpectrumEncoder
```

#### 修改点2：CSPDiffusion.__init__

```python
def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)

    # 原有代码（保留）：
    self.decoder = hydra.utils.instantiate(
        self.hparams.decoder,
        latent_dim = self.hparams.latent_dim + self.hparams.time_dim,
        pred_type = True, smooth = True)
    self.beta_scheduler = ...
    self.sigma_scheduler = ...
    self.time_dim = self.hparams.time_dim
    self.time_embedding = SinusoidalTimeEmbeddings(self.time_dim)
    self.keep_lattice = self.hparams.cost_lattice < 1e-5
    self.keep_coords  = self.hparams.cost_coord  < 1e-5

    # ★ 新增：SpectrumEncoder
    self.spectrum_encoder = SpectrumEncoder(
        xmu_dim   = self.hparams.get('xmu_dim',   150),
        chi_dim   = self.hparams.get('chi_dim',   200),
        feat_dim  = self.hparams.get('feat_dim',  73),
        latent_dim= self.hparams.get('spectrum_latent_dim', 256),
    )
```

#### 修改点3：forward() — 拼接 condition

```python
def forward(self, batch):
    batch_size = batch.num_graphs
    times = self.beta_scheduler.uniform_sample_t(batch_size, self.device)
    time_emb = self.time_embedding(times)   # (B, 256)

    # ★ 新增：谱条件编码
    spectrum_cond = self.spectrum_encoder(
        batch.xmu_xanes,       # (B, 150)
        batch.chi1,            # (B, 200)
        batch.feff_features,   # (B, 73)
    )                          # → (B, 256)
    condition = torch.cat([time_emb, spectrum_cond], dim=-1)  # (B, 512)

    # 原有代码（保留，只把 time_emb 替换为 condition）：
    alphas_cumprod = self.beta_scheduler.alphas_cumprod[times]
    # ... 加噪过程不变 ...

    pred_l, pred_x, pred_t = self.decoder(
        condition,          # ← 原来是 time_emb，现在是 condition(512)
        atom_type_probs, input_frac_coords, input_lattice,
        batch.num_atoms, batch.batch)

    # loss 计算不变
```

#### 修改点4：sample() — 同样替换 time_emb

```python
@torch.no_grad()
def sample(self, batch, diff_ratio=1.0, step_lr=1e-5):
    batch_size = batch.num_graphs

    # ★ 新增：在采样循环外计算谱编码（每次采样只算一次）
    spectrum_cond = self.spectrum_encoder(
        batch.xmu_xanes, batch.chi1, batch.feff_features)

    # 原有初始化代码不变...
    for t in tqdm(range(self.beta_scheduler.timesteps, 0, -1)):
        times = torch.full((batch_size,), t, device=self.device)
        time_emb = self.time_embedding(times)

        # ★ 拼接
        condition = torch.cat([time_emb, spectrum_cond], dim=-1)

        # 原有 decoder 调用，time_emb → condition
        pred_l, pred_x, pred_t = self.decoder(
            condition, t_t, x_t, l_t, batch.num_atoms, batch.batch)
        # ... 其余不变
```

**注意**：`keep_lattice` 在 yaml 中设 `cost_lattice: 0.0`，这样 `self.keep_lattice = True`，采样时晶格直接用输入值（diag(12,12,12)），不参与扩散。

---

### Step 3.4：新建 YAML 配置文件

#### `conf/data/xas_fe_local.yaml`（新建）

```yaml
_target_: experiment2.step3.xas_local_datamodule.XASDataModule
data_root: "C:/Users/T-Cat/Desktop/DiffCSP-main/site_dataset_Fe_only_oxide_one_site"
step1_dir: "C:/Users/T-Cat/Desktop/DiffCSP-main/experiment2/step1"
step2_dir: "C:/Users/T-Cat/Desktop/DiffCSP-main/experiment2/step2"
feff_feat_csv: "C:/Users/T-Cat/Desktop/DiffCSP-main/tesst_feff_features_all_full_v4.csv"
L: 12.0
N_neighbors: 20
batch_size:
  train: 16
  val: 16
  test: 16
num_workers: 0
```

#### `conf/model/diffusion_xas.yaml`（新建，参考 diffusion_w_type.yaml）

```yaml
_target_: diffcsp.pl_modules.diffusion_w_type_xas.CSPDiffusion

# 晶格固定不预测
cost_lattice: 0.0
cost_coord:   1.0
cost_type:    1.0

# 时间嵌入维度（不变）
time_dim: 256

# ★ 谱编码器输出维度
spectrum_latent_dim: 256
xmu_dim:  150
chi_dim:  200
feat_dim: 73

# latent_dim = spectrum_latent_dim，CSPNet 收到的是 time_dim+latent_dim=512
latent_dim: 256

# decoder 配置（参考原版 diffusion_w_type.yaml，几乎不变）
decoder:
  _target_: diffcsp.pl_modules.cspnet.CSPNet
  hidden_dim: 128
  latent_dim: 512     # time_dim(256) + spectrum_latent_dim(256)，这里覆盖
  num_layers: 4
  max_atoms: 100
  act_fn: silu
  dis_emb: sin
  num_freqs: 10
  edge_style: fc
  cutoff: 6.0
  max_neighbors: 20
  ln: false
  ip: true
  smooth: true
  pred_type: true

beta_scheduler:
  _target_: diffcsp.pl_modules.diff_utils.BetaScheduler
  timesteps: 1000
  scheduler_mode: cosine

sigma_scheduler:
  _target_: diffcsp.pl_modules.diff_utils.SigmaScheduler
  timesteps: 1000
  sigma_begin: 0.005
  sigma_end: 0.5
```

---

### Step 3.5：端到端前向测试

**文件名**：`step3_5_e2e_forward_test.py`

**任务**：不启动完整训练，只验证一个 batch 能跑通 forward()，loss 不为 NaN。

```python
# 步骤：
# 1. 加载 XASDataModule，取 train dataloader 的第一个 batch
# 2. 初始化 CSPDiffusion（用 diffusion_xas.yaml 配置）
# 3. 调用 model(batch)，检查返回的 loss
# 4. 调用 loss.backward()，检查梯度不为 NaN
# 5. 用 5 个样本调用 model.sample(batch)，检查 pred_lengths

assert not loss.isnan(), "loss is NaN"
assert loss.item() > 0,  "loss is zero"

# 健康检查（防止重蹈 Exp1 覆辙）：
pred_lattices = traj[0]['lattices']   # (B, 3, 3)
pred_lengths  = torch.stack([
    torch.tensor([m[0,0].item(), m[1,1].item(), m[2,2].item()])
    for m in pred_lattices])          # (B, 3)

# 由于 cost_lattice=0，晶格固定为 diag(12,12,12)
# pred_lengths 应精确等于 [12, 12, 12]
assert (pred_lengths - 12.0).abs().max() < 0.1, \
    f"晶格不固定！pred_lengths={pred_lengths}"

print("✅ forward pass 通过")
print("✅ loss 不为 NaN")
print("✅ 晶格固定为 12Å（cost_lattice=0 生效）")
print(f"   loss = {loss.item():.4f}")
print(f"   loss_coord = {output_dict['loss_coord'].item():.4f}")
print(f"   loss_type  = {output_dict['loss_type'].item():.4f}")
```

---

## 输出文件清单

```
experiment2/step3/
├── xas_local_dataset.py          ★ Dataset 类（Step4 使用）
├── xas_local_datamodule.py       ★ DataModule 类（Step4 使用）
├── diffusion_w_type_xas.py       ★ 改造后的扩散模型（Step4 使用）
├── step3_5_e2e_forward_test.py   前向测试脚本
└── conf_xas/                     ★ 配置文件目录
    ├── data/xas_fe_local.yaml
    └── model/diffusion_xas.yaml
```

---

## 注意事项

1. **不修改原项目任何文件**：所有改动产生新文件放在 experiment2/step3/，Step4 训练时用 sys.path 优先加载

2. **xmu.dat 列索引**：能量 = data[:,0]，μ(E) = data[:,3]（Step2 实测，不是文档里的 data[:,1]）

3. **cost_lattice = 0 的含义**：`self.keep_lattice = (cost_lattice < 1e-5) = True`，采样时晶格直接用输入的 diag(12,12,12)，不参与扩散。确认 yaml 里 `cost_lattice: 0.0`

4. **Fe 位点定位容错**：原胞转换后 Fe 数量可能因对称性减少，site_idx 越界时退回 index=0

5. **eval_cutoff 字段**：存入 batch 但不入 CSPNet，仅 Step5 评估时使用

6. **clone() 必须做**：frac_coords 和 atom_types 提取后必须 `.copy()`，参见 Exp1 的 embedding clone bug

7. **Step3.5 的晶格检查是 Exp1 经验的强制复现**：pred_lengths 爆炸是当年 265 epoch 白跑的根因，这次固定晶格后此检查必须通过

---

## 完成后向 Main Agent 汇报

重点汇报：
- Step3.5 前向测试的 loss / loss_coord / loss_type 数值（合理范围：各 0.1 ~ 2.0）
- 晶格固定检查是否通过（pred_lengths ≈ 12）
- Dataset 加载速度（每个样本的处理时间，估算训练 epoch 时长）
- 任何报错或需要决策的问题
