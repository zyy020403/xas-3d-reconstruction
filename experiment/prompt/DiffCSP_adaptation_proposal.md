# 基于 DiffCSP 的 XAS → Structure 适配方案

---

## 1. DiffCSP 框架解剖

### 1.1 它做什么

DiffCSP 解决的是**晶体结构预测 (CSP)** 问题：

```
原始任务:
  输入: 化学组成 (如 SrTiO₃ → 元素类型 + 数量)
  输出: 完整晶体结构 (原子坐标 + 晶格参数)

两种模式:
  CSP模式 (diffusion.py):     给定原子类型 → 预测坐标 + 晶格
  Ab Initio模式 (diffusion_w_type.py): 同时预测原子类型 + 坐标 + 晶格
```

### 1.2 核心架构

```
                 噪声调度
                    │
  输入条件 ────►  CSPNet (等变GNN) ────► 预测噪声
                    │
              扩散过程 (T步)
                    │
              去噪 → 晶体结构
```

### 1.3 每个文件的作用

```
diffcsp/
├── run.py                    # 训练入口 (PyTorch Lightning + Hydra)
├── common/
│   ├── data_utils.py         # ★ 数据处理: 读CSV、构建图、计算距离
│   ├── constants.py          # 元素属性表 (原子序数等)
│   └── utils.py              # 工具函数
├── pl_data/
│   ├── dataset.py            # ★ PyTorch Dataset: 加载晶体数据
│   └── datamodule.py         # ★ Lightning DataModule: train/val/test 分割
├── pl_modules/
│   ├── cspnet.py             # ★★★ 核心网络: 等变图神经网络 (denoiser)
│   ├── diffusion.py          # ★★ CSP 扩散模型 (已知原子类型)
│   ├── diffusion_w_type.py   # ★★★ Ab Initio 扩散模型 (同时预测类型)
│   ├── diff_utils.py         # 扩散工具 (噪声调度, 采样)
│   ├── gnn.py                # GNN 基础层 (消息传递)
│   ├── energy_model.py       # 能量预测模型 (用于优化)
│   └── model.py              # 基础模型类
└── prop_models/              # 预训练的属性预测模型

conf/
├── default.yaml              # 主配置
├── data/                     # 数据集配置 (perov_5, mp_20, ...)
├── model/
│   ├── diffusion.yaml        # CSP 模型配置
│   ├── diffusion_w_type.yaml # Ab Initio 模型配置
│   └── decoder/cspnet.yaml   # 网络超参数
└── train/default.yaml        # 训练超参数

scripts/
├── evaluate.py               # 评估脚本
├── compute_metrics.py        # 计算指标
├── generation.py             # 从头生成结构
├── sample.py                 # 给定组成采样结构
└── optimization.py           # 属性优化

data/
├── perov_5/                  # 钙钛矿数据集 (5种元素)
│   ├── train.csv             # 每行一个晶体结构
│   ├── val.csv
│   └── test.csv
├── mp_20/                    # Materials Project (≤20原子)
├── mpts_52/                  # MP 时间分割 (≤52原子)
└── carbon_24/                # 碳结构 (≤24原子)
```

### 1.4 数据格式 (关键!)

DiffCSP 的 CSV 每一行存储一个晶体：

```python
# data_utils.py 中的关键字段:
{
    'frac_coords':  分数坐标 (N, 3),        # 原子位置
    'atom_types':   原子类型 (N,),           # 原子序数
    'lengths':      晶格长度 (3,),           # a, b, c
    'angles':       晶格角度 (3,),           # α, β, γ
    'num_atoms':    原子数 N,
}
```

### 1.5 扩散过程 (diffusion_w_type.py)

```
前向 (加噪):
  坐标:  x_t = x_0 + σ_t · ε_x     (wrapped normal, 在分数坐标空间)
  类型:  a_t = α_t · a_0 + (1-α_t) · uniform    (类型扩散)
  晶格:  l_t = l_0 + σ_t · ε_l     (高斯噪声)

反向 (去噪):
  CSPNet(x_t, a_t, l_t, t) → (ε̂_x, â_0, ε̂_l)
  
  坐标: 预测噪声 ε̂_x → 更新 x_{t-1}
  类型: 直接预测干净类型 â_0 → 重新加噪到 t-1
  晶格: 预测噪声 ε̂_l → 更新 l_{t-1}
```

---

## 2. 我们的任务 vs DiffCSP 原始任务

```
DiffCSP 原始:
  条件:   化学组成 (原子类型列表)
  生成:   原子坐标 + 晶格
  
我们的任务:
  条件:   XAS 谱 (500维一维信号) + 物理先验特征 (~20维)
  生成:   局部结构 (6Å球内的原子类型 + 坐标)
  
关键差异:
  1. 条件从 "离散元素列表" 变成 "连续谱信号"
  2. 不需要预测晶格 (我们预测的是局部结构, 不是周期性晶体)
  3. 原子数可变 (不同结构 6Å 球内原子数不同)
  4. 必须同时预测类型 → 用 diffusion_w_type.py 作为基础
```

---

## 3. 需要修改的文件和修改内容

### 3.1 修改清单总览

```
需要修改     (改动大):
  ★ diffcsp/pl_data/dataset.py           — 数据加载方式完全重写
  ★ diffcsp/pl_data/datamodule.py        — 配合新数据
  ★ diffcsp/pl_modules/diffusion_w_type.py — 添加谱条件注入
  ★ diffcsp/pl_modules/cspnet.py         — 添加谱的 cross-attention
  ★ diffcsp/common/data_utils.py         — 新的数据处理函数
  ★ conf/data/xas_mp.yaml               — 新建数据集配置
  ★ conf/model/diffusion_xas.yaml       — 新建模型配置

需要修改     (改动小):
  · scripts/evaluate.py                  — 适配新评估指标
  · scripts/compute_metrics.py           — 新指标计算
  · scripts/sample.py                    — 从谱采样结构

基本不改:
  · diffcsp/run.py                       — Lightning 训练入口 (几乎不动)
  · diffcsp/pl_modules/diff_utils.py     — 噪声调度 (不动)
  · diffcsp/pl_modules/gnn.py            — GNN 基础层 (不动)
  · conf/train/default.yaml              — 训练配置 (微调参数)
```

### 3.2 详细修改方案

---

#### 文件 1: `diffcsp/common/data_utils.py` — 数据预处理

**原始**: 从 CSV 读取晶体结构 (分数坐标, 晶格, 原子类型)

**修改**: 添加 XAS 谱读取 + POSCAR → 局部结构截取

```python
# ===== 新增函数 =====

def load_xas_spectrum(chi_path, xmu_path, n_points=500):
    """读取 FEFF 计算的 XAS 谱, 插值到统一长度"""
    chi_data = np.loadtxt(chi_path)  # k, chi(k)
    k, chi = chi_data[:, 0], chi_data[:, 1]
    # k²χ(k) 加权
    chi_weighted = k**2 * chi
    # 插值到 n_points
    k_uniform = np.linspace(k.min(), k.max(), n_points)
    chi_interp = np.interp(k_uniform, k, chi_weighted)
    return chi_interp  # (500,)

def extract_local_environment(poscar_path, center_element='Fe', radius=6.0):
    """
    从 POSCAR 截取中心原子 6Å 球内的局部结构
    
    返回:
      cart_coords: (N, 3) 以 Fe 质心为原点的笛卡尔坐标
      atom_types:  (N,) 原子序数
      num_atoms:   N
    """
    from pymatgen.core import Structure
    structure = Structure.from_file(poscar_path)
    
    # 找所有 Fe 原子
    fe_indices = [i for i, site in enumerate(structure) 
                  if site.specie.symbol == center_element]
    fe_center = np.mean([structure[i].coords for i in fe_indices], axis=0)
    
    # 截取 6Å 球内所有原子 (含周期性镜像)
    atoms = []
    for site in structure:
        for image in structure.lattice.get_all_distances(
                site.frac_coords, 
                structure.lattice.get_fractional_coords(fe_center)):
            # 简化: 用 pymatgen 的 get_neighbors
            pass
    
    # 实际实现用 structure.get_neighbors
    all_neighbors = []
    for fi in fe_indices:
        neighbors = structure.get_neighbors(structure[fi], radius)
        all_neighbors.extend(neighbors)
    
    # 去重 + 加上 Fe 自身
    # ... (详细实现)
    
    # 以 Fe 质心为原点
    coords = np.array([...]) - fe_center
    types = np.array([site.specie.Z for site in ...])
    
    return coords, types, len(coords)

def load_physics_features(features_csv_row):
    """读取师兄提取的物理先验特征"""
    # pre_peak_E, edge_position, white_line_height, ...
    return np.array([...])  # (20,)

def build_xas_dataset(manifest_csv, spectrum_dir, poscar_dir, radius=6.0):
    """
    构建完整数据集
    
    manifest_csv 每行:
      mp_id, spectrum_path, poscar_path, physics_features...
    
    输出: 每个样本 = {
      'spectrum':     (500,),
      'physics':      (20,),
      'cart_coords':  (N, 3),
      'atom_types':   (N,),
      'num_atoms':    N,
    }
    """
    pass
```

---

#### 文件 2: `diffcsp/pl_data/dataset.py` — Dataset 类

**原始**: 从 CSV 读晶体, 构建 pymatgen Structure, 提取图

**修改**: 读谱 + 局部结构

```python
class XASStructureDataset(Dataset):
    """
    替换原始的 CrystDataset
    
    每个样本:
      - spectrum: (500,) XAS 谱
      - physics_features: (20,) 物理先验
      - cart_coords: (N, 3) 笛卡尔坐标 (以 Fe 为原点)
      - atom_types: (N,) 原子序数
      - num_atoms: int
    
    注意: 不需要 frac_coords 和 lattice
          因为我们预测的是局部结构, 不是周期性晶体
          用笛卡尔坐标, 不用分数坐标
    """
    def __init__(self, data_path, spectrum_dir, poscar_dir, 
                 physics_features_csv, radius=6.0):
        self.samples = self._preprocess(...)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'spectrum': torch.tensor(sample['spectrum'], dtype=torch.float32),
            'physics': torch.tensor(sample['physics'], dtype=torch.float32),
            'cart_coords': torch.tensor(sample['cart_coords'], dtype=torch.float32),
            'atom_types': torch.tensor(sample['atom_types'], dtype=torch.long),
            'num_atoms': sample['num_atoms'],
        }
```

---

#### 文件 3: `diffcsp/pl_modules/cspnet.py` — 核心网络 (最关键修改)

**原始**: CSPNet 是一个等变 GNN，输入是 (坐标, 类型, 晶格, 时间步)

**修改**: 添加 XAS 谱的 cross-attention 条件注入

```python
class CSPNet_XAS(CSPNet):
    """
    在 CSPNet 基础上添加谱条件注入
    
    原始 CSPNet 的前向:
      node_features = embed(atom_types) + time_embed(t)
      for layer in gnn_layers:
          node_features = layer(node_features, edge_index, edge_attr)
      output = readout(node_features)
    
    修改后:
      node_features = embed(atom_types) + time_embed(t)
      spectrum_tokens = tokenize(spectrum, physics_features)  # 新增
      for layer in gnn_layers:
          node_features = layer(node_features, edge_index, edge_attr)
          node_features = cross_attn(node_features, spectrum_tokens)  # 新增
      output = readout(node_features)
    """
    def __init__(self, original_params, spec_dim=500, n_physics=20):
        super().__init__(original_params)
        
        hidden_dim = self.hidden_dim  # CSPNet 的隐藏维度
        
        # 谱 tokenizer (不降维)
        self.spec_proj = nn.Linear(1, hidden_dim)
        self.spec_pos_enc = nn.Parameter(torch.randn(spec_dim, hidden_dim))
        self.phys_proj = nn.Linear(1, hidden_dim)
        self.phys_pos_enc = nn.Parameter(torch.randn(n_physics, hidden_dim))
        
        # 每个 GNN 层后面加一个 cross-attention
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
            for _ in range(self.num_layers)
        ])
        self.cross_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(self.num_layers)
        ])
    
    def tokenize_spectrum(self, spectrum, physics):
        """
        spectrum: (B, 500)
        physics: (B, 20)
        → tokens: (B, 520, hidden_dim)
        """
        spec_tokens = self.spec_proj(spectrum.unsqueeze(-1))  # (B, 500, H)
        spec_tokens = spec_tokens + self.spec_pos_enc.unsqueeze(0)
        
        phys_tokens = self.phys_proj(physics.unsqueeze(-1))   # (B, 20, H)
        phys_tokens = phys_tokens + self.phys_pos_enc.unsqueeze(0)
        
        return torch.cat([spec_tokens, phys_tokens], dim=1)   # (B, 520, H)
    
    def forward(self, atom_types, coords, t, spectrum, physics, batch):
        """
        在每个 GNN 消息传递层之后,
        做一次 cross-attention: 原子节点 attend to 谱 tokens
        """
        # 原始 CSPNet 的初始化
        node_feats = self.node_embedding(atom_types) + self.time_embedding(t)
        
        # 谱 tokens (每个 batch 元素有自己的 520 tokens)
        spec_tokens = self.tokenize_spectrum(spectrum, physics)
        
        # GNN + Cross-Attention 交替
        for i, gnn_layer in enumerate(self.gnn_layers):
            # 图消息传递 (原子间交互)
            node_feats = gnn_layer(node_feats, edge_index, edge_attr)
            
            # Cross-attention (原子 attend to 谱)
            # 需要处理 batch: 每个图的节点只 attend to 自己的谱
            node_feats_batched = self._batch_to_padded(node_feats, batch)
            attended, _ = self.cross_attn_layers[i](
                node_feats_batched, spec_tokens, spec_tokens)
            node_feats = self.cross_norms[i](
                node_feats + self._padded_to_batch(attended, batch))
        
        return self.output_head(node_feats)
```

---

#### 文件 4: `diffcsp/pl_modules/diffusion_w_type.py` — 扩散模型

**原始**: 条件是原子组成 (CSP 任务不需要)
**修改**: 条件是 XAS 谱

```python
class DiffusionXAS(DiffusionWithType):
    """
    主要修改点:
    
    1. 删除晶格预测 (我们预测局部结构, 不需要晶格)
    2. 坐标空间从分数坐标改为笛卡尔坐标
       (原始用 wrapped normal 在 [0,1]³ 上扩散
        我们用普通高斯在 [-6, +6]³ Å 上扩散)
    3. 条件注入: 谱 + 物理特征
    4. 原子数的处理:
       DiffCSP 原始: 原子数已知 (由组成决定)
       我们: 原子数未知, 用固定 N_max + exists mask
    """
    
    def __init__(self, ...):
        super().__init__(...)
        
        # 删除晶格相关组件
        # self.lattice_... 全部删除
        
        # 替换 CSPNet 为 CSPNet_XAS
        self.decoder = CSPNet_XAS(...)
        
        # 存在性预测头
        self.exists_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, batch):
        """
        训练时:
          1. 从 batch 取出真实坐标、类型、谱
          2. 对坐标和类型加噪
          3. 用 CSPNet_XAS 预测噪声/类型
          4. 计算 loss
        """
        spectrum = batch['spectrum']          # (B, 500)
        physics = batch['physics']            # (B, 20)
        cart_coords = batch['cart_coords']    # 真实坐标
        atom_types = batch['atom_types']      # 真实类型
        
        # 随机时间步
        t = torch.randint(0, self.T, (B,))
        
        # 加噪 (笛卡尔坐标用普通高斯, 不用 wrapped)
        noise_coords = torch.randn_like(cart_coords)
        sigma_t = self.sigma_schedule(t)
        noisy_coords = cart_coords + sigma_t * noise_coords
        
        # 类型扩散 (沿用 DiffCSP 的类型扩散)
        noisy_types = self.type_diffusion(atom_types, t)
        
        # 预测
        pred_noise, pred_types, pred_exists = self.decoder(
            noisy_types, noisy_coords, t, spectrum, physics, batch)
        
        # Loss
        loss_coords = F.mse_loss(pred_noise, noise_coords)
        loss_types = F.cross_entropy(pred_types, atom_types)
        loss_exists = F.binary_cross_entropy_with_logits(
            pred_exists, batch['exists_mask'])
        
        return loss_coords + loss_types + loss_exists
    
    @torch.no_grad()
    def sample(self, spectrum, physics, n_atoms_max=80, n_steps=50):
        """
        推理: 给一条谱, 生成结构
        
        从纯噪声开始, 逐步去噪
        """
        # 初始化 N_max 个噪声原子
        coords = torch.randn(1, n_atoms_max, 3) * 6.0  # [-6, +6] Å
        types = torch.randint(0, self.n_elements, (1, n_atoms_max))
        
        for step in reversed(range(n_steps)):
            t = torch.tensor([step])
            pred_noise, pred_types, pred_exists = self.decoder(
                types, coords, t, spectrum, physics)
            
            # DDIM 更新坐标
            coords = self.ddim_step(coords, pred_noise, t)
            
            # 更新类型 (取 argmax 或重新加噪)
            types = pred_types.argmax(dim=-1)
        
        # 筛选 exists > 0.5 的原子
        exists = torch.sigmoid(pred_exists) > 0.5
        final_coords = coords[0][exists[0]]
        final_types = types[0][exists[0]]
        
        return final_coords, final_types
```

---

#### 文件 5: 新建配置文件

**`conf/data/xas_mp.yaml`**:
```yaml
root_path: ${oc.env:PROJECT_ROOT}/data/xas_mp
datamodule:
  _target_: diffcsp.pl_data.datamodule.XASDataModule
  datasets:
    train:
      _target_: diffcsp.pl_data.dataset.XASStructureDataset
      data_path: ${data.root_path}/train.csv
      spectrum_dir: ${data.root_path}/spectra/
      poscar_dir: ${data.root_path}/poscars/
      physics_csv: ${data.root_path}/physics_features.csv
      radius: 6.0
    val:
      # 同上, 用 val.csv
    test:
      # 同上, 用 test.csv
  batch_size:
    train: 32
    val: 32
    test: 32
```

**`conf/model/diffusion_xas.yaml`**:
```yaml
_target_: diffcsp.pl_modules.diffusion_w_type.DiffusionXAS
spec_dim: 500
n_physics_features: 20
n_atoms_max: 80
hidden_dim: 256
num_layers: 6
num_heads: 8
beta_scheduler:
  _target_: diffcsp.pl_modules.diff_utils.CosineSchedule
  T: 1000
decoder:
  _target_: diffcsp.pl_modules.cspnet.CSPNet_XAS
  hidden_dim: 256
  num_layers: 6
  spec_dim: 500
  n_physics: 20
```

---

## 4. 数据准备步骤

```
Step 0: 整理师兄给的数据

  需要的目录结构:
  data/xas_mp/
  ├── spectra/
  │   ├── mp-xxxx_Fe_site0_chi.dat    # 每个 Fe 位点的 chi
  │   ├── mp-xxxx_Fe_site0_xmu.dat    # 每个 Fe 位点的 xmu
  │   └── ...
  ├── poscars/
  │   ├── mp-xxxx_POSCAR
  │   └── ...
  ├── physics_features.csv             # 表2: 物理先验
  ├── bond_lengths.csv                 # 表1: 键长范围
  ├── train.csv                        # 训练集 manifest
  ├── val.csv
  └── test.csv

  manifest CSV 格式:
    mp_id, fe_site_idx, spectrum_chi_path, spectrum_xmu_path, 
    poscar_path, n_atoms_in_sphere, [physics_feature_columns...]

Step 1: 预处理脚本 (新建)

  python scripts/preprocess_xas.py \
    --raw_dir /path/to/师兄的数据 \
    --output_dir data/xas_mp \
    --radius 6.0 \
    --val_ratio 0.15 \
    --test_ratio 0.15
  
  这个脚本:
    1. 遍历所有 POSCAR, 对每个 Fe 位点截取 6Å 球
    2. 读取对应的 chi/xmu 谱
    3. 计算/读取物理先验特征
    4. 随机分割 train/val/test
    5. 生成 CSV
```

---

## 5. 运行顺序

```
# ── Step 0: 环境 ──
cp .env.template .env
# 编辑 .env, 设置 PROJECT_ROOT 等路径
pip install torch==1.9.0 torch-geometric==1.7.2 pytorch_lightning==1.3.8 pymatgen

# ── Step 1: 数据预处理 ──
python scripts/preprocess_xas.py --raw_dir ... --output_dir data/xas_mp

# ── Step 2: 训练 ──
python diffcsp/run.py data=xas_mp model=diffusion_xas expname=xas_v1

# ── Step 3: 评估 ──
python scripts/evaluate.py --model_path <checkpoint> --dataset xas_mp
python scripts/compute_metrics.py --root_path <checkpoint> --tasks csp

# ── Step 4: 从谱采样结构 ──
python scripts/sample_from_spectrum.py \
  --model_path <checkpoint> \
  --spectrum_path test_spectrum.dat \
  --physics_path test_physics.csv \
  --num_samples 20 \
  --output_dir results/
```

---

## 6. 关键技术决策

### 6.1 坐标空间: 分数 vs 笛卡尔

```
DiffCSP 原始: 分数坐标 + wrapped normal diffusion
  原因: 晶体有周期性, 分数坐标自然在 [0,1]³ 上

我们的选择: 笛卡尔坐标 + 普通高斯 diffusion
  原因: 6Å 球内的局部结构没有周期性
  范围: [-6, +6]³ Å
  
  修改: diffusion_w_type.py 中
    删除 wrapped_normal 相关代码
    替换为普通高斯扩散
```

### 6.2 晶格预测: 保留 vs 删除

```
DiffCSP 原始: 同时预测坐标 + 晶格参数 (a,b,c,α,β,γ)

我们的选择: 删除晶格预测
  原因: 我们预测的是局部环境, 不是周期性晶体
  
  修改: diffusion_w_type.py 中
    删除所有 lattice 相关的扩散/预测/loss
    只保留 coords + types 的扩散
    
  但要注意: CIF 导出时需要一个假晶胞 (20×20×20 Å)
```

### 6.3 原子数: 固定 vs 可变

```
DiffCSP 原始: 原子数由组成决定, 是已知的

我们的选择: 原子数未知, 模型需要自己决定
  方案: 固定 N_max=80, 加 exists_mask
    - 预测每个位置是否有原子 (二分类)
    - exists > 0.5 的保留
    - 类似 DETR 的 object detection 思路
```

### 6.4 师兄三张表如何使用

```
表1 (键长范围):
  → bond_length_loss: 在 diffusion_w_type.py 的 loss 中添加
  → 采样后验证: 在 evaluate.py 中检查键长合法性

表2 (物理先验特征):
  → 作为额外输入 tokens: 在 cspnet.py 的 cross-attention 中
  → 已经在上面的方案中实现

表3 (推理逻辑):
  → 不编码进模型, 用于评估时的可解释性分析
  → 可视化 attention 权重, 验证模型关注了正确的谱区域
```

---

## 7. 修改工作量估算

| 文件 | 修改类型 | 工作量 | 说明 |
|------|----------|--------|------|
| `data_utils.py` | 新增函数 | 半天 | 谱读取 + POSCAR 截取 |
| `dataset.py` | 重写 | 半天 | 新 Dataset 类 |
| `datamodule.py` | 小改 | 1小时 | 适配新 Dataset |
| `cspnet.py` | 较大修改 | 1天 | 添加 cross-attention |
| `diffusion_w_type.py` | 较大修改 | 1天 | 改坐标空间 + 删晶格 + 加 exists |
| 新建配置文件 | 新增 | 1小时 | yaml 配置 |
| `preprocess_xas.py` | 新建 | 半天 | 数据预处理脚本 |
| `evaluate.py` | 小改 | 2小时 | 新评估指标 |
| `sample_from_spectrum.py` | 新建 | 2小时 | 从谱采样 |
| **总计** | | **~4-5 天** | |

---

## 8. DiffCSP 的优势 (为什么用它)

```
1. 等变性: CSPNet 保证了 SE(3) 等变性
   → 旋转输入结构 = 旋转输出结构
   → 物理合理性内置

2. 类型扩散: 已经实现了离散元素类型的扩散
   → 不需要自己设计元素如何 "加噪/去噪"

3. 代码成熟: NeurIPS 2023 论文, 代码稳定
   → 减少 debug 时间

4. 评估框架: 已有 compute_metrics.py
   → 结构匹配、Match Rate 等指标现成

5. Hydra 配置: 实验管理方便
   → 改配置文件就能跑不同实验
```

---

## 9. 需要师兄确认

```
1. 数据目录结构: 师兄给的 9000 个结构的文件组织方式？
   每个结构一个文件夹？命名规则？

2. 谱的对应关系: 每个 POSCAR 对应几条谱？
   如果有多个 Fe 位点, 是一条平均谱还是每个位点一条？

3. 表 2 的格式: 已经算好了还是需要我写代码提取？
   如果已有, 列名是什么？

4. GPU: 型号？显存？
   CSPNet + cross-attention, batch=32 约需 8-12 GB

5. 时间: 多久需要出第一个结果？
```
