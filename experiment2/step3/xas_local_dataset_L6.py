"""
Step 3.1 — XASLocalStructureDataset  (v5-L6)
==========================================
Dataset for XAS -> local Fe structure prediction (Experiment 2).

修改记录（v4）：
  - feff_features 索引键改为 (mp_id, site_nn)，从 sample_dir 路径末段解析
    格式：...mp_1001571_CaFeO3__feff_Fe_site_01
    只保留 _feff_Fe_site_ 行（排除其他元素位点）

修改记录（v5）：
  - 确认坐标系为 [-0.5, 0.5]：frac_coords = neighbor_carts / L（不做 % 1.0 折叠）
  - 新增过滤：任一 frac_coord 分量绝对值 > 0.5 的样本返回 None（极稀疏结构）
  - 两项改动使坐标分布为单峰（以 Fe 原点为中心），配合 diffusion_w_type_xas.py v3 使用
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# ── 常量 ──────────────────────────────────────────────────────────────────────
L = 6.0
N_NEIGHBORS = 20
NEIGHBOR_SEARCH_R = 10.0
FE_K_EDGE_DEFAULT = 7112.0

XMU_N_POINTS    = 150
XMU_EMIN_OFFSET = -50.0
XMU_EMAX_OFFSET = 150.0

CHI1_N_POINTS   = 200
FEFF_FEAT_NCOLS = 73    # iloc[3:76]

# sample_dir 末段格式：mp_1001571_CaFeO3__feff_Fe_site_01
_FEFF_DIR_RE = re.compile(r'(mp_\d+)_.*_feff_Fe_site_(\d+)', re.IGNORECASE)


def _parse_feff_key(sample_dir: str):
    """
    从 sample_dir 路径末段提取 (mp_id, site_nn)。
    例：'...mp_1001571_CaFeO3__feff_Fe_site_01' → ('mp_1001571', '01')
    非 Fe 位点（_feff_Lu_site_ 等）返回 None。
    """
    basename = os.path.basename(sample_dir.rstrip('/\\'))
    m = _FEFF_DIR_RE.search(basename)
    if m is None:
        return None
    mp_id   = m.group(1)                  # 'mp_1001571'
    site_nn = m.group(2).zfill(2)         # '01'
    return (mp_id, site_nn)


# ── 谱加载函数 ─────────────────────────────────────────────────────────────────

def load_xmu_xanes(xmu_path: str, e0: float,
                   n_points: int = XMU_N_POINTS,
                   emin_offset: float = XMU_EMIN_OFFSET,
                   emax_offset: float = XMU_EMAX_OFFSET) -> np.ndarray:
    """
    从 xmu.dat 加载 XANES 并插值到固定 n_points 网格。
    能量 = data[:,0]，mu(E) = data[:,3]（Step2.2 实测）。
    返回 shape (n_points,) float32。
    """
    data   = np.loadtxt(xmu_path, comments='#')
    energy = data[:, 0]
    mu     = data[:, 3]

    emin = max(e0 + emin_offset, energy[0])
    emax = min(e0 + emax_offset, energy[-1])
    if emin >= emax:
        warnings.warn(f"xmu.dat 能量窗口超出数据范围，使用原始范围: {xmu_path}")
        emin, emax = energy[0], energy[-1]

    e_grid = np.linspace(emin, emax, n_points)
    return np.interp(e_grid, energy, mu).astype(np.float32)


def load_chi1(chi1_path: str, n_points: int = CHI1_N_POINTS) -> np.ndarray:
    """
    从 chi1.dat 加载 chi(k) 并插值到固定 n_points 网格。
    k = data[:,0]，chi1 = data[:,1]。
    返回 shape (n_points,) float32。
    """
    data   = np.loadtxt(chi1_path, comments='#')
    k      = data[:, 0]
    chi1   = data[:, 1]
    k_grid = np.linspace(k[0], k[-1], n_points)
    return np.interp(k_grid, k, chi1).astype(np.float32)


# ── Dataset ───────────────────────────────────────────────────────────────────

class XASLocalStructureDataset(Dataset):
    """
    XAS -> 局部 Fe 结构 Dataset（Experiment 2，固定 20 邻居）。

    Parameters
    ----------
    data_root : str
    inventory_csv : str  — 含列 mp_id, folder_path, site_nn, E0
    ids_file : str       — 每行一个 mp_id
    feff_feat_csv : str  — tesst_feff_features_all_full_v4.csv
                           索引列 sample_dir 为完整路径，从末段解析 (mp_id, site_nn)
    feff_scaler_path : str or None
    L : float
    """

    def __init__(
        self,
        data_root: str,
        inventory_csv: str,
        ids_file: str,
        feff_feat_csv: str,
        feff_scaler_path: str = None,
        L: float = L,
    ):
        super().__init__()
        self.data_root = data_root
        self.L = L

        # ID 列表
        with open(ids_file, 'r') as f:
            self.ids = [ln.strip() for ln in f if ln.strip()]

        # inventory
        inv = pd.read_csv(inventory_csv)
        inv.columns = [c.strip() for c in inv.columns]
        self.inv = inv.set_index('mp_id')

        # feff_features：以 (mp_id, site_nn) 为键建立查找字典
        self.feat_df, self.feff_lookup = self._load_feff(feff_feat_csv)

        # scaler
        self.feff_scaler = None
        if feff_scaler_path and os.path.exists(feff_scaler_path):
            import pickle
            with open(feff_scaler_path, 'rb') as f:
                self.feff_scaler = pickle.load(f)

        self.samples = self._build_sample_list()

    # -------------------------------------------------------------------------

    def _load_feff(self, feff_feat_csv: str):
        """
        读取 feff CSV，建立 (mp_id, site_nn) -> row_index 字典。
        只保留 _feff_Fe_site_ 行。
        """
        feat_df = pd.read_csv(feff_feat_csv)
        feat_df.columns = [c.strip() for c in feat_df.columns]

        lookup = {}
        kept   = []
        for idx, row in feat_df.iterrows():
            sample_dir = str(row['sample_dir'])
            key = _parse_feff_key(sample_dir)
            if key is None:
                continue   # 非 Fe 位点，跳过
            mp_id, site_nn = key
            lookup[(mp_id, site_nn)] = idx
            kept.append(idx)

        feat_df_fe = feat_df.loc[kept].copy()
        print(f"[feff] 加载 {len(feat_df_fe)} 条 Fe 位点记录（共 {len(feat_df)} 行）")
        return feat_df_fe, lookup

    # -------------------------------------------------------------------------

    def _build_sample_list(self):
        samples = []
        missing = []
        for mp_id in self.ids:
            if mp_id not in self.inv.index:
                missing.append(mp_id)
                continue
            row         = self.inv.loc[mp_id]
            site_nn     = str(row['site_nn']).zfill(2)
            folder_path = str(row['folder_path'])
            e0          = float(row['E0']) if 'E0' in self.inv.columns else FE_K_EDGE_DEFAULT
            samples.append((mp_id, site_nn, folder_path, e0))

        if missing:
            warnings.warn(
                f"{len(missing)} 个 mp_id 不在 inventory 中，已跳过。示例：{missing[:3]}")
        return samples

    # -------------------------------------------------------------------------

    def _get_folder(self, mp_id: str, folder_path: str) -> str:
        if os.path.isabs(folder_path) and os.path.isdir(folder_path):
            return folder_path
        candidate = os.path.join(self.data_root, folder_path)
        if os.path.isdir(candidate):
            return candidate
        for name in os.listdir(self.data_root):
            if mp_id in name:
                full = os.path.join(self.data_root, name)
                if os.path.isdir(full):
                    return full
        raise FileNotFoundError(
            f"找不到 mp_id={mp_id} 的文件夹（folder_path={folder_path}）")

    # -------------------------------------------------------------------------

    def _load_local_structure(self, folder: str, site_nn: str):
        poscar_path = None
        for name in ('POSCAR_supercell_fixed', 'POSCAR_supercell',
                     'POSCAR_fixed', 'POSCAR'):
            p = os.path.join(folder, name)
            if os.path.exists(p):
                poscar_path = p
                break
        if poscar_path is None:
            raise FileNotFoundError(f"找不到 POSCAR 文件：{folder}")

        supercell = Structure.from_file(poscar_path)
        analyzer  = SpacegroupAnalyzer(supercell, symprec=0.1)
        primitive = analyzer.get_primitive_standard_structure()

        fe_indices = [i for i, s in enumerate(primitive)
                      if s.specie.symbol == 'Fe']
        if not fe_indices:
            raise ValueError(f"原胞中未找到 Fe 原子：{folder}")

        site_idx = int(site_nn) - 1
        if site_idx >= len(fe_indices):
            warnings.warn(
                f"{folder}: site_idx={site_idx} 越界"
                f"（原胞 Fe 数={len(fe_indices)}），退回 0")
            site_idx = 0
        fe_index = fe_indices[site_idx]

        neighbors = primitive.get_neighbors(
            primitive[fe_index], r=NEIGHBOR_SEARCH_R)
        if len(neighbors) < N_NEIGHBORS:
            return None

        neighbors_sorted = sorted(
            neighbors, key=lambda x: x.nn_distance)[:N_NEIGHBORS]

        fe_cart        = primitive[fe_index].coords
        neighbor_carts = np.array(
            [n.coords - fe_cart for n in neighbors_sorted])

        frac_coords = (neighbor_carts / self.L).copy()
        # ★ Step4d 修复：min-image 折叠到 [-0.5, 0.5]
        # handoff 设计意图："L=6 时 4Å 邻居 frac=0.67，中心化后=-0.33"
        # v5 原版未做此折叠，导致所有 >3Å 邻居被过滤，丢失率 98%
        # 折叠后坐标与 diffusion v3 sample() 的 x-round(x) 完全一致
        frac_coords = frac_coords - np.round(frac_coords)
        atom_types  = np.array(
            [n.specie.Z for n in neighbors_sorted]).copy()
        eval_cutoff = float(
            min(neighbors_sorted[N_NEIGHBORS - 1].nn_distance, 4.0))

        return frac_coords, atom_types, eval_cutoff

    # -------------------------------------------------------------------------

    def _load_spectrum(self, folder: str, mp_id: str, site_nn: str, e0: float):
        xmu_feat  = load_xmu_xanes(os.path.join(folder, 'xmu.dat'), e0)
        chi1_feat = load_chi1(os.path.join(folder, 'chi1.dat'))

        key = (mp_id, site_nn)
        if key in self.feff_lookup:
            row_idx = self.feff_lookup[key]
            feats   = self.feat_df.loc[row_idx].iloc[3:76].values.astype(np.float32)
        else:
            warnings.warn(
                f"feff_features 中未找到 key={key}，使用零向量")
            feats = np.zeros(FEFF_FEAT_NCOLS, dtype=np.float32)

        if len(feats) != FEFF_FEAT_NCOLS:
            buf = np.zeros(FEFF_FEAT_NCOLS, dtype=np.float32)
            buf[:min(len(feats), FEFF_FEAT_NCOLS)] = feats[:FEFF_FEAT_NCOLS]
            feats = buf

        if self.feff_scaler is not None:
            feats = self.feff_scaler.transform(
                feats.reshape(1, -1))[0].astype(np.float32)

        # ★ 拦截 NaN/Inf：CSV 原始缺失值或 scaler 零方差特征（std=0 → x/0=NaN）
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

        return xmu_feat, chi1_feat, feats

    # -------------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        mp_id, site_nn, folder_path, e0 = self.samples[index]

        try:
            folder = self._get_folder(mp_id, folder_path)
        except FileNotFoundError as e:
            warnings.warn(str(e))
            return None

        try:
            result = self._load_local_structure(folder, site_nn)
        except Exception as e:
            warnings.warn(f"结构加载失败 {mp_id}: {e}")
            return None

        if result is None:
            return None

        frac_coords, atom_types, eval_cutoff = result

        # ★ 安全检查：min-image 折叠后此条件不应触发
        # 若触发说明折叠逻辑有误，保留以便调试
        if np.abs(frac_coords).max() > 0.5:
            return None

        try:
            xmu_feat, chi1_feat, feats = self._load_spectrum(
                folder, mp_id, site_nn, e0)
        except Exception as e:
            warnings.warn(f"谱加载失败 {mp_id}: {e}")
            return None

        data = Data(
            frac_coords   = torch.tensor(frac_coords, dtype=torch.float32),
            atom_types    = torch.tensor(atom_types,  dtype=torch.long),
            lengths       = torch.tensor([self.L, self.L, self.L],   dtype=torch.float32).view(1, -1),
            angles        = torch.tensor([90., 90., 90.], dtype=torch.float32).view(1, -1),
            num_atoms     = N_NEIGHBORS,
            num_nodes     = N_NEIGHBORS,
            xmu_xanes     = torch.tensor(xmu_feat,  dtype=torch.float32).unsqueeze(0),  # (1,150)
            chi1          = torch.tensor(chi1_feat, dtype=torch.float32).unsqueeze(0),  # (1,200)
            feff_features = torch.tensor(feats,     dtype=torch.float32).unsqueeze(0),  # (1,73)
            eval_cutoff   = torch.tensor(eval_cutoff, dtype=torch.float32),
        )
        data.mp_id = mp_id
        return data

    def __repr__(self) -> str:
        return f"XASLocalStructureDataset(n_samples={len(self.samples)}, L={self.L}A)"


# ── 快速单样本测试 ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    DATA_ROOT = r"C:\Users\T-Cat\Desktop\DiffCSP-main\site_dataset_Fe_only_oxide_one_site"
    STEP1_DIR = r"C:\Users\T-Cat\Desktop\DiffCSP-main\experiment2\step1"
    FEFF_CSV  = r"C:\Users\T-Cat\Desktop\DiffCSP-main\tesst_feff_features_all_full_v4.csv"

    ds = XASLocalStructureDataset(
        data_root        = DATA_ROOT,
        inventory_csv    = os.path.join(STEP1_DIR, 'data_inventory.csv'),
        ids_file         = os.path.join(STEP1_DIR, 'train_ids.txt'),
        feff_feat_csv    = FEFF_CSV,
        feff_scaler_path = os.path.join(STEP1_DIR, 'feff_feature_scaler.pkl'),
    )
    print(f"Dataset 长度: {len(ds)}")

    hit = miss = 0
    for i in range(min(50, len(ds))):
        s = ds[i]
        if s is None:
            continue
        nz = s.feff_features.abs().sum().item()
        # 零向量经 scaler 后会产生固定值（约 71013），以此判断是否命中
        if nz > 80000 or nz < 1:
            miss += 1
        else:
            hit += 1

    print(f"前50样本 feff 命中率：{hit}/{hit+miss}")
    print()

    for i in range(3):
        s = ds[i]
        if s is not None:
            print(f"[{i}] mp_id={s.mp_id}  "
                  f"xmu_xanes={tuple(s.xmu_xanes.shape)}  "
                  f"feff_nonzero={s.feff_features.abs().sum().item():.2f}")