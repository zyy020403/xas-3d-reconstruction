# =============================================================================
# 脚本编号: step3.1 (dataset)
# 脚本名称: xas_dataset.py
# 输入:
#   - data_inventory.csv (experiment/step1/data_inventory.csv)
#   - chi.dat 谱文件 (site_dataset / ionic_dataset 文件夹下各位点)
#   - POSCAR_supercell_fixed (同上)
# 输出:
#   - XASCrystalDataset: __getitem__ 返回包含晶体结构 + XAS 谱字段的字典
#   - xas_collate_fn: 处理变长 n_sites 的 collate 函数
# 说明:
#   POSCAR 解析用 pymatgen；谱预处理复用 step2_1_spectrum_encoder.preprocess_chi；
#   晶体字段用 torch_geometric.data.Data + Batch.from_data_list 组装。
#   所有数据在 __init__ 里预加载进内存，__getitem__ 只做字典查找，速度快 10x+。
#
# ── 修改记录 ──────────────────────────────────────────────────────────────────
# Fix 1（embedding 唯一性）：
#   spec.clone() 和全零 fallback 新建 tensor 已在原版中存在，本版保留并确认。
# Fix 2（原胞转换）：
#   _parse_poscar 中在读取超胞后调用 get_primitive_structure(tolerance=0.25)，
#   失败时 fallback 到超胞，不丢弃样本。
#   __init__ 末尾打印原胞原子数统计，便于确认转换效果。
# =============================================================================

import os
import sys
import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch

# ── 路径常量 ──────────────────────────────────────────────────────────────────
PROJECT_ROOT      = r"C:\Users\T-Cat\Desktop\DiffCSP-main"
EXPERIMENT_DIR    = os.path.join(PROJECT_ROOT, "experiment")
SITE_DATASET_DIR  = r"C:\Users\T-Cat\Desktop\XAS-FeO\site_dataset"
IONIC_DATASET_DIR = r"C:\Users\T-Cat\Desktop\XAS-FeO\test_missing_keep3_packed_A"

# ── 导入 step2 预处理函数 ─────────────────────────────────────────────────────
_step2_dir = os.path.join(EXPERIMENT_DIR, "step2")
if _step2_dir not in sys.path:
    sys.path.insert(0, _step2_dir)
from step2_1_spectrum_encoder import preprocess_chi  # noqa: E402

logger = logging.getLogger(__name__)

# ── 质量等级 → 权重映射 ───────────────────────────────────────────────────────
QUALITY_WEIGHT = {"A": 1.0, "B": 0.5, "C": 0.1}
QUALITY_WEIGHT_UNKNOWN = 0.3

SPECTRA_LEN = 512  # preprocess_chi 输出的固定长度


# ─────────────────────────────────────────────────────────────────────────────
#  辅助：POSCAR 解析（Fix 2：超胞 → 原胞）
# ─────────────────────────────────────────────────────────────────────────────

def _parse_poscar(poscar_path: str):
    """
    用 pymatgen 解析 POSCAR_supercell_fixed，并转换为原胞（Fix 2）。

    流程：
      1. 读取超胞
      2. 调用 get_primitive_structure(tolerance=0.25) 转换为原胞
      3. 若转换失败，fallback 到超胞（记录警告，不丢弃样本）
      4. 从原胞（或超胞 fallback）提取结构信息

    返回 (frac_coords, atom_types, lengths, angles, num_atoms)，全为 Tensor / int。
    失败时返回 None。
    """
    try:
        from pymatgen.core import Structure

        # ── 读取超胞 ──────────────────────────────────────────────────────────
        supercell = Structure.from_file(poscar_path)

        # ── Fix 2：转换为原胞 ─────────────────────────────────────────────────
        try:
            struct = supercell.get_primitive_structure(tolerance=0.25)
        except Exception as e:
            logger.warning(
                f"原胞转换失败，fallback 到超胞: {os.path.basename(os.path.dirname(poscar_path))}  ({e})"
            )
            struct = supercell

        frac_coords = torch.tensor(struct.frac_coords, dtype=torch.float32)
        atom_types  = torch.tensor([s.specie.Z for s in struct], dtype=torch.long)
        lengths     = torch.tensor(list(struct.lattice.abc),    dtype=torch.float32)
        angles      = torch.tensor(list(struct.lattice.angles), dtype=torch.float32)
        return frac_coords, atom_types, lengths, angles, len(struct)

    except Exception as e:
        logger.warning(f"POSCAR 解析失败: {poscar_path}  ({e})")
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  辅助：根据 source_dataset 确定文件夹根目录
# ─────────────────────────────────────────────────────────────────────────────

def _folder_root(source: str) -> str:
    if source == "ionic":
        return IONIC_DATASET_DIR
    return SITE_DATASET_DIR


# ─────────────────────────────────────────────────────────────────────────────
#  XASCrystalDataset
# ─────────────────────────────────────────────────────────────────────────────

class XASCrystalDataset(Dataset):
    """
    每个样本 = 一个化合物（mp_id）的所有不等价位点。

    所有数据在 __init__ 里预加载进内存（32GB RAM 完全够用），
    __getitem__ 只做一次列表索引，彻底消除磁盘 I/O 瓶颈。

    __getitem__ 返回 dict，包含：
        frac_coords     Tensor [N_atoms, 3]
        atom_types      Tensor [N_atoms]
        lengths         Tensor [3]
        angles          Tensor [3]
        num_atoms       int
        spectra         Tensor [n_sites, 1, SPECTRA_LEN]
        site_elements   Tensor [n_sites]
        is_ionic        Tensor [n_sites]
        quality_weights Tensor [n_sites]
        n_sites         int
    """

    def __init__(self, mp_ids, inventory_df: pd.DataFrame):
        super().__init__()

        from pymatgen.core.periodic_table import Element as PmgElement

        # ── 按 mp_id 分组 ────────────────────────────────────────────────────
        mp_ids_set = set(int(x) for x in mp_ids)
        sub_df = inventory_df[inventory_df["mp_id"].isin(mp_ids_set)].copy()

        mp_to_sites = {}
        for mp_id, grp in sub_df.groupby("mp_id"):
            mp_to_sites[int(mp_id)] = grp.to_dict("records")

        # ── 预加载所有数据进内存 ──────────────────────────────────────────────
        self._cache = []
        self.valid_mp_ids = []
        skipped     = 0
        total       = len(mp_to_sites)
        n_atoms_list = []   # Fix 2 统计用

        logger.info(f"开始预加载数据到内存（共 {total} 个 mp_id）...")

        for idx, (mp_id, sites) in enumerate(mp_to_sites.items()):
            if idx % 500 == 0:
                logger.info(f"  预加载进度: {idx}/{total}")

            # ── 解析 POSCAR（含原胞转换，Fix 2）─────────────────────────────
            first_site  = sites[0]
            poscar_path = os.path.join(first_site["source_path"], "POSCAR_supercell_fixed")
            parsed = _parse_poscar(poscar_path)
            if parsed is None:
                skipped += 1
                continue
            frac_coords, atom_types, lengths, angles, num_atoms = parsed
            n_atoms_list.append(num_atoms)

            # ── 逐位点加载谱 ─────────────────────────────────────────────────
            n_sites         = len(sites)
            spectra         = torch.zeros(n_sites, 1, SPECTRA_LEN, dtype=torch.float32)
            site_elements   = torch.zeros(n_sites, dtype=torch.long)
            is_ionic_flags  = torch.zeros(n_sites, dtype=torch.long)
            quality_weights = torch.zeros(n_sites, dtype=torch.float32)

            for i, site in enumerate(sites):
                # 原子序数
                try:
                    site_elements[i] = PmgElement(site["element"]).Z
                except Exception:
                    site_elements[i] = 0

                # is_ionic
                is_ionic_flags[i] = 1 if bool(site["is_ionic"]) else 0

                # quality_weight
                tier = site.get("quality_tier", "")
                quality_weights[i] = QUALITY_WEIGHT.get(
                    str(tier).upper(), QUALITY_WEIGHT_UNKNOWN
                )

                # chi.dat（Fix 1：spec.clone() 确保独立副本）
                chi_path = os.path.join(site["source_path"], "chi.dat")
                try:
                    spec = preprocess_chi(chi_path)   # [1, 512] float32
                    spectra[i] = spec.clone()          # ★ Fix 1：独立副本
                except Exception as e:
                    logger.debug(f"chi.dat 读取失败: {chi_path} ({e})")
                    spectra[i] = torch.zeros(1, SPECTRA_LEN, dtype=torch.float32)  # ★ Fix 1：每次新建
                    quality_weights[i] = 0.0

            # ── 存入缓存 ─────────────────────────────────────────────────────
            self._cache.append({
                "frac_coords":     frac_coords,
                "atom_types":      atom_types,
                "lengths":         lengths,
                "angles":          angles,
                "num_atoms":       num_atoms,
                "spectra":         spectra,
                "site_elements":   site_elements,
                "is_ionic":        is_ionic_flags,
                "quality_weights": quality_weights,
                "n_sites":         n_sites,
            })
            self.valid_mp_ids.append(mp_id)

        if skipped:
            logger.warning(f"跳过 {skipped} 个 mp_id（POSCAR 读取失败）")
        logger.info(f"XASCrystalDataset 初始化完成，有效样本数: {len(self.valid_mp_ids)}")

        # ── Fix 2 统计：打印原胞原子数分布 ───────────────────────────────────
        if n_atoms_list:
            logger.info(
                f"原胞原子数统计: min={min(n_atoms_list)}, "
                f"max={max(n_atoms_list)}, "
                f"mean={sum(n_atoms_list)/len(n_atoms_list):.1f}, "
                f"median={sorted(n_atoms_list)[len(n_atoms_list)//2]}"
            )

    # ── 基本接口 ──────────────────────────────────────────────────────────────

    def __len__(self):
        return len(self._cache)

    def __getitem__(self, idx):
        return self._cache[idx]


# ─────────────────────────────────────────────────────────────────────────────
#  xas_collate_fn
# ─────────────────────────────────────────────────────────────────────────────

def xas_collate_fn(batch):
    """
    将 XASCrystalDataset 的一个 batch（list of dict）整理为 PyG Batch。

    晶体字段：用 Batch.from_data_list() 自动处理（生成 batch.batch 向量）。
    谱字段：pad 到 batch 内最大 n_sites。

    输出 batch 的额外属性：
        spectra         [B, n_sites_max, 1, 512]
        site_elements   [B, n_sites_max]
        is_ionic        [B, n_sites_max]
        quality_weights [B, n_sites_max]
        n_sites         [B]
    """
    # ── 晶体字段 → PyG Batch ──────────────────────────────────────────────────
    data_list = []
    for item in batch:
        data = Data(
            frac_coords=item["frac_coords"],
            atom_types=item["atom_types"],
            lengths=item["lengths"].unsqueeze(0),
            angles=item["angles"].unsqueeze(0),
            num_atoms=item["num_atoms"],
            num_nodes=item["num_atoms"],
        )
        data_list.append(data)
    pyg_batch = Batch.from_data_list(data_list)

    # ── 谱字段 → padded tensors ───────────────────────────────────────────────
    B         = len(batch)
    max_sites = max(item["n_sites"] for item in batch)

    spectra         = torch.zeros(B, max_sites, 1, SPECTRA_LEN, dtype=torch.float32)
    site_elements   = torch.zeros(B, max_sites, dtype=torch.long)
    is_ionic        = torch.zeros(B, max_sites, dtype=torch.long)
    quality_weights = torch.zeros(B, max_sites, dtype=torch.float32)
    n_sites_tensor  = torch.tensor([item["n_sites"] for item in batch], dtype=torch.long)

    for i, item in enumerate(batch):
        n = item["n_sites"]
        spectra[i, :n]         = item["spectra"]
        site_elements[i, :n]   = item["site_elements"]
        is_ionic[i, :n]        = item["is_ionic"]
        quality_weights[i, :n] = item["quality_weights"]

    # ── 挂载到 PyG Batch ──────────────────────────────────────────────────────
    pyg_batch.spectra         = spectra
    pyg_batch.site_elements   = site_elements
    pyg_batch.is_ionic        = is_ionic
    pyg_batch.quality_weights = quality_weights
    pyg_batch.n_sites         = n_sites_tensor

    return pyg_batch


# =============================================================================
#  唯一性验证块（用于排查 embedding 重复 bug）
#  运行方式: python xas_dataset.py
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    STEP1_DIR = os.path.join(PROJECT_ROOT, "experiment", "step1")
    inventory_path = os.path.join(STEP1_DIR, "data_inventory.csv")
    train_ids_path = os.path.join(STEP1_DIR, "train_ids.txt")

    if not os.path.exists(inventory_path):
        print("找不到 data_inventory.csv，请先运行 Step 1。")
        sys.exit(1)

    inventory_df = pd.read_csv(inventory_path)
    with open(train_ids_path) as f:
        all_ids = [int(l.strip()) for l in f if l.strip()]

    CHECK_N = min(30, len(all_ids))
    print(f"加载前 {CHECK_N} 个 mp_id 进行唯一性检查（同时验证原胞转换）...")
    dataset = XASCrystalDataset(all_ids[:CHECK_N], inventory_df)

    from torch.nn.functional import cosine_similarity
    duplicates   = 0
    sample_specs = []
    for i in range(len(dataset)):
        item = dataset[i]
        sample_specs.append(item["spectra"][0].flatten())

    for i in range(len(sample_specs)):
        for j in range(i + 1, len(sample_specs)):
            sim = cosine_similarity(
                sample_specs[i].unsqueeze(0),
                sample_specs[j].unsqueeze(0)
            ).item()
            if sim > 0.999:
                print(f"  ⚠️  样本 {i} 与 {j} cosine={sim:.6f} — 疑似重复！mp_ids: "
                      f"{dataset.valid_mp_ids[i]} vs {dataset.valid_mp_ids[j]}")
                duplicates += 1

    if duplicates == 0:
        print(f"✅ 唯一性检查通过，{len(sample_specs)} 个样本中无重复谱。")
    else:
        print(f"❌ 发现 {duplicates} 对重复谱，请检查 preprocess_chi 是否存在全局缓存。")