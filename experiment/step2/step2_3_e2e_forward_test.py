# =============================================================================
# 脚本编号: step2.3
# 脚本名称: step2.3_e2e_forward_test.py
# 输入:
#   - experiment/step1/data_inventory.csv
#   - 各位点文件夹下的 chi.dat 文件
#   - step2.1_spectrum_encoder.py    (SpectrumEncoder, preprocess_chi)
#   - step2.2_multisite_aggregator.py (MultiSiteAggregator, collate_multisite_batch)
# 输出:
#   - experiment/step2/e2e_test_log.txt   端到端测试日志
# 说明:
#   从 data_inventory.csv 中取 4 个不同 mp_id（位点数各异），
#   跑通完整的 preprocess_chi → SpectrumEncoder → collate → MultiSiteAggregator
#   前向传播，验证数据流无误，并检查排列不变性。
#   若清单文件不存在，则用合成数据替代（保证脚本可独立运行测试）。
# =============================================================================

import os
import sys
import logging
import random
from io import StringIO
from contextlib import redirect_stdout

import numpy as np
import torch
import pandas as pd

# ---------------------------------------------------------------------------
# 路径设置：将 step2 目录加入 sys.path 以 import 上游模块
# ---------------------------------------------------------------------------

PROJECT_ROOT   = r"C:\Users\T-Cat\Desktop\DiffCSP-main"
EXPERIMENT_DIR = os.path.join(PROJECT_ROOT, "experiment")
STEP1_DIR      = os.path.join(EXPERIMENT_DIR, "step1")
STEP2_DIR      = os.path.join(EXPERIMENT_DIR, "step2")
os.makedirs(STEP2_DIR, exist_ok=True)

# 将 step2 目录加入 path（以便 import）
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# 也加入 STEP2_DIR（当从其他目录运行时）
if STEP2_DIR not in sys.path:
    sys.path.insert(0, STEP2_DIR)

from step2_1_spectrum_encoder import (    # noqa: E402  — 可能需按实际文件名调整
    preprocess_chi, SpectrumEncoder, count_parameters as count_enc
)
from step2_2_multisite_aggregator import (  # noqa: E402
    MultiSiteAggregator, collate_multisite_batch, count_parameters as count_agg
)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

HOLDOUT_PATH = os.path.join(STEP1_DIR, "holdout_1000_ids.txt")
INVENTORY_PATH = os.path.join(STEP1_DIR, "data_inventory.csv")

D_SITE   = 256
D_STRUCT = 256

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 辅助：加载 holdout 集合（需排除）
# ---------------------------------------------------------------------------

def load_holdout_ids() -> set:
    if not os.path.isfile(HOLDOUT_PATH):
        return set()
    with open(HOLDOUT_PATH, "r") as f:
        return {line.strip() for line in f if line.strip()}


# ---------------------------------------------------------------------------
# 辅助：合成 4 个 mp_id 的位点数据（inventory 不存在时使用）
# ---------------------------------------------------------------------------

def make_synthetic_data():
    """返回 list of dict，每个 dict 模拟一个 mp_id 的所有位点信息"""
    import tempfile

    cases = []
    for mp_id, n_sites in [("mp_SYN_001", 1), ("mp_SYN_002", 2),
                            ("mp_SYN_003", 3), ("mp_SYN_004", 5)]:
        sites = []
        for site_idx in range(n_sites):
            # 写合成 chi.dat
            k_vals   = np.linspace(0.5, 18.0, 200)
            chi_vals = np.sin(k_vals + site_idx) * np.exp(-0.05 * k_vals)
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".dat", delete=False, encoding="utf-8"
            )
            tmp.write("# synthetic\n")
            for k, c in zip(k_vals, chi_vals):
                tmp.write(f"{k:.4f}  {c:.6f}\n")
            tmp.close()
            sites.append({
                "chi_path": tmp.name,
                "atomic_number": random.choice([26, 14, 8, 3, 38]),
                "is_ionic": random.choice([0, 1]),
                "quality_tier": random.choice(["A", "B", "C"]),
            })
        cases.append({"mp_id": mp_id, "sites": sites})
    return cases


# ---------------------------------------------------------------------------
# 辅助：从 inventory 读取 4 个不同位点数的 mp_id
# ---------------------------------------------------------------------------

QUALITY_WEIGHT_MAP = {"A": 1.0, "B": 0.5, "C": 0.1}

def load_real_data(holdout_ids: set):
    """
    从 data_inventory.csv 中选 4 个不同 mp_id（尽量位点数各异），
    返回与 make_synthetic_data() 相同格式的列表。
    """
    df = pd.read_csv(INVENTORY_PATH)

    # 过滤保留集
    if "mp_id" in df.columns:
        df = df[~df["mp_id"].astype(str).isin(holdout_ids)]

    # 确定 source_path 列名
    path_col = None
    for c in ["source_path", "folder_path", "folder"]:
        if c in df.columns:
            path_col = c
            break
    if path_col is None:
        raise ValueError("inventory 中未找到路径列（source_path/folder_path/folder）")

    # 按 mp_id 分组
    grouped = df.groupby("mp_id") if "mp_id" in df.columns else {None: df}
    mp_groups = {}
    if "mp_id" in df.columns:
        for mpid, grp in df.groupby("mp_id"):
            mp_groups[str(mpid)] = grp
    else:
        mp_groups = {"unknown": df}

    # 找 4 个位点数各异的 mp_id
    by_n = {}
    for mpid, grp in mp_groups.items():
        n = len(grp)
        if n not in by_n:
            by_n[n] = (mpid, grp)

    selected = []
    # 优先取位点数 1, 2, 3, 5+（或随机取 4 个）
    preferred = [1, 2, 3, 5]
    for target_n in preferred:
        if target_n in by_n and len(selected) < 4:
            selected.append(by_n[target_n])

    # 不够 4 个就随机补充
    all_items = list(mp_groups.items())
    random.shuffle(all_items)
    for mpid, grp in all_items:
        if len(selected) >= 4:
            break
        if not any(s[0] == mpid for s in selected):
            selected.append((mpid, grp))

    # 组装
    cases = []
    for mpid, grp in selected[:4]:
        sites = []
        for _, row in grp.iterrows():
            chi_path = os.path.join(str(row[path_col]), "chi.dat")
            atomic_num = int(row.get("atomic_number", 26))
            is_ionic   = int(row.get("is_ionic", 0))
            tier       = str(row.get("quality_tier", "A"))
            sites.append({
                "chi_path": chi_path,
                "atomic_number": atomic_num,
                "is_ionic": is_ionic,
                "quality_tier": tier,
            })
        cases.append({"mp_id": mpid, "sites": sites})
    return cases


# ---------------------------------------------------------------------------
# 主测试流程
# ---------------------------------------------------------------------------

def run_e2e_test(output_lines: list):
    def log(msg):
        print(msg)
        output_lines.append(msg)

    log("=" * 60)
    log("=== Step 2 端到端测试 ===")
    log("=" * 60)

    # ── 模型实例化 ─────────────────────────────────────────────────────────
    encoder    = SpectrumEncoder(d_site=D_SITE)
    aggregator = MultiSiteAggregator(d_site=D_SITE, d_struct=D_STRUCT)
    encoder.eval()
    aggregator.eval()

    enc_params = count_enc(encoder)
    agg_params = count_agg(aggregator)
    log(f"\n[模型参数量]")
    log(f"  SpectrumEncoder   : {enc_params:,}  ({enc_params/1e4:.1f} 万)")
    log(f"  MultiSiteAggregator: {agg_params:,}  ({agg_params/1e4:.1f} 万)")
    log(f"  合计              : {(enc_params+agg_params):,}  ({(enc_params+agg_params)/1e4:.1f} 万)")

    # ── 数据加载 ───────────────────────────────────────────────────────────
    holdout_ids = load_holdout_ids()
    log(f"\n[保留集] 已加载 {len(holdout_ids)} 个 holdout mp_id")

    using_synthetic = False
    if os.path.isfile(INVENTORY_PATH):
        log(f"[数据来源] 真实数据: {INVENTORY_PATH}")
        try:
            cases = load_real_data(holdout_ids)
        except Exception as e:
            log(f"  警告: 加载真实数据失败 ({e})，切换为合成数据")
            cases = make_synthetic_data()
            using_synthetic = True
    else:
        log(f"[数据来源] data_inventory.csv 未找到，使用合成数据")
        cases = make_synthetic_data()
        using_synthetic = True

    # ── 逐 mp_id 编码 ─────────────────────────────────────────────────────
    log(f"\n[逐位点编码]")
    all_site_embs  = []
    all_qw_tensors = []

    for case in cases:
        mp_id  = case["mp_id"]
        sites  = case["sites"]
        n_sites = len(sites)

        site_embs = []
        qw_vals   = []

        for site in sites:
            # 预处理谱
            spec = preprocess_chi(site["chi_path"])   # [1, 512]

            # 编码
            with torch.no_grad():
                emb = encoder(
                    spec.unsqueeze(0),                        # [1, 1, 512]
                    torch.tensor([site["atomic_number"]]),    # [1]
                    torch.tensor([site["is_ionic"]]),         # [1]
                )                                             # [1, 256]
            site_embs.append(emb.squeeze(0))          # [256]

            tier = site.get("quality_tier", "A")
            qw_vals.append(QUALITY_WEIGHT_MAP.get(tier, 1.0))

        site_embs_tensor = torch.stack(site_embs, dim=0)   # [n_sites, 256]
        qw_tensor        = torch.tensor(qw_vals)            # [n_sites]

        log(f"  mp_id={mp_id}: {n_sites} sites "
            f"→ site_embeddings shape: {list(site_embs_tensor.shape)}")
        assert site_embs_tensor.shape == (n_sites, D_SITE), "site_emb shape 错误"

        all_site_embs.append(site_embs_tensor)
        all_qw_tensors.append(qw_tensor)

    # ── collate ──────────────────────────────────────────────────────────
    padded, padding_mask, quality_weights = collate_multisite_batch(
        all_site_embs, all_qw_tensors
    )
    n_max = padded.shape[1]
    log(f"\n[Batch collation]")
    log(f"  padded shape       : {list(padded.shape)}")         # [4, N_max, 256]
    log(f"  padding_mask shape : {list(padding_mask.shape)}")   # [4, N_max]
    log(f"  quality_weights    : {list(quality_weights.shape) if quality_weights is not None else None}")

    # ── 聚合 ─────────────────────────────────────────────────────────────
    with torch.no_grad():
        struct_embs = aggregator(padded, padding_mask, quality_weights)

    log(f"\n[Structure embeddings]")
    log(f"  shape    : {list(struct_embs.shape)}")              # [4, 256]
    log(f"  min      : {struct_embs.min().item():.4f}")
    log(f"  max      : {struct_embs.max().item():.4f}")
    log(f"  has_nan  : {torch.isnan(struct_embs).any().item()}")
    log(f"  has_inf  : {torch.isinf(struct_embs).any().item()}")

    assert struct_embs.shape == (len(cases), D_STRUCT), "structure embedding shape 错误"
    assert not torch.isnan(struct_embs).any(), "structure embedding 包含 NaN！"
    assert not torch.isinf(struct_embs).any(), "structure embedding 包含 Inf！"

    # ── 排列不变性验证（对第一个样本，N > 1 时才有意义）─────────────────────
    log(f"\n[排列不变性验证]")
    test_idx = next((i for i, c in enumerate(cases) if len(c["sites"]) > 1), None)
    if test_idx is not None:
        n_i = len(cases[test_idx]["sites"])
        single_emb = all_site_embs[test_idx].unsqueeze(0)          # [1, n_i, 256]
        single_mask = padding_mask[test_idx:test_idx+1, :n_i]      # [1, n_i]（全 False）

        # 补充无 padding 的 mask
        no_pad_mask = torch.zeros(1, n_i, dtype=torch.bool)

        with torch.no_grad():
            emb_orig = aggregator(single_emb, no_pad_mask)
            perm     = torch.randperm(n_i)
            emb_perm = aggregator(single_emb[:, perm, :], no_pad_mask[:, perm])

        diff = (emb_orig - emb_perm).abs().max().item()
        log(f"  mp_id={cases[test_idx]['mp_id']}, N={n_i}, "
            f"max diff after permutation: {diff:.2e}")
        assert diff < 1e-4, f"排列不变性验证失败！diff={diff}"
        log("  排列不变性验证: PASSED ✓")
    else:
        log("  所有样本 N=1，跳过排列不变性验证（不适用）")

    # ── 数据来源提示 ──────────────────────────────────────────────────────
    if using_synthetic:
        log(f"\n[注意] 本次测试使用合成数据，请在 data_inventory.csv 就绪后重新运行")

    log("\n" + "=" * 60)
    log("=== 测试通过 ✓ ===")
    log("=" * 60)


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    output_lines = []

    # 捕获同时打印到控制台
    run_e2e_test(output_lines)

    # 写日志文件
    log_path = os.path.join(STEP2_DIR, "e2e_test_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines) + "\n")

    print(f"\n日志已保存至: {log_path}")