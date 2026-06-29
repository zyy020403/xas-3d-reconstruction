# verify_mp_spectrum_source_v2.py
# 修复版：mp-api 导入 fallback + 详细错误打印
# ---------------------------------------------------------

import os
import sys
import traceback
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ========== 路径 ==========
EXP4_ROOT   = r"C:\Users\T-Cat\Desktop\DiffCSP-main\experiment4"
DATA_ROOT   = os.path.join(EXP4_ROOT, "data")
XMU_DIR     = os.path.join(DATA_ROOT, r"MP_all_EXAFS_only_csv\MP_all_EXAFS_only_csv")
INVENTORY   = os.path.join(EXP4_ROOT, "step1", "data_inventory.csv")
OUT_DIR     = os.path.join(EXP4_ROOT, "verify_mp")
os.makedirs(OUT_DIR, exist_ok=True)

# ========== MP API key ==========
# 去 https://next-gen.materialsproject.org/api 登录后复制
MP_API_KEY = "miyEj2gxygXLK8vPJEJRdS3bi8345IxK"    # <-- 必须填！

# ========== 探针样本 ==========
PROBES = [
    # (mp_id, center_element, n_center_sites, tag_from_phase_c)
    ("mp-20846",  "Ca", 8,  "INCOMPAT"),
    ("mp-775145", "O",  16, "INCOMPAT"),
    ("mp-560146", "Y",  16, "INCOMPAT"),
    ("mp-559833", "Na", 16, "INCOMPAT"),
]


def try_import_mprester():
    """尝试多种方式导入 MPRester，任一成功即返回"""
    errors = []
    
    # 方式 1：新版 mp-api (推荐)
    try:
        from mp_api.client import MPRester
        return MPRester, "mp_api.client"
    except Exception as e:
        errors.append(f"  mp_api.client: {type(e).__name__}: {e}")
    
    # 方式 2：旧版 mp-api 兼容 shim
    try:
        from mp_api import MPRester
        return MPRester, "mp_api (top-level)"
    except Exception as e:
        errors.append(f"  mp_api: {type(e).__name__}: {e}")
    
    # 方式 3：pymatgen 自带
    try:
        from pymatgen.ext.matproj import MPRester
        return MPRester, "pymatgen.ext.matproj (legacy)"
    except Exception as e:
        errors.append(f"  pymatgen.ext.matproj: {type(e).__name__}: {e}")
    
    return None, errors


def load_local_spectrum(sample_name):
    xmu_path = os.path.join(XMU_DIR, f"{sample_name}.csv")
    if not os.path.isfile(xmu_path):
        return None, None, f"FILE_NOT_FOUND: {xmu_path}"
    df = pd.read_csv(xmu_path)
    if "x" not in df.columns or "y" not in df.columns:
        return None, None, f"BAD_COLUMNS: {df.columns.tolist()}"
    return df["x"].values, df["y"].values, None


def fetch_mp_xas(mpr, mp_id, element):
    """适配新旧 API 都试一下"""
    attempts = []
    
    # 新版 API：mpr.xas.search(...)
    try:
        if hasattr(mpr, "xas"):
            docs = mpr.xas.search(
                material_ids=[mp_id],
                absorbing_element=element,
                edge="K",
            )
            results = []
            for d in docs:
                spec = getattr(d, "spectrum", None)
                results.append({
                    "source_api": "mpr.xas.search",
                    "xas_id": getattr(d, "xas_id", None),
                    "absorbing_index": getattr(d, "absorbing_index", None),
                    "spectrum_type": getattr(d, "spectrum_type", None),
                    "E": np.array(spec.x) if spec is not None and hasattr(spec, "x") else None,
                    "mu": np.array(spec.y) if spec is not None and hasattr(spec, "y") else None,
                    "raw_doc_fields": list(d.model_dump().keys()) if hasattr(d, "model_dump") else None,
                })
            return results, None
    except Exception as e:
        attempts.append(f"mpr.xas.search failed: {type(e).__name__}: {e}")
    
    # 尝试旧版 get_xas_data
    try:
        if hasattr(mpr, "get_xas_data"):
            data = mpr.get_xas_data(mp_id, element, edge="K")
            # data 结构因版本而异，直接打印结构
            return [{"source_api": "mpr.get_xas_data", "raw": data}], None
    except Exception as e:
        attempts.append(f"mpr.get_xas_data failed: {type(e).__name__}: {e}")
    
    return None, "\n".join(attempts)


def compare_spectra(E1, mu1, E2, mu2):
    if E1 is None or E2 is None or len(E1) < 10 or len(E2) < 10:
        return None
    mask = (E1 >= E2.min()) & (E1 <= E2.max())
    if mask.sum() < 10:
        return {"overlap_points": int(mask.sum()), "status": "NO_OVERLAP"}
    E_common = E1[mask]
    mu1_common = mu1[mask]
    mu2_on_E1 = np.interp(E_common, E2, mu2)
    mu1_norm = (mu1_common - mu1_common.mean()) / (mu1_common.std() + 1e-9)
    mu2_norm = (mu2_on_E1 - mu2_on_E1.mean()) / (mu2_on_E1.std() + 1e-9)
    rmse = float(np.sqrt(np.mean((mu1_norm - mu2_norm) ** 2)))
    corr = float(np.corrcoef(mu1_common, mu2_on_E1)[0, 1])
    return {
        "overlap_points": int(mask.sum()),
        "normalized_rmse": rmse,
        "pearson_r": corr,
    }


def main():
    print("=" * 70)
    print("MP XAS Source Verification v2")
    print("=" * 70)

    if MP_API_KEY == "YOUR_API_KEY_HERE":
        print("\n❌ MP_API_KEY 没填!")
        print("   去 https://next-gen.materialsproject.org/api 拿 API key")
        print(f"   填到脚本第 19 行")
        sys.exit(1)

    # 导入 MPRester
    print("\n[1/4] 尝试导入 MPRester...")
    MPRester, source = try_import_mprester()
    if MPRester is None:
        print("❌ 所有导入方式都失败了:")
        for e in source:
            print(e)
        print("\n建议:")
        print("  pip install --upgrade mp-api --break-system-packages")
        print("  或")
        print("  pip install --upgrade 'mp-api>=0.40' --break-system-packages")
        sys.exit(1)
    print(f"✅ 从 {source} 导入成功")

    # 连接
    print("\n[2/4] 连接 MP...")
    try:
        mpr = MPRester(MP_API_KEY)
        print(f"✅ 连接成功，类型: {type(mpr).__name__}")
    except Exception as e:
        print(f"❌ 连接失败:")
        traceback.print_exc()
        sys.exit(1)

    # 检查可用方法
    print("\n[3/4] MPRester 可用子模块/方法:")
    methods = [m for m in dir(mpr) if not m.startswith("_")]
    xas_methods = [m for m in methods if "xas" in m.lower()]
    print(f"  XAS 相关: {xas_methods}")
    print(f"  有 .xas 属性: {hasattr(mpr, 'xas')}")
    print(f"  有 .get_xas_data: {hasattr(mpr, 'get_xas_data')}")

    # 处理探针
    print(f"\n[4/4] 处理 {len(PROBES)} 个探针样本...")
    all_results = []
    
    for mp_id, element, n_sites, phase_c_tag in PROBES:
        sample_name = f"{mp_id}__{mp_id}-EXAFS-{element}-K"
        print(f"\n--- {sample_name}  (n_sites={n_sites}, phase_c={phase_c_tag}) ---")

        E_local, mu_local, err = load_local_spectrum(sample_name)
        if err:
            print(f"  本地读取失败: {err}")
            continue
        print(f"  本地: {len(E_local)} 点, E ∈ [{E_local[0]:.1f}, {E_local[-1]:.1f}] eV, "
              f"μ_range=[{mu_local.min():.3f}, {mu_local.max():.3f}]")

        print(f"  拉 MP...")
        mp_data, err_msg = fetch_mp_xas(mpr, mp_id, element)
        
        if mp_data is None:
            print(f"  ❌ MP 错误:\n    {err_msg}")
            all_results.append({
                "sample_name": sample_name, "mp_id": mp_id, "element": element,
                "n_sites": n_sites, "phase_c_tag": phase_c_tag,
                "n_mp_spectra": -1, "error": err_msg
            })
            continue

        n_mp = len(mp_data)
        print(f"  ✅ MP 返回 {n_mp} 条")

        if n_mp == 0:
            all_results.append({
                "sample_name": sample_name, "mp_id": mp_id, "element": element,
                "n_sites": n_sites, "phase_c_tag": phase_c_tag,
                "n_mp_spectra": 0,
                "interpretation": "MP 上没这条数据—师兄自己跑的 FEFF",
            })
            continue

        # 打印 MP 每条谱的 metadata
        for i, d in enumerate(mp_data):
            print(f"    MP #{i}: type={d.get('spectrum_type')}, "
                  f"site={d.get('absorbing_index')}, "
                  f"xas_id={d.get('xas_id')}, "
                  f"api={d.get('source_api')}")
            if d.get("raw_doc_fields"):
                print(f"      doc fields: {d['raw_doc_fields']}")

        # 相似度对比
        similarities = []
        for i, d in enumerate(mp_data):
            if d.get("E") is None:
                continue
            sim = compare_spectra(E_local, mu_local, d["E"], d["mu"])
            if sim and "pearson_r" in sim:
                sim["mp_index"] = i
                sim["spectrum_type"] = d.get("spectrum_type")
                sim["absorbing_index"] = d.get("absorbing_index")
                similarities.append(sim)

        best = None
        if similarities:
            best = max(similarities, key=lambda s: s["pearson_r"])
            print(f"  最匹配: MP #{best['mp_index']} "
                  f"(type={best['spectrum_type']}, site={best['absorbing_index']}), "
                  f"pearson={best['pearson_r']:.4f}, rmse={best['normalized_rmse']:.4f}")

        # 画图
        try:
            fig, ax = plt.subplots(1, 1, figsize=(11, 6))
            ax.plot(E_local, mu_local, "k-", lw=2.2, label="本地 (你的数据)")
            for i, d in enumerate(mp_data):
                if d.get("E") is not None:
                    lbl = f"MP #{i}"
                    if d.get("spectrum_type"):
                        lbl += f" type={d['spectrum_type']}"
                    if d.get("absorbing_index") is not None:
                        lbl += f" site={d['absorbing_index']}"
                    ax.plot(d["E"], d["mu"], "--", alpha=0.75, label=lbl)
            ax.set_xlabel("E (eV)")
            ax.set_ylabel("μ(E)")
            ax.set_title(f"{sample_name}  (n_sites={n_sites}, phase_c={phase_c_tag})")
            ax.legend(fontsize=9, loc="best")
            ax.grid(alpha=0.3)
            png_path = os.path.join(OUT_DIR, f"compare_{mp_id}_{element}.png")
            fig.savefig(png_path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            print(f"  图: {png_path}")
        except Exception as e:
            print(f"  画图失败: {e}")

        all_results.append({
            "sample_name": sample_name, "mp_id": mp_id, "element": element,
            "n_sites": n_sites, "phase_c_tag": phase_c_tag,
            "n_mp_spectra": n_mp,
            "mp_spectrum_types": str([d.get("spectrum_type") for d in mp_data]),
            "mp_absorbing_indices": str([d.get("absorbing_index") for d in mp_data]),
            "best_match_pearson": best.get("pearson_r") if best else None,
            "best_match_type": best.get("spectrum_type") if best else None,
            "best_match_site": best.get("absorbing_index") if best else None,
        })

    if all_results:
        df_out = pd.DataFrame(all_results)
        csv_path = os.path.join(OUT_DIR, "mp_source_verification.csv")
        df_out.to_csv(csv_path, index=False)
        print(f"\n\n汇总 → {csv_path}")
        print(df_out.to_string())

    print("\n" + "=" * 70)
    print("解读")
    print("=" * 70)
    print("""
看结果决定:

① n_mp_spectra == 1 且 absorbing_index 是 None/空
   → SITE-AVERAGED (MP 默认)

② n_mp_spectra > 1，每条有不同 absorbing_index，本地只匹配到 site=0 或 None
   → MP 提供 per-site，师兄选了固定规则（看 best_match 里的 site 值）

③ n_mp_spectra == 0
   → 师兄自己跑的 FEFF，必须等他回复
""")


if __name__ == "__main__":
    main()