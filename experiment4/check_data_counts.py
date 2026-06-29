# save as: check_data_counts.py, 在本地跑
import os, re
import pandas as pd

CHI_DIR  = r"C:\Users\T-Cat\Desktop\DiffCSP-main\experiment4\data\MP_all_EXAFS_only_chi_csv\MP_all_EXAFS_only_chi_csv"
XMU_DIR  = r"C:\Users\T-Cat\Desktop\DiffCSP-main\experiment4\data\MP_all_EXAFS_only_csv\MP_all_EXAFS_only_csv"
POSCAR_DIR = r"C:\Users\T-Cat\Desktop\DiffCSP-main\experiment4\data\POSCAR_zip\MP_all_POSCAR_flat"
FEFF_CSV = r"C:\Users\T-Cat\Desktop\DiffCSP-main\experiment4\data\feff_features_all_csv_75cols(in).csv"

# 正则：从 chi 文件名抽 (mp_id, element)
# 例: mp-10__mp-10-EXAFS-As-K_chi.csv
pat = re.compile(r"(mp-\d+)__mp-\d+-EXAFS-([A-Z][a-z]?)-K(?:_chi)?\.csv")

def parse(dirpath, is_chi):
    out = set()
    for f in os.listdir(dirpath):
        m = pat.match(f)
        if m: out.add((m.group(1), m.group(2)))
    return out

chi_set = parse(CHI_DIR, True)
xmu_set = parse(XMU_DIR, False)
poscar_ids = {f.replace("_POSCAR","") for f in os.listdir(POSCAR_DIR) if f.endswith("_POSCAR")}

feff = pd.read_csv(FEFF_CSV)
feff_keys = set()
for name in feff["sample_name"]:
    m = re.match(r"(mp-\d+)__mp-\d+-EXAFS-([A-Z][a-z]?)-K", str(name))
    if m: feff_keys.add((m.group(1), m.group(2)))

print(f"chi  文件对应 (mp_id, element) 数: {len(chi_set)}")
print(f"xmu  文件对应 (mp_id, element) 数: {len(xmu_set)}")
print(f"POSCAR 文件对应 mp_id 数:           {len(poscar_ids)}")
print(f"feff_features 行数:                {len(feff)}")
print(f"feff_features 对应 (mp_id, element): {len(feff_keys)}")

# 交集
chi_xmu = chi_set & xmu_set
chi_xmu_poscar = {(m,e) for (m,e) in chi_xmu if m in poscar_ids}
all_four = chi_xmu_poscar & feff_keys
print(f"\nchi ∩ xmu                       : {len(chi_xmu)}")
print(f"chi ∩ xmu ∩ POSCAR              : {len(chi_xmu_poscar)}")
print(f"chi ∩ xmu ∩ POSCAR ∩ feff_feat : {len(all_four)}  ← Exp4 有效训练池")

# 每个 mp_id 平均几个元素
from collections import Counter
elem_per_mp = Counter(m for (m,e) in all_four)
dist = Counter(elem_per_mp.values())
print(f"\nmp_id → 元素数 分布:")
for k in sorted(dist.keys()): print(f"  {k} 种元素: {dist[k]} 个 mp_id")

# 元素分布
elem_dist = Counter(e for (m,e) in all_four)
print(f"\n元素覆盖（Top 15）:")
for e, c in elem_dist.most_common(15): print(f"  {e:3s}: {c}")
print(f"总共 {len(elem_dist)} 种元素")