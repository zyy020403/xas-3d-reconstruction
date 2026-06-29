# check_missing_compounds.py

import os
import pandas as pd

STEP1_DIR  = r"C:\Users\T-Cat\Desktop\DiffCSP-main\experiment2\step1"

df_scan   = pd.read_csv(os.path.join(STEP1_DIR, "step1_poscar_check.csv"), encoding="utf-8-sig")
df_atoms  = pd.read_csv(os.path.join(STEP1_DIR, "prim_natoms_all.csv"),    encoding="utf-8-sig")

print(f"step1_poscar_check 总行数     : {len(df_scan)}")
print(f"step1_poscar_check unique mp_id : {df_scan['mp_id'].nunique()}")
print(f"prim_natoms_all 总行数        : {len(df_atoms)}")
print(f"prim_natoms_all unique mp_id  : {df_atoms['mp_id'].nunique()  if 'mp_id' in df_atoms.columns else '无mp_id列'}")
print(f"prim_natoms_all 列名          : {df_atoms.columns.tolist()}")

# merge 后看丢了多少
df_merged = df_scan[["mp_id","folder_name","site_nn"]].merge(
    df_atoms[["folder_name","prim_n_atoms"]], on="folder_name", how="left"
)
print(f"\nmerge后总行数                 : {len(df_merged)}")
print(f"merge后 prim_n_atoms 为空的行  : {df_merged['prim_n_atoms'].isna().sum()}")
print(f"merge后 unique mp_id          : {df_merged['mp_id'].nunique()}")

# LVSI后
df_merged["site_nn_int"] = df_merged["site_nn"].astype(int)
df_lvsi = df_merged.sort_values("site_nn_int").groupby("mp_id").first().reset_index()
print(f"\nLVSI去重后化合物数            : {len(df_lvsi)}")
print(f"其中 prim_n_atoms 为空        : {df_lvsi['prim_n_atoms'].isna().sum()}")
print(f"其中 prim_n_atoms = -1        : {(df_lvsi['prim_n_atoms']==-1).sum()}")
print(f"其中 prim_n_atoms > 0         : {(df_lvsi['prim_n_atoms']>0).sum()}")