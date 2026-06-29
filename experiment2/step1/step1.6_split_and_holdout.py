# step1.6_split_and_holdout.py
# 输入：data_inventory.csv + feff_features CSV
# 输出：holdout_1000_ids.txt, train/val/test_ids.txt，并更新 data_inventory.csv 的 split 列

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

STEP1_DIR     = r"C:\Users\T-Cat\Desktop\DiffCSP-main\experiment2\step1"
INVENTORY_CSV = os.path.join(STEP1_DIR, "data_inventory.csv")
FEFF_FEAT_CSV = r"C:\Users\T-Cat\Desktop\DiffCSP-main\tesst_feff_features_all_full_v4.csv"

HOLDOUT_TXT   = os.path.join(STEP1_DIR, "holdout_1000_ids.txt")
TRAIN_TXT     = os.path.join(STEP1_DIR, "train_ids.txt")
VAL_TXT       = os.path.join(STEP1_DIR, "val_ids.txt")
TEST_TXT      = os.path.join(STEP1_DIR, "test_ids.txt")

RANDOM_SEED   = 42
N_CLUSTERS    = 100
TARGET_HOLDOUT= 1000

def parse_sample_name(s):
    try:
        parts = s.split("_")
        return parts[0]+"_"+parts[1], parts[-1].zfill(2)
    except:
        return None, None

def main():
    df = pd.read_csv(INVENTORY_CSV, encoding="utf-8-sig")
    print(f"data_inventory 行数 : {len(df)}")

    # 读 feff_features，取聚类用的4列
    df_feat = pd.read_csv(FEFF_FEAT_CSV, encoding="utf-8-sig")
    parsed  = df_feat["sample_name"].apply(parse_sample_name)
    df_feat["mp_id"]   = [x[0] for x in parsed]
    df_feat["site_nn"] = [x[1] for x in parsed]
    df_feat = df_feat.drop_duplicates(subset=["mp_id","site_nn"])

    # 聚类特征：E0(col6), white_line_I(col12), R1_peak_pos(col69), chi_kmax(col67)
    cluster_cols = [df_feat.columns[6], df_feat.columns[12],
                    df_feat.columns[69], df_feat.columns[67]]
    print(f"聚类特征列 : {cluster_cols}")

    df["site_nn"] = df["site_nn"].astype(str).str.zfill(2)
    df_feat["site_nn"] = df_feat["site_nn"].astype(str).str.zfill(2)

    df_cluster = df[["mp_id","site_nn"]].merge(
        df_feat[["mp_id","site_nn"] + cluster_cols],
        on=["mp_id","site_nn"], how="left"
    )

    # 用均值填充 NaN
    for c in cluster_cols:
        df_cluster[c] = df_cluster[c].fillna(df_cluster[c].mean())

    # 标准化 + KMeans
    X = df_cluster[cluster_cols].values
    X_scaled = StandardScaler().fit_transform(X)

    print(f"KMeans 聚类（n_clusters={N_CLUSTERS}）...")
    km = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_SEED, n_init=10)
    df["cluster"] = km.fit_predict(X_scaled)

    # 每个簇按 ~10% 抽 holdout
    np.random.seed(RANDOM_SEED)
    holdout_ids = []

    for cid, grp in df.groupby("cluster"):
        n = len(grp)
        if n < 5:
            continue
        k = max(1, min(20, round(n * 0.10)))
        sampled = grp.sample(n=k, random_state=RANDOM_SEED)["mp_id"].tolist()
        holdout_ids.extend(sampled)

    # 控制总数在 TARGET_HOLDOUT 左右
    np.random.shuffle(holdout_ids)
    if len(holdout_ids) > TARGET_HOLDOUT:
        holdout_ids = holdout_ids[:TARGET_HOLDOUT]

    print(f"Holdout 数量 : {len(holdout_ids)}")

    # 确认每个 holdout 所在簇仍有 ≥2 个样本留在训练集
    holdout_set = set(holdout_ids)
    df_remain   = df[~df["mp_id"].isin(holdout_set)]
    cluster_remain_counts = df_remain["cluster"].value_counts()
    holdout_clusters = df[df["mp_id"].isin(holdout_set)]["cluster"].unique()
    n_cluster_ok = sum(cluster_remain_counts.get(c, 0) >= 2 for c in holdout_clusters)
    print(f"holdout 涉及簇数 : {len(holdout_clusters)}，其中训练集仍 ≥2 个的簇 : {n_cluster_ok}")

    # 剩余样本 70:15:15 划分
    df_rest = df[~df["mp_id"].isin(holdout_set)].copy()
    df_rest = df_rest.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    n      = len(df_rest)
    n_val  = int(n * 0.15)
    n_test = int(n * 0.15)
    n_train= n - n_val - n_test

    train_ids = df_rest.iloc[:n_train]["mp_id"].tolist()
    val_ids   = df_rest.iloc[n_train:n_train+n_val]["mp_id"].tolist()
    test_ids  = df_rest.iloc[n_train+n_val:]["mp_id"].tolist()

    print(f"\n======= 数据集划分 =======")
    print(f"holdout : {len(holdout_ids)}")
    print(f"train   : {len(train_ids)}")
    print(f"val     : {len(val_ids)}")
    print(f"test    : {len(test_ids)}")
    print(f"合计    : {len(holdout_ids)+len(train_ids)+len(val_ids)+len(test_ids)}")

    # 写 txt
    for path, ids in [(HOLDOUT_TXT, holdout_ids), (TRAIN_TXT, train_ids),
                      (VAL_TXT, val_ids), (TEST_TXT, test_ids)]:
        with open(path, "w") as f:
            f.write("\n".join(ids))
        print(f"已保存 → {path}")

    # 更新 data_inventory 的 split 列
    split_map = {mid: "holdout" for mid in holdout_ids}
    split_map.update({mid: "train" for mid in train_ids})
    split_map.update({mid: "val"   for mid in val_ids})
    split_map.update({mid: "test"  for mid in test_ids})
    df["split"] = df["mp_id"].map(split_map).fillna("excluded")
    df.drop(columns=["cluster"], inplace=True)
    df.to_csv(INVENTORY_CSV, index=False, encoding="utf-8-sig")
    print(f"\ndata_inventory.csv split 列已更新 → {INVENTORY_CSV}")

if __name__ == "__main__":
    main()