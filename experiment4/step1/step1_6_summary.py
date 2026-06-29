# step1_6_summary.py
# ------------------------------------------------------------
# Exp4 Step 1.6
#   Run all 10 self-checks from spec §7:
#     1. total samples in 120K-130K
#     2. split counts (holdout in 3K-5K)
#     3. mp_id zero intersection
#     4. feff dim == 74
#     5. has_pre_edge ⊆ {0, 1}
#     6. scaler.center_ ≈ train 74-d median
#     7. holdout element health (O/Li <= 25%, distinct elements >= 30)
#     8. rare elements all in train
#     9. scaler.pkl reloads and transforms
#    10. exclusion reason breakdown vs Main Agent estimates
#   Outputs:
#     element_distribution.csv
#     step1_summary.txt  (full human-readable report)
# ------------------------------------------------------------

import os
import pandas as pd
import numpy as np
import joblib

STEP1_DIR = r"C:\Users\T-Cat\Desktop\DiffCSP-main\experiment4\step1"
RARE_THRESHOLD = 20


def main():
    # ---------- Load all artifacts ----------
    inv      = pd.read_csv(os.path.join(STEP1_DIR, "data_inventory.csv"))
    full_inv = pd.read_pickle(os.path.join(STEP1_DIR, "step1_4_full_inventory.pkl"))
    excl     = pd.read_csv(os.path.join(STEP1_DIR, "step1_excluded_log.csv"))
    scaler   = joblib.load(os.path.join(STEP1_DIR, "feff_feature_scaler.pkl"))
    with open(os.path.join(STEP1_DIR, "step1_3_feff_feature_names.txt"),
              encoding="utf-8") as f:
        feat_names = [l.strip() for l in f if l.strip()]

    lines = []
    def P(s=""):
        print(s)
        lines.append(s)

    P("=" * 72)
    P(" Experiment 4 — Step 1 Completion Report")
    P("=" * 72)

    # -------- [1] total samples --------
    P(f"\n[1] Total kept samples: {len(inv)}")
    P(f"    expected range: 120,000 - 130,000 (MA estimate 126K)")
    P(f"    verdict: {'OK' if 120_000 <= len(inv) <= 130_000 else 'DEVIATION'}")

    # -------- [2] splits --------
    P(f"\n[2] Split counts")
    for split in ("train", "val", "test", "holdout"):
        sub = inv[inv["split"] == split]
        mps = sub["mp_id"].nunique()
        n   = len(sub)
        pct = n / len(inv)
        P(f"    {split:8s}  mp_ids={mps:6d}  samples={n:7d}  ({pct:.2%})")
    hn = int((inv["split"] == "holdout").sum())
    P(f"    holdout range check (3,000-5,000): "
      f"{'OK' if 3000 <= hn <= 5000 else 'DEVIATION'}  (got {hn})")

    # -------- [3] zero intersection --------
    P(f"\n[3] mp_id zero-intersection assertion")
    mp_by = {s: set(inv[inv["split"] == s]["mp_id"])
             for s in ("train", "val", "test", "holdout")}
    all_ok = True
    for a in mp_by:
        for b in mp_by:
            if a >= b: continue
            ix = len(mp_by[a] & mp_by[b])
            mark = "OK" if ix == 0 else "FAIL"
            if ix: all_ok = False
            P(f"    {a:8s} ∩ {b:8s} = {ix:4d}  [{mark}]")
    P(f"    overall: {'PASS' if all_ok else 'FAIL'}")

    # -------- [4] feff dim 74 --------
    P(f"\n[4] feff feature dim")
    P(f"    feat_names count: {len(feat_names)}  [{'OK' if len(feat_names)==74 else 'FAIL'}]")
    P(f"    scaler.center_.shape: {scaler.center_.shape}")

    # -------- [5] has_pre_edge --------
    P(f"\n[5] has_pre_edge value check")
    vc = inv["has_pre_edge"].value_counts().to_dict()
    P(f"    value_counts: {vc}")
    P(f"    values ⊆ {{0,1}}: {'OK' if set(vc.keys()) <= {0,1} else 'FAIL'}")
    n1 = vc.get(1, 0)
    P(f"    has_pre_edge=1 share: {n1/len(inv):.2%}")

    # -------- [6] scaler.center_ ≈ train median --------
    P(f"\n[6] scaler.center_ vs train 74-d median")
    train_mat = full_inv[full_inv["split"] == "train"][feat_names].values
    train_med = np.median(train_mat, axis=0)
    max_diff = float(np.max(np.abs(scaler.center_ - train_med)))
    P(f"    max |scaler.center_ - train_median| = {max_diff:.6g}  "
      f"[{'OK' if max_diff < 1e-6 else 'WARN'}]")

    # -------- [7] holdout element health --------
    P(f"\n[7] Holdout element distribution")
    hold_inv = inv[inv["split"] == "holdout"]
    hold_vc = hold_inv["center_element"].value_counts()
    P(f"    distinct elements in holdout: {len(hold_vc)}  "
      f"[{'OK' if len(hold_vc) >= 30 else 'WARN (expect >=30)'}]")
    P(f"    top 5:")
    for e, n in hold_vc.head(5).items():
        P(f"      {e:3s}  n={n:5d}  pct={n/len(hold_inv):.2%}")
    for e in ("O", "Li"):
        if e in hold_vc.index:
            pct = hold_vc[e] / len(hold_inv)
            if pct > 0.30:  mark = "FAIL (>30%)"
            elif pct > 0.25: mark = "WARN (>25%)"
            else:            mark = "OK"
            P(f"    {e} pct in holdout: {pct:.2%}  [{mark}]")

    # -------- [8] rare elements all in train --------
    P(f"\n[8] Rare-element placement")
    elem_counts = inv["center_element"].value_counts()
    rare = elem_counts[elem_counts < RARE_THRESHOLD].index.tolist()
    P(f"    rare elements (count<{RARE_THRESHOLD}): {len(rare)}")
    if rare:
        P(f"    list: {sorted(rare)}")
        rare_inv = inv[inv["center_element"].isin(rare)]
        split_dist = rare_inv["split"].value_counts().to_dict()
        P(f"    rare-sample split distribution: {split_dist}")
        all_in_train = (rare_inv["split"] == "train").all()
        P(f"    all rare samples in train? {'YES [OK]' if all_in_train else 'NO [FAIL]'}")

    # -------- [9] scaler reload --------
    P(f"\n[9] Scaler reload sanity")
    reloaded = joblib.load(os.path.join(STEP1_DIR, "feff_feature_scaler.pkl"))
    z = reloaded.transform(train_mat[:5])
    P(f"    reloaded.transform(X[:5]).shape = {z.shape}  [{'OK' if z.shape == (5, 74) else 'FAIL'}]")

    # -------- [10] exclusion breakdown --------
    P(f"\n[10] Exclusion reason breakdown")
    ec = excl["reason"].value_counts()
    for reason, n in ec.items():
        P(f"    {reason:20s} {n:7d}")
    P(f"    TOTAL excluded: {len(excl)}")

    # Compare against MA estimates (§7 item 10 + revised §5.4b target ~800)
    expected = {
        "parse_fail":     (0, 5),
        "H_element":      (1300, 1800),
        "missing_poscar": (600, 1000),   # revised: ~800 via isfile check
        "iqr_outlier":    (3000, 10_000),
    }
    P(f"\n    vs MA estimates:")
    for reason, (lo, hi) in expected.items():
        actual = int(ec.get(reason, 0))
        mark = "OK" if (lo <= actual <= hi) else "DEVIATION"
        P(f"    {reason:20s} actual={actual:6d}  expected[{lo},{hi}]  [{mark}]")

    # -------- element_distribution.csv --------
    P(f"\n[11] element_distribution.csv")
    rows = []
    for e, n_total in elem_counts.items():
        sub = inv[inv["center_element"] == e]
        rows.append(dict(
            element=e,
            n_samples_total=int(n_total),
            n_train=int((sub["split"] == "train").sum()),
            n_val=int((sub["split"] == "val").sum()),
            n_test=int((sub["split"] == "test").sum()),
            n_holdout=int((sub["split"] == "holdout").sum()),
            is_rare=bool(n_total < RARE_THRESHOLD),
        ))
    edf = pd.DataFrame(rows).sort_values("n_samples_total", ascending=False)
    edf.to_csv(os.path.join(STEP1_DIR, "element_distribution.csv"), index=False)
    P(f"    written: {len(edf)} element rows")

    # -------- File list --------
    P(f"\n[12] Produced files in {STEP1_DIR}:")
    for fn in sorted(os.listdir(STEP1_DIR)):
        p = os.path.join(STEP1_DIR, fn)
        if os.path.isfile(p):
            kb = os.path.getsize(p) / 1024
            P(f"    {fn:55s} {kb:10.1f} KB")

    # -------- write step1_summary.txt --------
    with open(os.path.join(STEP1_DIR, "step1_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    P(f"\n[Save] step1_summary.txt written.")


if __name__ == "__main__":
    main()
