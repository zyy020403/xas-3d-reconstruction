# =============================================================================
# 脚本编号: step4.1 patch
# 脚本名称: step4.1_patch_mpid.py
# 输入:
#   - experiment/step4/predictions_val.pt   (键为 unk_X_Y，需修复)
#   - experiment/step4/predictions_test.pt  (同上)
#   - XASDataModule（从 step3/ 加载，获取有序 mp_id 列表）
# 输出:
#   - experiment/step4/predictions_val.pt   (覆盖，键改为真实 mp_id)
#   - experiment/step4/predictions_test.pt  (同上)
#   - experiment/step4/mpid_order_val.txt   (val 集 mp_id 顺序备份)
#   - experiment/step4/mpid_order_test.txt  (test 集 mp_id 顺序备份)
# 说明:
#   Step 4.1 采样时 batch 中无 mp_id 字段，导致所有 compound 键为 unk_X_Y。
#   本脚本从 XASDataModule 的 dataset 对象中读取有序 mp_id 列表，
#   与 predictions dict 中的 unk 键（按插入顺序）一一对应后重命名。
#   不重新采样，直接修复已保存的 .pt 文件。
# =============================================================================

import os
import sys
import torch

PROJECT_ROOT = r"C:\Users\T-Cat\Desktop\DiffCSP-main"
EXPERIMENT_DIR = os.path.join(PROJECT_ROOT, "experiment")
STEP4_DIR = os.path.join(EXPERIMENT_DIR, "step4")
STEP1_DIR = os.path.join(EXPERIMENT_DIR, "step1")
STEP3_DIR = os.path.join(EXPERIMENT_DIR, "step3")

os.environ["PROJECT_ROOT"] = PROJECT_ROOT
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, STEP3_DIR)

VAL_IDS_FILE      = os.path.join(STEP1_DIR, "val_ids.txt")
TEST_IDS_FILE     = os.path.join(STEP1_DIR, "test_ids.txt")
HOLDOUT_IDS_FILE  = os.path.join(STEP1_DIR, "holdout_1000_ids.txt")

# ─── 导入 XASDataModule ──────────────────────────────────────────────────────
try:
    from xas_datamodule import XASDataModule
    print("[Import] xas_datamodule from step3/")
except ImportError:
    from modified_diffcsp.xas_datamodule import XASDataModule
    print("[Import] xas_datamodule from step3/modified_diffcsp/")


def get_ordered_mp_ids(datamodule, split: str) -> list:
    """
    从 datamodule 的 dataset 中提取有序 mp_id 列表。
    split: 'val' 或 'test'
    """
    if split == "val":
        dataset = datamodule.val_dataset
    else:
        dataset = datamodule.test_dataset

    # 尝试多种可能的属性名
    for attr in ["mp_ids", "mp_id_list", "compound_ids", "ids", "sample_ids"]:
        if hasattr(dataset, attr):
            ids = getattr(dataset, attr)
            print(f"[{split}] Found mp_ids via dataset.{attr}, count={len(ids)}")
            return list(ids)

    # fallback：尝试直接遍历 dataset 的 samples 列表
    if hasattr(dataset, "samples"):
        ids = [s["mp_id"] for s in dataset.samples]
        print(f"[{split}] Found mp_ids via dataset.samples, count={len(ids)}")
        return ids

    if hasattr(dataset, "data_list"):
        ids = [d["mp_id"] for d in dataset.data_list]
        print(f"[{split}] Found mp_ids via dataset.data_list, count={len(ids)}")
        return ids

    # 最后 fallback：直接读 val_ids.txt / test_ids.txt（这是 datamodule 的来源）
    print(f"[{split}] WARNING: dataset 没有 mp_ids 属性，直接从 ids 文件读取。")
    id_file = VAL_IDS_FILE if split == "val" else TEST_IDS_FILE
    with open(id_file, "r") as f:
        ids = [line.strip() for line in f if line.strip()]
    
    # 过滤掉 holdout ids（datamodule 内部也会过滤）
    with open(HOLDOUT_IDS_FILE, "r") as f:
        holdout = set(line.strip() for line in f if line.strip())
    ids = [i for i in ids if i not in holdout]
    print(f"[{split}] Loaded {len(ids)} ids from file (after holdout filter).")
    return ids


def rekey_predictions(predictions: dict, ordered_ids: list, split: str) -> dict:
    """
    predictions 的键是 unk_X_Y，按插入顺序（Python 3.7+ dict 有序）
    与 ordered_ids 一一对应，重命名键。
    """
    current_keys = list(predictions.keys())
    n_pred = len(current_keys)
    n_ids  = len(ordered_ids)

    if n_pred != n_ids:
        print(f"[{split}] WARNING: predictions count ({n_pred}) != ids count ({n_ids}).")
        print(f"  将使用前 min({n_pred}, {n_ids}) 个条目匹配，多余的丢弃。")
        n = min(n_pred, n_ids)
        current_keys = current_keys[:n]
        ordered_ids  = ordered_ids[:n]

    new_dict = {}
    for old_key, mp_id in zip(current_keys, ordered_ids):
        new_dict[mp_id] = predictions[old_key]

    print(f"[{split}] Rekeyed {len(new_dict)} compounds.")
    return new_dict


def main():
    # 初始化 datamodule
    print("[Data] Initializing XASDataModule ...")
    datamodule = XASDataModule(
        batch_size=16,
        num_workers=0,
        step1_dir=STEP1_DIR,
    )
    datamodule.setup("fit")
    datamodule.setup("test")

    # ── Val ──────────────────────────────────────────────────────────────────
    print("\n=== Fixing Val predictions ===")
    val_path = os.path.join(STEP4_DIR, "predictions_val.pt")
    val_preds = torch.load(val_path, map_location="cpu")
    print(f"Loaded {len(val_preds)} val predictions (keys: {list(val_preds.keys())[:3]} ...)")

    val_ids = get_ordered_mp_ids(datamodule, "val")

    val_preds_fixed = rekey_predictions(val_preds, val_ids, "val")

    # 保存
    torch.save(val_preds_fixed, val_path)
    print(f"Saved fixed val predictions -> {val_path}")

    # 备份 mp_id 顺序
    order_file_val = os.path.join(STEP4_DIR, "mpid_order_val.txt")
    with open(order_file_val, "w") as f:
        f.write("\n".join(val_ids))
    print(f"Saved mp_id order -> {order_file_val}")

    # ── Test ─────────────────────────────────────────────────────────────────
    print("\n=== Fixing Test predictions ===")
    test_path = os.path.join(STEP4_DIR, "predictions_test.pt")
    test_preds = torch.load(test_path, map_location="cpu")
    print(f"Loaded {len(test_preds)} test predictions.")

    test_ids = get_ordered_mp_ids(datamodule, "test")

    test_preds_fixed = rekey_predictions(test_preds, test_ids, "test")

    torch.save(test_preds_fixed, test_path)
    print(f"Saved fixed test predictions -> {test_path}")

    order_file_test = os.path.join(STEP4_DIR, "mpid_order_test.txt")
    with open(order_file_test, "w") as f:
        f.write("\n".join(test_ids))
    print(f"Saved mp_id order -> {order_file_test}")

    # ── 验证 ──────────────────────────────────────────────────────────────────
    print("\n=== Verification ===")
    for split, path, ids in [
        ("val", val_path, val_ids),
        ("test", test_path, test_ids),
    ]:
        preds = torch.load(path, map_location="cpu")
        sample_id = list(preds.keys())[0]
        is_unk = sample_id.startswith("unk_")
        print(f"[{split}] count={len(preds)}, first_key={sample_id}, is_unk={is_unk}")
        if is_unk:
            print(f"  [WARNING] mp_id 仍为 unk，fallback 也失败了，请检查 datamodule 结构")
        else:
            print(f"  [OK] mp_id 已正确设置")

    print("\nPatch complete.")


if __name__ == "__main__":
    main()