# =============================================================================
# 脚本编号: step4.1_qt patch
# 脚本名称: step4.1_qt_patch_mpid.py
# 输入:
#   - experiment/quicktest/qt_step4/predictions_val.pt
#   - experiment/quicktest/qt_step4/predictions_test.pt
# 输出:
#   - 同上（如有 unk 键则修复；qt 版通常无需修复）
#   - experiment/quicktest/qt_step4/mpid_order_val.txt
#   - experiment/quicktest/qt_step4/mpid_order_test.txt
# 说明:
#   正式服需要此脚本是因为 xas_collate_fn 没有把 mp_id 放进 batch，
#   采样结果的键全是 unk_X_Y，需要事后对应修复。
#   Qt 版的 qt_collate_fn 已经在 batch 里保留了 mp_id，所以采样键直接就是
#   真实 mp_id（整数），本脚本主要做验证 + 备份顺序文件。
#   若发现 unk 键（说明采样脚本有变动），仍按正式服逻辑用 id 文件修复。
# =============================================================================

import os
import torch

PROJECT_ROOT   = r"C:\Users\T-Cat\Desktop\DiffCSP-main"
EXPERIMENT_DIR = os.path.join(PROJECT_ROOT, "experiment")
QT_DIR         = os.path.join(EXPERIMENT_DIR, "quicktest")
QT_STEP4_DIR   = os.path.join(QT_DIR, "qt_step4")

VAL_IDS_FILE  = os.path.join(QT_DIR, "qt_val_ids.txt")
TEST_IDS_FILE = os.path.join(QT_DIR, "qt_test_ids.txt")


def _read_ids(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def check_and_fix(pred_path: str, ids_file: str, split: str):
    preds = torch.load(pred_path, map_location="cpu", weights_only=False)
    keys  = list(preds.keys())
    n     = len(keys)
    print(f"\n[{split}] 共 {n} 个条目，前3键: {keys[:3]}")

    has_unk = any(str(k).startswith("unk_") for k in keys)

    if not has_unk:
        print(f"[{split}] ✓ 键已是真实 mp_id，无需修复。")
    else:
        print(f"[{split}] ⚠️  发现 unk 键，按 id 文件顺序修复...")
        ordered_ids = _read_ids(ids_file)
        n_fix = min(n, len(ordered_ids))
        new_dict = {}
        for old_key, mp_id in zip(keys[:n_fix], ordered_ids[:n_fix]):
            new_dict[mp_id] = preds[old_key]
        torch.save(new_dict, pred_path)
        print(f"[{split}] 修复完成，保存 {len(new_dict)} 个条目 → {pred_path}")
        preds = new_dict
        keys  = list(preds.keys())

    # 备份顺序文件
    order_path = os.path.join(QT_STEP4_DIR, f"mpid_order_{split}.txt")
    with open(order_path, "w") as f:
        f.write("\n".join(str(k) for k in keys))
    print(f"[{split}] mp_id 顺序已备份 → {order_path}")

    return preds, keys


def main():
    print("=" * 50)
    print("step4.1_qt_patch_mpid：验证 & 修复 mp_id 键")
    print("=" * 50)

    val_preds,  val_keys  = check_and_fix(
        os.path.join(QT_STEP4_DIR, "predictions_val.pt"),  VAL_IDS_FILE,  "val"
    )
    test_preds, test_keys = check_and_fix(
        os.path.join(QT_STEP4_DIR, "predictions_test.pt"), TEST_IDS_FILE, "test"
    )

    print("\n=== 最终验证 ===")
    for split, keys in [("val", val_keys), ("test", test_keys)]:
        sample_key = keys[0]
        is_unk = str(sample_key).startswith("unk_")
        status = "❌ 仍为 unk，请检查" if is_unk else "✓ OK"
        print(f"[{split}] count={len(keys)}, first_key={sample_key}  {status}")

    print("\nPatch 完成。")


if __name__ == "__main__":
    main()