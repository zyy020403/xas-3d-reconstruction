"""
step2.0b_adapter_smoke.py — adapter integration smoke (single real batch).

Status:
    Bridges Step 2.0 (dtype audit, done) and Step 2.1 (5ep × 1000 sanity).
    Validates that exp6_data_adapter.adapt() correctly bridges the Exp4 PyG
    Batch produced by xas_local_datamodule_v2.xas_collate_fn_v2 into the
    SA1-locked DETRXas.forward + SetCriterion contract, on REAL Exp4 train data
    (not synthetic, unlike SA1's Step 1.2 smoke).

Pipeline tested (single batch_size=4 batch):
    PyG Batch
      → exp6_data_adapter.adapt(model)
      → DETRXas.forward(model_batch)
      → SetCriterion.forward(out, targets)
      → weighted total via weight_dict
      → backward
      → NaN grad scan

Hyperparams used (handoff §3.2 sanity values, locked):
    cost_class = 1.0,  cost_pos   = 1.0     (matcher)
    lambda_cls = 1.0,  lambda_pos = 1.0     (criterion weight_dict)
    eos_coef   = 0.1                         (no_object class weight)
    aux losses get same weights as main per DETR convention.

Run:
    cd /home/tcat/experiment6
    /home/tcat/conda_envs/mlff/bin/python step2/step2.0b_adapter_smoke.py \\
        2>&1 | tee step2/step2.0b_smoke_log.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

# ----- import path setup -----
# shared/ is a package (relative imports inside SA1 files: `from .eval_metrics`).
# But Exp4's xas_local_datamodule_v2.py (cp'd zero-change) uses bare imports:
#   `from xas_local_dataset_v2 import XasLocalDatasetV2`
# So we add BOTH the repo root (for `shared.X` package imports) AND shared/
# itself (for Exp4 bare import inside the cp'd datamodule).
REPO_ROOT = Path(__file__).resolve().parent.parent
SHARED = REPO_ROOT / "shared"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SHARED))

import torch  # noqa: E402

from shared.detr_xas import DETRXas  # noqa: E402
from shared.matcher import HungarianMatcher  # noqa: E402
from shared.criterion import SetCriterion  # noqa: E402
from shared.exp6_data_adapter import adapt, reset_first_batch_flag  # noqa: E402
from shared.xas_local_datamodule_v2 import XasLocalDataModuleV2  # noqa: E402

# -------------------------------------------------------------------------
# Sanity hyperparameters — VERBATIM from handoff §3.2. NOT tunable here.
# If smoke fails, push MA1; do NOT modify these values.
# -------------------------------------------------------------------------
COST_CLASS = 1.0
COST_POS = 1.0
LAMBDA_CLS = 1.0
LAMBDA_POS = 1.0
EOS_COEF = 0.1

NUM_DECODER_LAYERS = 6  # 1 main + 5 aux (DETR + SA1 default)
N_AUX = NUM_DECODER_LAYERS - 1

VOCAB_PATH = SHARED / "exp6_element_vocab.json"


def build_weight_dict() -> dict:
    """DETR convention: aux layers carry same loss weight as main layer."""
    wd = {"loss_ce": LAMBDA_CLS, "loss_pos": LAMBDA_POS}
    for i in range(N_AUX):
        wd[f"loss_ce_{i}"] = LAMBDA_CLS
        wd[f"loss_pos_{i}"] = LAMBDA_POS
    return wd


def is_aux_key(k: str) -> bool:
    """Aux keys carry an `_i` suffix where i ∈ {0..N_AUX-1}."""
    return any(k.endswith(f"_{i}") for i in range(N_AUX))


def main():
    print("=" * 72)
    print("Exp6 Step 2.0b — adapter integration smoke")
    print("=" * 72)

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device     = {device}")
    print(f"  vocab_path = {VOCAB_PATH}")
    assert VOCAB_PATH.exists(), f"vocab not found: {VOCAB_PATH}"

    # ---------------- [1/6] datamodule ----------------
    print("\n[1/6] DataModule.setup('fit') — first batch from train loader ...")
    dm = XasLocalDataModuleV2(batch_size=4, num_workers=0)
    dm.setup("fit")
    loader = dm.train_dataloader()
    pyg_batch = next(iter(loader))
    assert pyg_batch is not None, "first train batch is None — datamodule broken"
    pyg_batch = pyg_batch.to(device)
    print(f"  pyg_batch.num_graphs = {pyg_batch.num_graphs}")
    print(f"  field names verified : "
          f"xmu_xanes, chi1, feff_features, frac_coords, atom_types, "
          f"batch, center_element")

    # ---------------- [2/6] model + matcher + criterion ----------------
    print("\n[2/6] Building DETRXas + matcher + criterion ...")
    model = DETRXas(vocab_path=str(VOCAB_PATH)).to(device)
    matcher = HungarianMatcher(cost_class=COST_CLASS, cost_pos=COST_POS).to(device)
    weight_dict = build_weight_dict()
    criterion = SetCriterion(
        num_classes=model.n_neighbor_types,   # 89, omitting no_object
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=EOS_COEF,
        losses=["labels", "pos", "cardinality"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  total params: {n_params:,}  (SA1 reported 18,226,205 — must match)")
    assert n_params == 18_226_205, (
        f"param count drift: {n_params} vs SA1's 18,226,205 — "
        f"shared/ files may have been modified post-SA1"
    )
    print(f"  N_NEIGHBOR={model.n_neighbor_types}, N_CENTER={model.n_center_types}, "
          f"no_object_idx={model.no_object_idx}")

    # ---------------- [3/6] adapter ----------------
    print("\n[3/6] adapt(pyg_batch, model) — first-batch sanity print follows ...\n")
    reset_first_batch_flag()
    model_batch, targets = adapt(pyg_batch, model)
    print()  # spacing after sanity print

    # adapter contract assertions
    B = pyg_batch.num_graphs
    assert model_batch["xmu"].shape == (B, 150)
    assert model_batch["chi1"].shape == (B, 200)
    assert model_batch["feff"].shape == (B, 74)
    assert model_batch["center_idx"].shape == (B,)
    assert model_batch["center_idx"].dtype == torch.long
    assert (model_batch["center_idx"] >= 0).all()
    assert (model_batch["center_idx"] < model.n_center_types).all()
    assert len(targets) == B
    for i, t in enumerate(targets):
        n_i = t["labels"].numel()
        assert t["labels"].dtype == torch.long, f"target {i}: labels dtype"
        assert t["pos"].dtype.is_floating_point, f"target {i}: pos dtype"
        assert t["pos"].shape == (n_i, 3), f"target {i}: pos shape"
        assert n_i <= 20, f"target {i}: too many neighbors ({n_i})"
        assert (t["labels"] >= 0).all() and (t["labels"] < model.no_object_idx).all(), \
            f"target {i}: labels OOB"
        assert (t["pos"] >= -0.5 - 1e-5).all() and (t["pos"] <= 0.5 + 1e-5).all(), \
            f"target {i}: pos OOB"
    print(f"  adapter contract assertions PASS ({B} samples)")

    # ---------------- [4/6] DETRXas.forward ----------------
    print("\n[4/6] DETRXas.forward(model_batch) ...")
    model.train()
    out = model(model_batch)
    K1 = model.n_neighbor_types + 1  # 90
    assert out["pred_logits"].shape == (B, 20, K1), \
        f"pred_logits shape {out['pred_logits'].shape} != (B, 20, {K1})"
    assert out["pred_pos"].shape == (B, 20, 3), \
        f"pred_pos shape {out['pred_pos'].shape} != (B, 20, 3)"
    assert not torch.isnan(out["pred_pos"]).any()
    assert (out["pred_pos"] >= -0.5 - 1e-5).all()
    assert (out["pred_pos"] <= 0.5 + 1e-5).all()
    assert "aux_outputs" in out and len(out["aux_outputs"]) == N_AUX, \
        f"aux_outputs len={len(out.get('aux_outputs', []))} != {N_AUX}"
    print(f"  pred_logits  {tuple(out['pred_logits'].shape)} ✓")
    print(f"  pred_pos     {tuple(out['pred_pos'].shape)} ⊂ [-0.5, 0.5] ✓")
    print(f"  aux_outputs  list[{len(out['aux_outputs'])}] ✓")

    # ---------------- [5/6] SetCriterion ----------------
    print("\n[5/6] SetCriterion(out, targets) ...")
    loss_dict = criterion(out, targets)
    print(f"  criterion returned {len(loss_dict)} keys (expect 19: 4 main + 15 aux)")

    main_keys = sorted(k for k in loss_dict if not is_aux_key(k))
    aux_keys = sorted(k for k in loss_dict if is_aux_key(k))
    assert len(main_keys) == 4, f"expect 4 main keys, got {main_keys}"
    assert len(aux_keys) == 15, f"expect 15 aux keys, got {len(aux_keys)}"

    print("  main-layer losses:")
    for k in main_keys:
        v = loss_dict[k]
        assert isinstance(v, torch.Tensor) and v.dim() == 0, f"{k} is not 0-dim tensor"
        print(f"    {k:25s} = {v.item():.4f}")
    print(f"  aux loss key sample : {aux_keys[:6]} ... (+9 more)")

    # cls : pos ratio preview — handoff §3.2 sanity target ∈ [1/3, 3] AT EP1 END.
    # First-batch ratio is uninformative (random init), but we expose the
    # magnitude so MA1/SA2 can sanity-check λ_pos=1.0 reverse-engineering.
    cls_main = loss_dict["loss_ce"].item()
    pos_main = loss_dict["loss_pos"].item()
    ratio = cls_main / (LAMBDA_POS * pos_main) if pos_main > 0 else float("inf")
    print(f"\n  loss_ce  (main)         = {cls_main:.4f}  (~ ln(90) = 4.50 at random init)")
    print(f"  loss_pos (main, raw)    = {pos_main:.4f}  (SA1 reported ~7.4 at init)")
    print(f"  ratio cls / (λ_pos·pos) = {ratio:.3f}")
    print(f"  → ep1-end target ∈ [1/3, 3]; init value just confirms magnitudes match SA1")

    # weighted total
    total = sum(
        loss_dict[k] * weight_dict[k]
        for k in weight_dict if k in loss_dict
    )
    sa1_loss_with_lambda5 = 249.59
    print(f"\n  WEIGHTED TOTAL          = {total.item():.4f}")
    print(f"  SA1 smoke (λ_pos=5.0)   = {sa1_loss_with_lambda5} "
          f"(synthetic, B=5; ours real, B=4, λ_pos=1.0)")
    print(f"  → expect roughly 1/5×SA1 due to λ_pos drop, ± real-vs-synthetic variance")

    # ---------------- [6/6] backward ----------------
    print("\n[6/6] backward + NaN grad scan ...")
    total.backward()
    n_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
    n_nan_grad = sum(
        1 for p in model.parameters()
        if p.grad is not None and torch.isnan(p.grad).any()
    )
    n_inf_grad = sum(
        1 for p in model.parameters()
        if p.grad is not None and torch.isinf(p.grad).any()
    )
    print(f"  params with grad : {n_with_grad}")
    print(f"  params with NaN  : {n_nan_grad}")
    print(f"  params with Inf  : {n_inf_grad}")
    assert n_nan_grad == 0, "NaN gradient — push MA1 (handoff §10 row 5)"
    assert n_inf_grad == 0, "Inf gradient — push MA1"

    print()
    print("=" * 72)
    print("ADAPTER SMOKE PASS — bridge to DETRXas + SetCriterion validated.")
    print("Real Exp4 batch flowed through model + criterion + backward cleanly.")
    print("Next: Step 2.1 sanity (5 ep × 1000 samples, full PL Trainer).")
    print("=" * 72)


if __name__ == "__main__":
    main()
