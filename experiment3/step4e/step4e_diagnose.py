"""
step4e_diagnose.py
==================
逐步诊断 Exp3 的 coord_loss 为什么是 10^7 量级。

运行方式：
    python step4e_diagnose.py

诊断思路：
    coord_loss = F.mse_loss(pred_x, tar_x)
    tar_x = d_log_p_wrapped_normal(delta, sigma) * sqrt(sigma_norm)
    
    问题只可能出在三处：
      A. tar_x 本身就是 10^4 量级（sigma 参数导致 score 极大）
      B. pred_x 是 10^4 量级（decoder 输出异常）
      C. 两者都是合理量级但差异极大（模型完全没学）
    
    本脚本依次排查 A → B → C，并对比 Exp2 同一 batch 的值。
"""

import os, sys, json, warnings
import numpy as np
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")

# ── 路径 ───────────────────────────────────────────────────────────────────────
PROJECT_ROOT = r"C:\Users\T-Cat\Desktop\DiffCSP-main"
EXP2_ROOT    = os.path.join(PROJECT_ROOT, "experiment2")
EXP3_ROOT    = os.path.join(PROJECT_ROOT, "experiment3")
STEP1_DIR    = os.path.join(EXP2_ROOT, "step1")
STEP3C_DIR   = os.path.join(EXP3_ROOT, "step3c")
CONF_DIR     = os.path.join(EXP2_ROOT, "step3", "conf_xas")
VOCAB_PATH   = os.path.join(EXP3_ROOT, "step3b", "elem_vocab.json")
FEFF_CSV     = os.path.join(PROJECT_ROOT, "tesst_feff_features_all_full_v4.csv")
INVENTORY_CSV = os.path.join(STEP1_DIR, "data_inventory.csv")
TRAIN_IDS    = os.path.join(STEP1_DIR, "train_ids.txt")
DATA_ROOT    = r"C:\Users\T-Cat\Desktop\DiffCSP-main\site_dataset_Fe_only_oxide_one_site"

EXP2_CKPT_TXT = os.path.join(EXP2_ROOT, "step4b", "best_checkpoint_path.txt")

os.environ["PROJECT_ROOT"] = PROJECT_ROOT
for _p in [PROJECT_ROOT,
           os.path.join(EXP2_ROOT, "step2"),
           os.path.join(EXP2_ROOT, "step3"),
           STEP3C_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ── 工具函数 ───────────────────────────────────────────────────────────────────

def stat(name, t):
    """打印 tensor 的统计信息"""
    if t is None:
        print(f"  {name}: None")
        return
    t = t.float()
    print(f"  {name:40s}  min={t.min().item():+12.4f}  max={t.max().item():+12.4f}"
          f"  mean={t.mean().item():+12.4f}  std={t.std().item():12.4f}  "
          f"nan={t.isnan().any().item()}  inf={t.isinf().any().item()}")


def load_mini_batch(n=4):
    """从训练集取 n 个有效样本，返回 PyG Batch"""
    from torch_geometric.data import Batch
    sys.path.insert(0, os.path.join(EXP2_ROOT, "step3"))
    from xas_local_dataset_L6 import XASLocalStructureDataset
    ds = XASLocalStructureDataset(
        data_root        = DATA_ROOT,
        inventory_csv    = INVENTORY_CSV,
        ids_file         = TRAIN_IDS,
        feff_feat_csv    = FEFF_CSV,
        feff_scaler_path = os.path.join(STEP1_DIR, "feff_feature_scaler.pkl"),
        L                = 6.0,
    )
    samples = []
    for i in range(min(500, len(ds))):
        s = ds[i]
        if s is not None:
            samples.append(s)
        if len(samples) >= n:
            break
    print(f"  加载 {len(samples)} 个样本")
    return Batch.from_data_list(samples)


def load_exp3_model():
    import hydra
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf, open_dict

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=os.path.join(CONF_DIR, "model"),
                               job_name="diag", version_base=None):
        _raw = compose(config_name="diffusion_xas")
    model_cfg = OmegaConf.create({"model": OmegaConf.to_container(_raw, resolve=False)}).model
    with open_dict(model_cfg):
        model_cfg._target_ = "diffusion_w_type_xas_exp3.CSPDiffusion"
        model_cfg.vocab_path = VOCAB_PATH

    optim_cfg = OmegaConf.create({
        "optimizer": {"_target_": "torch.optim.Adam", "lr": 1e-4},
        "use_lr_scheduler": False,
    })
    model = hydra.utils.instantiate(model_cfg, optim=optim_cfg, _recursive_=False)
    model.lattice_scaler = model.scaler = None
    return model, model_cfg


def get_exp2_ckpt():
    if os.path.exists(EXP2_CKPT_TXT):
        with open(EXP2_CKPT_TXT) as f:
            p = f.read().strip()
        if os.path.exists(p):
            return p
    fallback = os.path.join(EXP2_ROOT, "step4b", "checkpoints", "last.ckpt")
    return fallback if os.path.exists(fallback) else None


def load_exp2_weights(model):
    ckpt = get_exp2_ckpt()
    if ckpt is None:
        print("  ⚠️  找不到 Exp2 checkpoint，跳过热启动")
        return False
    print(f"  载入 Exp2 权重: {ckpt}")
    sd = torch.load(ckpt, map_location="cpu")["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  missing={len(missing)}  unexpected={len(unexpected)}")
    return True


# ── 核心诊断：手动逐步拆解 forward() ──────────────────────────────────────────

def diagnose_forward(model, batch, label=""):
    """
    手动执行 forward 中的每一步，在关键节点打印统计信息。
    不调用 model.forward()，而是逐行复现。
    """
    from diffcsp.common.data_utils import lattice_params_to_matrix_torch
    from diffcsp.pl_modules.diff_utils import d_log_p_wrapped_normal

    print(f"\n{'='*60}")
    print(f"诊断：{label}")
    print(f"{'='*60}")

    model.eval()
    with torch.no_grad():
        B = batch.num_graphs
        times = model.beta_scheduler.uniform_sample_t(B, torch.device("cpu"))
        time_emb = model.time_embedding(times)

        # ── 1. SpectrumEncoder 输出 ──────────────────────────────────────────
        print("\n[1] SpectrumEncoder 输入/输出")
        stat("xmu_xanes", batch.xmu_xanes)
        stat("chi1", batch.chi1)
        stat("feff_features", batch.feff_features)
        spectrum_cond = model.spectrum_encoder(
            batch.xmu_xanes, batch.chi1, batch.feff_features)
        stat("spectrum_cond (latent)", spectrum_cond)
        stat("time_emb", time_emb)
        condition = torch.cat([time_emb, spectrum_cond], dim=-1)
        stat("condition = cat(time_emb, latent)", condition)

        # ── 2. Sigma scheduler ───────────────────────────────────────────────
        print("\n[2] Sigma/Beta scheduler 参数")
        sigmas      = model.sigma_scheduler.sigmas[times]
        sigmas_norm = model.sigma_scheduler.sigmas_norm[times]
        print(f"  times (sampled t):  {times.tolist()}")
        stat("sigmas (per batch)", sigmas)
        stat("sigmas_norm (per batch)", sigmas_norm)

        sigmas_per_atom      = sigmas.repeat_interleave(batch.num_atoms)[:, None]
        sigmas_norm_per_atom = sigmas_norm.repeat_interleave(batch.num_atoms)[:, None]
        stat("sigmas_per_atom", sigmas_per_atom)
        stat("sigmas_norm_per_atom", sigmas_norm_per_atom)
        stat("sqrt(sigmas_norm_per_atom)", sigmas_norm_per_atom.sqrt())

        # ── 3. 加噪后坐标 ────────────────────────────────────────────────────
        print("\n[3] 加噪坐标")
        frac_coords = batch.frac_coords
        rand_x = torch.randn_like(frac_coords)
        stat("frac_coords (clean)", frac_coords)
        stat("rand_x", rand_x)
        input_frac_coords = frac_coords + sigmas_per_atom * rand_x
        delta = input_frac_coords - frac_coords   # = sigmas_per_atom * rand_x
        stat("delta = sigma * rand_x", delta)
        stat("input_frac_coords (noisy)", input_frac_coords)

        # ── 4. Score target tar_x ────────────────────────────────────────────
        print("\n[4] Score target: tar_x = d_log_p(...) * sqrt(sigma_norm)")
        raw_score = d_log_p_wrapped_normal(delta, sigmas_per_atom)
        stat("d_log_p_wrapped_normal (raw score)", raw_score)
        tar_x = raw_score * torch.sqrt(sigmas_norm_per_atom)
        stat("tar_x (normalized score target)", tar_x)
        print(f"  tar_x L2 norm (mean over atoms): {tar_x.pow(2).mean().item():.4f}")

        # ── 5. Decoder 输出 pred_x ───────────────────────────────────────────
        print("\n[5] Decoder 输出: pred_x")
        alphas_cumprod = model.beta_scheduler.alphas_cumprod[times]
        c0 = torch.sqrt(alphas_cumprod)
        c1 = torch.sqrt(1. - alphas_cumprod)
        lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
        input_lattice = c0[:, None, None] * lattices + c1[:, None, None] * torch.randn_like(lattices)
        if model.keep_lattice:
            input_lattice = lattices

        gt_types_oh = F.one_hot(batch.atom_types - 1, num_classes=100).float()
        rand_t = torch.randn_like(gt_types_oh)
        atom_type_probs = (
            c0.repeat_interleave(batch.num_atoms)[:, None] * gt_types_oh
            + c1.repeat_interleave(batch.num_atoms)[:, None] * rand_t)

        pred_l, pred_x, pred_t = model.decoder(
            condition, atom_type_probs, input_frac_coords, input_lattice,
            batch.num_atoms, batch.batch)
        stat("pred_x (decoder raw output)", pred_x)

        # ── 6. 最终 loss 计算 ────────────────────────────────────────────────
        print("\n[6] Loss 计算")
        loss_coord = F.mse_loss(pred_x, tar_x)
        diff_sq = (pred_x - tar_x).pow(2)
        print(f"  MSE(pred_x, tar_x) = {loss_coord.item():.4f}")
        print(f"  (pred_x - tar_x)^2 max  = {diff_sq.max().item():.4f}")
        print(f"  (pred_x - tar_x)^2 mean = {diff_sq.mean().item():.4f}")
        print(f"  tar_x^2 mean (signal)   = {tar_x.pow(2).mean().item():.4f}")
        print(f"  pred_x^2 mean           = {pred_x.pow(2).mean().item():.4f}")

        # ── 7. 诊断结论 ──────────────────────────────────────────────────────
        print("\n[7] 诊断结论")
        tar_scale  = tar_x.pow(2).mean().item()
        pred_scale = pred_x.pow(2).mean().item()

        if tar_scale > 1e5:
            print(f"  ❌ 根本原因 A：tar_x 量级异常大（{tar_scale:.2e}）")
            print(f"     → sigma_scheduler 参数导致 score target 超大")
            print(f"     → 请检查 YAML 中 sigma_scheduler 的 sigma_begin/sigma_end")
            raw_score_scale = raw_score.pow(2).mean().item()
            sigma_norm_scale = sigmas_norm_per_atom.mean().item()
            print(f"     → raw_score^2 mean = {raw_score_scale:.2e}")
            print(f"     → sigma_norm mean  = {sigma_norm_scale:.2e}")
            print(f"     → sqrt(sigma_norm) mean = {sigmas_norm_per_atom.sqrt().mean().item():.4f}")
        elif pred_scale > 1e5:
            print(f"  ❌ 根本原因 B：pred_x 量级异常大（{pred_scale:.2e}）")
            print(f"     → Decoder 输出爆炸，检查 condition 向量的 scale")
        elif loss_coord > 10:
            print(f"  ⚠️  根本原因 C：tar_x 和 pred_x 量级都正常，但差异大")
            print(f"     tar_x^2={tar_scale:.4f}  pred_x^2={pred_scale:.4f}")
            print(f"     → 这是正常的「未收敛」，继续训练即可")
        else:
            print(f"  ✅ loss_coord={loss_coord.item():.4f} 正常（< 10）")

    print()


# ── 额外：打印 sigma_scheduler 全部参数 ──────────────────────────────────────

def print_scheduler_params(model, label=""):
    print(f"\n{'─'*50}")
    print(f"Scheduler 参数：{label}")
    print(f"{'─'*50}")
    ss = model.sigma_scheduler
    bs = model.beta_scheduler
    # sigma_scheduler
    for attr in ['sigma_begin', 'sigma_end', 'num_steps', 'sigmas']:
        if hasattr(ss, attr):
            v = getattr(ss, attr)
            if torch.is_tensor(v):
                print(f"  sigma_scheduler.{attr}: shape={tuple(v.shape)}"
                      f"  min={v.min():.6f}  max={v.max():.6f}")
            else:
                print(f"  sigma_scheduler.{attr}: {v}")
    # sigmas_norm
    if hasattr(ss, 'sigmas_norm'):
        v = ss.sigmas_norm
        print(f"  sigma_scheduler.sigmas_norm: shape={tuple(v.shape)}"
              f"  min={v.min():.6f}  max={v.max():.6f}")
    # beta_scheduler
    for attr in ['timesteps', 'betas']:
        if hasattr(bs, attr):
            v = getattr(bs, attr)
            if torch.is_tensor(v):
                print(f"  beta_scheduler.{attr}: shape={tuple(v.shape)}"
                      f"  min={v.min():.6f}  max={v.max():.6f}")
            else:
                print(f"  beta_scheduler.{attr}: {v}")
    print()


# ── 主程序 ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Step 4e 诊断脚本")
    print("=" * 60)

    print("\n[加载数据]")
    batch = load_mini_batch(n=4)

    print("\n[加载 Exp3 模型（随机初始化）]")
    model_rand, _ = load_exp3_model()
    model_rand.eval()

    print("\n[加载 Exp3 模型（Exp2 权重热启动）]")
    model_warm, _ = load_exp3_model()
    has_exp2 = load_exp2_weights(model_warm)
    model_warm.eval()

    # ── Scheduler 参数对比 ─────────────────────────────────────────────────
    print_scheduler_params(model_rand, "Exp3 随机初始化")

    # ── 诊断 1：随机初始化 ─────────────────────────────────────────────────
    diagnose_forward(model_rand, batch, label="Exp3 随机初始化")

    # ── 诊断 2：Exp2 热启动 ────────────────────────────────────────────────
    if has_exp2:
        diagnose_forward(model_warm, batch, label="Exp3 Exp2 热启动")

    # ── 额外：直接检查 tar_x 与 sigma 的关系 ──────────────────────────────
    print("\n" + "=" * 60)
    print("补充诊断：tar_x scale vs sigma 的关系")
    print("=" * 60)
    from diffcsp.pl_modules.diff_utils import d_log_p_wrapped_normal

    ss = model_rand.sigma_scheduler
    print("  抽取 sigma scheduler 的若干典型值：")
    T = len(ss.sigmas)
    for idx in [1, T//4, T//2, 3*T//4, T-1]:
        sig  = ss.sigmas[idx].item()
        norm = ss.sigmas_norm[idx].item()
        # 用典型 delta = 0.1 计算 score
        delta_test = torch.tensor([[0.1]])
        sig_t      = torch.tensor([[sig]])
        score      = d_log_p_wrapped_normal(delta_test, sig_t).item()
        tar_x_val  = score * (norm ** 0.5)
        print(f"  t={idx:4d}  sigma={sig:.6f}  sigma_norm={norm:.6f}"
              f"  score(delta=0.1)={score:+12.4f}  tar_x={tar_x_val:+12.4f}")

    print("\n  若 tar_x 列存在 >100 的值，则根本原因是 sigma_norm 参数异常。")
    print("  若 tar_x 列全部 <10，则问题在 pred_x（decoder），与 weights 相关。")
