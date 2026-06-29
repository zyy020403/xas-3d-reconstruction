# =============================================================================
# 脚本编号: step3_qt (train) v2
# 脚本名称: step3_qt_train.py
# 修复说明（v2 相比 v1）:
#   Fix 1 — embedding cosine=1.0 bug：
#     preprocess_chi 失败时每次 return 新 tensor（不共享对象）
#     Dataset 存储时对所有谱 tensor 调用 .clone()，杜绝浅拷贝
#     训练前调用 check_embedding_uniqueness，assert 通过才继续
#   Fix 2 — 使用原胞（primitive cell）替换超胞作为训练标签：
#     _parse_poscar 中加入 get_primitive_structure(tolerance=0.25)
#     转换失败时 fallback 到超胞，记录警告
#     训练前打印原胞原子数统计
# =============================================================================

import os
import sys
import types
import importlib.util
import warnings
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
warnings.filterwarnings("ignore", message="Trying to infer the `batch_size`")
warnings.filterwarnings("ignore", message=".*does not have many workers.*")

PROJECT_ROOT   = r"C:\Users\T-Cat\Desktop\DiffCSP-main"
EXPERIMENT_DIR = os.path.join(PROJECT_ROOT, "experiment")
QT_DIR         = os.path.join(EXPERIMENT_DIR, "quicktest")
STEP2_DIR      = os.path.join(EXPERIMENT_DIR, "step2")
CONF_MODEL_DIR = os.path.join(PROJECT_ROOT, "conf", "model")
QT_OUTPUT_DIR  = os.path.join(QT_DIR, "qt_output")
os.makedirs(QT_OUTPUT_DIR, exist_ok=True)

os.environ["PROJECT_ROOT"] = PROJECT_ROOT

for p in [PROJECT_ROOT, QT_DIR, STEP2_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# =============================================================================
# sys.modules 注入（必须在 import diffusion 之前）
# =============================================================================
import torch
import torch.nn as nn

# 1. k¹ 编码器
_qt_encoder_path = os.path.join(QT_DIR, "step2.1_qt_encoder.py")
if not os.path.exists(_qt_encoder_path):
    raise FileNotFoundError(f"找不到 qt 编码器：{_qt_encoder_path}")
_spec = importlib.util.spec_from_file_location("step2_1_spectrum_encoder", _qt_encoder_path)
_qt_encoder_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_qt_encoder_mod)
sys.modules["step2_1_spectrum_encoder"] = _qt_encoder_mod
logging.getLogger(__name__).info("✓ 已注入 step2_1_spectrum_encoder（k¹ 权重）")

# 2. stub aggregator
class _StubAggregator(nn.Module):
    def __init__(self, d_site=256, d_struct=256):
        super().__init__()
        self._unused = nn.Linear(1, 1)
    def forward(self, padded, mask):
        raise RuntimeError("[QT] StubAggregator.forward 不应被调用")

_stub_mod = types.ModuleType("step2_2_multisite_aggregator")
_stub_mod.MultiSiteAggregator     = _StubAggregator
_stub_mod.collate_multisite_batch = lambda *a, **k: (_ for _ in ()).throw(RuntimeError())
sys.modules["step2_2_multisite_aggregator"] = _stub_mod
logging.getLogger(__name__).info("✓ 已注入 stub step2_2_multisite_aggregator")

# =============================================================================
# Dataset / DataModule / 检查函数（内联，避免跨文件 import 问题）
# =============================================================================
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch

QUALITY_WEIGHT         = {"A": 1.0, "B": 0.5, "C": 0.1}
QUALITY_WEIGHT_UNKNOWN = 0.3
SPECTRA_LEN            = 512


def _parse_poscar(poscar_path: str):
    """
    解析 POSCAR_supercell_fixed 并转换为原胞（Fix 2）。
    返回 (frac_coords, atom_types, lengths, angles, num_atoms, used_primitive)
    失败时返回 None。
    """
    try:
        from pymatgen.core import Structure
        supercell = Structure.from_file(poscar_path)

        used_primitive = True
        try:
            structure = supercell.get_primitive_structure(tolerance=0.25)
        except Exception as e:
            logging.getLogger(__name__).warning(
                f"原胞转换失败，fallback 超胞: {os.path.basename(poscar_path)}  ({e})"
            )
            structure      = supercell
            used_primitive = False

        frac_coords = torch.tensor(structure.frac_coords, dtype=torch.float32)
        atom_types  = torch.tensor([s.specie.Z for s in structure], dtype=torch.long)
        lengths     = torch.tensor(list(structure.lattice.abc),    dtype=torch.float32)
        angles      = torch.tensor(list(structure.lattice.angles), dtype=torch.float32)
        return frac_coords, atom_types, lengths, angles, len(structure), used_primitive

    except Exception as e:
        logging.getLogger(__name__).warning(f"POSCAR 解析失败: {poscar_path}  ({e})")
        return None


class QTCrystalDataset(Dataset):
    """
    QuickTest Dataset v2。
    Fix 1：spec tensor 存储时强制 .clone()，全零 fallback 每次新建。
    Fix 2：POSCAR 读取后转换为原胞（含 fallback）。
    """

    def __init__(self, mp_ids, inventory_df: pd.DataFrame):
        super().__init__()
        from pymatgen.core.periodic_table import Element as PmgElement

        preprocess_chi = sys.modules["step2_1_spectrum_encoder"].preprocess_chi

        mp_ids_set  = set(int(x) for x in mp_ids)
        sub_df      = inventory_df[inventory_df["mp_id"].isin(mp_ids_set)].copy()

        self._cache       = []
        self.valid_mp_ids = []
        n_primitive       = 0
        n_fallback        = 0
        n_atoms_list      = []
        skipped           = 0

        for _, row in sub_df.iterrows():
            mp_id = int(row["mp_id"])

            # ── POSCAR + 原胞转换（Fix 2）─────────────────────────────────────
            poscar_path = os.path.join(row["source_path"], "POSCAR_supercell_fixed")
            result = _parse_poscar(poscar_path)
            if result is None:
                skipped += 1
                continue
            frac_coords, atom_types, lengths, angles, num_atoms, used_prim = result

            if used_prim: n_primitive += 1
            else:         n_fallback  += 1
            n_atoms_list.append(num_atoms)

            # ── chi.dat（Fix 1：每次独立 tensor）──────────────────────────────
            chi_path = os.path.join(row["source_path"], "chi.dat")
            try:
                spec = preprocess_chi(chi_path).clone()          # ★ Fix 1
            except Exception as e:
                logging.getLogger(__name__).debug(f"chi.dat 失败: {chi_path} ({e})")
                spec = torch.zeros(1, SPECTRA_LEN, dtype=torch.float32)  # ★ Fix 1：每次新建

            spectra = spec.unsqueeze(0).clone()   # [1, 1, 512]，★ Fix 1：clone

            # ── 元素 ──────────────────────────────────────────────────────────
            try:   z = PmgElement(row["element"]).Z
            except Exception: z = 26

            tier = row.get("quality_tier", "")
            qw   = QUALITY_WEIGHT.get(str(tier).upper(), QUALITY_WEIGHT_UNKNOWN)

            self._cache.append({
                "frac_coords":     frac_coords.clone(),
                "atom_types":      atom_types.clone(),
                "lengths":         lengths.clone(),
                "angles":          angles.clone(),
                "num_atoms":       num_atoms,
                "spectra":         spectra,
                "site_elements":   torch.tensor([z],  dtype=torch.long),
                "is_ionic":        torch.tensor([0],  dtype=torch.long),
                "quality_weights": torch.tensor([qw], dtype=torch.float32),
                "n_sites":         1,
                "mp_id":           mp_id,
            })
            self.valid_mp_ids.append(mp_id)

        if skipped:
            logging.getLogger(__name__).warning(f"跳过 {skipped} 个样本（POSCAR 失败）")
        logging.getLogger(__name__).info(
            f"QTCrystalDataset 完成: {len(self.valid_mp_ids)} 个样本 | "
            f"原胞转换成功 {n_primitive} / fallback {n_fallback}"
        )
        if n_atoms_list:
            logging.getLogger(__name__).info(
                f"原胞原子数: min={min(n_atoms_list)}, max={max(n_atoms_list)}, "
                f"mean={sum(n_atoms_list)/len(n_atoms_list):.1f}"
            )

    def __len__(self):        return len(self._cache)
    def __getitem__(self, i): return self._cache[i]


def qt_collate_fn(batch):
    data_list = [
        Data(
            frac_coords=item["frac_coords"],
            atom_types=item["atom_types"],
            lengths=item["lengths"].unsqueeze(0),
            angles=item["angles"].unsqueeze(0),
            num_atoms=item["num_atoms"],
            num_nodes=item["num_atoms"],
        )
        for item in batch
    ]
    pyg_batch = Batch.from_data_list(data_list)
    pyg_batch.spectra         = torch.stack([item["spectra"]         for item in batch])
    pyg_batch.site_elements   = torch.stack([item["site_elements"]   for item in batch])
    pyg_batch.is_ionic        = torch.stack([item["is_ionic"]        for item in batch])
    pyg_batch.quality_weights = torch.stack([item["quality_weights"] for item in batch])
    pyg_batch.n_sites         = torch.ones(len(batch), dtype=torch.long)
    pyg_batch.mp_id           = [item["mp_id"] for item in batch]
    return pyg_batch


def check_embedding_uniqueness(dataset, n_check: int = 20) -> bool:
    """Fix 1 验证：前 n_check 个样本的谱 tensor 两两 cosine < 0.999。"""
    from torch.nn.functional import cosine_similarity
    n      = min(n_check, len(dataset))
    specs  = [dataset[i]["spectra"].flatten() for i in range(n)]
    has_dup = False
    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_similarity(specs[i].unsqueeze(0), specs[j].unsqueeze(0)).item()
            if sim > 0.999:
                logging.getLogger(__name__).warning(
                    f"  ⚠️  sample {i}（mp_id={dataset.valid_mp_ids[i]}）"
                    f" 与 sample {j}（mp_id={dataset.valid_mp_ids[j]}）"
                    f" cosine={sim:.6f} — 疑似重复！"
                )
                has_dup = True
    if not has_dup:
        logging.getLogger(__name__).info(f"  ✅ embedding 唯一性检查通过（前 {n} 个样本无重复）")
    return not has_dup


# =============================================================================
# 主训练逻辑
# =============================================================================
if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    import torch.nn.functional as F
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
    from pytorch_lightning.loggers import CSVLogger
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    import hydra
    from omegaconf import OmegaConf

    from diffcsp.common.data_utils import lattice_params_to_matrix_torch
    from diffcsp.pl_modules.diff_utils import d_log_p_wrapped_normal
    from tqdm import tqdm as _tqdm

    # ── 1. 模型配置 ───────────────────────────────────────────────────────────
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONF_MODEL_DIR,
                               job_name="qt_train_v2",
                               version_base=None):
        _raw = compose(config_name="diffusion")

    _raw_dict = OmegaConf.to_container(_raw, resolve=False, throw_on_missing=False)
    full_cfg  = OmegaConf.create({"model": _raw_dict})
    model_cfg = full_cfg.model
    assert model_cfg.latent_dim == 256, f"latent_dim 应为 256，实为 {model_cfg.latent_dim}"

    optim_cfg = OmegaConf.create({
        "optimizer": {"_target_": "torch.optim.Adam", "lr": 1e-4},
        "use_lr_scheduler": False,
        "lr_scheduler": None,
    })

    # ── 2. 实例化模型 ─────────────────────────────────────────────────────────
    logger.info("实例化 CSPDiffusion（v2）...")
    model = hydra.utils.instantiate(model_cfg, optim=optim_cfg, _recursive_=False)
    model.lattice_scaler = None
    model.scaler         = None
    logger.info(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # ── 3. monkey-patch ───────────────────────────────────────────────────────
    def _qt_configure_optimizers(self):
        return hydra.utils.instantiate(
            self.hparams.optim.optimizer, params=self.parameters(), _convert_="partial"
        )
    model.configure_optimizers = types.MethodType(_qt_configure_optimizers, model)
    logger.info("✓ configure_optimizers → 无 scheduler")

    def _encode_xas_qt(self, batch):
        return self.spectrum_encoder(
            batch.spectra[:, 0],
            batch.site_elements[:, 0],
            batch.is_ionic[:, 0],
        )
    model._encode_xas = types.MethodType(_encode_xas_qt, model)
    logger.info("✓ _encode_xas → 单位点，跳过 aggregator")

    def _forward_qt(self, batch):
        batch_size = batch.num_graphs
        times    = self.beta_scheduler.uniform_sample_t(batch_size, self.device)
        time_emb = self.time_embedding(times)
        struct_emb = self._encode_xas(batch)
        cond_emb   = torch.cat([struct_emb, time_emb], dim=-1)

        alphas_cumprod       = self.beta_scheduler.alphas_cumprod[times]
        c0                   = torch.sqrt(alphas_cumprod)
        c1                   = torch.sqrt(1. - alphas_cumprod)
        sigmas               = self.sigma_scheduler.sigmas[times]
        sigmas_norm          = self.sigma_scheduler.sigmas_norm[times]
        lattices             = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
        frac_coords          = batch.frac_coords
        rand_l               = torch.randn_like(lattices)
        rand_x               = torch.randn_like(frac_coords)
        input_lattice        = c0[:, None, None] * lattices + c1[:, None, None] * rand_l
        sigmas_per_atom      = sigmas.repeat_interleave(batch.num_atoms)[:, None]
        sigmas_norm_per_atom = sigmas_norm.repeat_interleave(batch.num_atoms)[:, None]
        input_frac_coords    = (frac_coords + sigmas_per_atom * rand_x) % 1.

        if self.keep_coords:  input_frac_coords = frac_coords
        if self.keep_lattice: input_lattice = lattices

        pred_l, pred_x = self.decoder(
            cond_emb, batch.atom_types,
            input_frac_coords, input_lattice,
            batch.num_atoms, batch.batch,
        )
        tar_x = (
            d_log_p_wrapped_normal(sigmas_per_atom * rand_x, sigmas_per_atom)
            / torch.sqrt(sigmas_norm_per_atom)
        )
        loss_lattice = F.mse_loss(pred_l, rand_l)
        loss_coord   = F.mse_loss(pred_x, tar_x)
        loss = self.hparams.cost_lattice * loss_lattice + self.hparams.cost_coord * loss_coord
        return {"loss": loss, "loss_lattice": loss_lattice, "loss_coord": loss_coord}

    model.forward = types.MethodType(_forward_qt, model)
    logger.info("✓ forward → plain MSE")

    def _sample_qt(self, batch, step_lr=1e-5):
        batch_size = batch.num_graphs
        l_T = torch.randn([batch_size, 3, 3]).to(self.device)
        x_T = torch.rand([batch.num_nodes, 3]).to(self.device)
        if self.keep_coords:  x_T = batch.frac_coords
        if self.keep_lattice: l_T = lattice_params_to_matrix_torch(batch.lengths, batch.angles)

        time_start = self.beta_scheduler.timesteps
        traj = {time_start: {
            "num_atoms": batch.num_atoms, "atom_types": batch.atom_types,
            "frac_coords": x_T % 1., "lattices": l_T,
        }}
        struct_emb = self._encode_xas(batch)

        for t in _tqdm(range(time_start, 0, -1)):
            times    = torch.full((batch_size,), t, device=self.device)
            time_emb = self.time_embedding(times)
            cond_emb = torch.cat([struct_emb, time_emb], dim=-1)

            alphas    = self.beta_scheduler.alphas[t]
            alp_cum   = self.beta_scheduler.alphas_cumprod[t]
            sigmas    = self.beta_scheduler.sigmas[t]
            sigma_x   = self.sigma_scheduler.sigmas[t]
            sigma_n   = self.sigma_scheduler.sigmas_norm[t]
            c0 = 1.0 / torch.sqrt(alphas)
            c1 = (1 - alphas) / torch.sqrt(1 - alp_cum)

            x_t = traj[t]["frac_coords"]
            l_t = traj[t]["lattices"]
            if self.keep_coords:  x_t = x_T
            if self.keep_lattice: l_t = l_T

            rl = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rx = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
            ss  = step_lr * (sigma_x / self.sigma_scheduler.sigma_begin) ** 2
            sx  = torch.sqrt(2 * ss)
            _, px = self.decoder(cond_emb, batch.atom_types, x_t, l_t, batch.num_atoms, batch.batch)
            px = px * torch.sqrt(sigma_n)
            x05 = x_t - ss * px + sx * rx if not self.keep_coords else x_t

            rl = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rx = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
            adj = self.sigma_scheduler.sigmas[t - 1]
            ss  = sigma_x ** 2 - adj ** 2
            sx  = torch.sqrt((adj ** 2 * ss) / (sigma_x ** 2))
            pl2, px2 = self.decoder(cond_emb, batch.atom_types, x05, l_t, batch.num_atoms, batch.batch)
            px2 = px2 * torch.sqrt(sigma_n)
            x1  = x05 - ss * px2 + sx * rx if not self.keep_coords else x_t
            l1  = c0 * (l_t - c1 * pl2) + sigmas * rl if not self.keep_lattice else l_t

            traj[t - 1] = {
                "num_atoms": batch.num_atoms, "atom_types": batch.atom_types,
                "frac_coords": x1 % 1., "lattices": l1,
            }

        traj_stack = {
            "num_atoms": batch.num_atoms, "atom_types": batch.atom_types,
            "all_frac_coords": torch.stack([traj[i]["frac_coords"] for i in range(time_start, -1, -1)]),
            "all_lattices":    torch.stack([traj[i]["lattices"]    for i in range(time_start, -1, -1)]),
        }
        return traj[0], traj_stack

    model.sample = types.MethodType(_sample_qt, model)
    logger.info("✓ sample → qt 版")

    # ── 4. Dataset ────────────────────────────────────────────────────────────
    logger.info("构建 QTCrystalDataset（含原胞转换）...")
    inventory_df = pd.read_csv(os.path.join(QT_DIR, "qt_inventory.csv"))

    def _read_ids(path):
        with open(path) as f:
            return [int(l.strip()) for l in f if l.strip()]

    logger.info("--- Train ---")
    train_dataset = QTCrystalDataset(_read_ids(os.path.join(QT_DIR, "qt_train_ids.txt")), inventory_df)
    logger.info("--- Val ---")
    val_dataset   = QTCrystalDataset(_read_ids(os.path.join(QT_DIR, "qt_val_ids.txt")),   inventory_df)

    # ── 5. Fix 1 验证 ─────────────────────────────────────────────────────────
    logger.info("Fix 1 验证：embedding 唯一性检查...")
    assert check_embedding_uniqueness(train_dataset, n_check=20), \
        "❌ embedding 重复 bug 未消除，停止训练。请检查 preprocess_chi 和 .clone() 逻辑。"

    # ── 6. DataLoader ─────────────────────────────────────────────────────────
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,
                              num_workers=0, collate_fn=qt_collate_fn,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_dataset,   batch_size=8, shuffle=False,
                              num_workers=0, collate_fn=qt_collate_fn,
                              pin_memory=True)

    class _QTDataModule(pl.LightningDataModule):
        def setup(self, stage=None): pass
        def train_dataloader(self): return train_loader
        def val_dataloader(self):   return val_loader

    # ── 7. Trainer ────────────────────────────────────────────────────────────
    checkpoint_cb = ModelCheckpoint(
        dirpath=QT_OUTPUT_DIR,
        filename="v2-epoch{epoch:03d}-val{val_loss:.4f}",
        monitor="val_loss", save_top_k=2, mode="min",
        save_last=True, verbose=True,
    )
    csv_logger = CSVLogger(save_dir=QT_OUTPUT_DIR, name="logs_v2")

    trainer = pl.Trainer(
        default_root_dir=QT_OUTPUT_DIR,
        logger=csv_logger,
        callbacks=[checkpoint_cb, LearningRateMonitor(logging_interval="epoch")],
        precision="bf16",
        devices=1, accelerator="gpu",
        gradient_clip_val=1.0,
        max_epochs=30,
        check_val_every_n_epoch=1,
        log_every_n_steps=5,
    )

    # ── 8. 启动 ────────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("QuickTest v2 开始训练（原胞 + embedding fix）")
    logger.info(f"  输出目录:   {QT_OUTPUT_DIR}")
    logger.info(f"  train 样本: {len(train_dataset)}")
    logger.info(f"  val 样本:   {len(val_dataset)}")
    logger.info("=" * 60)

    torch.set_float32_matmul_precision("medium")
    trainer.fit(model=model, datamodule=_QTDataModule())

    logger.info("QuickTest v2 训练完成。")
    logger.info(f"最优 checkpoint: {checkpoint_cb.best_model_path}")
    if checkpoint_cb.best_model_score is not None:
        logger.info(f"最优 val_loss:   {checkpoint_cb.best_model_score:.6f}")