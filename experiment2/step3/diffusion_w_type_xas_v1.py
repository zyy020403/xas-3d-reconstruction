"""
Step 3.3 — diffusion_w_type_xas.py  (v2)
==========================================
在原版 diffusion_w_type.py 基础上最小手术：
  1. import SpectrumEncoder（来自 experiment2/step2/）
  2. __init__：新增 self.spectrum_encoder
  3. forward()：condition = cat([time_emb, spectrum_cond], dim=-1)，传给 decoder
  4. sample()：在循环外算 spectrum_cond，循环内拼接 condition

修改记录（v2）：
  - hparams.get() → getattr(self.hparams, key, default)
    OmegaConf DictConfig 不支持 Python dict 的 .get() 方法

关键约束：
  cost_lattice = 0.0 → keep_lattice = True
  采样时晶格全程保持输入的 diag(12,12,12)，不参与扩散。
"""

import math
import sys
import os

# ── SpectrumEncoder 路径 ──────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'step2'))
from spectrum_encoder import SpectrumEncoder  # noqa: E402

import torch
import torch.nn.functional as F
from typing import Any

import hydra
import pytorch_lightning as pl
from tqdm import tqdm

from diffcsp.common.data_utils import (
    lattice_params_to_matrix_torch)
from diffcsp.pl_modules.diff_utils import d_log_p_wrapped_normal

MAX_ATOMIC_NUM = 100


# ── 基类 ──────────────────────────────────────────────────────────────────────

class BaseModule(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()
        if hasattr(self.hparams, "model"):
            self._hparams = self.hparams.model

    def configure_optimizers(self):
        opt = hydra.utils.instantiate(
            self.hparams.optim.optimizer,
            params=self.parameters(), _convert_="partial")
        if not self.hparams.optim.use_lr_scheduler:
            return [opt]
        scheduler = hydra.utils.instantiate(
            self.hparams.optim.lr_scheduler, optimizer=opt)
        return {"optimizer": opt, "lr_scheduler": scheduler,
                "monitor": "val_loss"}


# ── 时间嵌入 ──────────────────────────────────────────────────────────────────

class SinusoidalTimeEmbeddings(pl.LightningModule):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device   = time.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = time[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


# ── 主模型 ────────────────────────────────────────────────────────────────────

class CSPDiffusion(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # 原版组件（不变）
        self.decoder = hydra.utils.instantiate(
            self.hparams.decoder,
            latent_dim = self.hparams.latent_dim + self.hparams.time_dim,
            pred_type  = True,
            smooth     = True)
        self.beta_scheduler  = hydra.utils.instantiate(self.hparams.beta_scheduler)
        self.sigma_scheduler = hydra.utils.instantiate(self.hparams.sigma_scheduler)
        self.time_dim        = self.hparams.time_dim
        self.time_embedding  = SinusoidalTimeEmbeddings(self.time_dim)
        self.keep_lattice    = self.hparams.cost_lattice < 1e-5
        self.keep_coords     = self.hparams.cost_coord  < 1e-5

        # ★ 新增：SpectrumEncoder
        # 用 getattr 而非 .get()，因为 OmegaConf DictConfig 不支持 .get()
        self.spectrum_encoder = SpectrumEncoder(
            xmu_dim    = getattr(self.hparams, 'xmu_dim',   150),
            chi_dim    = getattr(self.hparams, 'chi_dim',   200),
            feat_dim   = getattr(self.hparams, 'feat_dim',  73),
            latent_dim = getattr(self.hparams, 'spectrum_latent_dim', 256),
        )

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, batch):
        batch_size = batch.num_graphs
        times      = self.beta_scheduler.uniform_sample_t(batch_size, self.device)
        time_emb   = self.time_embedding(times)          # (B, 256)

        # ★ 谱条件
        spectrum_cond = self.spectrum_encoder(
            batch.xmu_xanes,       # (B, 150)
            batch.chi1,            # (B, 200)
            batch.feff_features,   # (B, 73)
        )                          # (B, 256)
        condition = torch.cat([time_emb, spectrum_cond], dim=-1)  # (B, 512)

        # 加噪（与原版相同）
        alphas_cumprod = self.beta_scheduler.alphas_cumprod[times]
        beta           = self.beta_scheduler.betas[times]
        c0 = torch.sqrt(alphas_cumprod)
        c1 = torch.sqrt(1. - alphas_cumprod)

        sigmas           = self.sigma_scheduler.sigmas[times]
        sigmas_norm      = self.sigma_scheduler.sigmas_norm[times]

        lattices    = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
        frac_coords = batch.frac_coords

        rand_l, rand_x = torch.randn_like(lattices), torch.randn_like(frac_coords)
        input_lattice  = c0[:, None, None] * lattices + c1[:, None, None] * rand_l

        sigmas_per_atom      = sigmas.repeat_interleave(batch.num_atoms)[:, None]
        sigmas_norm_per_atom = sigmas_norm.repeat_interleave(batch.num_atoms)[:, None]
        input_frac_coords    = (frac_coords + sigmas_per_atom * rand_x) % 1.

        gt_atom_types_onehot = F.one_hot(
            batch.atom_types - 1, num_classes=MAX_ATOMIC_NUM).float()
        rand_t = torch.randn_like(gt_atom_types_onehot)
        atom_type_probs = (
            c0.repeat_interleave(batch.num_atoms)[:, None] * gt_atom_types_onehot
            + c1.repeat_interleave(batch.num_atoms)[:, None] * rand_t)

        if self.keep_coords:
            input_frac_coords = frac_coords
        if self.keep_lattice:
            input_lattice = lattices

        # ★ decoder：传 condition 而非 time_emb
        pred_l, pred_x, pred_t = self.decoder(
            condition,
            atom_type_probs, input_frac_coords, input_lattice,
            batch.num_atoms, batch.batch)

        tar_x = d_log_p_wrapped_normal(
            sigmas_per_atom * rand_x, sigmas_per_atom
        ) / torch.sqrt(sigmas_norm_per_atom)

        loss_lattice = F.mse_loss(pred_l, rand_l)
        loss_coord   = F.mse_loss(pred_x, tar_x)
        loss_type    = F.mse_loss(pred_t, rand_t)
        loss = (self.hparams.cost_lattice * loss_lattice
                + self.hparams.cost_coord  * loss_coord
                + self.hparams.cost_type   * loss_type)

        return {
            'loss':         loss,
            'loss_lattice': loss_lattice,
            'loss_coord':   loss_coord,
            'loss_type':    loss_type,
        }

    # ── sample ────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def sample(self, batch, diff_ratio=1.0, step_lr=1e-5):
        batch_size = batch.num_graphs

        # ★ 谱编码在循环外只算一次
        spectrum_cond = self.spectrum_encoder(
            batch.xmu_xanes, batch.chi1, batch.feff_features)  # (B, 256)

        l_T = torch.randn([batch_size, 3, 3]).to(self.device)
        x_T = torch.rand([batch.num_nodes, 3]).to(self.device)
        t_T = torch.randn([batch.num_nodes, MAX_ATOMIC_NUM]).to(self.device)

        if self.keep_coords:
            x_T = batch.frac_coords
        if self.keep_lattice:
            l_T = lattice_params_to_matrix_torch(batch.lengths, batch.angles)

        traj = {self.beta_scheduler.timesteps: {
            'num_atoms':   batch.num_atoms,
            'atom_types':  t_T,
            'frac_coords': x_T % 1.,
            'lattices':    l_T,
        }}

        for t in tqdm(range(self.beta_scheduler.timesteps, 0, -1)):
            times     = torch.full((batch_size,), t, device=self.device)
            time_emb  = self.time_embedding(times)
            condition = torch.cat([time_emb, spectrum_cond], dim=-1)  # (B, 512)

            alphas         = self.beta_scheduler.alphas[t]
            alphas_cumprod = self.beta_scheduler.alphas_cumprod[t]
            sigmas         = self.beta_scheduler.sigmas[t]
            sigma_x        = self.sigma_scheduler.sigmas[t]
            sigma_norm     = self.sigma_scheduler.sigmas_norm[t]

            c0 = 1.0 / torch.sqrt(alphas)
            c1 = (1 - alphas) / torch.sqrt(1 - alphas_cumprod)

            x_t = traj[t]['frac_coords']
            l_t = traj[t]['lattices']
            t_t = traj[t]['atom_types']

            if self.keep_coords:
                x_t = x_T
            if self.keep_lattice:
                l_t = l_T

            # Corrector
            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_t = torch.randn_like(t_T) if t > 1 else torch.zeros_like(t_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            step_size = step_lr * (sigma_x / self.sigma_scheduler.sigma_begin) ** 2
            std_x     = torch.sqrt(2 * step_size)

            pred_l, pred_x, pred_t = self.decoder(
                condition, t_t, x_t, l_t, batch.num_atoms, batch.batch)
            pred_x = pred_x * torch.sqrt(sigma_norm)

            x_t_minus_05 = (x_t - step_size * pred_x + std_x * rand_x
                            if not self.keep_coords else x_t)
            l_t_minus_05 = l_t
            t_t_minus_05 = t_t

            # Predictor
            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_t = torch.randn_like(t_T) if t > 1 else torch.zeros_like(t_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            adjacent_sigma_x = self.sigma_scheduler.sigmas[t - 1]
            step_size = sigma_x ** 2 - adjacent_sigma_x ** 2
            std_x = torch.sqrt(
                adjacent_sigma_x ** 2 * step_size / sigma_x ** 2)

            pred_l, pred_x, pred_t = self.decoder(
                condition, t_t_minus_05, x_t_minus_05, l_t_minus_05,
                batch.num_atoms, batch.batch)
            pred_x = pred_x * torch.sqrt(sigma_norm)

            x_t_minus_1 = (x_t_minus_05 - step_size * pred_x + std_x * rand_x
                           if not self.keep_coords else x_t)
            l_t_minus_1 = (c0 * (l_t_minus_05 - c1 * pred_l) + sigmas * rand_l
                           if not self.keep_lattice else l_t)
            t_t_minus_1 = c0 * (t_t_minus_05 - c1 * pred_t) + sigmas * rand_t

            traj[t - 1] = {
                'num_atoms':   batch.num_atoms,
                'atom_types':  t_t_minus_1,
                'frac_coords': x_t_minus_1 % 1.,
                'lattices':    l_t_minus_1,
            }

        traj_stack = {
            'num_atoms': batch.num_atoms,
            'atom_types': torch.stack([
                traj[i]['atom_types']
                for i in range(self.beta_scheduler.timesteps, -1, -1)
            ]).argmax(dim=-1) + 1,
            'all_frac_coords': torch.stack([
                traj[i]['frac_coords']
                for i in range(self.beta_scheduler.timesteps, -1, -1)
            ]),
            'all_lattices': torch.stack([
                traj[i]['lattices']
                for i in range(self.beta_scheduler.timesteps, -1, -1)
            ]),
        }
        return traj[0], traj_stack

    # ── Lightning hooks ───────────────────────────────────────────────────────

    def training_step(self, batch: Any, batch_idx: int):
        output_dict = self(batch)
        self.log_dict({
            'train_loss':   output_dict['loss'],
            'lattice_loss': output_dict['loss_lattice'],
            'coord_loss':   output_dict['loss_coord'],
            'type_loss':    output_dict['loss_type'],
        }, on_step=True, on_epoch=True, prog_bar=True)
        if output_dict['loss'].isnan():
            return None
        return output_dict['loss']

    def validation_step(self, batch: Any, batch_idx: int):
        output_dict = self(batch)
        log_dict, loss = self.compute_stats(output_dict, prefix='val')
        self.log_dict(log_dict, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch: Any, batch_idx: int):
        output_dict = self(batch)
        log_dict, loss = self.compute_stats(output_dict, prefix='test')
        self.log_dict(log_dict)
        return loss

    def compute_stats(self, output_dict, prefix):
        log_dict = {
            f'{prefix}_loss':         output_dict['loss'],
            f'{prefix}_lattice_loss': output_dict['loss_lattice'],
            f'{prefix}_coord_loss':   output_dict['loss_coord'],
            f'{prefix}_type_loss':    output_dict['loss_type'],
        }
        return log_dict, output_dict['loss']