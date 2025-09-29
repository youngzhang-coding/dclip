# train/model_wrapper.py
import copy
from typing import Any, List, Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn, optim
from PIL import Image
import pytorch_lightning as pl

from models.model import DCLIP
from models.clip.clip import tokenize
from models.utils import Box
from train.lr_scheduler import build_cosine_with_warmup, cosine_scheduler
from train.data_wrapper import vae_preprocess, clip_preprocess


class DCLIPLightningWrapper(pl.LightningModule):
    """
    Lightning wrapper for DCLIP:
      - Cosine LR with linear warmup (precomputed).
      - (Optional) EMA / momentum encoder with cosine momentum schedule toward 1.0.
      - Parameter grouping for weight decay.
      - Safe clamping of logit_scale (and optional logit_scale_e) if present.
    """

    def __init__(
        self,
        # --- DCLIP architecture hyperparameters ---
        embed_dim: int,
        image_resolution: int,
        vision_layers: int,
        vision_width: int,
        vision_patch_size: int,
        context_length: int,
        vocab_size: int,
        transformer_width: int,
        transformer_heads: int,
        transformer_layers: int,
        model_id: str = "runwayml/stable-diffusion-inpainting",
        contrastive_loss_weight: float = 1.0,
        diffusion_loss_weight: float = 1.0,
        normalized: bool = False,
        region_iou_threshold: float = 0.5,
        # --- Optimization hyperparameters ---
        lr: float = 1e-4,  # peak LR after warmup
        lr_end: float = 1e-6,  # final LR
        lr_start: float = 0.0,  # start of warmup
        weight_decay: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.98,
        eps: float = 1e-8,
        warmup_epochs: int = 5,
        epochs: int = 100,
        update_freq: int = 1,  # == accumulate_grad_batches
        # --- EMA (momentum encoder) ---
        support_ema: bool = True,
        momentum_ema_base: float = 0.996,  # starting momentum (will cosine->1.0)
        momentum_warmup_epochs: int = 0,  # if >0 can warmup momentum schedule start
        validate_with_ema: bool = True,  # also compute EMA losses in validation
        ema_update_after_step: int = 0,  # delay EMA until some steps passed
        ema_include_diffusion: bool = False,  # copy diffusion module into EMA (False saves memory)
        # --- Logging / misc ---
        clamp_logit_scale: bool = True,
        logit_scale_max: float = 4.6052,  # ln(100),
        # --- CUDA Memory Optimization ---
        enable_gradient_checkpoint: bool = False,  # enable gradient checkpointing to save memory
    ):
        super().__init__()
        self.save_hyperparameters()

        # Build base model
        self.model = DCLIP(
            embed_dim=embed_dim,
            image_resolution=image_resolution,
            vision_layers=vision_layers,
            vision_width=vision_width,
            vision_patch_size=vision_patch_size,
            context_length=context_length,
            vocab_size=vocab_size,
            transformer_width=transformer_width,
            transformer_heads=transformer_heads,
            transformer_layers=transformer_layers,
            model_id=model_id,
            device="cuda" if torch.cuda.is_available() else "cpu",
            contrastive_loss_weight=contrastive_loss_weight,
            diffusion_loss_weight=diffusion_loss_weight,
            normalized=normalized,
            region_iou_threshold=region_iou_threshold,
        )

        if enable_gradient_checkpoint:
            self.model.enable_gradient_checkpointing()
            print("Gradient checkpointing enabled.") # No Trainer Attached ! Can't use self.print !

        # Optimization config
        self.base_lr = lr
        self.final_lr = lr_end
        self.start_lr = lr_start
        self.weight_decay = weight_decay
        self.betas = (beta1, beta2)
        self.eps = eps
        self.warmup_epochs = warmup_epochs
        self.planned_epochs = epochs
        self.update_freq = max(1, update_freq)

        # EMA config
        self.support_ema = support_ema
        self.momentum_ema_base = momentum_ema_base
        self.momentum_warmup_epochs = momentum_warmup_epochs
        self.validate_with_ema = validate_with_ema
        self.ema_update_after_step = ema_update_after_step
        self.ema_include_diffusion = ema_include_diffusion

        # Runtime placeholders
        self.lr_schedule: Optional[np.ndarray] = None
        self.steps_per_epoch: Optional[int] = None
        self.total_opt_steps: Optional[int] = None

        self.momentum_schedule: Optional[np.ndarray] = None
        self.ema_model: Optional[nn.Module] = None

        # Logit scale clamp
        self.clamp_logit_scale = clamp_logit_scale
        self.logit_scale_max = logit_scale_max

    # -------------------------------------------------
    # Forward helpers
    # -------------------------------------------------
    def forward(
        self, clip_inputs: torch.Tensor, vae_inputs: torch.Tensor, texts: torch.Tensor, boxes: List[List[Box]]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward with base model.
        Inputs:
          - clip_inputs: preprocessed images for CLIP (B, 3, H, W), float tensor [0, 1]
          - vae_inputs: preprocessed images for VAE (B, 3, H, W), float tensor [-1, 1]
          - texts: tokenized text input (B, T) or raw list of strings
          - boxes: List of List of boxes per image; each box is [x1, y1, x2, y2] in pixels
        Returns a dict with at least "loss" key; may contain others like "contrastive_loss", "diffusion_loss".
        """
        return self.model(clip_inputs, vae_inputs, texts, boxes)

    def forward_ema(
        self, clip_inputs: torch.Tensor, vae_inputs: torch.Tensor, texts: torch.Tensor, boxes: List[List[Box]]
    ) -> Dict[str, torch.Tensor]:
        """
        Run forward with EMA parameters (no grad).
        Returns the same dict keys as base model if available.
        """
        if self.ema_model is None:
            raise RuntimeError("EMA model not initialized.")
        with torch.no_grad():
            return self.ema_model(clip_inputs, vae_inputs, texts, boxes)

    def _maybe_tokenize(self, texts):
        if isinstance(texts, torch.Tensor):
            return texts.to(self.device)
        if isinstance(texts, list):
            return tokenize(texts).to(self.device)
        raise TypeError("texts must be List[str] or torch.Tensor")

    def _prepare_training_input(self, batch):
        images = batch["images"]  # List of PIL images
        texts = self._maybe_tokenize(batch["texts"])  # Tensor or tokenized
        boxes = [det["boxes"] for det in batch["dets"]]
        original_sizes = [img.size for img in images]
        boxes = self._rescale_boxes(boxes, original_sizes)
        clip_inputs = clip_preprocess(images).to(self.device)
        vae_inputs = vae_preprocess(images).to(self.device)
        return clip_inputs, vae_inputs, texts, boxes

    def _rescale_boxes(
        self, boxes: List[List[Box]], original_sizes: List[Tuple[int, int]]
    ):
        if not boxes:
            return []

        img_res = self.hparams.image_resolution
        if isinstance(img_res, (list, tuple)):
            assert len(img_res) == 2, "image_resolution should be int or (H, W)"
            target_h, target_w = float(img_res[0]), float(img_res[1])
        else:
            target_h = target_w = float(img_res)

        scaled: List[List[Box]] = []
        for sample_boxes, (orig_w, orig_h) in zip(boxes, original_sizes):
            if orig_w <= 0 or orig_h <= 0:
                raise ValueError(f"Invalid original size: {(orig_w, orig_h)}")
            w_scale, h_scale = target_w / orig_w, target_h / orig_h
            scaled_sample = []
            for box in sample_boxes:
                if isinstance(box, torch.Tensor):
                    box = box.cpu().numpy()
                x1, y1, x2, y2 = box
                x1 = np.clip(x1 * w_scale, 0, target_w)
                x2 = np.clip(x2 * w_scale, 0, target_w)
                y1 = np.clip(y1 * h_scale, 0, target_h)
                y2 = np.clip(y2 * h_scale, 0, target_h)
                scaled_sample.append([x1, y1, x2, y2])
            scaled.append(scaled_sample)
        return scaled

    # -------------------------------------------------
    # Training / Validation
    # -------------------------------------------------
    def training_step(self, batch, batch_idx):
        clip_inputs, vae_inputs, texts, boxes = self._prepare_training_input(batch)

        out = self(clip_inputs, vae_inputs, texts, boxes)

        loss = out["loss"]

        self.log("train/loss", loss, prog_bar=True)
        if "contrastive_loss" in out:
            self.log("train/contrastive_loss", out["contrastive_loss"])
        if "diffusion_loss" in out:
            self.log("train/diffusion_loss", out["diffusion_loss"])

        if self.lr_schedule is not None:
            self.log("train/lr", self._current_lr_value(), prog_bar=True)
        if self.support_ema and self.momentum_schedule is not None:
            self.log(
                "train/momentum_ema", self._current_momentum_value(), prog_bar=False
            )

        return loss

    # -------------------------------------------------
    # Optimizer & LR schedule
    # -------------------------------------------------
    def configure_optimizers(self):
        """
        Returns optimizer only; LR & EMA schedules applied manually.
        """
        p_wd, p_non_wd = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if (
                p.ndim < 2
                or "bias" in n.lower()
                or "ln" in n.lower()
                or "bn" in n.lower()
            ):
                p_non_wd.append(p)
            else:
                p_wd.append(p)

        self.print(f"[ParamGroups] wd={len(p_wd)}  no_wd={len(p_non_wd)}")

        optim_groups = [
            {"params": p_wd, "weight_decay": self.weight_decay},
            {"params": p_non_wd, "weight_decay": 0.0},
        ]
        optimizer = optim.AdamW(
            optim_groups,
            lr=self.base_lr,
            betas=self.betas,
            eps=self.eps,
        )
        return optimizer

    def on_fit_start(self):
        self.model.diffusion.device = self.device
        self._maybe_build_schedules_and_ema()

    def on_train_start(self):
        self._maybe_build_schedules_and_ema()

    def _maybe_build_schedules_and_ema(self):
        if self.lr_schedule is not None:
            return

        train_loader = (
            self.trainer.datamodule.train_dataloader()
            if self.trainer.datamodule
            else self.trainer._data_connector._train_dataloader_source.dataloader()
        )

        #TODO handle steps_per_epoch when using IterableDataset
        # num_batches = len(train_loader)
        # self.steps_per_epoch = max(1, num_batches // self.update_freq)
        self.steps_per_epoch = 1000

        total_epochs = self.planned_epochs
        if (
            self.trainer.max_epochs is not None
            and self.trainer.max_epochs != total_epochs
        ):
            self.print(
                f"[Warning] epochs={total_epochs} != Trainer.max_epochs={self.trainer.max_epochs}. Using Trainer.max_epochs."
            )
            total_epochs = self.trainer.max_epochs

        # LR schedule
        self.lr_schedule = build_cosine_with_warmup(
            lr=self.base_lr,
            lr_end=self.final_lr,
            epochs=total_epochs,
            steps_per_epoch=self.steps_per_epoch,
            warmup_epochs=self.warmup_epochs,
            lr_start=self.start_lr,
        ).values
        self.total_opt_steps = len(self.lr_schedule)

        # Momentum (EMA) schedule
        if self.support_ema:
            # We do NOT warmup momentum unless specified. Usually directly cosine base->1.
            self.momentum_schedule = cosine_scheduler(
                base_value=self.momentum_ema_base,
                final_value=1.0,
                epochs=total_epochs,
                niter_per_ep=self.steps_per_epoch,
                warmup_epochs=self.momentum_warmup_epochs,
                start_warmup_value=(
                    self.momentum_ema_base
                    if self.momentum_warmup_epochs > 0
                    else self.momentum_ema_base
                ),
            )
            self._build_ema_model()

        self.print(
            f"[LR] schedule steps={self.total_opt_steps}  steps_per_epoch={self.steps_per_epoch}  warmup_epochs={self.warmup_epochs}"
        )
        if self.support_ema:
            self.print(
                f"[EMA] enabled base_m={self.momentum_ema_base}  schedule_len={len(self.momentum_schedule)}  validate_with_ema={self.validate_with_ema}"
            )

    # -------------------------------------------------
    # EMA utilities
    # -------------------------------------------------
    def _build_ema_model(self):
        """
        Build EMA model (deep copy) optionally excluding large diffusion parts.
        """
        # Deepcopy ensures buffers & params are detached
        self.ema_model = copy.deepcopy(self.model).to(self.device)
        for p in self.ema_model.parameters():
            p.requires_grad = False

        if not self.ema_include_diffusion:
            # Optionally drop diffusion to save memory; keep placeholder attr for safety.
            if hasattr(self.ema_model, "diffusion"):
                self.ema_model.diffusion = None  # type: ignore
        self.print("[EMA] model created.")

    @torch.no_grad()
    def _update_ema(self, momentum: float):
        """
        EMA update: ema = ema * m + (1 - m) * model
        Skips if diffusion excluded (those params absent).
        """
        if self.ema_model is None:
            return
        msd = self.model.state_dict()
        esd = self.ema_model.state_dict()
        for k, v in msd.items():
            if k not in esd:
                continue  # skipped keys (e.g., removed diffusion)
            # Only update floating params/buffers
            if not torch.is_floating_point(v):
                esd[k].copy_(v)
                continue
            esd[k].lerp_(v, 1.0 - momentum)

    # -------------------------------------------------
    # Optimizer step override
    # -------------------------------------------------
    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_closure,
        on_tpu: bool = False,
        using_native_amp: bool = False,
        using_lbfgs: bool = False,
    ):
        # Perform optimizer step
        optimizer.step(closure=optimizer_closure)

        # Set LR for NEXT step
        if self.lr_schedule is not None:
            step_index = min(self.global_step, self.total_opt_steps - 1)
            lr_value = float(self.lr_schedule[step_index])
            for pg in optimizer.param_groups:
                pg["lr"] = lr_value

        # Clamp logit scale(s) if present
        if self.clamp_logit_scale:
            self._clamp_logit_scale()

        # EMA update
        if self.support_ema and self.momentum_schedule is not None:
            if self.global_step >= self.ema_update_after_step:
                m = self._current_momentum_value()
                self._update_ema(momentum=m)

    def _clamp_logit_scale(self):
        """
        Clamp logit_scale (and logit_scale_e if exists) to avoid exploding logits.
        """
        target = self.model
        max_v = self.logit_scale_max
        for attr in ["logit_scale", "logit_scale_e"]:
            if hasattr(target, attr):
                val = getattr(target, attr)
                if isinstance(val, torch.Tensor):
                    val.data.clamp_(0, max_v)

    # -------------------------------------------------
    # Helpers for schedules
    # -------------------------------------------------
    def _current_lr_value(self) -> float:
        if self.lr_schedule is None:
            return self.base_lr
        idx = min(self.global_step, len(self.lr_schedule) - 1)
        return float(self.lr_schedule[idx])

    def _current_momentum_value(self) -> float:
        if self.momentum_schedule is None:
            return self.momentum_ema_base
        idx = min(self.global_step, len(self.momentum_schedule) - 1)
        return float(self.momentum_schedule[idx])

    # -------------------------------------------------
    # Checkpoint EMA state (Lightning automatically saves model.state_dict)
    # -------------------------------------------------
    def on_save_checkpoint(self, checkpoint):
        if self.support_ema and self.ema_model is not None:
            checkpoint["ema_state_dict"] = self.ema_model.state_dict()

    def on_load_checkpoint(self, checkpoint):
        if self.support_ema and "ema_state_dict" in checkpoint:
            if self.ema_model is None:
                # Need to build base + EMA schedule before loading
                self._build_ema_model()
            self.ema_model.load_state_dict(checkpoint["ema_state_dict"], strict=False)
            self.print("[EMA] state loaded from checkpoint.")

