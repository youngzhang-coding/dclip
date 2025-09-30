import argparse
import os
import random
from datetime import datetime

import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from swanlab.integration.pytorch_lightning import SwanLabLogger

from train.data_wrapper import get_datamodule, WebDatasetDataModule
from train.model_wrapper import DCLIPLightningWrapper


def parse_args():
    parser = argparse.ArgumentParser("DCLIP Training")

    # Data
    parser.add_argument("--train_tar", type=str, required=True,
                        help="Training WebDataset tar path (pattern allowed, e.g. /path/shards/{000000..000999}.tar)")
    parser.add_argument("--val_tar", type=str, default=None, help="Validation tar (optional)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    # top_k for train stablity and memory saving
    parser.add_argument("--top_k", type=int, default=5, help="Number of top detections to use from the object detector")

    # Model
    parser.add_argument("--embed_dim", type=int, default=768)
    parser.add_argument("--image_resolution", type=int, default=224)
    parser.add_argument("--vision_layers", type=int, default=12)
    parser.add_argument("--vision_width", type=int, default=768)
    parser.add_argument("--vision_patch_size", type=int, default=16)
    parser.add_argument("--context_length", type=int, default=77)
    parser.add_argument("--vocab_size", type=int, default=49408)
    parser.add_argument("--transformer_width", type=int, default=512)
    parser.add_argument("--transformer_heads", type=int, default=8)
    parser.add_argument("--transformer_layers", type=int, default=12)
    parser.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-inpainting")

    # Loss / misc
    parser.add_argument("--contrastive_loss_weight", type=float, default=1.0)
    parser.add_argument("--diffusion_loss_weight", type=float, default=1.0)
    parser.add_argument("--normalized", action="store_true")
    parser.add_argument("--region_iou_threshold", type=float, default=0.5)

    # Optim
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_end", type=float, default=1e-6)
    parser.add_argument("--lr_start", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.98)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--update_freq", type=int, default=1,
                        help="Gradient accumulation steps (= accumulate_grad_batches)")

    # EMA
    parser.add_argument("--no_ema", action="store_true")
    parser.add_argument("--momentum_ema_base", type=float, default=0.996)
    parser.add_argument("--momentum_warmup_epochs", type=int, default=0)
    parser.add_argument("--validate_with_ema", action="store_true")
    parser.add_argument("--ema_update_after_step", type=int, default=0)
    parser.add_argument("--ema_include_diffusion", action="store_true")

    # Runtime / Trainer
    parser.add_argument("--precision", type=str, default="16-mixed",
                        help="Choices: 16-mixed / 32 / bf16-mixed")
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument("--accelerator", type=str, default="gpu" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every_n_steps", type=int, default=50)
    parser.add_argument("--val_check_interval", type=float, default=1.0)
    parser.add_argument("--limit_val_batches", type=float, default=0.0,
                        help="0 disables validation")
    parser.add_argument("--gradient_clip_val", type=float, default=0.0)
    parser.add_argument("--detect_anomaly", action="store_true")
    parser.add_argument("--enable_gradient_checkpoint", action="store_true", help="Enable gradient checkpointing to save memory")

    # IO
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--project", type=str, default="dclip")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--resume_from", type=str, default=None)

    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_datamodule(args):
    if args.val_tar:
        return WebDatasetDataModule(
            train_tar=args.train_tar,
            val_tar=args.val_tar,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            top_k=args.top_k,
        )
    else:
        return get_datamodule(
            tar_path=args.train_tar,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            top_k=args.top_k,
        )


def build_model(args):
    model = DCLIPLightningWrapper(
        embed_dim=args.embed_dim,
        image_resolution=args.image_resolution,
        vision_layers=args.vision_layers,
        vision_width=args.vision_width,
        vision_patch_size=args.vision_patch_size,
        context_length=args.context_length,
        vocab_size=args.vocab_size,
        transformer_width=args.transformer_width,
        transformer_heads=args.transformer_heads,
        transformer_layers=args.transformer_layers,
        model_id=args.model_id,
        contrastive_loss_weight=args.contrastive_loss_weight,
        diffusion_loss_weight=args.diffusion_loss_weight,
        normalized=args.normalized,
        region_iou_threshold=args.region_iou_threshold,
        lr=args.lr,
        lr_end=args.lr_end,
        lr_start=args.lr_start,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        eps=args.eps,
        warmup_epochs=args.warmup_epochs,
        epochs=args.epochs,
        update_freq=args.update_freq,
        support_ema=not args.no_ema,
        momentum_ema_base=args.momentum_ema_base,
        momentum_warmup_epochs=args.momentum_warmup_epochs,
        validate_with_ema=args.validate_with_ema,
        ema_update_after_step=args.ema_update_after_step,
        ema_include_diffusion=args.ema_include_diffusion,
        enable_gradient_checkpoint=args.enable_gradient_checkpoint,
    )
    return model


def build_trainer(args):
    os.makedirs(args.output_dir, exist_ok=True)
    run_name = args.name or datetime.now().strftime("%Y%m%d-%H%M%S")

    logger = SwanLabLogger(
        project=args.project,
        experiment_name=run_name,
        save_dir=os.path.join(args.output_dir, args.project, run_name, "swanlab"),
    )
    logger.log_hyperparams(vars(args))

    ckpt_cb = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, args.project, run_name, "checkpoints"),
        filename="epoch{epoch:03d}-step{step}",
        save_top_k=3,
        monitor="train/loss",
        mode="min",
        save_last=True,
        auto_insert_metric_name=False,
    )
    lr_cb = LearningRateMonitor(logging_interval="step", log_momentum=True, log_weight_decay=True)

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.epochs,
        precision=args.precision,
        accumulate_grad_batches=args.update_freq,
        logger=logger,
        callbacks=[ckpt_cb, lr_cb],
        log_every_n_steps=args.log_every_n_steps,
        val_check_interval=args.val_check_interval,
        limit_val_batches=args.limit_val_batches,
        gradient_clip_val=args.gradient_clip_val if args.gradient_clip_val > 0 else None,
        detect_anomaly=args.detect_anomaly,
        default_root_dir=args.output_dir,
        enable_progress_bar=True,
    )
    return trainer


def main():
    args = parse_args()
    set_seed(args.seed)

    datamodule = build_datamodule(args)
    model = build_model(args)
    trainer = build_trainer(args)

    ckpt_path = args.resume_from if args.resume_from and os.path.isfile(args.resume_from) else None

    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()