# models/diffusion/model.py
from diffusers.pipelines import DiffusionPipeline
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def freeze(module: nn.Module):
    for param in module.parameters():
        param.requires_grad = False


class FrozenDiffusionWrapper(nn.Module):
    """
    A frozen wrapper around StableDiffusionInpaintPipeline that exposes a simple
    training-style loss: predict added noise at random timesteps under inpainting masks,
    conditioned on external context embeddings.
    """

    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", device="cuda"):
        super().__init__()
        self.pipe = DiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, safe_checker=None,
        ).to(device)
        self.device = device

        # Drop components we don't use
        del self.pipe.text_encoder
        del self.pipe.tokenizer
        del self.pipe.feature_extractor
        del self.pipe.image_encoder

        self.vae = self.pipe.vae
        self.unet = self.pipe.unet
        self.scheduler = self.pipe.scheduler

        # Spatial downsample factor for latents (usually 8 for SD1.x)
        self.spatial_downsample = getattr(self.pipe, "vae_scale_factor", 8)
        # Latent scaling factor used by VAE (usually 0.18215 for SD1.x)
        self.latent_scaling = getattr(self.vae.config, "scaling_factor", 0.18215)

        freeze(self.vae)
        freeze(self.unet)

        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
        self.pipe.enable_attention_slicing()
        self.pipe.enable_vae_slicing()

    @torch.no_grad()
    def _encode_vae(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to latents with the VAE, applying scaling factor.
        Images are expected in [0, 1]; they will be mapped to [-1, 1] for VAE.
        """
        images = images.clamp(0, 1)
        images = images * 2.0 - 1.0
        images = images.to(dtype=self.vae.dtype, device=self.device)
        latents = self.vae.encode(images).latent_dist.sample()
        latents = latents * self.latent_scaling
        return latents

    def forward(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute diffusion inpainting loss.

        Args:
            images: (B, 3, H, W) float in [0, 1].
            masks:  (B, 1, H, W) binary mask; 1 indicates region to inpaint.
            context: (B, K, C) encoder hidden states for cross-attention.
            context_mask: (B, K) bool; True for valid tokens. Also used to build attention_mask.
            timesteps: Optional LongTensor of shape (B,) for noise schedule. If None, sampled uniformly.

        Returns:
            loss: scalar tensor (MSE between predicted noise and ground-truth noise).
        """
        assert images.dim() == 4 and images.size(1) == 3, "images must be (B,3,H,W)"
        if masks.dim() == 3:
            masks = masks.unsqueeze(1)
        assert masks.dim() == 4 and masks.size(1) == 1, "masks must be (B,1,H,W)"

        B, _, H, W = images.shape

        # Encode to latents
        with torch.no_grad():
            latents_orig = self._encode_vae(images)  # (B, 4, H/8, W/8)
        

        # Resize mask to latent resolution
        latent_h, latent_w = H // self.spatial_downsample, W // self.spatial_downsample
        masks_latent = F.interpolate(
            masks.float(), size=(latent_h, latent_w), mode="nearest"
        ).to(device=self.device, dtype=latents_orig.dtype)

        latents_orig = latents_orig * masks_latent

        # Prepare noise and timesteps
        noise = torch.randn_like(latents_orig, device=self.device)
        if timesteps is None:
            num_steps = getattr(self.scheduler, "num_train_timesteps", 1000)
            timesteps = torch.randint(
                0, num_steps, (B,), device=self.device, dtype=torch.long
            )

        # Add noise
        latents_noisy = self.scheduler.add_noise(latents_orig, noise, timesteps)

        latents_input = latents_noisy * masks_latent # only input masked region

        # Prepare context
        if context is None or context.numel() == 0:
            # Fallback to a single zero token so UNet cross-attn still works
            cross_dim = getattr(self.unet.config, "cross_attention_dim", None)
            if cross_dim is None:
                raise ValueError(
                    "UNet cross_attention_dim is undefined; provide valid context."
                )
            context = torch.zeros(
                (B, 1, cross_dim), device=self.device, dtype=latents_input.dtype
            )
            attention_mask = None
        else:
            # Ensure device/dtype
            context = context.to(device=self.device, dtype=latents_input.dtype)
            attention_mask = None
            if context_mask is not None and context_mask.numel() > 0:
                # Build boolean mask on the correct device
                cm = context_mask.to(device=self.device).bool()  # (B, K)
                # Ensure at least one valid token per sample to avoid all -inf rows in softmax
                empty = ~cm.any(dim=1)  # (B,)
                if empty.any():
                    cm = cm.clone()
                    cm[empty, 0] = True  # force the first token valid

                # Zero out invalid positions (safe even if attention_mask not consumed)
                context = context * cm.unsqueeze(-1).to(context.dtype)

                # Build additive attention mask over keys (K): 0 for valid, very negative for invalid
                mask_value = torch.finfo(
                    latents_input.dtype
                ).min  # e.g., -65504 for fp16
                attention_mask = (~cm).unsqueeze(1).to(
                    latents_input.dtype
                ) * mask_value  # (B,1,K)

            # Optional: sanity check on feature dim
            cross_dim = getattr(self.unet.config, "cross_attention_dim", None)
            if cross_dim is not None and context.size(-1) != cross_dim:
                raise ValueError(
                    f"context hidden size {context.size(-1)} != UNet cross_attention_dim {cross_dim}"
                )

        # Predict noise (pass attention mask if supported; fallback otherwise)
        try:
            kwargs = {}
            if attention_mask is not None:
                kwargs = {"cross_attention_kwargs": {"attention_mask": attention_mask}}
            noise_pred = self.unet(
                latents_input, timesteps, encoder_hidden_states=context, **kwargs
            ).sample
        except TypeError:
            # Older diffusers without cross_attention_kwargs
            noise_pred = self.unet(
                latents_input, timesteps, encoder_hidden_states=context
            ).sample

        # Loss
        loss = F.mse_loss(noise_pred * masks_latent, noise * masks_latent) # output a scalar
        return loss