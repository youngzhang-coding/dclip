# models/model.py
from models.clip import RegionCLIPOutput, RegionCLIP
from models.diffusion import FrozenDiffusionWrapper
import torch
import torch.nn as nn
from typing import List, Tuple, Union
from models.utils import extract_tokens_by_regions_batch, create_masks_from_regions, flatten_and_pad_regions, memory_tracker

Box = Union[Tuple[float, float, float, float], torch.Tensor]

class DCLIP(nn.Module):
    """
    DCLIP: Joint region-level contrastive + diffusion loss.
    Now simplified by leveraging utils.extract_tokens_by_regions_batch for IoU-based
    patch membership. We enforce disjoint partitions (earliest region wins) and
    rebuild background accordingly.
    """
    def __init__(
        self,
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
        device: str = "cuda",
        contrastive_loss_weight: float = 1.0,
        diffusion_loss_weight: float = 1.0,
        normalized: bool = False,
        region_iou_threshold: float = 0.5,
    ):
        super().__init__()
        self.clip = RegionCLIP(
            embed_dim,
            image_resolution,
            vision_layers,
            vision_width,
            vision_patch_size,
            context_length,
            vocab_size,
            transformer_width,
            transformer_heads,
            transformer_layers,
            True,
        )
        self.diffusion = FrozenDiffusionWrapper(model_id=model_id, device=device)
        self.device = device
        self.contrastive_loss_weight = float(contrastive_loss_weight)
        self.diffusion_loss_weight = float(diffusion_loss_weight)
        self.image_resolution = image_resolution
        self.vision_patch_size = vision_patch_size
        self.patch_grid = image_resolution // vision_patch_size
        self.normalized = normalized
        self.region_iou_threshold = region_iou_threshold
    
    def enable_gradient_checkpointing(self):
        self.clip.enable_gradient_checkpointing()
    
    def disable_gradient_checkpointing(self):
        self.clip.disable_gradient_checkpointing()
        
    def set_num_checkpoint_segments(self, num_segments: int):
        self.clip.set_num_checkpoint_segments(num_segments)
        
    def _prepare_regions_per_image(self, groups_batch: List[List[dict]]) -> List[List[torch.Tensor]]:
        regions_per_image = []
        for groups in groups_batch: # group keys: 'type', 'index', 'box', 'patch_ids', 'tokens'
            regions = []
            for g in groups:
                tokens = g['tokens']
                regions.append(tokens)
            regions_per_image.append(regions)
        return regions_per_image

    def forward(
        self,
        clip_inputs: torch.Tensor,
        vae_inputs: torch.Tensor,
        texts: torch.Tensor,
        boxes: List[List[Box]],
        **kwargs,
    ):
        clip_out: RegionCLIPOutput = self.clip(clip_inputs, texts)
        contrastive_loss = clip_out.loss
        tokens = clip_out.tokens
        B, N, C = tokens.shape
        H = W = self.image_resolution
        assert N == self.patch_grid ** 2, f"Expected {self.patch_grid**2} tokens, got {N}"

        groups_batch = extract_tokens_by_regions_batch(
            image_features=tokens,
            img_sz=(H, W),
            patch_sz=(self.vision_patch_size, self.vision_patch_size),
            batch_regions=boxes,
            thr=self.region_iou_threshold,
            regions_normalized=self.normalized,
        )

        regions_per_image = self._prepare_regions_per_image(groups_batch)

        out_flat, attn_mask, splits, _ = flatten_and_pad_regions(regions_per_image)
        
        with memory_tracker("repeat_interleave"):
            images_repeated = vae_inputs.repeat_interleave(torch.tensor(splits, device=vae_inputs.device), dim=0)

        masks = create_masks_from_regions(regions=boxes, image_size=(H, W))
        
        diffusion_loss = self.diffusion(images=images_repeated, masks=masks, context=out_flat, context_mask=attn_mask)

        loss = self.contrastive_loss_weight * contrastive_loss + self.diffusion_loss_weight * diffusion_loss
        
        return {
            "loss": loss,
            "contrastive_loss": contrastive_loss,
            "diffusion_loss": diffusion_loss,
        }