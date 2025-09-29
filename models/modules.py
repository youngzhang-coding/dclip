# models/modules.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import math
from typing import List


class MultiHeadAttnPooling(nn.Module):
    """
    Multi-head attention pooling: (B, N, D) -> (B, C, D)
    - Uses C learnable queries (slots) as the query sequence to aggregate inputs into C representations.
    - Supports optional attention_mask: (B, N), where 1/True means valid and 0/False means masked.
    """
    def __init__(
        self,
        d_model: int,
        n_pools: int,     # C
        n_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_pools = n_pools
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # C learnable queries (slots)
        self.query = nn.Parameter(torch.randn(n_pools, d_model))

        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W_q.weight); nn.init.zeros_(self.W_q.bias)
        nn.init.xavier_uniform_(self.W_k.weight); nn.init.zeros_(self.W_k.bias)
        nn.init.xavier_uniform_(self.W_v.weight); nn.init.zeros_(self.W_v.bias)
        nn.init.xavier_uniform_(self.W_o.weight); nn.init.zeros_(self.W_o.bias)
        nn.init.normal_(self.query, mean=0.0, std=1.0 / math.sqrt(self.d_model))

    def forward(
        self,
        x: torch.Tensor,                    # (B, N, D)
        attention_mask: torch.Tensor=None,  # (B, N), 1/True = valid, 0/False = masked
        return_attn: bool=False
    ):
        B, N, D = x.shape
        H, Dh, C = self.n_heads, self.d_head, self.n_pools
        assert D == self.d_model, "Last dim of input must equal d_model"

        # Q: (B, C, D) -> (B, H, C, Dh)
        q = self.W_q(self.query)                            # (C, D)
        q = q.unsqueeze(0).expand(B, C, D)                  # (B, C, D)
        q = q.view(B, C, H, Dh).permute(0, 2, 1, 3)         # (B, H, C, Dh)

        # K/V: (B, N, D) -> (B, H, N, Dh)
        k = self.W_k(x).view(B, N, H, Dh).permute(0, 2, 1, 3)
        v = self.W_v(x).view(B, N, H, Dh).permute(0, 2, 1, 3)

        # Attention logits: (B, H, C, N)
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Dh)

        if attention_mask is not None:
            # mask: (B, 1, 1, N); True/1 keeps, False/0 masks
            mask = attention_mask.bool().unsqueeze(1).unsqueeze(1)  # (B, 1, 1, N)
            attn_logits = attn_logits.masked_fill(~mask, torch.finfo(attn_logits.dtype).min)

        attn = F.softmax(attn_logits, dim=-1)
        attn = self.dropout(attn)

        # (B, H, C, Dh)
        pooled = torch.matmul(attn, v)
        # -> (B, C, D)
        pooled = pooled.transpose(1, 2).contiguous().view(B, C, D)
        out = self.W_o(pooled)  # (B, C, D)

        if return_attn:
            # Return attention averaged over heads: (B, C, N)
            attn_mean = attn.mean(dim=1)  # (B, C, N)
            return out, attn_mean
        return out


def _flatten_and_pad_regions(regions_per_image) -> tuple[torch.Tensor, torch.Tensor, list, List]:
    """
    regions_per_image: List[List[Tensor(N_i, D)]], len = batch_size, inner len = K_i
    Returns:
      x: Tensor (M, max_N, D) with padding
      mask: Bool Tensor (M, max_N)
      splits: List[int] = [K_1, K_2, ...] (to reconstruct per-image)
      lengths: Long Tensor (M,)
    """
    # Flatten regions
    flat_regions = []
    splits = []
    for regions in regions_per_image:
        splits.append(len(regions))
        flat_regions.extend(regions)

    if len(flat_regions) == 0:
        raise ValueError("No regions found: got empty regions_per_image.")

    # Ensure same device/dtype
    device = flat_regions[0].device
    dtype = flat_regions[0].dtype
    for t in flat_regions:
        if t.ndim != 2:
            raise ValueError(f"Each region tensor must be 2D (N, D), got {t.shape}")
        if t.device != device:
            raise ValueError("All region tensors must be on the same device.")
        if t.dtype != dtype:
            raise ValueError("All region tensors must share the same dtype.")

    lengths = torch.tensor([t.size(0) for t in flat_regions], device=device, dtype=torch.long)
    # Pad to (M, max_N, D)
    x = pad_sequence(flat_regions, batch_first=True)  # (M, max_N, D)
    max_N = x.size(1)
    # Build mask: True for valid tokens, False for padding
    mask = torch.arange(max_N, device=device).unsqueeze(0) < lengths.unsqueeze(1)  # (M, max_N)
    return x, mask, splits, lengths.tolist()


class BatchRaggedAttnPooling(nn.Module):
    def __init__(self, d_model: int, n_pools: int, n_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.pool = MultiHeadAttnPooling(d_model=d_model, n_pools=n_pools, n_heads=n_heads, dropout=dropout)

    @torch.no_grad()
    def infer_output_shape(self, regions_per_image):
        x, _, splits, _ = _flatten_and_pad_regions(regions_per_image)
        M, _, D = x.shape
        return (M, self.pool.n_pools, D), splits

    def forward(self, regions_per_image, return_attn: bool = False):
        x, mask, splits, _ = _flatten_and_pad_regions(regions_per_image)  # x:(M,max_N,D), mask:(M,max_N)
        if return_attn:
            out_flat, attn_flat = self.pool(x, attention_mask=mask, return_attn=True)
            return out_flat, splits, attn_flat
        else:
            out_flat = self.pool(x, attention_mask=mask, return_attn=False)
            return out_flat, splits


if __name__ == "__main__":
    # Simple test
    batch_size = 3
    d_model = 16
    n_pools = 32
    n_heads = 4

    regions_per_image = [
        [torch.randn(5, d_model), torch.randn(3, d_model)],  # Image 1 with 2 region sets
        [torch.randn(4, d_model)],                             # Image 2 with 1 region set,
        [torch.randn(2, d_model), torch.randn(6, d_model), torch.randn(1, d_model)]  # Image 3 with 3 region sets
    ]

    model = BatchRaggedAttnPooling(d_model=d_model, n_pools=n_pools, n_heads=n_heads, dropout=0.1)
    out, splits = model(regions_per_image)
    print("Output shape:", out.shape)
    print("Splits:", splits)