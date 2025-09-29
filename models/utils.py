import torch
from torch.nn.utils.rnn import pad_sequence
from typing import Iterable, List, Sequence, Tuple, Union

# Types
Box = Union[Sequence[float], torch.Tensor]  # (x1, y1, x2, y2)


# ----- helpers (simplified names) -----
def _norm_boxes(boxes: torch.Tensor) -> torch.Tensor:
    """Put boxes in (x1<=x2, y1<=y2); shape (..., 4)"""
    tl = torch.minimum(boxes[..., :2], boxes[..., 2:])
    br = torch.maximum(boxes[..., :2], boxes[..., 2:])
    return torch.cat([tl, br], dim=-1)


def _area(boxes: torch.Tensor) -> torch.Tensor:
    wh = (boxes[..., 2:] - boxes[..., :2]).clamp(min=0)
    return wh[..., 0] * wh[..., 1]


def _iou_mat(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    IoU between two sets of boxes.
    a: (N, 4), b: (M, 4) -> (N, M)
    """
    a = _norm_boxes(a)
    b = _norm_boxes(b)

    tl = torch.maximum(a[:, None, :2], b[None, :, :2])  # (N, M, 2)
    br = torch.minimum(a[:, None, 2:], b[None, :, 2:])  # (N, M, 2)
    wh = (br - tl).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    ua = _area(a)[:, None]
    ub = _area(b)[None, :]
    union = ua + ub - inter
    return torch.where(union > 0, inter / union, inter.new_zeros(()).expand_as(inter))


def _grid_boxes(
    img_sz: Tuple[int, int],
    patch_sz: Tuple[int, int],
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """
    Build all patch boxes for an image as a grid.
    Returns:
      - boxes: (N, 4)
      - ids: (N,)
      - nh, nw
    """
    H, W = img_sz
    h, w = patch_sz
    assert h > 0 and w > 0
    nh, nw = H // h, W // w
    assert nh > 0 and nw > 0

    ys = torch.arange(nh, device=device, dtype=dtype) * float(h)
    xs = torch.arange(nw, device=device, dtype=dtype) * float(w)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    x1 = gx.reshape(-1)
    y1 = gy.reshape(-1)
    x2 = x1 + float(w)
    y2 = y1 + float(h)
    boxes = torch.stack([x1, y1, x2, y2], dim=1)
    ids = torch.arange(nh * nw, device=device, dtype=torch.long)
    return boxes, ids, nh, nw


def iou(box1: Box, box2: Box) -> float:
    """
    IoU of two boxes (x1, y1, x2, y2) -> float in [0, 1]
    """
    b1 = torch.as_tensor(box1, dtype=torch.float32).reshape(1, 4)
    b2 = torch.as_tensor(box2, dtype=torch.float32).reshape(1, 4)
    return float(_iou_mat(b1, b2)[0, 0].item())


def check(box1: Box, box2: Box, thr: float = 0.5) -> bool:
    return iou(box1, box2) >= thr


def ids_in_region(
    img_sz: Tuple[int, int],
    patch_sz: Tuple[int, int],
    region: Box,
    thr: float = 0.5,
) -> List[int]:
    """
    Patch ids whose IoU with region >= thr (single image).
    """
    device = torch.device("cpu")
    boxes, ids, _, _ = _grid_boxes(img_sz, patch_sz, device=device)
    r = torch.as_tensor(region, dtype=boxes.dtype, device=device).reshape(1, 4)
    m = (_iou_mat(boxes, r).squeeze(1) >= thr)
    return ids[m].tolist()


def ids_in_regions(
    img_sz: Tuple[int, int],
    patch_sz: Tuple[int, int],
    regions: Iterable[Box],
    thr: float = 0.5,
) -> List[int]:
    """
    Patch ids whose IoU with ANY region >= thr (single image).
    """
    device = torch.device("cpu")
    boxes, ids, _, _ = _grid_boxes(img_sz, patch_sz, device=device)
    regs = torch.as_tensor(list(regions), dtype=boxes.dtype, device=device).reshape(-1, 4)
    if regs.numel() == 0:
        return []
    m = (_iou_mat(boxes, regs) >= thr).any(dim=1)
    return ids[m].tolist()


def batch_ids_in_regions(
    img_szs: Sequence[Tuple[int, int]],
    patch_szs: Sequence[Tuple[int, int]],
    batch_regs: Sequence[Iterable[Box]],
    thr: float = 0.5,
) -> List[List[int]]:
    """
    Patch ids per image for multiple images and regions.
    """
    assert len(img_szs) == len(patch_szs) == len(batch_regs)
    out: List[List[int]] = []
    for sz, pz, regs in zip(img_szs, patch_szs, batch_regs):
        out.append(ids_in_regions(sz, pz, regs, thr))
    return out


def pad_ids(pids: Sequence[Sequence[int]], max_len: int) -> torch.Tensor:
    """
    Pad variable-length id lists to (B, max_len) with -1.
    """
    B = len(pids)
    out = torch.full((B, max_len), -1, dtype=torch.long)
    for i, ids in enumerate(pids):
        if not ids:
            continue
        t = torch.as_tensor(ids, dtype=torch.long)
        n = min(t.numel(), max_len)
        out[i, :n] = t[:n]
    return out


def extract_tokens_by_regions(
    image_features: torch.Tensor,
    img_sz: Tuple[int, int],
    patch_sz: Tuple[int, int],
    regions: Iterable[Box],
    thr: float = 0.5,
    regions_normalized: bool = False,
) -> List[dict]:
    """
    Extract token groups per region from a single image, plus one background group.
    Assumes image_features has NO CLS token (length == nh*nw).

    Args:
        image_features: (N, D) or (1, N, D) features for a single image (no CLS).
        img_sz: (H, W) in pixels.
        patch_sz: (h, w) in pixels.
        regions: iterable of boxes (x1, y1, x2, y2). If regions_normalized=True, coords in [0,1].
        thr: IoU threshold to select patches.
        regions_normalized: whether regions are normalized.

    Returns:
        A list of dicts for each region and one background:
            {
              "type": "region" | "background",
              "index": int | None,
              "box": Tuple[float,float,float,float] | None,  # pixel coords for regions
              "patch_ids": LongTensor (Ki,),
              "tokens": Tensor (Ki, D)
            }
        The last element is the background group.
    """
    # Normalize features to (N, D)
    if image_features.dim() == 3:
        assert image_features.size(0) == 1, "Expect a single image; got batch size > 1."
        feats = image_features[0]
    elif image_features.dim() == 2:
        feats = image_features
    else:
        raise ValueError("image_features must be (N, D) or (1, N, D) without CLS.")
    N, D = feats.shape

    H, W = img_sz
    h, w = patch_sz
    assert h > 0 and w > 0 and (H % h == 0) and (W % w == 0), "Image size must be divisible by patch size."
    nh, nw = H // h, W // w
    n_patches = nh * nw
    if N != n_patches:
        raise ValueError(f"Feature length {N} != nh*nw ({n_patches}). Make sure no CLS is present.")

    # Prepare pixel-space regions
    regs_px: List[Tuple[float, float, float, float]] = []
    for r in regions:
        x1, y1, x2, y2 = map(float, r)
        if regions_normalized:
            x1, x2 = x1 * W, x2 * W
            y1, y2 = y1 * H, y2 * H
        regs_px.append((x1, y1, x2, y2))

    groups: List[dict] = []
    covered: set = set()

    # Per-region groups
    for i, box in enumerate(regs_px):
        pids = ids_in_region(img_sz, patch_sz, box, thr=thr)  # List[int]
        covered.update(pids)
        if len(pids) == 0:
            pid_t = torch.empty((0,), dtype=torch.long, device=feats.device)
            tok_t = feats.new_zeros((0, D))
        else:
            pid_t = torch.as_tensor(pids, dtype=torch.long, device=feats.device)
            tok_t = feats.index_select(0, pid_t)
        groups.append(
            {"type": "region", "index": i, "box": box, "patch_ids": pid_t, "tokens": tok_t}
        )

    # Background group = all patches not covered by any region
    bg_ids = sorted(set(range(n_patches)) - covered)
    if len(bg_ids) == 0:
        bg_pid_t = torch.empty((0,), dtype=torch.long, device=feats.device)
        bg_tok_t = feats.new_zeros((0, D))
    else:
        bg_pid_t = torch.as_tensor(bg_ids, dtype=torch.long, device=feats.device)
        bg_tok_t = feats.index_select(0, bg_pid_t)

    groups.append(
        {"type": "background", "index": None, "box": None, "patch_ids": bg_pid_t, "tokens": bg_tok_t}
    )
    return groups


def extract_tokens_by_regions_batch(
    image_features: torch.Tensor,
    img_sz: Union[Tuple[int, int], Sequence[Tuple[int, int]]],
    patch_sz: Union[Tuple[int, int], Sequence[Tuple[int, int]]],
    batch_regions: Sequence[Iterable[Box]],
    thr: float = 0.5,
    regions_normalized: bool = False,
) -> List[List[dict]]:
    """
    Batched version of extract_tokens_by_regions. Assumes features have NO CLS.

    Args:
        image_features: (B, N, D) features for a batch of images (no CLS).
        img_sz: (H, W) shared across batch, or a sequence of length B.
        patch_sz: (h, w) shared across batch, or a sequence of length B.
        batch_regions: sequence of regions per image; each is iterable of boxes.
        thr: IoU threshold.
        regions_normalized: whether regions are normalized per image.

    Returns:
        List over batch; each item is the same structure returned by extract_tokens_by_regions.
    """
    assert image_features.dim() == 3, "image_features must be (B, N, D) without CLS."
    B, N, D = image_features.shape
    assert len(batch_regions) == B, "batch_regions length must match batch size."

    # Normalize img_sz and patch_sz to per-sample lists
    if isinstance(img_sz, tuple):
        img_szs = [img_sz] * B
    else:
        assert len(img_sz) == B, "img_sz must be (H,W) or a sequence of length B."
        img_szs = list(img_sz)

    if isinstance(patch_sz, tuple):
        patch_szs = [patch_sz] * B
    else:
        assert len(patch_sz) == B, "patch_sz must be (h,w) or a sequence of length B."
        patch_szs = list(patch_sz)

    out: List[List[dict]] = []
    for i in range(B):
        feats_i = image_features[i]  # (N, D)
        groups_i = extract_tokens_by_regions(
            feats_i, img_sz=img_szs[i], patch_sz=patch_szs[i],
            regions=batch_regions[i], thr=thr, regions_normalized=regions_normalized
        )
        out.append(groups_i)
    return out


def scale_box(box: Box, image_size: Tuple[int, int]) -> Box:
    """
    Scale a box from normalized [0,1] to pixel space.
    """
    w, h = image_size
    x1, y1, x2, y2 = box
    if not (0 <= x1 <= 1 and 0 <= x2 <= 1 and 0 <= y1 <= 1 and 0 <= y2 <= 1):
        return box
    return (x1 * w, y1 * h, x2 * w, y2 * h)


def create_masks_from_regions(regions: List[List[Box]], image_size: Tuple[int, int]):
    img_h, img_w = image_size
    batch_masks = []
    for region in regions:
        masks = []
        for box in region:
            mask = torch.zeros((1, img_h, img_w), dtype=torch.long)
            x1, y1, x2, y2 = map(int, scale_box(box, (img_w, img_h)))
            mask[:, y1 : y2 + 1, x1 : x2 + 1] = 1 #plus one to include the boundary
            masks.append(mask)
        background_mask = 1 - torch.clamp(torch.sum(torch.cat(masks, dim=0), dim=0, keepdim=True), 0, 1)
        masks.append(background_mask)
        batch_masks.append(torch.cat(masks, dim=0)) # (num_regions, H, W)
    return torch.cat(batch_masks, dim=0)


def flatten_and_pad_regions(regions_per_image) -> tuple[torch.Tensor, torch.Tensor, list, List]:
    """
    regions_per_image: List[List[Tensor(N_i, D)]], len = batch_size, inner len = K_i
    Returns:
      x: Tensor (M, max_N, D) with padding
      mask: Bool Tensor (M, max_N)
      splits: List[int] = [K_1, K_2, ...] (to reconstruct per-image)
      lengths: Long Tensor (M,). Each region's length before padding.
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