import os
import sys
import math
import torch
import pytest

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(ROOT_DIR, "models")
if MODELS_DIR not in sys.path:
    sys.path.insert(0, MODELS_DIR)

from models.utils import (
    _iou_mat,
    _grid_boxes,
    _format_bytes,
    iou,
    check,
    ids_in_region,
    ids_in_regions,
    batch_ids_in_regions,
    pad_ids,
    extract_tokens_by_regions,
    extract_tokens_by_regions_batch,
)

def test_iou_abd_check_basic():
    a = (0, 0, 2, 2)
    b = (0, 0, 2, 2)
    c = (3, 3, 4, 4)
    
    assert math.isclose(iou(a, b), 1.0)
    assert math.isclose(iou(a, c), 0.0)

    assert check(a, b, 0.5) is True
    assert check(a, c, 0.5) is False


def test_iou_handles_unordered_boxes_via_norm():
    # Reversed coordinates should be normalized internally
    a = (3, 3, 1, 1)  # same box as (1,1,3,3)
    b = (1, 1, 3, 3)
    assert math.isclose(iou(a, b), 1.0)


def test_iou_mat_values():
    a = torch.tensor([[0, 0, 2, 2], [0, 0, 1, 1]], dtype=torch.float32)
    b = torch.tensor([[0, 0, 1, 1], [1, 1, 2, 2]], dtype=torch.float32)
    m = _iou_mat(a, b)
    assert m.shape == (2, 2)

    # Expected:
    # a0 vs b0: inter=1, union=4+1-1=4 -> 0.25
    # a0 vs b1: inter=1, union=4+1-1=4 -> 0.25
    # a1 vs b0: inter=1, union=1+1-1=1 -> 1.0
    # a1 vs b1: inter=0, union=1+1-0=2 -> 0.0
    exp = torch.tensor([[0.25, 0.25], [1.0, 0.0]], dtype=torch.float32)
    assert torch.allclose(m, exp, atol=1e-6)
    
def test_iou_mat_values_2():
    a = torch.tensor([[0, 0, 2, 2], [1, 1, 3, 3]], dtype=torch.float32)
    b = torch.tensor([[1.5, 1.5, 2.5, 4.5]], dtype=torch.float32)
    m = _iou_mat(a, b)
    assert m.shape == (2, 1)
    
    # Expected:
    # a0 vs b0: inter=0.25, union=4+3-0.25=6.75 -> ~0.037037
    # a1 vs b0: inter=1.5, union=4+3-1.5=5.5 -> ~0.272727
    exp = torch.tensor([[0.037037], [0.272727]], dtype=torch.float32)
    assert torch.allclose(m, exp, atol=1e-6)


def test_grid_boxes_layout_and_ids():
    H, W = 8, 8
    h, w = 4, 4
    boxes, ids, nh, nw = _grid_boxes((H, W), (h, w), device=torch.device("cpu"))
    assert nh == 2 and nw == 2
    assert boxes.shape == (nh * nw, 4)
    assert ids.shape == (nh * nw,)

    # Expected positions in row-major with indexing='ij'
    exp_boxes = torch.tensor(
        [
            [0, 0, 4, 4],  # id 0
            [4, 0, 8, 4],  # id 1
            [0, 4, 4, 8],  # id 2
            [4, 4, 8, 8],  # id 3
        ],
        dtype=boxes.dtype,
    )
    assert torch.allclose(boxes, exp_boxes, atol=0)


def test_ids_in_region_thresholding():
    H, W = 8, 8
    h, w = 4, 4

    # Exact overlap with top-left patch -> id 0
    ids0 = ids_in_region((H, W), (h, w), (0, 0, 4, 4), thr=0.5)
    assert ids0 == [0]

    # Center box overlaps 4 patches but IoU per patch is 4/28 ~ 0.1429
    ids_low_thr = ids_in_region((H, W), (h, w), (2, 2, 6, 6), thr=0.1)
    assert sorted(ids_low_thr) == [0, 1, 2, 3]

    ids_high_thr = ids_in_region((H, W), (h, w), (2, 2, 6, 6), thr=0.5)
    assert ids_high_thr == []


def test_ids_in_regions_union_behavior():
    H, W = 8, 8
    h, w = 4, 4
    regs = [(0, 0, 4, 4), (4, 4, 8, 8)]
    ids_all = ids_in_regions((H, W), (h, w), regs, thr=0.5)
    assert sorted(ids_all) == [0, 3]

    # Empty regions -> empty result
    ids_empty = ids_in_regions((H, W), (h, w), [], thr=0.5)
    assert ids_empty == []


def test_batch_ids_in_regions_multiple_samples():
    img_szs = [(8, 8), (8, 8)]
    patch_szs = [(4, 4), (4, 4)]
    regs = [
        [(2, 2, 6, 6)],  # low IoU per patch unless low threshold
        [],              # empty
    ]

    # Default thr=0.5 -> first becomes empty, second empty
    out_default = batch_ids_in_regions(img_szs, patch_szs, regs, thr=0.5)
    assert out_default == [[], []]

    # Lower threshold -> first covers 4 patches
    out_low = batch_ids_in_regions(img_szs, patch_szs, regs, thr=0.1)
    assert [sorted(x) for x in out_low] == [[0, 1, 2, 3], []]


def test_pad_ids_shape_and_values():
    pids = [[0, 1, 2], [], [3]]
    out = pad_ids(pids, max_len=4)
    assert out.shape == (3, 4)
    # Row 0
    assert torch.equal(out[0], torch.tensor([0, 1, 2, -1], dtype=torch.long))
    # Row 1
    assert torch.equal(out[1], torch.tensor([-1, -1, -1, -1], dtype=torch.long))
    # Row 2
    assert torch.equal(out[2], torch.tensor([3, -1, -1, -1], dtype=torch.long))


def make_feats(n_patches: int, dim: int = 2):
    # Deterministic features: token i -> [i, i*10]
    idx = torch.arange(n_patches, dtype=torch.float32)
    if dim == 2:
        return torch.stack([idx, idx * 10.0], dim=1)
    return torch.arange(n_patches * dim, dtype=torch.float32).view(n_patches, dim)


def test_extract_tokens_basic_pixels():
    H, W = 8, 8
    h, w = 4, 4
    n_patches = (H // h) * (W // w)  # 4
    feats = make_feats(n_patches, dim=2)  # (4,2)

    # One region exactly the top-left patch box -> id 0
    regions = [(0, 0, 4, 4)]
    groups = extract_tokens_by_regions(
        feats, img_sz=(H, W), patch_sz=(h, w), regions=regions, thr=0.5, regions_normalized=False
    )

    # Expect 1 region group + 1 background group
    assert len(groups) == 2
    g0, gbg = groups[0], groups[1]

    # Region 0
    assert g0["type"] == "region" and g0["index"] == 0
    assert torch.equal(g0["patch_ids"], torch.tensor([0], dtype=torch.long))
    assert torch.allclose(g0["tokens"], feats[0:1])

    # Background ids are [1,2,3]
    assert gbg["type"] == "background"
    assert torch.equal(gbg["patch_ids"], torch.tensor([1, 2, 3], dtype=torch.long))
    assert torch.allclose(gbg["tokens"], feats[1:])


def test_extract_tokens_normalized_multi_regions():
    H, W = 8, 8
    h, w = 4, 4
    feats = make_feats(4, dim=2)

    # Two normalized regions: top-left (id 0) and top-right (id 1)
    regions = [
        (0.0, 0.0, 0.5, 0.5),  # -> id 0
        (0.5, 0.0, 1.0, 0.5),  # -> id 1
    ]
    groups = extract_tokens_by_regions(
        feats, img_sz=(H, W), patch_sz=(h, w), regions=regions, thr=0.5, regions_normalized=True
    )

    assert len(groups) == 3
    g0, g1, gbg = groups

    assert torch.equal(g0["patch_ids"], torch.tensor([0], dtype=torch.long))
    assert torch.equal(g1["patch_ids"], torch.tensor([1], dtype=torch.long))
    # Background should contain ids 2,3 (bottom row)
    assert torch.equal(gbg["patch_ids"], torch.tensor([2, 3], dtype=torch.long))


def test_extract_tokens_empty_regions_returns_background_only():
    H, W = 8, 8
    h, w = 4, 4
    feats = make_feats(4, dim=2)

    groups = extract_tokens_by_regions(
        feats, img_sz=(H, W), patch_sz=(h, w), regions=[], thr=0.5, regions_normalized=False
    )
    assert len(groups) == 1
    gbg = groups[0]
    assert gbg["type"] == "background"
    assert torch.equal(gbg["patch_ids"], torch.tensor([0, 1, 2, 3], dtype=torch.long))


def test_extract_tokens_raises_on_length_mismatch():
    H, W = 8, 8
    h, w = 4, 4
    # Wrong N: expect 4, give 5
    feats_bad = make_feats(5, dim=2)
    with pytest.raises(ValueError):
        extract_tokens_by_regions(
            feats_bad, img_sz=(H, W), patch_sz=(h, w), regions=[], thr=0.5, regions_normalized=False
        )


def test_extract_tokens_batch_mixed():
    H, W = 8, 8
    h, w = 4, 4
    # Batch of 2, each with N=4, D=2
    feats0 = make_feats(4, dim=2)
    feats1 = make_feats(4, dim=2) + 100.0  # distinguish second image
    feats = torch.stack([feats0, feats1], dim=0)  # (B=2, N=4, D=2)

    # Image 0: region -> id 0 (top-left)
    # Image 1: region -> id 3 (bottom-right)
    regs_batch = [
        [(0, 0, 4, 4)],      # pixels
        [(4, 4, 8, 8)],      # pixels
    ]

    groups_batch = extract_tokens_by_regions_batch(
        feats, img_sz=(H, W), patch_sz=(h, w), batch_regions=regs_batch, thr=0.5, regions_normalized=False
    )
    assert len(groups_batch) == 2

    g0 = groups_batch[0]
    g1 = groups_batch[1]

    # Each has 1 region group + 1 background group
    assert len(g0) == 2 and len(g1) == 2

    # Image 0 checks
    assert torch.equal(g0[0]["patch_ids"], torch.tensor([0], dtype=torch.long))
    assert torch.allclose(g0[0]["tokens"], feats0[0:1])
    assert torch.equal(g0[1]["patch_ids"], torch.tensor([1, 2, 3], dtype=torch.long))

    # Image 1 checks
    assert torch.equal(g1[0]["patch_ids"], torch.tensor([3], dtype=torch.long))
    assert torch.allclose(g1[0]["tokens"], feats1[3:4])
    assert torch.equal(g1[1]["patch_ids"], torch.tensor([0, 1, 2], dtype=torch.long))


def test_extract_tokens_background_union_after_multiple_regions():
    H, W = 8, 8
    h, w = 4, 4
    feats = make_feats(4, dim=2)
    # Regions covering ids {0,1}
    regions = [(0, 0, 4, 4), (4, 0, 8, 4)]
    groups = extract_tokens_by_regions(
        feats, img_sz=(H, W), patch_sz=(h, w), regions=regions, thr=0.5, regions_normalized=False
    )
    # Background should be the remaining {2,3}
    assert torch.equal(groups[-1]["patch_ids"], torch.tensor([2, 3], dtype=torch.long))


def test_format_bytes_units():
    """Test that _format_bytes correctly formats bytes to human-readable units"""
    # Test basic units
    assert _format_bytes(0) == "    0.00 B"
    assert _format_bytes(512) == "  512.00 B"
    assert _format_bytes(1024) == "    1.00 KB"
    assert _format_bytes(1048576) == "    1.00 MB"
    assert _format_bytes(1073741824) == "    1.00 GB"
    
    # Test fractional values
    assert _format_bytes(1536) == "    1.50 KB"
    assert _format_bytes(268435456) == "  256.00 MB"
    
    # Test negative values
    assert _format_bytes(-1024) == "   -1.00 KB"
    
    # Test edge cases
    assert _format_bytes(1023) == " 1023.00 B"
    assert _format_bytes(1025) == "    1.00 KB"