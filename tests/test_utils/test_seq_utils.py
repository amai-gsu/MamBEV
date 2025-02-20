import torch
from torch import nn
from einops import pack, repeat, rearrange
import pytest

from projects.MAMBEV.mambev.layers import Traverse2D
from projects.MAMBEV.mambev.utils.sequence_utils import (
    batch_seq_flatten_ex,
    batch_seq_flatten,
    reverse_batch_seq_flatten,
)  # from sequence_utils import


def test_batch_seq_flatten(traversals, data):
    patch_size = data["patch_size"]
    dtype = data["dtype"]
    device = data["device"]
    H = data["H"]
    W = data["W"]
    seq = data["seq"]

    corners = dict(
        tl=torch.tensor((0, 0), device=device, dtype=dtype),
        tr=torch.tensor((0, W - 1), device=device, dtype=dtype),
        br=torch.tensor((H - 1, W - 1), device=device, dtype=dtype),
        bl=torch.tensor((H - 1, 0), device=device, dtype=dtype),
    )

    for traversal in traversals:
        flat, idx, H_rev, W_rev = batch_seq_flatten_ex(
            seq.clone(),
            H,  # type:ignore
            W,  # type:ignore
            traversal,
            patch_size,
        )

        if not isinstance(traversal, tuple):
            assert torch.equal(flat[0, 0, 0], corners[traversal[:2]])
        reverse = reverse_batch_seq_flatten(flat, H_rev, W_rev, traversal, patch_size)
        print("Original Shape:", seq.shape)
        print("Reverse Shape:", reverse.shape)
        assert torch.equal(seq, reverse)
        print(f"PASSED: {traversal}")


@pytest.fixture
def data():
    bs, nc, H, W = 1, 1, 2, 2
    device = "cuda"
    dtype = torch.long

    x = torch.arange(H, dtype=dtype)
    y = torch.arange(W, dtype=dtype)
    row, col = torch.meshgrid(x, y, indexing="ij")
    seq, _ = pack((row, col), "H W *")
    seq = seq.to(device)
    seq = repeat(seq, "H W C -> bs nc H W C", bs=bs, nc=nc)

    return dict(patch_size=(2, 2), dtype=dtype, device=device, H=H, W=W, seq=seq)


def test_Traverse2D(traversals, data):
    patch_size = data["patch_size"]
    dtype = data["dtype"]
    device = data["device"]
    H = data["H"]
    W = data["W"]
    seq = data["seq"]

    trav_gen = Traverse2D(traversals, patch_size, reverse=True)
    flat1, idx1, sizes1 = trav_gen.forward(seq.clone())
    og1 = trav_gen.forward_reverse([f.clone() for f in flat1], sizes1)
    corners = dict(
        tl=torch.tensor((0, 0), device=device, dtype=dtype),
        tr=torch.tensor((0, W - 1), device=device, dtype=dtype),
        br=torch.tensor((H - 1, W - 1), device=device, dtype=dtype),
        bl=torch.tensor((H - 1, 0), device=device, dtype=dtype),
    )
    for i, traversal in enumerate(traversals):
        (flat, idx) = batch_seq_flatten(
            seq.clone(),
            H,
            W,
            traversal,
            patch_size,
        )
        print(trav_gen.flatten_pipes[i])

        # flat1 = rearrange(flat1, "... H W C -> ... (H W) C")

        if not isinstance(traversal, tuple):
            assert torch.equal(flat1[i][0, 0, 0], corners[traversal[:2]])
        assert og1[i].shape == seq.shape
        assert torch.equal(og1[i], seq), f"Reverse FAIL: {traversal}, {traversals}"
        assert torch.equal(flat1[i], flat), f"Parity FAIL: {traversal}, {traversals}"
