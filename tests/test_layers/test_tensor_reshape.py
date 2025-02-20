from einops import pack, repeat
from projects.MAMBEV.mambev.layers import SkipFlip
import pytest
import torch
from torch import nn


@pytest.fixture
def sample():
    bs, nc, H, W = 1, 1, 2, 2
    device = "cuda"
    dtype = torch.long

    x = torch.arange(H, dtype=dtype)
    y = torch.arange(W, dtype=dtype)
    row, col = torch.meshgrid(x, y, indexing="ij")
    seq, _ = pack((row, col), "H W *")
    seq = seq.to(device)
    seq = repeat(seq, "H W C -> bs nc H W C", bs=bs, nc=nc)

    return seq


def test_SkipFlip(sample: torch.Tensor):
    answer = torch.tensor(
        [[[[[0, 0], [0, 1]], [[1, 1], [1, 0]]]]],
        device=sample.device,
        dtype=sample.dtype,
    )
    op = SkipFlip()
    out = op(sample.clone())

    assert torch.equal(out, answer)
    sequential = nn.Sequential()
    sequential.append(op)
    out1 = sequential(sample.clone())
    assert torch.equal(out1, answer)
    assert torch.equal(out1, out)
