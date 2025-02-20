from typing import List, Tuple

import pytest
import torch
from einops import rearrange
from projects.MAMBEV.mambev.modules.hydra_blocks.conv_gate_ablations.post_pre_hydra_block import (
    PostPreHydraBlock,
)
from test_utils import device, create_inputs  # type:ignore

from projects.MAMBEV.mambev.utils.sequence_utils import (
    batch_seq_flatten,
)


def _traverse_image_features(
    self,
    vxBCdt: torch.Tensor,
    level_sizes: List[int],
    spatial_shapes: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    value_level_list: List[torch.Tensor] = vxBCdt.split(level_sizes, dim=2)
    value_level_list = [
        rearrange(
            value_level, "bs num_cams (H W) dim -> bs num_cams H W dim", H=H_, W=W_
        )
        for (H_, W_), value_level in zip(spatial_shapes, value_level_list)
    ]

    # traversals of each image feature level
    # order: F_0 tr0, F_0 tr1, .... F_0 trN, F_1 tr0, ....
    values_traversals = []
    idx_maps = []
    count = 0
    seq_lens = value_level_list[0].new_zeros(
        len(value_level_list) * self.ntraversals + 1, dtype=torch.int64
    )
    for level in range(len(value_level_list)):
        value_level = value_level_list[level]
        H_, W_ = value_level.shape[-3:-1]
        added_len = H_ * W_
        for traversal in self.traversal_methods:
            count += 1
            vt, idx_map = batch_seq_flatten(
                value_level_list[level], H_, W_, traversal, self.patch_size
            )
            values_traversals.append(vt)
            idx_maps.append(idx_map)
            seq_lens[count] += seq_lens[count - 1] + added_len
    values_traversals = torch.cat(values_traversals, dim=-2)
    idx_maps = torch.cat(idx_maps)
    # Value Traversals: Reordered Feature Maps
    return values_traversals, idx_maps, seq_lens


@pytest.fixture
def base_cfg(device):
    return dict(
        d_model=256,
        d_state=8,
        d_conv=7,
        expand=1,
        headdim=256,
        ngroups=1,
        bias=False,
        conv_bias=True,
        use_post_norm=True,
        layer_idx=0,
        chunk_size=32,
        sequence_parallel=True,
        x_og_activation=False,
        device=device,
        dtype=torch.bfloat16,
        feature_levels=[0],
        # collect_method="batch_cam",
        collect_method="bct",
        traversal_methods=("br0cross",),
    )


def test_TraversalParity(base_cfg, traversals, device):
    base_cfg["traversal_methods"] = traversals
    model = PostPreHydraBlock(**base_cfg)  # type:ignore
    model = model.to(device)
    H, W = 12, 24
    (
        inp_res,
        qxBCdt,
        vxBCdt,
        reference_points_cam,
        bev_mask,
        ref_int,
        spatial_shapes,
    ) = create_inputs(model, 4, model.num_cams, 4, 50, 50, H, W)
    level_sizes = [int(H * W) for H, W in spatial_shapes]
    vxBCdt1, vmaps1, seq_lens1 = model._traverse_image_features(
        vxBCdt.clone(), level_sizes, spatial_shapes
    )
    vxBCdt2, vmaps2, seq_lens2 = _traverse_image_features(
        model, vxBCdt.clone(), level_sizes, spatial_shapes
    )
    tensor_diff(vxBCdt1, vxBCdt2, 1)
    tensor_diff(vmaps1, vmaps2)
    tensor_diff(seq_lens1, seq_lens2)


def tensor_diff(src, trg, dim=0):
    mask_nequal = src != trg
    ind_nequal = torch.where(mask_nequal)[dim]
    num_diff = mask_nequal.sum()
    assert not num_diff, f"""Outputs are not the same:
    #Different: {num_diff} / {mask_nequal.numel()}
    Bounds: [{torch.min(ind_nequal)}, {torch.max(ind_nequal)}]"""
