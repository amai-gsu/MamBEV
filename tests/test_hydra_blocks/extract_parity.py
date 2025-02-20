from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Tuple, Literal

import pytest
import torch
from einops import parse_shape, rearrange, reduce, repeat
from projects.MAMBEV.mambev.modules.hydra_blocks.conv_gate_ablations.post_pre_hydra_block import (
    PostPreHydraBlock,
)
from test_utils import device, create_inputs  # type:ignore

from projects.MAMBEV.mambev.layers.traversal2d_layer import batched
from projects.MAMBEV.mambev.modules.hydra_blocks.base_hydra_block import ref2int_factory
from projects.MAMBEV.mambev.utils.sequence_utils import (
    batch_seq_flatten,
    get_merge_mask_and_sort,
)

ref2int = ref2int_factory("ceil", 1)


@dataclass()
class RefPtsObj:
    ref_pts: torch.Tensor
    int_ref_pts: torch.Tensor
    bev_mask: torch.Tensor
    HW: Tuple[int, int]


def _bev_aware_merge(
    self,
    qxBCdt: torch.Tensor,
    vxBCdt: torch.Tensor,
    reference_points_cam: torch.Tensor,
    query_grid_hit_mask: torch.Tensor,
    spatial_shapes: torch.Tensor,
    value_map: torch.Tensor,
    seq_lens: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    extract_map, vmask, seq_lens_new = _build_extractmap_mask(
        self,
        qxBCdt,
        reference_points_cam,
        query_grid_hit_mask,
        spatial_shapes,
        value_map,
        seq_lens,
    )
    input_seq = torch.zeros(
        (1, sum(seq_lens_new), self.d_x + 2 * (self.d_B + self.d_C + self.d_dt)),
        dtype=qxBCdt.dtype,
        device=qxBCdt.device,
    )

    vxBCdt = rearrange(vxBCdt, "bs ncntnv c -> (bs ncntnv) c")
    qxBCdt = rearrange(qxBCdt, "bs nq c -> (bs nq) c")

    input_seq[:, vmask] = vxBCdt
    input_seq[:, ~vmask] = qxBCdt[extract_map[1]]
    return input_seq, extract_map, vmask


def _build_extractmap_mask_new_gen(
    self,
    bev_mask: torch.Tensor,
    reference_points: torch.Tensor,
    spatial_shapes: torch.Tensor,
    value_map: torch.Tensor,
    seq_lens: torch.Tensor,
):
    """
    bev_mask (bs, nc, q, z): Mask for the reference_points that fall in bounds
    reference_points (bs, nc, q, z, 2): Locations on the image corresponding to each query
    spatial_shapes (nl, 2): shapes of the image feature maps by level
    value_map (nl, (tr,H*W)): locations of each image feature after traversal and flattening
    ## GOAL ##
    # reorder query_ind so that the correct query is inserted into the sequence
    # update the insert position based on the traversal method

    """

    # bs = bev_mask.size(0)
    # tr_bev_mask = repeat(bev_mask, "bs nc q z -> tr bs nc q z", tr=self.ntraversals)
    # reference_points = repeat(reference_points, "...  -> tr ...", tr=self.ntraversals)
    # # trav_ind, batch_ind, cam_ind, query_ind, pillar_ind = torch.where(bev_mask)[:5]
    # all_inds = torch.nonzero(tr_bev_mask)
    # nquery_per_tr_batch_cam = reduce(
    #     tr_bev_mask.int(), "tr bs nc q z -> (tr bs nc)", "sum"
    # ).tolist()
    # int_ref_bylevel_defaulttr = [
    #     self.ref2int(reference_points[tr_bev_mask], H_, W_, False).split_with_sizes(  # type:ignore
    #         nquery_per_tr_batch_cam, dim=-1
    #     )
    #     for H_, W_ in spatial_shapes
    # ]
    # # assert isinstance(int_ref_bylevel_defaulttr[1], torch.Tensor)
    #
    # all_inds_tr_batch_cam = all_inds.split_with_sizes(nquery_per_tr_batch_cam, dim=0)
    #
    # vmap_level_tr = value_map.split_with_sizes(seq_lens.diff().tolist(), dim=-1)
    #
    # # iterate over levels
    # for init_refs, vtr in zip(
    #     int_ref_bylevel_defaulttr, batched(vmap_level_tr, n=self.ntraversals)
    # ):
    #     # iterate over batch and camera
    #     for a, refs in zip(all_inds_tr_batch_cam, init_refs):
    #         # iterate over num traversals
    #         for vmap in vtr:
    #             update_refs = vmap[refs]
    #             yield update_refs, vmap
    #
    tr_bev_mask = bev_mask
    # tr_bev_mask = repeat(bev_mask, "bs nc q z -> tr bs nc q z", tr=self.ntraversals)
    #
    # reference_points = repeat(
    #     reference_points, "...  -> tr ...", tr=self.ntraversals
    # )
    # trav_ind, batch_ind, cam_ind, query_ind, pillar_ind = torch.where(bev_mask)[:5]
    all_inds = torch.nonzero(tr_bev_mask)
    nquery_per_tr_batch_cam = reduce(
        tr_bev_mask.int(), "bs nc q z -> (bs nc)", "sum"
    ).tolist()
    # nquery_per_tr_batch_cam = reduce(
    #     tr_bev_mask.int(), "tr bs nc q z -> (tr bs nc)", "sum"
    # ).tolist()
    int_ref_bylevel_defaulttr = [
        self.ref2int(reference_points[tr_bev_mask], H_, W_, False).split_with_sizes(  # type:ignore
            nquery_per_tr_batch_cam, dim=-1
        )
        for H_, W_ in spatial_shapes
    ]
    # assert isinstance(int_ref_bylevel_defaulttr[1], torch.Tensor)

    all_inds_tr_batch_cam = all_inds.split_with_sizes(nquery_per_tr_batch_cam, dim=0)

    vmap_level_tr = value_map.split_with_sizes(seq_lens.diff().tolist(), dim=-1)

    update_inds = []
    seq_lens_new = []
    vmasks = []
    q_expand_map = []
    running_q_count = 0
    for init_refs, vtr in zip(
        int_ref_bylevel_defaulttr, batched(vmap_level_tr, n=self.ntraversals)
    ):
        # iterate over batch and camera
        for a, refs in zip(all_inds_tr_batch_cam, init_refs):
            tr_a = []
            # iterate over num traversals
            for vmap in vtr:
                update_refs = vmap[refs]
                sort_offsets, vmask, new_len = get_merge_mask_and_sort(
                    len(vmap),
                    update_refs,
                )
                yield (
                    update_refs,
                    vmap,
                    a[sort_offsets][:, 2],
                    new_len,
                    vmask,
                    sort_offsets + running_q_count,
                )


def _build_extractmap_mask_gen(
    self,
    qxBCdt: torch.Tensor,
    reference_points_cam: torch.Tensor,
    query_grid_hit_mask: torch.Tensor,
    spatial_shapes: torch.Tensor,
    value_map: torch.Tensor,
    seq_lens: torch.Tensor,
):
    batch_size, nquery, channels = qxBCdt.shape

    q_shape = parse_shape(query_grid_hit_mask, "bs nc nq z")
    device = qxBCdt.device
    query_indices = repeat(
        torch.arange(0, nquery, dtype=torch.int64, device=device),
        "nq -> nq z",
        z=q_shape["z"],
    )
    running_q_count = 0
    for i, (H_, W_) in enumerate(spatial_shapes):
        i = i * self.ntraversals
        # for traversal in self.traversal_methods:
        for batch_idx in range(batch_size):
            for camera_index in range(q_shape["nc"]):
                bev_mask = query_grid_hit_mask[batch_idx, camera_index]

                ref_hits = reference_points_cam[batch_idx, camera_index][bev_mask]
                initial_ref_locs = self.ref2int(ref_hits, H_, W_)
                query_level_cam_ind = query_indices[bev_mask]
                for tr_idx in range(self.ntraversals):
                    v_map = value_map[seq_lens[i + tr_idx] : seq_lens[i + tr_idx + 1]]
                    ref_level_cam_trav = v_map[initial_ref_locs]
                    sort_offsets, vmask, new_len = get_merge_mask_and_sort(
                        H_ * W_,  # type:ignore
                        ref_level_cam_trav,
                    )
                    yield (
                        ref_level_cam_trav,
                        v_map,
                        query_level_cam_ind[sort_offsets],
                        new_len,
                        vmask,
                        sort_offsets + running_q_count,
                    )
                running_q_count += len(query_level_cam_ind)


def _build_extractmap_mask(
    self,
    qxBCdt: torch.Tensor,
    reference_points_cam: torch.Tensor,
    query_grid_hit_mask: torch.Tensor,
    spatial_shapes: torch.Tensor,
    value_map: torch.Tensor,
    seq_lens: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    batch_size, nquery, channels = qxBCdt.shape

    q_shape = parse_shape(query_grid_hit_mask, "bs nc nq z")
    device = qxBCdt.device
    query_indices = repeat(
        torch.arange(0, nquery, dtype=torch.int64, device=device),
        "nq -> nq z",
        z=q_shape["z"],
    )
    seq_lens_new = []
    v_masks = []
    extract_map = []
    q_expand_map = []
    running_q_count = 0
    for i, (H_, W_) in enumerate(spatial_shapes):
        i = i * self.ntraversals
        # for traversal in self.traversal_methods:
        for batch_idx in range(batch_size):
            for camera_index in range(q_shape["nc"]):
                bev_mask = query_grid_hit_mask[batch_idx, camera_index]

                ref_hits = reference_points_cam[batch_idx, camera_index][bev_mask]
                initial_ref_locs = self.ref2int(ref_hits, H_, W_)
                query_level_cam_ind = query_indices[bev_mask]
                for tr_idx in range(self.ntraversals):
                    v_map = value_map[seq_lens[i + tr_idx] : seq_lens[i + tr_idx + 1]]
                    ref_level_cam_trav = v_map[initial_ref_locs]
                    sort_offsets, vmask, new_len = get_merge_mask_and_sort(
                        H_ * W_,  # type:ignore
                        ref_level_cam_trav,
                    )
                    extract_map.append(
                        query_level_cam_ind[sort_offsets] + (nquery * batch_idx)
                    )
                    v_masks.append(vmask)

                    # q_expand_map.append(sort_offsets + running_q_count)
                    q_expand_map.append(sort_offsets + (nquery * batch_idx))
                    seq_lens_new.append(new_len)
                    running_q_count += len(sort_offsets)
    vmask = torch.cat(v_masks, dim=0)
    q_expand_map = torch.cat(q_expand_map)
    extract_map = torch.cat(extract_map)
    extract_map = torch.vstack(
        [
            torch.arange(len(vmask), dtype=torch.int64, device=device)[~vmask],
            extract_map,
            q_expand_map,
        ]
    )
    return extract_map, vmask, seq_lens_new


def _add_fb_masked_pregate_old(
    self,
    y: torch.Tensor,
    x_og: torch.Tensor,
    mask: torch.Tensor,
    extract_map: torch.Tensor,
    batch_size: int,
):
    mask = ~mask
    y = torch.roll(y, shifts=1, dims=1)
    y[:, 0, :] = 0.0
    # batches are collapsed (unpadding)
    ## y = (
    #     y[0:batch_size][:, mask]
    #     + torch.flip(y[batch_size : 2 * batch_size], (1,))[:, mask]
    # )
    y = y[0:batch_size][:, mask]
    # x_og = x_og * repeat(
    #     F.linear(x_og, self.fc_D.weight, bias=self.D),
    #     "b l h -> b l (h p)",
    #     p=self.headdim,
    # )

    x_og = rearrange(x_og, "bs nq d -> (bs nq) d")
    # y = rearrange(y, "bs t d -> (bs t) d")
    x_og.index_add_(0, extract_map[1], y.squeeze(0))

    return x_og


def _add_fb_masked_pregate_new(
    self,
    y: torch.Tensor,
    x_og: torch.Tensor,
    mask: torch.Tensor,
    extract_map: Dict[str, torch.Tensor],
    batch_size: int,
):
    mask = ~mask
    y = torch.roll(y, shifts=1, dims=1)
    y[:, 0, :] = 0.0
    y = y[0:batch_size][:, mask]
    # batches are collapsed (unpadding)
    # y = (
    #     y[0:batch_size][:, mask]
    #     + torch.flip(y[batch_size : 2 * batch_size], (1,))[:, mask]
    # )
    # x_og = x_og * repeat(
    #     F.linear(x_og, self.fc_D.weight, bias=self.D),
    #     "b l h -> b l (h p)",
    #     p=self.headdim,
    # )
    #
    x_og.index_put_(
        [extract_map["bs"], extract_map["q"]], y.squeeze(0), accumulate=True
    )

    return x_og


#
# @pytest.fixture
# @pytest.mark.parametrize("bs,nc,bev,z,H,W", [[2, 6, 2500, 4, 12, 23]])
# def refpts(bs, nc, bev, z, H, W):
#     reference_points_cam = torch.normal(torch.zeros([bs, nc, bev, z, 2]) + 0.5, 0.25)
#     bev_mask = (
#         (reference_points_cam[..., 1:2] > 0.0)
#         & (reference_points_cam[..., 1:2] < 1.0)
#         & (reference_points_cam[..., 0:1] < 1.0)
#         & (reference_points_cam[..., 0:1] > 0.0)
#     ).squeeze(-1)
#
#     ref_int = ref2int(reference_points_cam, H, W, False)  # type:ignore[Tuple[torch.Tensor, torch.Tensor]]
#     assert isinstance(ref_int, torch.Tensor)
#     return RefPtsObj(reference_points_cam, ref_int, bev_mask, (H, W))
#


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
        average_slots=True,
        device=device,
        dtype=torch.bfloat16,
        feature_levels=[0],
        collect_method="bct",
        # collect_method="batch_cam",
        traversal_methods=("br0cross",),
    )


@pytest.mark.parametrize(
    "updates",
    [
        dict(),
        dict(
            traversal_methods=("tl0cross", "tl1cross"),
        ),
        dict(
            traversal_methods=("tl0snake", "tl1snake"),
        ),
        dict(
            use_post_norm=False,
        ),
        dict(mem_eff_inference=True),
        dict(
            ref2int_convert_method="round",
            ref2int_offset=1,
        ),
        dict(
            x_og_activation=True,
        ),
        dict(
            v_zero_params={
                "z": True,
                "C": True,
            },
            q_zero_params={
                "B": True,
                "dt": True,
            },
        ),
    ],
)
def test_MergeParity(base_cfg, updates, device):
    base_cfg.update(updates)
    model = PostPreHydraBlock(**base_cfg)  # type:ignore
    model = model.to(device)
    H, W = 12, 23
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
    vxBCdt, vmaps, seq_lens = model._traverse_image_features(
        vxBCdt, level_sizes, spatial_shapes
    )
    data = dict(
        qxBCdt=qxBCdt,
        vxBCdt=rearrange(vxBCdt, "bs nc nv c -> bs (nc nv) c"),
        reference_points_cam=reference_points_cam,
        query_grid_hit_mask=bev_mask,
        spatial_shapes=spatial_shapes,
        value_map=vmaps,
        seq_lens=seq_lens,
    )
    ids1 = _build_extractmap_mask_gen(
        model,
        qxBCdt.clone(),
        reference_points_cam.clone(),
        bev_mask.clone(),
        spatial_shapes.clone(),
        vmaps.clone(),
        seq_lens.clone(),
    )
    ids2 = _build_extractmap_mask_new_gen(
        model,
        bev_mask.clone(),
        reference_points_cam.clone(),
        spatial_shapes.clone(),
        vmaps.clone(),
        seq_lens.clone(),
    )
    # vm1 = vmaps.split_with_sizes(seq_lens.diff().tolist(), dim=-1)
    for (a1, v1, u1, l1, m1, e1), (a2, v2, u2, l2, m2, e2) in zip(ids1, ids2):
        assert v1.shape == v2.shape
        assert torch.equal(v1, v2)
        assert torch.equal(a1, a2)
        assert l1 == l2
        assert torch.equal(m1, m2)
        assert torch.equal(u1, u2), "Updated locations are different"
        # assert torch.equal(e1, e2)

    input_seq, extract_map, vmask = _bev_aware_merge(
        model, **{s: d.clone() for s, d in data.items()}
    )  # type:ignore

    input_seq1, extract_map1, vmask1 = model._bev_aware_merge(
        **{s: d.clone() for s, d in data.items()}
    )  # type:ignore
    # assert torch.equal(vmask, vmask1)
    # assert torch.equal(input_seq, input_seq1)
    output_seq = input_seq[..., : model.d_x]
    print(input_seq.shape)
    output_seq1 = input_seq1[..., : model.d_x]
    x_og = qxBCdt[..., : model.d_x]
    yQ = _add_fb_masked_pregate_old(
        model, output_seq, x_og.clone(), vmask, extract_map, batch_size=1
    )
    yQ1 = _add_fb_masked_pregate_new(
        model, output_seq1, x_og.clone(), vmask1, extract_map1, batch_size=1
    )
    yQ = rearrange(yQ, "(bs nq) d -> bs nq d", bs=qxBCdt.size(0))
    mask_nequal = yQ != yQ1
    ind_nequal = torch.where(mask_nequal)[1]
    # incorrect_bounds = torch.aminmax(ind_nequal)
    num_diff = mask_nequal.sum()
    assert not num_diff, f"Outputs are not the same:\n  #Different: {num_diff} / {mask_nequal.numel()}\n  Bounds: [{torch.min(ind_nequal)}, {torch.max(ind_nequal)}]\n  #Different divmod d_x: {num_diff % model.d_x}, {num_diff // model.d_x}"


@pytest.mark.parametrize(
    "updates",
    [
        dict(),
        dict(
            traversal_methods=("tl0cross", "tl1cross"),
        ),
        dict(
            use_post_norm=False,
        ),
        dict(mem_eff_inference=True),
        dict(
            ref2int_convert_method="round",
            ref2int_offset=1,
        ),
        dict(
            x_og_activation=True,
        ),
        dict(
            v_zero_params={
                "z": True,
                "C": True,
            },
            q_zero_params={
                "B": True,
                "dt": True,
            },
        ),
    ],
)
def test_MergeSpeedNew(base_cfg, updates, device):
    base_cfg.update(updates)
    model = PostPreHydraBlock(**base_cfg)  # type:ignore
    model = model.to(device)
    H, W = 12, 23
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
    vxBCdt, vmaps, seq_lens = model._traverse_image_features(
        vxBCdt, level_sizes, spatial_shapes
    )
    data = dict(
        qxBCdt=qxBCdt,
        vxBCdt=rearrange(vxBCdt, "bs nc nv c -> bs (nc nv) c"),
        reference_points_cam=reference_points_cam,
        query_grid_hit_mask=bev_mask,
        spatial_shapes=spatial_shapes,
        value_map=vmaps,
        seq_lens=seq_lens,
    )

    input_seq1, extract_map1, vmask1 = model._bev_aware_merge(**deepcopy(data))  # type:ignore
    # assert torch.equal(vmask, vmask1)
    # assert torch.equal(input_seq, input_seq1)
    output_seq1 = input_seq1[..., : model.d_x]
    x_og = qxBCdt[..., : model.d_x]
    yQ1 = _add_fb_masked_pregate_new(
        model, output_seq1, x_og, vmask1, extract_map1, batch_size=1
    )


@pytest.mark.parametrize(
    "updates",
    [
        dict(),
        dict(
            traversal_methods=("tl0cross", "tl1cross"),
        ),
        dict(
            use_post_norm=False,
        ),
        dict(mem_eff_inference=True),
        dict(
            ref2int_convert_method="round",
            ref2int_offset=1,
        ),
        dict(
            x_og_activation=True,
        ),
        dict(
            v_zero_params={
                "z": True,
                "C": True,
            },
            q_zero_params={
                "B": True,
                "dt": True,
            },
        ),
    ],
)
def test_MergeSpeedOld(base_cfg, updates, device):
    base_cfg.update(updates)
    model = PostPreHydraBlock(**base_cfg)  # type:ignore
    model = model.to(device)
    H, W = 12, 23
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
    vxBCdt, vmaps, seq_lens = model._traverse_image_features(
        vxBCdt, level_sizes, spatial_shapes
    )
    data = dict(
        qxBCdt=qxBCdt,
        vxBCdt=rearrange(vxBCdt, "bs nc nv c -> bs (nc nv) c"),
        reference_points_cam=reference_points_cam,
        query_grid_hit_mask=bev_mask,
        spatial_shapes=spatial_shapes,
        value_map=vmaps,
        seq_lens=seq_lens,
    )

    input_seq, extract_map, vmask = _bev_aware_merge(model, **deepcopy(data))  # type:ignore

    output_seq = input_seq[..., : model.d_x]
    x_og = qxBCdt[..., : model.d_x]
    yQ = _add_fb_masked_pregate_old(
        model, output_seq, x_og, vmask, extract_map, batch_size=1
    )
    yQ = rearrange(yQ, "(bs nq) d -> bs nq d", bs=qxBCdt.size(0))
