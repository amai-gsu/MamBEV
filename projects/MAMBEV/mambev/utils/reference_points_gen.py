from typing import List, Optional, Union, Sequence

from mmengine.runner import autocast
import numpy as np
import torch
from torch._prims_common import DeviceLikeType
from einops import pack, rearrange, repeat
from mmdet3d.structures import Det3DDataSample


def get_reference_points_3d(
    H: int,
    W: int,
    Z: Union[int, float],
    D: int,
    bs: int,
    device: DeviceLikeType,
    dtype: torch.dtype,
):
    """

    Args:
        H (int): Height of BEV Grid
        W (int): Width of BEV Grid
        Z (int): Point Cloud Height
        D (int): Number of Points in a Point Pillar
        bs (int): Batch Size
        device (DeviceLikeType): Device of ref points
        dtype (torch.dtype): Dtype of reference points

    """
    zs = (
        torch.linspace(0.5, Z - 0.5, D, dtype=dtype, device=device)
        .view(-1, 1, 1)
        .expand(D, H, W)
        / Z
    )
    xs = (
        torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device)
        .view(1, 1, W)
        .expand(D, H, W)
        / W
    )

    ys = (
        torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device)
        .view(1, H, 1)
        .expand(D, H, W)
        / H
    )
    ref_3d = torch.stack((xs, ys, zs), -1)
    ref_3d = repeat(ref_3d, "np H W c -> bs np (H W) c", bs=bs)
    return ref_3d


def get_reference_points_2d(
    bs: int,
    H: int,
    W: int,
    device: DeviceLikeType,
    dtype: torch.dtype,
):
    # reference points on 2D bev plane, used in temporal self-attention (TSA).
    ref_y, ref_x = torch.meshgrid(
        torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device),
        torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device),
        indexing="ij",
    )
    ref_y = ref_y.reshape(-1)[None] / H
    ref_x = ref_x.reshape(-1)[None] / W
    ref_2d = torch.stack((ref_x, ref_y), -1)
    ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
    return ref_2d


def get_reference_points_2d_update(
    bs: int,
    H: int,
    W: int,
    device: DeviceLikeType,
    dtype: torch.dtype,
):
    ref_y, ref_x = torch.meshgrid(
        torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device),
        torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device),
        indexing="ij",
    )
    ref_y = repeat(ref_x / W, "H W -> B (H W)", B=bs)
    ref_y = repeat(ref_y / H, "H W -> B (H W)", B=bs)
    return torch.stack((ref_x, ref_y), -1).unsqueeze(2)


def get_reference_points(dim: str, **kwargs) -> torch.Tensor:
    """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (pillar_pointsevice where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

    # reference points in 3D space, used in spatial cross-attention (SCA)
    if dim == "3d":
        return get_reference_points_3d(**kwargs)
    # reference points on 2D bev plane, used in temporal self-attention (TSA).
    elif dim == "2d":
        return get_reference_points_2d(**kwargs)
    else:
        raise TypeError("Dim must be either `2d` or `3d`")


def get_reference_points_update(
    H: int,
    W: int,
    Z: int,
    dim: str,
    bs: int,
    device: DeviceLikeType,
    dtype: torch.dtype,
    num_points_in_pillar: Optional[int] = None,
) -> torch.Tensor:
    """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

    def make_3d_mesh_component(
        upper_lim: int,
        idx: int,
        shape: Sequence[int],
        steps: Optional[int] = None,
    ):
        if steps is None:
            steps = upper_lim
        nums = (
            torch.linspace(0.5, upper_lim - 0.5, steps, device=device, dtype=dtype)
            / upper_lim
        )
        view_idx = [1, 1, 1]
        view_idx[idx] = steps
        return nums.view(*view_idx).expand(*shape)

    # reference points in 3D space, used in spatial cross-attention (SCA)
    if dim == "3d":
        assert num_points_in_pillar is not None
        shape = num_points_in_pillar, H, W
        zs = make_3d_mesh_component(Z, 0, shape, steps=num_points_in_pillar)
        ys = make_3d_mesh_component(H, 1, shape)
        xs = make_3d_mesh_component(W, 2, shape)

        ref_3d = torch.stack((xs, ys, zs), -1)
        ref_3d = repeat(ref_3d, "np H W c -> bs np (H W) c", bs=bs)
        return ref_3d
    raise BaseException


@autocast(dtype=torch.float32)
def point_sampling(
    reference_points: torch.Tensor,
    pc_range: Sequence[float],
    img_metas: List[Det3DDataSample],
):
    lidar2img = [meta.lidar2img for meta in img_metas]  # type:ignore
    lidar2img = np.asarray(lidar2img)
    lidar2img = reference_points.new_tensor(lidar2img)  # (B, N, 4, 4)
    # WARNING: Do not remove
    reference_points = reference_points.clone()

    # scale xs
    reference_points[..., 0:1] = (
        reference_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
    )
    # scale ys
    reference_points[..., 1:2] = (
        reference_points[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
    )
    # scale zs
    reference_points[..., 2:3] = (
        reference_points[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
    )

    reference_points = torch.cat(
        (reference_points, torch.ones_like(reference_points[..., :1])), -1
    )

    # N = len([x,y,z]) + 1 = 4
    # _ = Batch Size
    num_points_in_pillar, num_query = reference_points.shape[1:3]
    num_cam = lidar2img.size(1)

    # permute, view, repeat, unsqueeze in one op
    reference_points = repeat(
        reference_points,
        "bs pp q N -> pp bs nc q N 1",
        nc=num_cam,
    )
    # lidar2img is a stack of 4x4 matrices
    lidar2img = repeat(
        lidar2img,
        "bs nc N F -> pp bs nc q N F",
        pp=num_points_in_pillar,
        q=num_query,
    )
    reference_points_cam = torch.matmul(
        lidar2img.to(torch.float32), reference_points.to(torch.float32)
    ).squeeze(-1)
    # print_log(reference_points_cam.shape, "current", logging.INFO)
    eps = 1e-5

    bev_mask = reference_points_cam[..., 2:3] > eps
    reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
        reference_points_cam[..., 2:3],
        torch.ones_like(reference_points_cam[..., 2:3]) * eps,
    )
    # scale reference points to range [0,1] for x,y
    reference_points_cam[..., 0] /= img_metas[0].img_shape[0][1]  # type:ignore
    reference_points_cam[..., 1] /= img_metas[0].img_shape[0][0]  # type:ignore

    # true where reference points x,y lie between 0,1
    bev_mask = (
        bev_mask
        & (reference_points_cam[..., 1:2] > 0.0)
        & (reference_points_cam[..., 1:2] < 1.0)
        & (reference_points_cam[..., 0:1] < 1.0)
        & (reference_points_cam[..., 0:1] > 0.0)
    )
    bev_mask = torch.nan_to_num(bev_mask)
    # "num_points_in_pillar batchsize num_cam num_query N -> num_cam batchsize num_query num_points_in_pillar N"
    reference_points_cam = rearrange(reference_points_cam, "pp bs c q N -> c bs q pp N")
    bev_mask = rearrange(bev_mask, "a b c d e -> c b d (a e)")

    return reference_points_cam, bev_mask


if __name__ == "__main__":
    H, W, Z, pillar_points, device, dtype = 6, 6, 8, 4, "cpu", torch.float32
    from time import perf_counter

    start = perf_counter()
    print(f"New finished in {perf_counter()-start}")

    start = perf_counter()
    print(f"OG finished in {perf_counter()-start}")
    # try:
    #     assert torch.allclose(gt, pred)
    #     print("Equivalent")
    # except AssertionError:
    #     print("Faile Assertion")
    #     print("GT", gt.shape)
    #     print("PR", pred.shape)
    # except RuntimeError:
    #     print("Shape or Type mismatch")
    #     print("GT", gt.shape)
    #     print("PR", pred.shape)
