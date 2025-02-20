from pathlib import Path
import copy
from typing import Dict, Sequence, Tuple


import seaborn as sns
import matplotlib.pyplot as plt

from einops import rearrange, repeat, pack
import torch
from mmdet3d.registry import HOOKS
from mmengine.hooks.hook import DATA_BATCH, Hook
from mmengine.runner import Runner
from mmengine.runner.runner import print_log
from projects.MAMBEV.mambev.utils.reference_points_gen import (
    get_reference_points_3d,
    point_sampling,
)


@HOOKS.register_module()
class ReferencePointsVisHook(Hook):
    def __init__(
        self,
        active_mode: str,
        pc_range: Sequence[float],
        bev_h: int,
        bev_w: int,
        pillar_points: int,
        save_dir: str,
        interval: int = 50,
        active_iters: int = 40,
    ) -> None:
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.pc_range = pc_range
        self.pillar_points = pillar_points
        self.active_mode = active_mode
        self.interval: int = interval
        self.active_iters: int = active_iters
        self.save_dir: Path = Path(save_dir)
        self.inactive = False
        super().__init__()

    def create_vis(self, data_batch: DATA_BATCH, batch_idx: int):
        assert isinstance(data_batch, Tuple)
        batch_inputs_dict, batch_data_samples = copy.deepcopy(data_batch)

        imgs = batch_inputs_dict.pop(0).get("img").cpu()
        img_metas = batch_data_samples[0]
        bs, num_cams, C, H, W = imgs.shape

        ref_3d = get_reference_points_3d(
            H=self.bev_h,
            W=self.bev_w,
            Z=self.pc_range[5] - self.pc_range[2],
            D=self.pillar_points,
            bs=bs,
            device="cpu",
            dtype=torch.float64,
        )
        reference_points_cam, bev_mask = point_sampling(
            ref_3d, self.pc_range, img_metas
        )
        indices = []
        for i, mask_per_img in enumerate(bev_mask):
            index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
            indices.append(index_query_per_img)
        max_len = max([len(each) for each in indices])

        # each camera only interacts with its corresponding BEV queries. This step can  greatly save GPU memory.

        reference_points_rebatch = reference_points_cam.new_zeros(
            [bs, num_cams, max_len, self.pillar_points, 2]
        )

        # ref points rebatch gets a list of queries corresponding to the image
        # also get reference points corresponding to the queries on each image
        for j in range(bs):
            for i, reference_points_per_img in enumerate(reference_points_cam):
                index_query_per_img = indices[i]
                reference_points_rebatch[j, i, : len(index_query_per_img)] = (
                    reference_points_per_img[j, index_query_per_img]
                )
        ref_points = rearrange(
            reference_points_rebatch, "bs nc nr zl xy -> (bs nc) (nr zl) xy"
        )

        for i in range(num_cams):
            xy = (
                (
                    (ref_points[0, :, 0] >= 0)
                    & (ref_points[0, :, 0] <= 1)
                    & (ref_points[0, :, 1] >= 0)
                    & (ref_points[0, :, 1] <= 1)
                )
                .nonzero()
                .squeeze()
            )
            ref_points_frame = ref_points[0, xy]
            hue = repeat(
                (torch.arange(self.pillar_points) + 0.5) / self.pillar_points,
                "Z -> (nr Z)",
                nr=reference_points_rebatch.shape[2],
            )[xy]
            a_c = torch.ones_like(hue)

            # ax = sns.scatterplot(x=ref_points[0,:,0],y=ref_points[0,:,1], alpha=0.09,palette="Spectral",hue = hue,hue_norm=(0, 3), size=10 , legend=False)
            ax = sns.scatterplot(
                x=ref_points_frame[:, 0].numpy(),
                y=ref_points_frame[:, 1].numpy(),
                alpha=0.12,
                c=pack([ref_points_frame, hue, a_c], "nrz *")[0],
                size=10,
                legend=False,
            )

            # FIX:
            plt.imread()
            bg_image = data_batch

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.imshow(
                bg_image, extent=ax.get_xlim() + ax.get_ylim(), aspect="auto", zorder=-1
            )
            ax.set_axis_off()
            ax.figure.set_size_inches(16, 9)
            plt.savefig()

        return

    def _before_iter(
        self,
        runner: Runner,
        batch_idx: int,
        data_batch: DATA_BATCH = None,
        mode: str = "train",
    ) -> None:
        if mode == "train":
            return
        if not self.inactive and self.active_mode not in ("interval", "first_n"):
            runner.logger.warn(
                "Invalid active mode set for ReferencePointsVisHook, falling back to always inactive"
            )
            self.inactive = True

        interval_hit = self.active_mode == "interval" and self.every_n_train_iters(
            runner, self.interval
        )
        first_n_hit = self.active_mode == "first_n" and runner.iter < self.active_iters
        if interval_hit or first_n_hit:
            self.create_vis(data_batch, batch_idx)
