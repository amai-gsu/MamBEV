from itertools import chain
from pathlib import Path
from typing import Callable, List, Tuple, TypeAlias, Union

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from einops import pack, rearrange, reduce, repeat
from matplotlib import rcParams
from mmdet3d.structures import Det3DDataSample

from projects.MAMBEV.mambev.layers.ref2int_layer import Ref2Int
from projects.MAMBEV.mambev.utils.sequence_utils import get_merge_mask_and_sort

rcParams["figure.figsize"] = 16, 9

PltFuncType: TypeAlias = Callable[[torch.Tensor, torch.Tensor, Path, Path], None]


class RefPtsVis:
    def __init__(
        self,
        layer_idx: int,
        attn_idx: int,
        dest_dir: str,
        ref2int: Ref2Int,
        bev_h: int = 50,
        bev_w: int = 50,
        proportion: int = 1,
        active: bool = False,
    ) -> None:
        self.layer_idx = layer_idx
        self.attn_idx = attn_idx
        self.sublayer_id = f"{layer_idx}_{attn_idx}"
        self.dest_dir = Path(dest_dir)
        if not self.dest_dir.exists():
            self.dest_dir.mkdir(parents=True)
            print(f"Making save location {self.dest_dir}")
        self.bev_w = bev_w
        self.bev_h = bev_h
        self.proportion = proportion
        self.ref2int = ref2int

        self.post_debug = self._dummy
        self.layer0_debug = self._dummy
        self.debug_offsets = self._dummy
        if active:
            self.post_debug = self._post_debug
            self.debug_offsets = self._debug_offsets
            if layer_idx == 0:
                self.layer0_debug = self._layer0_debug

    def _dummy(
        self,
        *args,
        # reference_points_cam: torch.Tensor,
        # bev_mask: torch.Tensor,
        # img_metas: List[Det3DDataSample],
        # spatial_shapes: torch.Tensor,
        # points: int,
    ):
        return None

    def _layer0_debug(
        self,
        reference_points_cam: torch.Tensor,
        bev_mask: torch.Tensor,
        img_metas: List[Det3DDataSample],
        spatial_shapes: torch.Tensor,
        points: int,
    ):
        self.debug_ref_pts(
            reference_points_cam.detach(),
            bev_mask,
            meta_info=img_metas,
            pillar_points=points,
            prefix="OG",
        )
        self.debug_ref2int(
            reference_points_cam.detach(),
            bev_mask,
            meta_info=img_metas,
            pillar_points=points,
            prefix="OG_r2i_",
            spatial_shape=tuple(spatial_shapes[0].tolist()),
        )

        floor_ref_pts = torch.zeros_like(reference_points_cam).detach()

        H_, W_ = spatial_shapes[0]

        floor_ref_pts[..., :] = self.ref2int.ref2int_sep(reference_points_cam, H_, W_)
        floor_ref_pts[..., 0] /= W_ - 1
        floor_ref_pts[..., 1] /= H_ - 1

        self.debug_ref_pts(
            floor_ref_pts.detach(),
            bev_mask,
            meta_info=img_metas,
            pillar_points=points,
            prefix="OG_FLOOR",
        )

        self.vis_input_seq(
            reference_points_cam.detach(),
            bev_mask,
            meta_info=img_metas,
            pillar_points=points,
            spatial_shape=tuple(spatial_shapes[0].tolist()),
            prefix="flattened_",
            ref2int=self.ref2int,
        )

        self.plot_pallettes(bev_mask, img_metas)

    def _post_debug(
        self,
        reference_points_cam: torch.Tensor,
        bev_mask: torch.Tensor,
        img_metas: List[Det3DDataSample],
        spatial_shapes: torch.Tensor,
        points: int,
    ):
        self.debug_ref_pts(
            reference_points_cam.detach(),
            bev_mask,
            meta_info=img_metas,
            pillar_points=points,
            prefix=f"LAYER{self.layer_idx}_{self.attn_idx}_OFFSET",
        )

        floor_ref_pts = torch.zeros_like(reference_points_cam).detach()

        H_, W_ = spatial_shapes[0]

        floor_ref_pts[..., :] = self.ref2int.ref2int_sep(reference_points_cam, H_, W_)
        floor_ref_pts[..., 0] /= W_ - 1
        floor_ref_pts[..., 1] /= H_ - 1

        self.debug_ref_pts(
            floor_ref_pts.detach(),
            bev_mask,
            meta_info=img_metas,
            pillar_points=points,
            prefix=f"LAYER{self.layer_idx}_{self.attn_idx}_FLOOR_OFFSET",
        )

    def _debug_offsets(
        self,
        sampling_offsets: torch.Tensor,
        seq_lens: List,
        meta_info: List[Det3DDataSample],
    ):
        rcParams["figure.figsize"] = 16, 9
        meta_info_sample = meta_info[0]
        scene_token = meta_info_sample.scene_token  # type:ignore
        frame_idx = str(meta_info_sample.sample_idx)  # type:ignore
        sls = list(chain(*seq_lens))
        soffs = torch.split(sampling_offsets, sls)

        so = rearrange(soffs[0], "q nh nl np ... -> (q nh nl np) ...")
        data = {}
        if so.dim() != 5 and so.size(-1) != 2:
            data["x"] = so.squeeze().numpy()
            sns.histplot(
                **data,
                # cmap="mako",
                # color="#4CB391"
            )
        else:
            data["x"], data["y"] = so[:, 0].numpy(), so[:, 1].numpy()
            sns.jointplot(
                **data,
                kind="kde",
                fill=True,
                thresh=0,
                levels=100,
                cmap="mako",
                warn_singular=False,
            )
        file_dest = self.dest_dir / scene_token / frame_idx
        try:
            file_dest.mkdir(parents=True, exist_ok=True)
        except FileExistsError:
            pass

        plt.savefig(
            file_dest / f"LAYER{self.sublayer_id}_offsets.png",
            bbox_inches="tight",
            pad_inches=0,
            dpi=100,
        )
        plt.close()

    def _plot_pallette(
        self,
        bev_c: torch.Tensor,
        file_dest: Path,
        grid_size: int,
        hues: torch.Tensor,
        i: str,
        max_uint8: torch.Tensor,
    ):
        color = pack(
            [
                hues,
                repeat(max_uint8 // 2, "h -> (q h)", q=grid_size),
                (bev_c / bev_c.max() * 255).to(torch.uint8),
            ],
            "qh *",
        )[0].unsqueeze(0)

        color = cv.cvtColor(color.numpy(), cv.COLOR_HSV2RGB)
        # color = torch.tensor(color, dtype=torch.float) / 255

        cgrid = repeat(
            color.squeeze(0),
            "(h w) rgb -> (h h1) (w w1) rgb",
            h=self.bev_h,
            w=self.bev_w,
            h1=16,
            w1=16,
        )

        fig, ax = plt.subplots()
        assert not isinstance(ax, np.ndarray)

        ax.figure.set_size_inches(8, 8)  # type:ignore
        ax.imshow(
            cgrid,
            extent=ax.get_xlim() + ax.get_ylim(),
            aspect="auto",
            # aspect="equal",
            zorder=-1,
        )
        ax.set_axis_off()
        plt.savefig(
            file_dest / f"bev_grid_cam{i}.png",
            bbox_inches="tight",
            pad_inches=0,
            dpi=100,
        )
        plt.close()

    def plot_pallettes(
        self,
        bev_mask: torch.Tensor,
        meta_info,
    ):
        bev_count = reduce(bev_mask.int(), "bs nc q pp -> bs nc q", "sum")[0]
        grid_size = bev_count.shape[-1]
        hues = (torch.arange(grid_size) / grid_size * 255).to(torch.uint8)
        max_uint8 = torch.tensor([255], dtype=torch.uint8)

        meta_info = meta_info[0]
        scene_token = meta_info.scene_token
        frame_idx = str(meta_info.sample_idx)

        file_dest = self.dest_dir / scene_token / frame_idx
        for i, bev_c in enumerate(bev_count):
            self._plot_pallette(bev_c, file_dest, grid_size, hues, str(i), max_uint8)

        bev_count = reduce(bev_count, "nc q -> q", "sum")
        self._plot_pallette(bev_count, file_dest, grid_size, hues, "_all", max_uint8)

    def _plot_cam(
        self,
        reference_points: torch.Tensor,
        bev_mask: torch.Tensor,
        meta_info: List[Det3DDataSample],
        pillar_points: int,
        prefix: str,
        plt_func: PltFuncType,
    ):
        """
        Creates plots using the corresponding images, and reference points using
        the plt_func input
        """
        meta_info_sample = meta_info[0]
        filenames = meta_info_sample.filename  # type:ignore
        scene_token = meta_info_sample.scene_token  # type:ignore
        frame_idx = str(meta_info_sample.sample_idx)  # type:ignore
        reference_points = rearrange(
            reference_points, "bs nc q pp xy -> bs nc pp q xy"
        )[0]
        bev_mask = rearrange(bev_mask, "bs nc q pp -> bs nc pp q ")[0]
        rpc_ind = [
            [
                (
                    zpoints[zmask][:: self.proportion],
                    zmask.nonzero()[:: self.proportion],
                )
                for zmask, zpoints in zip(mask, points)
            ]
            for mask, points in zip(bev_mask, reference_points)
        ]
        nc = len(rpc_ind)
        reference_points_cam = []
        indices: List[torch.Tensor] = []
        for rpc, ind in chain.from_iterable(rpc_ind):
            reference_points_cam.append(rpc)
            indices.append(ind)

        max_c = torch.tensor([255], dtype=torch.uint8)

        hue = ((torch.arange(pillar_points) + 0.5) / pillar_points * 255).to(
            torch.uint8
        )

        file_dest: Path = self.dest_dir / scene_token / frame_idx
        try:
            file_dest.mkdir(parents=True, exist_ok=True)
        except FileExistsError:
            pass

        for cam_idx in range(nc):
            cam_colors = []

            for pillar_idx in range(pillar_points):
                idx = cam_idx * pillar_points + pillar_idx
                cind = (indices[idx] / bev_mask.shape[-1] * 255).to("cpu", torch.uint8)
                color = pack(
                    [
                        cind,
                        repeat(
                            hue[pillar_idx : pillar_idx + 1],
                            "h -> (q h)",
                            q=len(cind),
                        ),
                        repeat(max_c, "h -> (q h)", q=len(cind)),
                    ],
                    "qh *",
                )[0].unsqueeze(0)  # type:ignore
                color = cv.cvtColor(color.numpy(), cv.COLOR_HSV2RGB)
                color = torch.tensor(color, dtype=torch.float) / 255
                cam_colors.append(color)

            camref = pack(
                reference_points_cam[
                    cam_idx * pillar_points : (cam_idx + 1) * pillar_points
                ],
                "* xy",
            )[0].cpu()

            camcolor = pack(cam_colors, "* rgb")[0]
            plt_func(
                camref,
                camcolor,
                filenames[cam_idx],
                file_dest / f"{prefix}_CAM_{cam_idx}.png",
            )
            # print(filenames[cam_idx])

    def vis_input_seq(
        self,
        reference_points: torch.Tensor,
        bev_mask: torch.Tensor,
        meta_info,
        pillar_points: int,
        spatial_shape: Tuple[int, int],
        prefix: str,
        ref2int: Callable[
            [torch.Tensor, Union[torch.Tensor, int], Union[torch.Tensor, int], bool],
            torch.Tensor,
        ],
    ):
        """
        Creates visualizatin of input sequence using patches to represent the image features
        and color coded squares to represent the BEV queries
        """
        head = 50

        ## find row lengths

        def plt_func(
            camref: torch.Tensor,
            camcolor: torch.Tensor,
            filename: Path,
            file_dest: Path,
        ) -> None:
            H_, W_ = spatial_shape

            camref_int = self.ref2int.ref2int_sep(camref, H_, W_)
            val, count = camref_int[..., 1].unique(return_counts=True, sorted=True)
            count_map = {v.item(): c.item() for v, c in zip(val, count)}

            row_lens = [W_ + count_map[i] if i in count_map else W_ for i in range(H_)]
            camref_1d = self.ref2int(camref, H_, W_)
            bg_img = plt.imread(filename)
            bg_img = rearrange(
                bg_img,
                "(h h1) (w w1) c -> (h w) h1 w1 c",
                h=spatial_shape[0],
                w=spatial_shape[1],
            )

            cam_chunks = repeat(
                (camcolor * 255).to(torch.uint8),
                "i rgb -> i h1 w1 rgb",
                h1=bg_img.shape[1],
                w1=bg_img.shape[2],
            )
            sort_offsets, vmask, new_len = get_merge_mask_and_sort(
                H_ * W_, camref_1d.squeeze(-1)
            )

            input_seq = torch.zeros(
                (new_len, bg_img.shape[1], bg_img.shape[2], 3),
                dtype=torch.uint8,
            )

            input_seq[vmask] = torch.from_numpy(bg_img)
            input_seq[~vmask] = cam_chunks[sort_offsets]

            save_loc = file_dest.parent / file_dest.stem
            try:
                save_loc.mkdir()
            except FileExistsError:
                pass
            for i, row in enumerate(input_seq.split_with_sizes(row_lens)):
                fig, ax = plt.subplots()
                assert not isinstance(ax, np.ndarray)

                this_head = min(head, len(row))
                rcParams["figure.figsize"] = this_head, 1
                rcParams["image.aspect"] = this_head
                ax.figure.set_size_inches(this_head, 1)  # type:ignore

                ax.imshow(
                    rearrange(row[:this_head], "n h w rgb -> h (n w) rgb"),
                    extent=ax.get_xlim() + ax.get_ylim(),
                    aspect="auto",
                    # aspect="equal",
                    zorder=-1,
                )

                ax.set_axis_off()

                plt.savefig(
                    save_loc / f"{ i }.png",
                    bbox_inches="tight",
                    pad_inches=0,
                    dpi=100,
                )
                plt.close()
            # raise ValueError

            # print(filenames[cam_idx])
            # return ax

        self._plot_cam(
            reference_points,
            bev_mask,
            meta_info,
            pillar_points,
            prefix,
            plt_func,
        )

    def debug_ref_pts(
        self,
        reference_points: torch.Tensor,
        bev_mask: torch.Tensor,
        meta_info,
        pillar_points: int,
        prefix: str,
    ):
        rcParams["figure.figsize"] = 16, 9

        def plt_func(
            camref: torch.Tensor,
            camcolor: torch.Tensor,
            filename: Path,
            file_dest: Path,
        ):
            # print(len(camcolor))
            ax = sns.scatterplot(
                x=camref[..., 0],  # type:ignore
                y=camref[..., 1],  # type:ignore
                c=camcolor,
                color=None,
                alpha=1,
                size=10,
                legend=False,
            )

            # ax = g.ax_joint

            bg_img = plt.imread(filename)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

            plt.gca().invert_yaxis()  # type:ignore
            ax.imshow(
                bg_img, extent=ax.get_xlim() + ax.get_ylim(), aspect="auto", zorder=-1
            )
            ax.set_axis_off()
            ax.figure.set_size_inches(16, 9)  # type:ignore
            plt.savefig(
                file_dest,
                bbox_inches="tight",
                pad_inches=0,
                dpi=100,
            )
            plt.close()

        self._plot_cam(
            reference_points, bev_mask, meta_info, pillar_points, prefix, plt_func
        )

    def debug_ref2int(
        self,
        reference_points: torch.Tensor,
        bev_mask: torch.Tensor,
        meta_info,
        pillar_points: int,
        prefix: str,
        # ref2int: Callable[
        #     [torch.Tensor, Union[torch.Tensor, int], Union[torch.Tensor, int], bool],
        #     torch.Tensor,
        # ],
        spatial_shape: Tuple[int, int],
    ):
        ### TODO: Compare the actual insert location to the reference location
        rcParams["figure.figsize"] = 16, 9
        H_, W_ = spatial_shape

        def plt_func(
            camref: torch.Tensor,
            camcolor: torch.Tensor,
            filename: Path,
            file_dest: Path,
        ):
            camref_int = self.ref2int.ref2int_sep(camref, H_, W_).float()  # type:ignore

            # account for insert to the left
            # camref_int[..., 0] -= 1
            camref_int[..., 0] /= W_ - 1
            camref_int[..., 1] /= H_ - 1

            dcam_ref = camref_int.float() - camref
            ax: plt.Axes
            # fig, ax = plt.subplots()  # type:ignore
            ax = sns.scatterplot(
                x=camref[..., 0],  # type:ignore
                y=camref[..., 1],  # type:ignore
                c=camcolor,
                alpha=1,
                size=10,
                legend=False,
            )
            # sns.scatterplot(
            #     x=camref_int[..., 0],  # type:ignore
            #     y=camref_int[..., 1],  # type:ignore
            #     c=camcolor,
            #     color=None,
            #     alpha=1,
            #     size=10,
            #     legend=False,
            #     ax=ax,
            # )

            for ref, dref, cc in zip(camref, dcam_ref, camcolor):
                refx, refy = ref[..., 0], ref[..., 1]
                drefx, drefy = dref[..., 0], dref[..., 1]
                cc = cc.tolist() + [1]
                ax.arrow(refx, refy, drefx, drefy, color=cc)
            #
            bg_img = plt.imread(filename)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            plt.gca().invert_yaxis()  # type:ignore
            ax.imshow(
                bg_img, extent=ax.get_xlim() + ax.get_ylim(), aspect="auto", zorder=-1
            )

            ax.set_axis_off()
            ax.figure.set_size_inches(16, 9)  # type:ignore
            # print(filenames[cam_idx])
            #

            plt.savefig(
                file_dest,
                bbox_inches="tight",
                pad_inches=0,
                dpi=100,
            )
            plt.close()

        self._plot_cam(
            reference_points, bev_mask, meta_info, pillar_points, prefix, plt_func
        )
