import copy
import logging
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple, Union

from mmengine.model import ModuleDict
from mmengine.runner import autocast
import torch

from mmdet.models.layers.transformer import PatchEmbed
from mmdet3d.structures.det3d_data_sample import ForwardResults
from mmdet3d.utils.typing_utils import OptConfigType, OptMultiConfig
from einops import pack, rearrange, unpack
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from mmdet3d.structures import Det3DDataSample
from mmdet3d.utils.typing_utils import ConfigDict
from mmengine.logging import print_log
from mmengine.optim import OptimWrapper
from mmengine.registry import HOOKS, MODELS


from projects.MAMBEV.mambev.utils.grid_mask import GridMask

PrevBEVDict = OrderedDict[int, Optional[Tuple[torch.Tensor, List[bool]]]]


@MODELS.register_module()
class MamBEV(MVXTwoStageDetector):
    def __init__(
        self,
        embed_dim: int,
        # levels of features
        num_levels: Optional[int],
        num_mono_levels: Optional[int],
        frames: Tuple[int, ...],
        pts_bbox_head: Union[ConfigDict, Dict],
        img_backbone: Union[ConfigDict, Dict],
        fcos3d_bbox_head: OptConfigType = None,
        compute_graph_vis_hook: OptConfigType = None,
        img_neck: OptConfigType = None,
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
        data_preprocessor: OptConfigType = None,
        use_grid_mask: bool = True,
        mono_loss_weight: float = 1.0,
        # levels of features
        with_bptt: Optional[bool] = False,
        patch_levels: List[int] = [],
        patch_cfg: Optional[Union[List[Dict], Dict]] = None,
        init_cfg: OptMultiConfig = None,
    ) -> None:
        super(MamBEV, self).__init__(
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            img_backbone=img_backbone,
            img_neck=img_neck,
            pts_bbox_head=pts_bbox_head,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg,  # type:ignore
        )
        if use_grid_mask:
            self.grid_mask = GridMask(
                use_h=True,
                use_w=True,
                rotate=1,
                offset=False,
                ratio=0.5,
                mode=1,
                prob=0.7,
            )
        self.patch_levels = patch_levels
        if self.patch_levels:
            self.patchify = ModuleDict()
            if isinstance(patch_cfg, List):
                assert (
                    len(patch_cfg) == len(patch_levels) or len(patch_cfg) == 1
                ), "Must have either the same number of patch configs as levels being patched or a single patch config"
                if len(patch_cfg) == 1:
                    patch_cfg = [copy.deepcopy(patch_cfg[0]) for _ in self.patch_levels]

                for i, p_cfg in zip(patch_levels, patch_cfg):
                    self.patchify[f"patch_embed_level{ i }"] = PatchEmbed(**p_cfg)

            elif isinstance(patch_cfg, Dict):
                print_log(
                    "Using the same patch embedding weights for every level",
                    "current",
                    logging.INFO,
                )

                patch_embedder = PatchEmbed(**patch_cfg)

                for i in self.patch_levels:
                    self.patchify[f"patch_embed_level{ i }"] = patch_embedder
            else:
                print_log(
                    "Using the same patch embedding weights for every level",
                    "current",
                    logging.INFO,
                )

                patch_cfg = dict(
                    # type="mmdet.PatchEmbed",
                    in_channels=embed_dim,
                    embed_dims=embed_dim,
                    conv_type="Conv2d",
                    kernel_size=4,
                    stride=4,
                    padding="corner",
                    dilation=1,
                    bias=True,
                    norm_cfg=None,
                    input_size=None,
                    init_cfg=None,
                )
                patch_embedder = PatchEmbed(**patch_cfg)

                for i in self.patch_levels:
                    self.patchify[f"patch_embed_level{ i }"] = patch_embedder

        self.use_grid_mask = use_grid_mask
        self.prev_frame_info = {
            "prev_bev": None,
            "scene_token": None,
            "prev_pos": 0,
            "prev_angle": 0,
        }
        assert hasattr(
            self.pts_bbox_head, "assigner"
        ), "BBox head must have an assigner"
        try:
            self.fcos3d_bbox_head = (
                MODELS.build(fcos3d_bbox_head) if fcos3d_bbox_head else None
            )
            # loss weight
            self.mono_loss_weight = mono_loss_weight
            self.num_mono_levels = num_mono_levels

        except TypeError as te:
            print_log(
                f"Initializing without BEVFormerV2 monocular perception head: \t{te}",
                "current",
                level=logging.WARNING,
            )
            self.fcos3d_bbox_head = None
            self.mono_loss_weight = None
            self.num_mono_levels = None

        # levels of features
        self.num_levels = num_levels
        self.frames = frames
        self.with_bptt = with_bptt
        if compute_graph_vis_hook:
            self.viz_compute_graph = True
            self.compute_graph_hook = HOOKS.build(compute_graph_vis_hook)
        else:
            self.viz_compute_graph = False

    # @autocast(dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    @autocast(dtype=torch.float16)
    def extract_img_feat(  # type:ignore
        self, img: torch.Tensor, input_metas: List[Det3DDataSample]
    ) -> Tuple[torch.Tensor, ...]:
        """Extract features of images. When batch size > 1 collapse the
        batch into a 4d tensor so all images can be passed to the backbone
        in a single pass"""
        if img is None:
            return tuple()
        input_shape = img.shape[-2:]
        # update real input shape of each single img
        for img_meta in input_metas:
            img_meta.input_shape = input_shape

        if img.dim() == 5:
            B = img.size(0)
            img = rearrange(img, "B N C H W -> (B N) C H W")
        elif img.dim() == 4:
            B = 1
        else:
            raise ValueError("Expected img to be either a 4D or 5D tensor")

        if self.use_grid_mask:
            img = self.grid_mask(img)

        img_feats = self.img_backbone(img)
        if isinstance(img_feats, dict):
            # returns a multilevel features in a tuple
            img_feats = tuple(img_feats.values())

        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for i, img_feat in enumerate(img_feats):
            if i in self.patch_levels:
                # print_log(
                #     f"Patchifying level {i} with shape {img_feat.shape}",
                #     "current",
                #     logging.DEBUG,
                # )
                # ignore the returned size tuple
                img_feat, shape = self.patchify[f"patch_embed_level{i}"](img_feat)
                img_feat = rearrange(
                    img_feat, "(B N) (H W) C -> B N C H W", H=shape[0], B=B
                )
                # print_log(
                #     f"Patch output shape {img_feat.shape}", "current", logging.DEBUG
                # )
                img_feats_reshaped.append(img_feat)
            else:
                # print_log(
                #     f"Not patchifying level {i} with shape {img_feat.shape}",
                #     "current",
                #     logging.DEBUG,
                # )
                img_feats_reshaped.append(
                    rearrange(img_feat, "(B N) C H W -> B N C H W", B=B)
                )
        return tuple(img_feats_reshaped)

    def obtain_history_bev(
        self, img_dict: Dict[int, Union[torch.Tensor, None]], img_metas: Dict[int, List]
    ):
        """
        Obtain previous frames BEV features in a single batch.
        To save GPU memory, gradients are not calculated.
        may add option to use old verison without batching
        if the memory req is too much
        """
        # Modify: roll back to previous version for single frame
        is_training = self.training
        self.eval()
        prev_bev: PrevBEVDict = OrderedDict({i: None for i in self.frames})
        unpack_img_metas = {}
        img_metas_packed = []
        imgs = []
        for i, img_meta in img_metas.items():
            unpack_img_metas[i] = len(img_meta)
            img_metas_packed.extend(img_meta)
            imgs.append(img_dict[i])

        try:
            imgs_tens, unpack_shapes = pack(imgs, "* N C H W")
        except IndexError:
            print_log("No previous frames were found", "current", logging.WARNING)
            return None

        with torch.no_grad():
            img_feats = self.extract_img_feat(imgs_tens, img_metas_packed)

            assert img_feats is not None, "Failed to extract image features"
            if self.num_levels:
                img_feats = img_feats[: self.num_levels]
            bev = self.pts_bbox_head(img_feats, img_metas_packed, None, only_bev=True)
            bev = unpack(bev, packed_shapes=unpack_shapes, pattern="* L C")

            for t in unpack_img_metas:
                prev_bev[t] = bev[t].detach(), [meta.mask for meta in img_metas[t]]
        if is_training:
            self.train()
        return prev_bev

    def extract_feat(  # type:ignore
        self, batch_inputs_dict: torch.Tensor, batch_input_metas: List[Det3DDataSample]
    ):
        """Extract features from images and points."""

        img_feats = self.extract_img_feat(batch_inputs_dict, batch_input_metas)  # type:ignore
        try:
            assert batch_input_metas[0].aug_param["CropResizeFlipImage_param"][-1]  # type:ignore
            img_feats = [torch.flip(x, dims=(-1,)) for x in img_feats]
        except (KeyError, AssertionError, AttributeError):
            print_log(
                "CropResizeFlipImage_param not found in input metas",
                "current",
                logging.DEBUG,
            )
        return img_feats

    def loss_mono(
        self,
        img_feats: Sequence[torch.Tensor],
        mono_input_dict: List[Dict],
        ann_idx: List[int],
    ):
        """

        Auxiliary monocular loss calcuation
        """
        assert self.fcos3d_bbox_head is not None
        batch_size = img_feats[0].shape[0]
        num_lvls = len(img_feats)
        # print_log(
        #     "Mono Loss Inputs:\n"
        #     f"  IMG Feats shape: {img_feats[0].shape}\n"
        #     f"  Number of Levels: {num_lvls}\n"
        #     f"  Batch size: {batch_size}\n",
        #     "current",
        #     logging.DEBUG,
        # )

        img_feats_select: list[torch.Tensor] = [None for _ in range(num_lvls)]  # type:ignore
        for lvl, img_feat in enumerate(img_feats):
            img_feats_select[lvl] = torch.cat(
                [img_feat[i, ann_idx[i]] for i in range(batch_size)],
                dim=0,
            )

        bsz_new = img_feats_select[0].shape[0]
        # print_log(
        #     "Mono Loss Concat:\n"
        #     f"  IMG Feats select shape: {img_feats_select[0].shape}\n"
        #     f"  Number of Levels: {num_lvls}\n"
        #     f"  Batch size new: {bsz_new}\n"
        #     f"  Mono Dict size: {len(mono_input_dict)}\n",
        #     "current",
        #     logging.DEBUG,
        # )
        assert batch_size == len(mono_input_dict)

        input_dict = []
        for i in range(batch_size):
            input_dict.extend(mono_input_dict[i])

        assert (
            bsz_new == len(input_dict)
        ), f"Batch size: {batch_size}\nNew Batch size: {bsz_new}\nInput length: {len(input_dict)}\n"
        losses = self.fcos3d_bbox_head.forward(img_feats_select, input_dict)
        return losses

    def bptt_loss(
        self,
        batch_inputs_dict: Dict[str, Dict[int, List[torch.Tensor]]],
        batch_data_samples: List[Det3DDataSample],
        **kwargs,
    ) -> List[Det3DDataSample]:
        # TODO: Implement BPTT
        # If the output of the previous time step is used as input,
        # and the current cuda implementation of MAMBA doesnt allow the
        # passage of hidden states into the inner S6 block, we cannot pass
        # S6 hidden states through time quickly without a custom kernel
        # However, the HF Transformers implementation does allow this though
        # it is likely MUCH slower

        raise NotImplementedError("Have not implemented Back Propogation through time")

    def loss(  # type:ignore
        self,
        batch_inputs_dict: Dict[int, Dict[str, torch.Tensor]],
        batch_data_samples: Dict[int, List[Det3DDataSample]],
        **kwargs,
    ) -> List[Det3DDataSample]:
        """
        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points' and `imgs` keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
                - imgs (torch.Tensor): Tensor of batch images, has shape
                  (Batch, Views, Channels, H ,W)
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, .

        Returns:
            dict[str, Tensor]: A dictionary of loss components.

        """

        if self.with_bptt:
            return self.bptt_loss(batch_inputs_dict, batch_data_samples)  # type: ignore
        # Split into current and previous frames
        prev_frames = tuple(k for k in batch_inputs_dict if k != 0)
        has_prev_frames = len(prev_frames)

        imgs = batch_inputs_dict.pop(0).get("img")
        assert imgs is not None, "No images found for target index"
        img_dict = {i: d.get("img") for i, d in batch_inputs_dict.items()}

        prev_img_metas = copy.deepcopy(batch_data_samples)
        img_metas = prev_img_metas.pop(0)
        img_feats = self.extract_feat(imgs, img_metas)
        assert img_feats is not None, "Failed to extract image features"
        # get BEV history
        if has_prev_frames:
            prev_bev = self.obtain_history_bev(img_dict, prev_img_metas)
        else:
            prev_bev = None

        # get current BEV predictions
        outs = self.pts_bbox_head.forward(
            mlvl_feats=img_feats
            if self.num_levels is None
            else img_feats[: self.num_levels],
            img_metas=img_metas,
            prev_bev=prev_bev,
        )

        batch_gt_instances_3d = [item.gt_instances_3d for item in batch_data_samples[0]]
        # Get loss dict
        losses = self.pts_bbox_head.loss_by_feat(batch_gt_instances_3d, outs)
        if self.fcos3d_bbox_head:
            mono_img_feats = (
                img_feats
                if self.num_mono_levels is None
                else img_feats[: self.num_mono_levels]
            )

            mono_input_dicts = [
                metas.mono_input_dict  # type:ignore
                for metas in img_metas
                if metas.mono_input_dict is not None  # type:ignore
            ]
            mono_ann_idx = [metas.mono_ann_idx for metas in img_metas]  # type:ignore
            # get mono head loss
            losses_mono = self.loss_mono(
                img_feats=mono_img_feats,
                mono_input_dict=mono_input_dicts,
                ann_idx=mono_ann_idx,
            )
            for k, v in losses_mono.items():
                losses[f"{k}_mono"] = v * self.mono_loss_weight

            losses.update(losses_mono)
        return losses

    def train_step(
        self, data: Union[dict, tuple, list], optim_wrapper: OptimWrapper
    ) -> Dict[str, torch.Tensor]:
        """
        1. Calls ``self.data_processor(data, training=False)`` to collect
           batch_inputs and corresponding data_samples(labels).
        2. Calls ``self(batch_inputs, data_samples, mode='loss')`` to get raw
           loss
        3. Calls ``self.parse_losses`` to get ``parsed_losses`` tensor used to
           backward and dict of loss tensor used to log messages.
        4. Calls ``optim_wrapper.update_params(loss)`` to update model.

        Args:
            data (dict or tuple or list): Data sampled from dataset.
            optim_wrapper (OptimWrapper): OptimWrapper instance
                used to update model parameters.

        Returns:
            Dict[str, torch.Tensor]: A ``dict`` of tensor for logging.
        """

        # Enable automatic mixed precision training context.
        with optim_wrapper.optim_context(self):
            data = self.data_preprocessor(data, True)
            losses = self._run_forward(data, mode="loss")  # type: ignore
        parsed_losses, log_vars = self.parse_losses(losses)  # type: ignore
        if self.viz_compute_graph:
            self.compute_graph_hook(parsed_losses, dict(self.named_parameters()))

        optim_wrapper.update_params(parsed_losses)
        return log_vars

    def aug_test(  # type:ignore
        self,
        batch_inputs_dict: Dict[int, Dict[str, torch.Tensor]],
        batch_data_samples: Dict[int, List[Det3DDataSample]],
        **kwargs,
    ):
        return self.predict(
            batch_inputs_dict,
            batch_data_samples,
            **kwargs,
        )

    def predict(  # type:ignore
        self,
        batch_inputs_dict: Dict[int, Dict[str, torch.Tensor]],
        batch_data_samples: Dict[int, List[Det3DDataSample]],
        **kwargs,
    ):
        prev_frames = tuple(k for k in batch_inputs_dict if k != 0)
        # print_log(f"Previous frames: {prev_frames}", "current", logging.DEBUG)
        has_prev_frames = len(prev_frames)
        imgs = batch_inputs_dict.pop(0).get("img", {})
        img_dict = {i: d.get("img") for i, d in batch_inputs_dict.items()}

        img_metas = batch_data_samples.pop(0)
        prev_img_metas = copy.deepcopy(batch_data_samples)

        img_feats = self.extract_feat(imgs, img_metas)  # type:ignore
        assert img_feats is not None, "Failed to extract image features"
        if has_prev_frames:
            prev_bev = self.obtain_history_bev(img_dict, prev_img_metas)
        else:
            prev_bev = None
        preds_dicts = self.pts_bbox_head.forward(
            mlvl_feats=img_feats
            if self.num_levels is None
            else img_feats[: self.num_levels],
            img_metas=img_metas,
            prev_bev=prev_bev,
            mode="training",
        )

        img_metas_dicts = [meta_info.to_dict() for meta_info in img_metas]

        results_list_3d = self.pts_bbox_head.predict_by_feat(
            preds_dicts, img_metas_dicts
        )
        # change the bboxes' format
        detsamples = self.add_pred_to_datasample(img_metas, results_list_3d)

        return detsamples

    def _forward(self, *args, **kwargs):  # type:ignore
        if self.training:
            return self.loss(*args, **kwargs)
        else:
            return self.predict(*args, **kwargs)

    def forward(  # type:ignore
        self,
        inputs: Dict[int, Dict[str, torch.Tensor]],
        data_samples: Dict[int, List[Det3DDataSample]],
        mode: str = "tensor",
        **kwargs,
    ) -> ForwardResults:
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - "predict": Forward and return the predictions, which are fully
        processed to a list of :obj:`Det3DDataSample`.
        - "loss": Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle neither back propagation nor
        optimizer updating, which are done in the :meth:`train_step`.

        Args:
            inputs  (dict | list[dict]): When it is a list[dict], the
                outer list indicate the test time augmentation. Each
                dict contains batch inputs
                which include 'points' and 'imgs' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
                - imgs (torch.Tensor): Image tensor has shape (B, C, H, W).
            data_samples (list[:obj:`Det3DDataSample`],
                list[list[:obj:`Det3DDataSample`]], optional): The
                annotation data of every samples. When it is a list[list], the
                outer list indicate the test time augmentation, and the
                inter list indicate the batch. Otherwise, the list simply
                indicate the batch. Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of :obj:`Det3DDataSample`.
            - If ``mode="loss"``, return a dict of tensor.
        """
        if mode == "loss":
            return self.loss(inputs, data_samples, **kwargs)
        elif mode == "predict":
            # if isinstance(data_samples[0], list):
            #     # aug test
            #     # assert len(data_samples[0]) == 1, (
            #     #     "Only support "
            #     #     "batch_size 1 "
            #     #     "in mmdet3d when "
            #     #     "do the test"
            #     #     "time augmentation."
            #     # )
            #     return self.aug_test(inputs, data_samples, **kwargs)
            # else:
            return self.predict(inputs, data_samples, **kwargs)
        elif mode == "tensor":
            return self._forward(inputs, data_samples, **kwargs)
        else:
            raise RuntimeError(
                f'Invalid mode "{mode}". ' "Only supports loss, predict and tensor mode"
            )
