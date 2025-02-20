from collections.abc import Sequence
import copy
import logging
import pickle
from collections import OrderedDict
from typing import Any, Dict, Optional, Set, Tuple, Union
import os
from os import path as osp

from mmdet3d.datasets.transforms import LoadMultiViewImageFromFiles
import numpy as np
from mmdet3d.datasets import WaymoDataset
from mmengine.logging import print_log
from mmengine.registry import DATASETS


@DATASETS.register_module()
class WaymoDatasetMultiFrame(WaymoDataset):
    def __init__(
        self,
        frames: Tuple[int, ...],
        debug_pipeline_keys: Set[str] = set(),
        mono_cfg: Optional[Dict] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.debug_pipeline_keys = debug_pipeline_keys
        self.frames = np.array(frames, dtype=int)
        self.queue_length = len(frames)
        if not self.test_mode and mono_cfg is not None:
            raise NotImplementedError("Mono head is not supported for waymo dataset")

    def union2one(self, queue: Dict[int, Dict[str, Any]]):
        # lidar2ego = queue[0]["data_samples"].lidar2ego
        egocurr2global = queue[0]["data_samples"].ego2global
        for i, each in queue.items():
            if i == 0:
                continue
            egoadj2global = each["data_samples"].ego2global

            lidaradj2lidarcurr = (
                # np.linalg.inv(lidar2ego)
                np.linalg.inv(egocurr2global) @ egoadj2global
                # @ lidar2ego
            )
            each["data_samples"].lidaradj2lidarcurr = lidaradj2lidarcurr
            for i_cam in range(len(each["data_samples"].lidar2img)):
                each["data_samples"].lidar2img[i_cam] = each["data_samples"].lidar2img[
                    i_cam
                ] @ np.linalg.inv(lidaradj2lidarcurr)

        return queue

    def _load_info(self, idx):
        # super()._load_info()
        if self.serialize_data:
            start_addr = 0 if idx == 0 else self.data_address[idx - 1].item()
            end_addr = self.data_address[idx].item()
            dbytes = memoryview(self.data_bytes[start_addr:end_addr])  # type: ignore
            data_info = pickle.loads(dbytes)  # type: ignore
        else:
            data_info = copy.deepcopy(self.data_list[idx])
        if idx >= 0:
            data_info["sample_idx"] = idx
        else:
            data_info["sample_idx"] = len(self) + idx
        return data_info

    def add_rot_trans_mats(self, input_dict: dict) -> None:
        lidar2img_rts = []
        cam2img = []
        lidar2cam_rts = []

        # input_dict["lidar2ego"] = input_dict["lidar_points"]["lidar2ego"]
        for cam_type, cam_info in input_dict["images"].items():
            lidar2cam_rts.append(np.array(cam_info["lidar2cam"]))
            cam2img.append(np.array(cam_info["cam2img"]))
            intrinsic = cam2img[-1]
            viewpad = np.eye(4)
            viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
            lidar2img = viewpad @ lidar2cam_rts[-1]
            lidar2img_rts.append(lidar2img)

        input_dict.update(
            dict(
                lidar2img=np.array(lidar2img_rts),
                cam2img=np.array(cam2img),
                lidar2cam=np.array(lidar2cam_rts),
            )
        )

    def prepare_data_single(
        self,
        index: int,
        frame_num: int,
        aug_param: Optional[Dict] = None,
    ) -> Union[Dict, None]:
        """Data preparation for both training and testing stage.

        Called by `__getitem__`  of dataset.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict or None: Data dict of the corresponding index.
        """
        ori_input_dict = self.get_data_info(index)

        # deepcopy here to avoid inplace modification in pipeline.
        input_dict = copy.deepcopy(ori_input_dict)

        # box_type_3d (str): 3D box type.
        input_dict["box_type_3d"] = self.box_type_3d
        # box_mode_3d (str): 3D box mode.
        input_dict["box_mode_3d"] = self.box_mode_3d

        if aug_param is None:
            if "aug_param" not in input_dict:
                input_dict["aug_param"] = {}
        else:
            input_dict["aug_param"] = aug_param
        # pre-pipline return None to random another in `__getitem__`
        if not self.test_mode and self.filter_empty_gt:
            if len(input_dict["ann_info"]["gt_labels_3d"]) == 0:
                return None
        self.add_rot_trans_mats(input_dict)

        if not input_dict:
            return None

        if not self.debug_pipeline_keys:
            example = self.pipeline(input_dict)
        else:
            example = self._debug_pipeline(input_dict)
        if example is None:
            return None

        if not self.test_mode and self.filter_empty_gt:
            # after pipeline drop the example with empty annotations
            # return None to random another in `__getitem__`

            try:
                assert len(example["data_samples"].gt_instances_3d.labels_3d)
            except (AttributeError, IndexError, AssertionError):
                if frame_num == 0:
                    print_log(
                        f"No labels found for sample index {index}, rerolling.",
                        "current",
                        logging.DEBUG,
                    )
                    return None

            except TypeError:
                print(f"Incorrect pipeline output type: {type(example)}")
                exit()
        if self.show_ins_var:
            if "ann_info" in ori_input_dict:
                self._show_ins_var(
                    ori_input_dict["ann_info"]["gt_labels_3d"],
                    example["data_samples"].gt_instances_3d.labels_3d,
                )
            else:
                print_log(
                    "'ann_info' is not in the input dict. It's probably that "
                    "the data is not in training mode",
                    "current",
                    level=30,
                )

        return example

    def prepare_data(  # type:ignore
        self, index: int
    ) -> Union[Tuple[Dict[int, Any], Dict[int, Any]], None]:
        """Data preparation for both training and testing stage.

        Called by `__getitem__`  of dataset.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict or None: Data dict of the corresponding index.
        """
        example = self.prepare_data_single(index, 0)
        if example is None:
            return None
        if isinstance(example, list):
            example = example[0]
            list_output = True
        else:
            list_output = False
        cur_context_name = example["data_samples"].context_name
        if hasattr(example["data_samples"], "aug_param"):
            aug_param = copy.deepcopy(example["data_samples"].aug_param)
        else:
            aug_param = {}
        data_dict = OrderedDict()
        for i, list_idx in zip(self.frames + index, self.frames):
            if list_idx == 0:
                data_dict[list_idx] = example
                continue

            if 0 <= i < len(self):
                data_info = self.prepare_data_single(
                    index=i, frame_num=list_idx, aug_param=aug_param
                )
                # add frame to queue iff part of the same scene as index frame
                if data_info is None:
                    data_info = copy.deepcopy(example)
                    data_info["data_samples"].mask = True

                elif list_output:
                    data_info = data_info[0]

                if data_info["data_samples"].context_name != cur_context_name:
                    data_info["data_samples"].mask = True
                else:
                    data_info["data_samples"].mask = False

            else:
                data_info = copy.deepcopy(example)
                data_info["data_samples"].mask = True

            data_dict[list_idx] = data_info
        output = self.union2one(data_dict)

        return {i: o["inputs"] for i, o in output.items()}, {
            i: o["data_samples"] for i, o in output.items()
        }

    def parse_data_info(self, info: dict) -> dict:
        """Process the raw data info.

        Convert all relative path of needed modality data file to
        the absolute path. And process the `instances` field to
        `ann_info` in training stage.

        Args:
            info (dict): Raw info dict.

        Returns:
            dict: Has `ann_info` in training stage. And
            all path has been converted to absolute path.
        """

        if self.modality["use_lidar"]:
            info["lidar_points"]["lidar_path"] = osp.join(
                self.data_prefix.get("pts", ""), info["lidar_points"]["lidar_path"]
            )

            info["num_pts_feats"] = info["lidar_points"]["num_pts_feats"]
            info["lidar_path"] = info["lidar_points"]["lidar_path"]
            if "lidar_sweeps" in info:
                for sweep in info["lidar_sweeps"]:
                    file_suffix = sweep["lidar_points"]["lidar_path"].split(os.sep)[-1]
                    if "samples" in sweep["lidar_points"]["lidar_path"]:
                        sweep["lidar_points"]["lidar_path"] = osp.join(
                            self.data_prefix["pts"], file_suffix
                        )
                    else:
                        sweep["lidar_points"]["lidar_path"] = osp.join(
                            self.data_prefix["sweeps"], file_suffix
                        )

        if self.modality["use_camera"]:
            for cam_id, img_info in info["images"].items():
                if "img_path" in img_info:
                    if cam_id in self.data_prefix:
                        cam_prefix = self.data_prefix[cam_id]
                    else:
                        cam_prefix = self.data_prefix.get("img", "")
                    img_info["img_path"] = osp.join(cam_prefix, img_info["img_path"])

            if self.default_cam_key is not None:
                info["img_path"] = info["images"][self.default_cam_key]["img_path"]
                if "lidar2cam" in info["images"][self.default_cam_key]:
                    info["lidar2cam"] = np.array(
                        info["images"][self.default_cam_key]["lidar2cam"]
                    )
                if "cam2img" in info["images"][self.default_cam_key]:
                    info["cam2img"] = np.array(
                        info["images"][self.default_cam_key]["cam2img"]
                    )
                if "lidar2img" in info["images"][self.default_cam_key]:
                    info["lidar2img"] = np.array(
                        info["images"][self.default_cam_key]["lidar2img"]
                    )
                else:
                    info["lidar2img"] = info["cam2img"] @ info["lidar2cam"]

        if not self.test_mode:
            # used in training
            info["ann_info"] = self.parse_ann_info(info)
        if self.test_mode and self.load_eval_anns:
            info["eval_ann_info"] = self.parse_ann_info(info)

        return info

    def _debug_pipeline(self, data: Dict):
        for name in self.debug_pipeline_keys:
            if name in data:
                print_log(
                    f"{name} -- { type( data[name] ) }",
                    "current",
                    logging.INFO,
                )
                if hasattr(data[name], "shape"):
                    print_log(
                        f"{name} -- { data[name].shape }",
                        "current",
                        logging.INFO,
                    )
                elif isinstance(data[name], Sequence):
                    print_log(
                        f"{name} -- {  np.array( data[name] ).shape}",
                        "current",
                        logging.INFO,
                    )

            else:
                print_log(
                    f"{name} -- not found",
                    "current",
                    logging.INFO,
                )

        for t in self.pipeline.transforms:
            data = t(data)
            for name in self.debug_pipeline_keys:
                if name in data:
                    print_log(
                        f"{t.__class__.__name__}: {name} -- { type( data[name] ) }",
                        "current",
                        logging.INFO,
                    )
                    if hasattr(data[name], "shape"):
                        print_log(
                            f"{t.__class__.__name__}: {name} -- { data[name].shape }",
                            "current",
                            logging.INFO,
                        )
                    elif isinstance(data[name], Sequence):
                        print_log(
                            f"{t.__class__.__name__}: {name} -- {  np.array( data[name] ).shape}",
                            "current",
                            logging.INFO,
                        )
                else:
                    print_log(
                        f"{t.__class__.__name__}: {name} -- not found",
                        "current",
                        logging.INFO,
                    )
        return data
