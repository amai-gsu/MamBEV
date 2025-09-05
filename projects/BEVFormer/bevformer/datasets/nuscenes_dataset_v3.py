import copy
import logging
import pickle
from collections import OrderedDict
import pprint
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from mmdet3d.datasets import NuScenesDataset
from mmengine.dataset import force_full_init, default_collate
from mmengine.logging import print_log
from mmengine.registry import DATASETS

from ..dd3d.datasets.nuscenes import DD3DNuscenesDataset


@DATASETS.register_module()
class CustomNuScenesDatasetV3(NuScenesDataset):
    def __init__(
        self,
        frames: Sequence[int],
        mono_cfg: Optional[Dict] = None,
        v1_input: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.v1_input = v1_input
        self.frames = np.array(frames, dtype=int)
        self.queue_length = len(frames)
        self.mono_cfg = mono_cfg
        if not self.test_mode and mono_cfg is not None:
            self.mono_dataset = DD3DNuscenesDataset(**mono_cfg)

    def prepare_data(self, index: int) -> Union[Tuple, None]:
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

        assert input_dict is not None, "Error occured while copying data"

        input_dict = self.prepare_input_dict(input_dict)

        if not self.test_mode and self.mono_cfg is not None:
            input_dict = self.prepare_mono_input(idx=index, input_dict=input_dict)

        assert input_dict is not None, "Error occured during parsing of mono input"

        cur_scene_token = input_dict["scene_token"]
        data_queue = OrderedDict()
        for i, list_idx in zip(self.frames + index, self.frames):
            if i < 0:
                continue
            elif i >= len(self):
                break
            data_info = self.prepare_input_dict(self.get_data_info(i))

            if not self.test_mode and self.mono_cfg is not None and list_idx == 0:
                data_info = self.prepare_mono_input(idx=i, input_dict=data_info)
            else:
                data_info["mono_input_dict"] = None

            # add frame to queue iff part of the same scene as index frame
            if data_info is not None and data_info["scene_token"] == cur_scene_token:
                if not self.test_mode and self.filter_empty_gt:
                    if len(data_info["ann_info"]["gt_labels_3d"]) == 0:
                        return None

                    fr = self.pipeline(data_info)

                    if isinstance(fr, list):
                        assert (
                            len(fr) == 1
                        ), "Currently, batchsize > 1 in test/val loop is not supported"
                        fr = fr[0]

                    elif fr is None or (
                        isinstance(fr, dict)
                        and len(fr["data_samples"].gt_instances_3d.labels_3d) == 0
                    ):
                        return None
                else:
                    fr = self.pipeline(data_info)
                    assert fr is not None
                    fr = fr[0]

                data_queue[list_idx] = fr

        data_queue = OrderedDict(sorted(data_queue.items()))

        if self.v1_input:
            ret = self.union2one(data_queue)
            return ret["imgs"], ret["img_metas"]
        else:
            ret = self.union2onev2(data_queue)

            return {i: o["inputs"] for i, o in ret.items()}, {
                i: o["data_samples"] for i, o in ret.items()
            }

    def union2onev2(self, queue: Dict[int, Dict[str, Any]]):
        lidar2ego = queue[0]["data_samples"].lidar2ego
        egocurr2global = queue[0]["data_samples"].ego2global
        for i, each in queue.items():
            each["data_samples"].set_field(
                value=self.box_type_3d, name="box_type_3d", field_type="metainfo"
            )

            each["data_samples"].set_field(
                value=self.box_mode_3d, name="box_mode_3d", field_type="metainfo"
            )
            if i == 0:
                continue
            egoadj2global = each["data_samples"].ego2global

            lidaradj2lidarcurr = (
                np.linalg.inv(lidar2ego)
                @ np.linalg.inv(egocurr2global)
                @ egoadj2global
                @ lidar2ego
            )
            each["data_samples"].lidaradj2lidarcurr = lidaradj2lidarcurr
            for i_cam in range(len(each["data_samples"].lidar2img)):
                each["data_samples"].lidar2img[i_cam] = each["data_samples"].lidar2img[
                    i_cam
                ] @ np.linalg.inv(lidaradj2lidarcurr)

        return queue

    def union2one(self, queue: Dict[int, Dict[str, Any]]):
        """
        convert sample queue into one single sample.
        """
        lidar2ego = queue[0]["data_samples"].lidar2ego
        egocurr2global = queue[0]["data_samples"].ego2global
        metas_map = {}
        for i, each in queue.items():
            metas_map[i] = dict(each["data_samples"].metainfo_items())
            metas_map[i]["timestamp"] = each["data_samples"].timestamp
            try:
                metas_map[i]["aug_param"] = each["data_samples"].aug_param
            except AttributeError:
                pass

            try:
                metas_map[i]["gt_bboxes_3d"] = each["data_samples"].gt_instances_3d[
                    "bboxes_3d"
                ]
                metas_map[i]["gt_labels_3d"] = each["data_samples"].gt_instances_3d[
                    "labels_3d"
                ]
            except AttributeError:
                pass

            # find relative rotation of each frame to curr frame
            if i == 0:
                metas_map[i]["lidaradj2lidarcurr"] = None
            else:
                egoadj2global = each["data_samples"].ego2global

                lidaradj2lidarcurr = (
                    np.linalg.inv(lidar2ego)
                    @ np.linalg.inv(egocurr2global)
                    @ egoadj2global
                    @ lidar2ego
                )
                each["data_samples"].lidaradj2lidarcurr = lidaradj2lidarcurr
                metas_map[i]["lidaradj2lidarcurr"] = lidaradj2lidarcurr
                for i_cam in range(len(each["data_samples"].lidar2img)):
                    metas_map[i]["lidar2img"][i_cam] = each["data_samples"].lidar2img[
                        i_cam
                    ] @ np.linalg.inv(lidaradj2lidarcurr)

        queue[0]["imgs"] = torch.stack(
            [each["inputs"]["img"] for each in queue.values()]
        )
        queue[0]["img_metas"] = metas_map
        del queue[0]["data_samples"]
        del queue[0]["inputs"]
        return queue[0]

    def prepare_input_dict(self, info):
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info["sample_idx"],
            token=info["token"],
            pts_filename=info["lidar_points"]["lidar_path"],
            # sweeps were not created during originial because I didnt download them #
            # may need to fix ... but not sure if used #
            # sweeps=info["lidar_sweeps"],
            #################################
            # this was converted from quaternion to rotation_matrix
            # see mmdetection3d/tools/dataset_converters/update_infos_to_v2.py line 274
            ego2global=info["ego2global"],
            lidar2ego=info["lidar_points"]["lidar2ego"],
            #### deleted in update need to fix ####
            prev=info["prev"],
            next=info["next"],
            scene_token=info["scene_token"],
            #######################################
            # frame_idx=info["frame_idx"],
            timestamp=info["timestamp"],
            ann_info=info["ann_info"],
            images=info["images"],
        )

        if self.modality["use_camera"]:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []

            for cam_type, cam_info in info["images"].items():
                image_paths.append(cam_info["img_path"])
                lidar2cam = np.array(cam_info["lidar2cam"])
                lidar2cam_rts.append(lidar2cam)
                intrinsic = np.array(cam_info["cam2img"])
                viewpad = np.eye(4)
                viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
                lidar2img = viewpad @ lidar2cam
                lidar2img_rts.append(np.array(lidar2img))
                cam_intrinsics.append(viewpad)
            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam2img=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                )
            )

        return input_dict

    def filter_crowd_annotations(self, data_dict):
        for ann in data_dict["annotations"]:
            if ann.get("iscrowd", 0) == 0:
                return True
        return False

    def _load_info(self, idx):
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

    def prepare_mono_input(self, idx: int, input_dict: Dict):
        if input_dict is None:
            return None
        data_info = self._load_info(idx)
        img_ids = []
        for cam_type, cam_info in data_info["images"].items():
            img_ids.append(cam_info["sample_data_token"])

        mono_input_dict = []
        mono_ann_index = []
        for i, img_id in enumerate(img_ids):
            tmp_dict = self.mono_dataset.getitem_by_datumtoken(img_id)
            if tmp_dict is not None:
                if self.filter_crowd_annotations(tmp_dict):
                    mono_input_dict.append(tmp_dict)
                    mono_ann_index.append(i)

        # filter empty annotation
        if len(mono_ann_index) == 0:
            return None

        mono_ann_index = mono_ann_index
        input_dict["mono_input_dict"] = mono_input_dict
        input_dict["mono_ann_idx"] = mono_ann_index

        return input_dict

    @force_full_init
    def get_data_info(self, idx: int):
        # assert isinstance(idx, int), f"idx must be an int, instead got {idx}"
        data_info = self._load_info(idx)
        if not self.test_mode:
            if "ann_info" not in data_info:
                data_info["ann_info"] = self.parse_ann_info(data_info)
        else:
            data_info["ann_info"] = self.parse_ann_info(data_info)

        return data_info

    def __getitem__(self, idx: int):  # type: ignore
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if not self._fully_initialized:
            print_log(
                "Please call `full_init()` method manually to accelerate " "the speed.",
                logger="current",
                level=logging.WARNING,
            )
            self.full_init()
        for _ in range(self.max_refetch + 1):
            data = self.prepare_data(idx)
            # Broken images or random augmentations may cause the returned data
            # to be None
            if data is None:
                idx = self._rand_another()
                continue
            return data

        raise Exception(
            f"Cannot find valid image after {self.max_refetch}! "
            "Please check your image path and pipeline"
        )
