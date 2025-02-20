# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmcv.transforms.base import BaseTransform
from mmengine.registry import TRANSFORMS
from mmengine.structures import InstanceData

from projects.MAMBEV.mambev.datasets.nuscenes_dataset import NuScenesDatasetMultiFrame
from mmdet3d.structures import Det3DDataSample, LiDARInstance3DBoxes


def _generate_nus_dataset_config():
    data_root = "./data/nuscenes"
    ann_file = "nuscenes_infos_temporal_val_v3.pkl"
    classes = [
        "car",
        "truck",
        "trailer",
        "bus",
        "construction_vehicle",
        "bicycle",
        "motorcycle",
        "pedestrian",
        "traffic_cone",
        "barrier",
    ]
    modality = dict(use_lidar=False, use_camera=True)

    data_prefix = dict(
        pts="samples/LIDAR_TOP",
        sweeps="sweeps/LIDAR_TOP",
        CAM_FRONT="samples/CAM_FRONT",
        CAM_FRONT_LEFT="samples/CAM_FRONT_LEFT",
        CAM_FRONT_RIGHT="samples/CAM_FRONT_RIGHT",
        CAM_BACK="samples/CAM_BACK",
        CAM_BACK_RIGHT="samples/CAM_BACK_RIGHT",
        CAM_BACK_LEFT="samples/CAM_BACK_LEFT",
    )

    meta_keys = (
        # rot scale trans mats
        "ego2global",
        "lidar2ego",
        "lidar2img",
        "lidar2cam",
        "cam2img",
        "axis_align_matrix",
        # mono data
        "mono_input_dict",
        # indices
        "mono_ann_idx",
        "sample_idx",
        "frame_idx",
        "scene_token",
        # augmentation
        "aug_param",
        "box_mode_3d",
        "box_type_3d",
        "pad_shape",
        "ori_shape",
        "img_shape",
        "crop_offset",
        "img_crop_offset",
        "img_norm_cfg",
        "resize_img_shape",
        "scale_factor",
        # other
        "filename",
        "num_pts_feats",
        "pts_filename",
        "timestamp",
    )
    pipeline = [
        dict(type="mmdet3d.LoadMultiViewImageFromFiles", to_float32=True),
        dict(
            type="NormalizeMultiviewImage",
            **dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False),
        ),
        dict(type="PadMultiViewImage", size_divisor=32),
        dict(
            type="mmdet3d.MultiScaleFlipAug3D",
            img_scale=(1600, 900),
            pts_scale_ratio=1,
            flip=False,
            transforms=[
                dict(type="RandomScaleImageMultiViewImage", scales=[0.5]),
                dict(type="PadMultiViewImage", size_divisor=32),
                dict(
                    type="mmdet3d.Pack3DDetInputs",
                    keys=[
                        "gt_bboxes_3d",
                        "gt_labels_3d",
                        "img",
                    ],
                    meta_keys=meta_keys,
                ),
            ],
        ),
    ]

    frames = (-2, -1, 0)
    return (
        data_root,
        ann_file,
        classes,
        data_prefix,
        pipeline,
        modality,
        frames,
    )


def test_getitem():
    np.random.seed(0)
    data_root, ann_file, classes, data_prefix, pipeline, modality, frames = (
        _generate_nus_dataset_config()
    )
    nus_dataset = NuScenesDatasetMultiFrame(
        data_root=data_root,
        ann_file=ann_file,
        data_prefix=data_prefix,
        pipeline=pipeline,
        test_mode=True,
        metainfo=dict(classes=classes),
        modality=modality,
        frames=frames,
        indices=2,
        use_valid_flag=True,
        box_type_3d="LiDAR",
    )

    input_dict0, data_samples0 = nus_dataset.prepare_data(0)
    input_dict1, data_samples1 = nus_dataset.prepare_data(1)
    # {i: o["inputs"] for i, o in output.items()}, {
    #             i: o["data_samples"] for i, o in output.items()
    #         }
    # input_dict = nus_dataset.get_data_info(0)
    # assert the the path should contains data_prefix and data_root
    # assert data_prefix["pts"] in input_dict["lidar_points"]["lidar_path"]
    # assert data_root in input_dict["lidar_points"]["lidar_path"]
    # print(list(input_dict[0]["img"].keys()))
    #
    # print(dir(data_samples[0]))
    # print(list(data_samples[0].eval_ann_info.keys()))
    # print(dir(data_samples))
    for files in zip(data_samples1[0].filename, data_samples0[0].filename):
        assert len(files) == len(set(files)), f"Files: { files }"
        print(*files, sep="\n", end="\n\n")

    #
    # for files in zip(*[d.filename for d in data_samples0.values()]):
    #     assert len(files) == len(set(files))
    #
    # for cam_id, img_info in input_dict["images"].items():
    #     if "img_path" in img_info:
    #         assert data_prefix["img"] in img_info["img_path"]
    #         assert data_root in img_info["img_path"]
    #
    # ann_info = nus_dataset.parse_ann_info(input_dict)
    #
    # # assert the keys in ann_info and the type
    # assert "gt_labels_3d" in ann_info
    # assert ann_info["gt_labels_3d"].dtype == np.int64
    # # assert len(ann_info["gt_labels_3d"]) == 37
    #
    # assert "gt_bboxes_3d" in ann_info
    # assert isinstance(ann_info["gt_bboxes_3d"], LiDARInstance3DBoxes)
    #
    # assert len(nus_dataset.metainfo["classes"]) == 10
    #
    # # assert input_dict["token"] == "fd8420396768425eabec9bdddf7e64b6"
    # # assert input_dict["timestamp"] == 1533201470.448696
    #


if __name__ == "__main__":
    test_getitem()
