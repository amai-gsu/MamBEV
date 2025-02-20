from typing import Dict
import numpy as np
import pprint
import torch

# import mmcv
# # from mmdet.datasets.builder import PIPELINESfrom mmdet.registry import TRANSFORMS -> Transforms
from mmengine.registry import TRANSFORMS
from PIL import Image
from PIL.Image import Transpose
import random


@TRANSFORMS.register_module()
class CropResizeFlipImage:
    """Fixed Crop and then randim resize and flip the image. Note the flip requires to flip the feature in the network
        ida_aug_conf = {
            "resize": [576, 608, 640, 672, 704]  # stride of 32 based on 640 (0.9, 1.1)
            "resize": [512, 544, 576, 608, 640, 672, 704, 736, 768]  #  (0.8, 1.2)
            "resize": [448, 480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800, 832]  #  (0.7, 1.3)
            "crop": (0, 260, 1600, 900),
            "H": 900,
            "W": 1600,
            "rand_flip": True,
    }
        Args:
            size (tuple, optional): Fixed padding size.
    """

    def __init__(
        self,
        data_aug_conf: Dict,
        training=True,
    ):
        self.data_aug_conf = data_aug_conf
        self.training = training

    def __call__(self, results:Dict):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        if "aug_param" not in results.keys():
            results["aug_param"] = {}
        imgs = results["img"]
        N = len(imgs)
        new_imgs = []
        resize, resize_dims, crop, flip = self._sample_augmentation(results)

        for i in range(N):
            img = Image.fromarray(np.uint8(imgs[i]))

            # augmentation (resize, crop, horizontal flip, rotate)
            # resize, resize_dims, crop, flip, rotate = self._sample_augmentation()  ###different view use different aug (BEV Det)
            img, ida_mat = self._img_transform(
                img,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
            )
            new_imgs.append(np.array(img).astype(np.float32))
            temp = np.eye(4)
            temp[:3, :3] = np.matmul(ida_mat, np.array(results["cam2img"][i]))

            results["cam2img"][i] = temp
            # print(results["cam2img"][i].shape)
            # results["cam2img"][i][:3, :3] = np.matmul(
            #     ida_mat, results["cam2img"][i][:3, :3]
            # )

            # print(results["cam2img"][i].shape)
            if "mono_ann_idx" in results.keys() and results["mono_ann_idx"] is not None:
                # apply transform to dd3d intrinsics
                if i in results["mono_ann_idx"]:
                    mono_index = results["mono_ann_idx"].index(i)
                    intrinsics = results["mono_input_dict"][mono_index]["intrinsics"]
                    if torch.is_tensor(intrinsics):
                        intrinsics = intrinsics.numpy().reshape(3, 3).astype(np.float32)
                    elif isinstance(intrinsics, np.ndarray):
                        intrinsics = intrinsics.reshape(3, 3).astype(np.float32)
                    else:
                        intrinsics = np.array(intrinsics, dtype=np.float32).reshape(
                            3, 3
                        )
                    results["mono_input_dict"][mono_index]["intrinsics"] = np.matmul(
                        ida_mat, intrinsics
                    )
                    results["mono_input_dict"][mono_index]["height"] = img.size[1]
                    results["mono_input_dict"][mono_index]["width"] = img.size[0]
                    # apply transform to dd3d box
                    for ann in results["mono_input_dict"][mono_index]["annotations"]:
                        # bbox_mode = BoxMode.XYXY_ABS
                        box = self._box_transform(
                            ann["bbox"], resize, crop, flip, img.size[0]
                        )[0]
                        box = box.clip(min=0)
                        box = np.minimum(box, list(img.size + img.size))
                        ann["bbox"] = box

        results["img"] = new_imgs
        # print("LIDAR2CAM", [res.shape for res in np.array(results["lidar2cam"])])
        # print("CAM2IMG", [res.shape for res in results["cam2img"]])

        results["lidar2img"] = [
            np.matmul(results["cam2img"][j], results["lidar2cam"][j])
            for j in range(len(results["lidar2cam"]))
        ]

        return results

    def _box_transform(self, box, resize, crop, flip, img_width):
        box = np.array([box])
        idxs = np.array([(0, 1), (2, 1), (0, 3), (2, 3)]).flatten()
        coords = np.asarray(box).reshape(-1, 4)[:, idxs].reshape(-1, 2)

        # crop
        coords[:, 0] -= crop[0]
        coords[:, 1] -= crop[1]

        # resize
        coords[:, 0] = coords[:, 0] * resize
        coords[:, 1] = coords[:, 1] * resize

        coords = coords.reshape((-1, 4, 2))
        minxy = coords.min(axis=1)
        maxxy = coords.max(axis=1)
        trans_box = np.concatenate((minxy, maxxy), axis=1)

        return trans_box

    def _img_transform(self, img, resize, resize_dims, crop, flip):
        ida_rot = np.eye(2)
        ida_tran = np.zeros(2)
        # adjust image
        img = img.crop(crop)
        img = img.resize(resize_dims)
        if flip:
            img = img.transpose(method=Transpose.FLIP_LEFT_RIGHT)

        # post-homography transformation
        ida_rot *= resize
        ida_tran -= np.array(crop[:2]) * resize
        ida_mat = np.eye(3)
        ida_mat[:2, :2] = ida_rot
        ida_mat[:2, 2] = ida_tran
        return img, ida_mat

    def _sample_augmentation(self, results):
        if "CropResizeFlipImage_param" in results["aug_param"].keys():
            return results["aug_param"]["CropResizeFlipImage_param"]
        crop = self.data_aug_conf["crop"]

        if self.training:
            resized_h = random.choice(self.data_aug_conf["resize"])
            resized_w = resized_h / (crop[3] - crop[1]) * (crop[2] - crop[0])
            resize = resized_h / (crop[3] - crop[1])
            resize_dims = (int(resized_w), int(resized_h))
            flip = False
            if self.data_aug_conf["rand_flip"] and np.random.choice([0, 1]):
                flip = True
        else:
            resized_h = random.choice(self.data_aug_conf["resize"])
            assert len(self.data_aug_conf["resize"]) == 1
            resized_w = resized_h / (crop[3] - crop[1]) * (crop[2] - crop[0])
            resize = resized_h / (crop[3] - crop[1])
            resize_dims = (int(resized_w), int(resized_h))
            flip = False
        results["aug_param"]["CropResizeFlipImage_param"] = (
            resize,
            resize_dims,
            crop,
            flip,
        )

        return resize, resize_dims, crop, flip


@TRANSFORMS.register_module()
class GlobalScaleRotTransImage:
    """Random resize, Crop and flip the image
    Args:
        size (tuple, optional): Fixed padding size.
    """

    def __init__(
        self,
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0],
        reverse_angle=False,
        training=True,
        flip_dx_ratio=0.5,
        flip_dy_ratio=0.5,
        only_gt=False,
    ):
        self.rot_range = rot_range
        self.scale_ratio_range = scale_ratio_range
        self.translation_std = translation_std

        self.reverse_angle = reverse_angle
        self.training = training

        self.flip_dx_ratio = flip_dx_ratio
        self.flip_dy_ratio = flip_dy_ratio
        self.only_gt = only_gt

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        # pprint.pp(results, depth=2)
        if "aug_param" not in results.keys():
            results["aug_param"] = {}

        rot_angle, scale_ratio, flip_dx, flip_dy, _, _ = self._sample_augmentation(
            results
        )

        # random rotate
        if not self.only_gt:
            self.rotate_bev_along_z(results["ann_info"], rot_angle)
        if self.reverse_angle:
            rot_angle *= -1
        results["ann_info"]["gt_bboxes_3d"].rotate(np.array(rot_angle))

        # random scale
        if not self.only_gt:
            self.scale_xyz(results, scale_ratio)
        results["ann_info"]["gt_bboxes_3d"].scale(scale_ratio)

        # random flip
        if flip_dx:
            if not self.only_gt:
                self.flip_along_x(results["ann_info"])
            results["ann_info"]["gt_bboxes_3d"].flip(bev_direction="vertical")
        if flip_dy:
            if not self.only_gt:
                self.flip_along_y(results["ann_info"])
            results["ann_info"]["gt_bboxes_3d"].flip(bev_direction="horizontal")

        # TODO: support translation
        return results

    def _sample_augmentation(self, results):
        if "GlobalRotScaleTransImage_param" in results["aug_param"].keys():
            return results["aug_param"]["GlobalRotScaleTransImage_param"]
        else:
            rot_angle = np.random.uniform(*self.rot_range) / 180 * np.pi
            scale_ratio = np.random.uniform(*self.scale_ratio_range)
            flip_dx = np.random.uniform() < self.flip_dx_ratio
            flip_dy = np.random.uniform() < self.flip_dy_ratio
        # generate bda_mat

        rot_sin = torch.sin(torch.tensor(rot_angle))
        rot_cos = torch.cos(torch.tensor(rot_angle))
        rot_mat = torch.Tensor(
            [[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]]
        )
        scale_mat = torch.Tensor(
            [[scale_ratio, 0, 0], [0, scale_ratio, 0], [0, 0, scale_ratio]]
        )
        flip_mat = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if flip_dx:
            flip_mat = flip_mat @ torch.Tensor([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if flip_dy:
            flip_mat = flip_mat @ torch.Tensor([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
        bda_mat = flip_mat @ (scale_mat @ rot_mat)
        bda_mat = torch.inverse(bda_mat)
        results["aug_param"]["GlobalRotScaleTransImage_param"] = (
            rot_angle,
            scale_ratio,
            flip_dx,
            flip_dy,
            bda_mat,
            self.only_gt,
        )

        return rot_angle, scale_ratio, flip_dx, flip_dy, bda_mat, self.only_gt

    def rotate_bev_along_z(self, results, angle):
        rot_cos = np.cos(angle)
        rot_sin = np.sin(angle)

        rot_mat = np.array(
            [
                [rot_cos, -rot_sin, 0, 0],
                [rot_sin, rot_cos, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        rot_mat_inv = np.linalg.inv(rot_mat)

        num_view = len(results["lidar2img"])
        for view in range(num_view):
            results["lidar2img"][view] = np.matmul(
                results["lidar2img"][view], rot_mat_inv
            )
            results["lidar2cam"][view] = np.matmul(
                results["lidar2cam"][view], rot_mat_inv
            )

        return

    def scale_xyz(self, results, scale_ratio):
        scale_mat = np.array(
            [
                [scale_ratio, 0, 0, 0],
                [0, scale_ratio, 0, 0],
                [0, 0, scale_ratio, 0],
                [0, 0, 0, 1],
            ]
        )

        scale_mat_inv = np.linalg.inv(scale_mat)

        num_view = len(results["lidar2img"])
        for view in range(num_view):
            results["lidar2img"][view] = np.matmul(
                results["lidar2img"][view], scale_mat_inv
            )
            results["lidar2cam"][view] = np.matmul(
                results["lidar2cam"][view], scale_mat_inv
            )
        return

    def flip_along_x(self, results):
        flip_mat = np.array(
            [
                [-1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        ).astype(np.float32)

        flip_mat_inv = np.linalg.inv(flip_mat)

        num_view = len(results["lidar2img"])
        for view in range(num_view):
            results["lidar2img"][view] = np.matmul(
                results["lidar2img"][view], flip_mat_inv
            )
            results["lidar2cam"][view] = np.matmul(
                results["lidar2cam"][view], flip_mat_inv
            )
        return

    def flip_along_y(self, results):
        flip_mat = np.array(
            [
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        ).astype(np.float32)

        flip_mat_inv = np.linalg.inv(flip_mat)

        num_view = len(results["lidar2img"])
        for view in range(num_view):
            results["lidar2img"][view] = np.matmul(
                results["lidar2img"][view], flip_mat_inv
            )
            results["lidar2cam"][view] = np.matmul(
                results["lidar2cam"][view], flip_mat_inv
            )
        return
