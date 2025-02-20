from mmdet3d.datasets import RandomResize3D, Resize3D
from mmdet3d.registry import TRANSFORMS
import numpy as np


@TRANSFORMS.register_module()
class Resize3DLidar2Img(Resize3D):
    def _resize_3d(self, results: dict) -> None:
        scale_factor = np.eye(4)
        scale_factor[0,0] *= results["scale_factor"][0]
        scale_factor[1,1] *= results["scale_factor"][1]
        results["lidar2img"] = scale_factor @ results["lidar2img"]
        results["img_shape"] = results["img"].shape
        results["ori_shape"] = results["img"].shape
        # super()._resize_3d(results)


@TRANSFORMS.register_module()
class RandomResize3DLidar2Img(RandomResize3D):
    def _resize_3d(self, results: dict) -> None:
        scale_factor = np.eye(4)
        scale_factor[0,0] *= results["scale_factor"][0]
        scale_factor[1,1] *= results["scale_factor"][1]
        results["lidar2img"] = scale_factor @ results["lidar2img"]
        results["img_shape"] = results["img"].shape
        results["ori_shape"] = results["img"].shape
        # super()._resize_3d(results)

