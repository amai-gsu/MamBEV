# Based on https://github.com/nutonomy/nuscenes-devkit
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

from collections import Counter
from dataclasses import dataclass
from functools import partial
import multiprocessing as mp
from pathlib import Path
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from mmengine.fileio import load
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.detection.render import visualize_sample
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box, LidarPointCloud
from nuscenes.utils.geometry_utils import BoxVisibility, box_in_image, view_points
from PIL import Image
from pyquaternion import Quaternion

cams = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT",
]


def render_annotation(
    anntoken: str,
    nusc: NuScenes,
    margin: float = 10,
    view: np.ndarray = np.eye(4),
    box_vis_level: BoxVisibility = BoxVisibility.ANY,
    out_path: str = "render.png",
    extra_info: bool = False,
) -> None:
    """
    Render selected annotation.
    :param anntoken: Sample_annotation token.
    :param margin: How many meters in each direction to include in LIDAR view.
    :param view: LIDAR view point.
    :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
    :param out_path: Optional path to save the rendered figure to disk.
    :param extra_info: Whether to render extra information below camera view.
    """
    ann_record = nusc.get("sample_annotation", anntoken)
    sample_record = nusc.get("sample", ann_record["sample_token"])
    assert (
        "LIDAR_TOP" in sample_record["data"].keys()
    ), "Error: No LIDAR_TOP in data, unable to render."

    # Figure out which camera the object is fully visible in (this may return nothing).
    boxes, cam = [], []
    cams = [key for key in sample_record["data"].keys() if "CAM" in key]
    all_bboxes = []
    select_cams = []
    for cam in cams:
        _, boxes, _ = nusc.get_sample_data(
            sample_record["data"][cam],
            box_vis_level=box_vis_level,
            selected_anntokens=[anntoken],
        )
        if len(boxes) > 0:
            all_bboxes.append(boxes)
            select_cams.append(cam)
            # We found an image that matches. Let's abort.
    # assert len(boxes) > 0, 'Error: Could not find image where annotation is visible. ' \
    #                      'Try using e.g. BoxVisibility.ANY.'
    # assert len(boxes) < 2, 'Error: Found multiple annotations. Something is wrong!'

    num_cam = len(all_bboxes)

    fig, axes = plt.subplots(1, num_cam + 1, figsize=(18, 9))
    select_cams = [sample_record["data"][cam] for cam in select_cams]
    print("bbox in cams:", select_cams)
    # Plot LIDAR view.
    lidar = sample_record["data"]["LIDAR_TOP"]
    data_path, boxes, camera_intrinsic = nusc.get_sample_data(
        lidar, selected_anntokens=[anntoken]
    )
    LidarPointCloud.from_file(data_path).render_height(axes[0], view=view)
    for box in boxes:
        c = np.array(get_color(box.name)) / 255.0
        box.render(axes[0], view=view, colors=(c, c, c))
        corners = view_points(boxes[0].corners(), view, False)[:2, :]
        axes[0].set_xlim(
            [np.min(corners[0, :]) - margin, np.max(corners[0, :]) + margin]
        )
        axes[0].set_ylim(
            [np.min(corners[1, :]) - margin, np.max(corners[1, :]) + margin]
        )
        axes[0].axis("off")
        axes[0].set_aspect("equal")

    # Plot CAMERA view.
    for i in range(1, num_cam + 1):
        cam = select_cams[i - 1]
        data_path, boxes, camera_intrinsic = nusc.get_sample_data(
            cam, selected_anntokens=[anntoken]
        )
        im = Image.open(data_path)
        axes[i].imshow(im)
        axes[i].set_title(nusc.get("sample_data", cam)["channel"])
        axes[i].axis("off")
        axes[i].set_aspect("equal")
        for box in boxes:
            c = np.array(get_color(box.name)) / 255.0
            box.render(axes[i], view=camera_intrinsic, normalize=True, colors=(c, c, c))

        # Print extra information about the annotation below the camera view.
        axes[i].set_xlim(0, im.size[0])
        axes[i].set_ylim(im.size[1], 0)

    if extra_info:
        rcParams["font.family"] = "monospace"

        w, l, h = ann_record["size"]
        category = ann_record["category_name"]
        lidar_points = ann_record["num_lidar_pts"]
        radar_points = ann_record["num_radar_pts"]

        sample_data_record = nusc.get("sample_data", sample_record["data"]["LIDAR_TOP"])
        pose_record = nusc.get("ego_pose", sample_data_record["ego_pose_token"])
        dist = np.linalg.norm(
            np.array(pose_record["translation"]) - np.array(ann_record["translation"])
        )

        information = " \n".join(
            [
                "category: {}".format(category),
                "",
                "# lidar points: {0:>4}".format(lidar_points),
                "# radar points: {0:>4}".format(radar_points),
                "",
                "distance: {:>7.3f}m".format(dist),
                "",
                "width:  {:>7.3f}m".format(w),
                "length: {:>7.3f}m".format(l),
                "height: {:>7.3f}m".format(h),
            ]
        )

        plt.annotate(
            information,
            (0, 0),
            (0, -20),
            xycoords="axes fraction",
            textcoords="offset points",
            va="top",
        )

    if out_path is not None:
        plt.savefig(out_path)


def get_sample_data(
    sample_data_token: str,
    nusc: NuScenes,
    box_vis_level: BoxVisibility = BoxVisibility.ANY,
    selected_anntokens=None,
    use_flat_vehicle_coordinates: bool = False,
):
    """
    Returns the data path as well as all annotations related to that sample_data.
    Note that the boxes are transformed into the current sensor's coordinate frame.
    :param sample_data_token: Sample_data token.
    :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
    :param selected_anntokens: If provided only return the selected annotation.
    :param use_flat_vehicle_coordinates: Instead of the current sensor's coordinate frame, use ego frame which is
                                         aligned to z-plane in the world.
    :return: (data_path, boxes, camera_intrinsic <np.array: 3, 3>)
    """

    # Retrieve sensor & pose records
    sd_record = nusc.get("sample_data", sample_data_token)
    cs_record = nusc.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
    sensor_record = nusc.get("sensor", cs_record["sensor_token"])
    pose_record = nusc.get("ego_pose", sd_record["ego_pose_token"])

    data_path = nusc.get_sample_data_path(sample_data_token)

    if sensor_record["modality"] == "camera":
        cam_intrinsic = np.array(cs_record["camera_intrinsic"])
        imsize = (sd_record["width"], sd_record["height"])
    else:
        cam_intrinsic = None
        imsize = None

    # Retrieve all sample annotations and map to sensor coordinate system.
    if selected_anntokens is not None:
        boxes = list(map(nusc.get_box, selected_anntokens))
    else:
        boxes = nusc.get_boxes(sample_data_token)

    # Make list of Box objects including coord system transforms.
    box_list = []
    for box in boxes:
        if use_flat_vehicle_coordinates:
            # Move box to ego vehicle coord system parallel to world z plane.
            yaw = Quaternion(pose_record["rotation"]).yaw_pitch_roll[0]
            box.translate(-np.array(pose_record["translation"]))
            box.rotate(
                Quaternion(
                    scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]
                ).inverse
            )
        else:
            # Move box to ego vehicle coord system.
            box.translate(-np.array(pose_record["translation"]))
            box.rotate(Quaternion(pose_record["rotation"]).inverse)

            #  Move box to sensor coord system.
            box.translate(-np.array(cs_record["translation"]))
            box.rotate(Quaternion(cs_record["rotation"]).inverse)

        if sensor_record["modality"] == "camera" and not box_in_image(
            box, cam_intrinsic, imsize, vis_level=box_vis_level
        ):
            continue

        box_list.append(box)

    return data_path, box_list, cam_intrinsic


@dataclass
class NuscData:
    sample_data: Any
    calibrated_sensor: Any
    sensor: Any
    ego_pose: Any
    anns: Any
    contents: Any


def get_predicted_data(
    sample_data_token: str,
    nusc: NuScenes,
    box_vis_level: BoxVisibility = BoxVisibility.ANY,
    selected_anntokens=None,
    use_flat_vehicle_coordinates: bool = False,
    pred_anns=None,
):
    """
    Returns the data path as well as all annotations related to that sample_data.
    Note that the boxes are transformed into the current sensor's coordinate frame.
    :param sample_data_token: Sample_data token.
    :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
    :param selected_anntokens: If provided only return the selected annotation.
    :param use_flat_vehicle_coordinates: Instead of the current sensor's coordinate frame, use ego frame which is
                                         aligned to z-plane in the world.
    :return: (data_path, boxes, camera_intrinsic <np.array: 3, 3>)
    """

    # Retrieve sensor & pose records
    sd_record = nusc.get("sample_data", sample_data_token)
    cs_record = nusc.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
    sensor_record = nusc.get("sensor", cs_record["sensor_token"])
    pose_record = nusc.get("ego_pose", sd_record["ego_pose_token"])

    data_path = nusc.get_sample_data_path(sample_data_token)

    if sensor_record["modality"] == "camera":
        cam_intrinsic = np.array(cs_record["camera_intrinsic"])
        imsize = (sd_record["width"], sd_record["height"])
    else:
        cam_intrinsic = None
        imsize = None

    # Retrieve all sample annotations and map to sensor coordinate system.
    # if selected_anntokens is not None:
    #    boxes = list(map(nusc.get_box, selected_anntokens))
    # else:
    #    boxes = nusc.get_boxes(sample_data_token)
    boxes = pred_anns
    # Make list of Box objects including coord system transforms.
    box_list = []
    for box in boxes:
        if use_flat_vehicle_coordinates:
            # Move box to ego vehicle coord system parallel to world z plane.
            yaw = Quaternion(pose_record["rotation"]).yaw_pitch_roll[0]
            box.translate(-np.array(pose_record["translation"]))
            box.rotate(
                Quaternion(
                    scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]
                ).inverse
            )
        else:
            # Move box to ego vehicle coord system.
            box.translate(-np.array(pose_record["translation"]))
            box.rotate(Quaternion(pose_record["rotation"]).inverse)

            #  Move box to sensor coord system.
            box.translate(-np.array(cs_record["translation"]))
            box.rotate(Quaternion(cs_record["rotation"]).inverse)

        if sensor_record["modality"] == "camera" and not box_in_image(
            box, cam_intrinsic, imsize, vis_level=box_vis_level
        ):
            continue
        box_list.append(box)

    return data_path, box_list, cam_intrinsic


def lidar_render(sample_token, data, out_path: Path, nusc: NuScenes):
    bbox_gt_list = []
    bbox_pred_list = []
    anns = nusc.get("sample", sample_token)["anns"]
    for ann in anns:
        content = nusc.get("sample_annotation", ann)
        try:
            bbox_gt_list.append(
                DetectionBox(
                    sample_token=content["sample_token"],
                    translation=tuple(content["translation"]),
                    size=tuple(content["size"]),
                    rotation=tuple(content["rotation"]),
                    velocity=nusc.box_velocity(content["token"])[:2],
                    ego_translation=(0.0, 0.0, 0.0)
                    if "ego_translation" not in content
                    else tuple(content["ego_translation"]),
                    num_pts=-1 if "num_pts" not in content else int(content["num_pts"]),
                    detection_name=category_to_detection_name(content["category_name"]),
                    detection_score=-1.0
                    if "detection_score" not in content
                    else float(content["detection_score"]),
                    attribute_name="",
                )
            )
        except:
            pass

    bbox_anns = data["results"][sample_token]
    for content in bbox_anns:
        bbox_pred_list.append(
            DetectionBox(
                sample_token=content["sample_token"],
                translation=tuple(content["translation"]),
                size=tuple(content["size"]),
                rotation=tuple(content["rotation"]),
                velocity=tuple(content["velocity"]),
                ego_translation=(0.0, 0.0, 0.0)
                if "ego_translation" not in content
                else tuple(content["ego_translation"]),
                num_pts=-1 if "num_pts" not in content else int(content["num_pts"]),
                detection_name=content["detection_name"],
                detection_score=-1.0
                if "detection_score" not in content
                else float(content["detection_score"]),
                attribute_name=content["attribute_name"],
            )
        )
    gt_annotations = EvalBoxes()
    pred_annotations = EvalBoxes()
    gt_annotations.add_boxes(sample_token, bbox_gt_list)
    pred_annotations.add_boxes(sample_token, bbox_pred_list)
    print("green is ground truth")
    print("blue is the predited result")
    visualize_sample(
        nusc,
        sample_token,
        gt_annotations,
        pred_annotations,
        savepath=out_path / f"{sample_token}_bev",
    )


def get_color(category_name: str):
    """
    Provides the default colors based on the category names.
    This method works for the general nuScenes categories, as well as the nuScenes detection categories.
    """
    a = [
        "noise",
        "animal",
        "human.pedestrian.adult",
        "human.pedestrian.child",
        "human.pedestrian.construction_worker",
        "human.pedestrian.personal_mobility",
        "human.pedestrian.police_officer",
        "human.pedestrian.stroller",
        "human.pedestrian.wheelchair",
        "movable_object.barrier",
        "movable_object.debris",
        "movable_object.pushable_pullable",
        "movable_object.trafficcone",
        "static_object.bicycle_rack",
        "vehicle.bicycle",
        "vehicle.bus.bendy",
        "vehicle.bus.rigid",
        "vehicle.car",
        "vehicle.construction",
        "vehicle.emergency.ambulance",
        "vehicle.emergency.police",
        "vehicle.motorcycle",
        "vehicle.trailer",
        "vehicle.truck",
        "flat.driveable_surface",
        "flat.other",
        "flat.sidewalk",
        "flat.terrain",
        "static.manmade",
        "static.other",
        "static.vegetation",
        "vehicle.ego",
    ]
    class_names = [
        "car",
        "truck",
        "construction_vehicle",
        "bus",
        "trailer",
        "barrier",
        "motorcycle",
        "bicycle",
        "pedestrian",
        "traffic_cone",
    ]
    # print(category_name)
    if category_name == "bicycle":
        return nusc.colormap["vehicle.bicycle"]
    elif category_name == "construction_vehicle":
        return nusc.colormap["vehicle.construction"]
    elif category_name == "traffic_cone":
        return nusc.colormap["movable_object.trafficcone"]

    for key in nusc.colormap.keys():
        if category_name in key:
            return nusc.colormap[key]
    return [0, 0, 0]


def render_sample_data(
    sample_token: str,
    out_path: Path,
    nusc: NuScenes,
    with_anns: bool = True,
    box_vis_level: BoxVisibility = BoxVisibility.ANY,
    axes_limit: float = 40,
    ax=None,
    nsweeps: int = 1,
    underlay_map: bool = True,
    use_flat_vehicle_coordinates: bool = True,
    show_lidarseg: bool = False,
    show_lidarseg_legend: bool = False,
    filter_lidarseg_labels=None,
    lidarseg_preds_bin_path: str = None,
    verbose: bool = True,
    show_panoptic: bool = False,
    pred_data=None,
) -> None:
    """
    Render sample data onto axis.
    :param sample_data_token: Sample_data token.
    :param with_anns: Whether to draw box annotations.
    :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
    :param axes_limit: Axes limit for lidar and radar (measured in meters).
    :param ax: Axes onto which to render.
    :param nsweeps: Number of sweeps for lidar and radar.
    :param out_path: Optional path to save the rendered figure to disk.
    :param underlay_map: When set to true, lidar data is plotted onto the map. This can be slow.
    :param use_flat_vehicle_coordinates: Instead of the current sensor's coordinate frame, use ego frame which is
        aligned to z-plane in the world. Note: Previously this method did not use flat vehicle coordinates, which
        can lead to small errors when the vertical axis of the global frame and lidar are not aligned. The new
        setting is more correct and rotates the plot by ~90 degrees.
    :param show_lidarseg: When set to True, the lidar data is colored with the segmentation labels. When set
        to False, the colors of the lidar data represent the distance from the center of the ego vehicle.
    :param show_lidarseg_legend: Whether to display the legend for the lidarseg labels in the frame.
    :param filter_lidarseg_labels: Only show lidar points which belong to the given list of classes. If None
        or the list is empty, all classes will be displayed.
    :param lidarseg_preds_bin_path: A path to the .bin file which contains the user's lidar segmentation
                                    predictions for the sample.
    :param verbose: Whether to display the image after it is rendered.
    :param show_panoptic: When set to True, the lidar data is colored with the panoptic labels. When set
        to False, the colors of the lidar data represent the distance from the center of the ego vehicle.
        If show_lidarseg is True, show_panoptic will be set to False.
    """
    lidar_render(sample_token, pred_data, out_path=out_path, nusc=nusc)
    sample = nusc.get("sample", sample_token)
    # sample = data['results'][sample_token_list[0]][0]
    cams = [
        "CAM_FRONT_LEFT",
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK_LEFT",
        "CAM_BACK",
        "CAM_BACK_RIGHT",
    ]
    if ax is None:
        _, ax = plt.subplots(4, 3, figsize=(24, 18))
    j = 0
    boxes_all = Counter()
    for ind, cam in enumerate(cams):
        sample_data_token = sample["data"][cam]

        sd_record = nusc.get("sample_data", sample_data_token)
        sensor_modality = sd_record["sensor_modality"]

        assert sensor_modality == "camera"

        # Load boxes and image.
        # NOTE: Only renders detections scores above threshold of 0.2
        boxes = [
            Box(
                record["translation"],
                record["size"],
                Quaternion(record["rotation"]),
                name=record["detection_name"],
                token="predicted",
            )
            for record in pred_data["results"][sample_token]
            if record["detection_score"] > 0.2
        ]

        data_path, boxes_pred, camera_intrinsic = get_predicted_data(
            sample_data_token, box_vis_level=box_vis_level, pred_anns=boxes, nusc=nusc
        )
        _, boxes_gt, _ = nusc.get_sample_data(
            sample_data_token, box_vis_level=box_vis_level
        )
        # set grid coords in plot
        if ind == 3:
            j += 1
        ind = ind % 3
        data = Image.open(data_path)
        # mmcv.imwrite(np.array(data)[:,:,::-1], f'{cam}.png')
        # Init axes.

        # Show image.
        ax[j, ind].imshow(data)
        ax[j + 2, ind].imshow(data)

        couts_pred = Counter([b.name for b in boxes_pred])
        couts_gt = Counter([b.name for b in boxes_gt])
        boxes_all += couts_pred + couts_gt

        # Show boxes.
        if with_anns:
            for box in boxes_pred:
                c = np.array(get_color(box.name)) / 255.0
                box.render(
                    ax[j, ind],
                    view=camera_intrinsic,
                    normalize=True,
                    colors=(c, c, c),
                )
            for box in boxes_gt:
                c = np.array(get_color(box.name)) / 255.0
                box.render(
                    ax[j + 2, ind],
                    view=camera_intrinsic,  # type:ignore
                    normalize=True,
                    colors=(c, c, c),
                )

        # Limit visible range.
        ax[j, ind].set_xlim(0, data.size[0])
        ax[j, ind].set_ylim(data.size[1], 0)
        ax[j + 2, ind].set_xlim(0, data.size[0])
        ax[j + 2, ind].set_ylim(data.size[1], 0)

        ax[j, ind].axis("off")
        ax[j, ind].set_title(
            "PRED: {} {labels_type}".format(
                sd_record["channel"],
                labels_type="(predictions)" if lidarseg_preds_bin_path else "",
            )
        )
        ax[j, ind].set_aspect("equal")

        ax[j + 2, ind].axis("off")
        ax[j + 2, ind].set_title(
            "GT:{} {labels_type}".format(
                sd_record["channel"],
                labels_type="(predictions)" if lidarseg_preds_bin_path else "",
            )
        )
        ax[j + 2, ind].set_aspect("equal")

    # create a legend which displays all mappings from color to category in order of most to least common
    # right now this excludes all predictions under the threshold
    patches = {}

    for lab, c in map(
        lambda box: (box[0], np.array(get_color(box[0])) / 255.0),
        boxes_all.most_common(),
    ):
        if lab in patches:
            continue
        patch = mpatches.Patch(color=c, label=lab)
        patches[lab] = patch

    plt.legend(
        handles=list(patches.values()),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        fancybox=True,
        ncol=len(patches),
    )
    if out_path is not None:
        plt.savefig(
            out_path / f"{sample_token}_camera",
            bbox_inches="tight",
            pad_inches=0,
            dpi=200,
        )
    if verbose:
        plt.show()
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", dest="data_root", type=str, default="./data/nuscenes"
    )
    parser.add_argument("--results-json", dest="results_json", type=str)
    parser.add_argument("--num-render", dest="num_render", type=int, default=10)
    parser.add_argument("--offset-render", dest="offset_render", type=int, default=0)
    parser.add_argument("--save-path", dest="save_path", default=".")
    parser.add_argument("--prop-cpus", dest="prop_cpus", type=float, default=0.9)
    cfg = parser.parse_args()
    nusc = NuScenes(version="v1.0-trainval", dataroot=cfg.data_root, verbose=True)
    # render_annotation('7603b030b42a4b1caa8c443ccc1a7d52')
    bevformer_results = load(cfg.results_json)
    sample_token_list = list(bevformer_results["results"].keys())

    mp.set_start_method("spawn")
    workers = int(cfg.prop_cpus * mp.cpu_count())
    af_render_sample_data = partial(
        render_sample_data,
        pred_data=bevformer_results,
        out_path=Path(cfg.save_path),
        nusc=nusc,
    )

    # with mp.Pool(workers) as p:
    #     p.map(
    #         af_render_sample_data, sample_token_list[cfg.offset_render : cfg.num_render]
    #     )
    for id in range(cfg.offset_render, cfg.num_render):
        render_sample_data(
            sample_token_list[id],
            pred_data=bevformer_results,
            out_path=Path(cfg.save_path),
            nusc=nusc,
        )
