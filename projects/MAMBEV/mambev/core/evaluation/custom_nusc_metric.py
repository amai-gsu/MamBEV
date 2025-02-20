# Copyright (c) OpenMMLab. All rights reserved.
# Modified by Jack Morris
from pathlib import Path
from typing import Dict, List, Optional, Union

import mmengine
from nuscenes import NuScenes
from .custom_nusc_eval import NuScenesEval

from mmdet3d.registry import METRICS
from mmdet3d.evaluation.metrics import NuScenesMetric


@METRICS.register_module()
class CustomNuScenesMetric(NuScenesMetric):
    def __init__(
        self,
        data_root: str,
        ann_file: str,
        render_tp_curves: bool = False,
        plot_examples: int = 0,
        metric: Union[str, List[str]] = "bbox",
        modality: dict = dict(use_camera=False, use_lidar=True),
        prefix: Optional[str] = None,
        format_only: bool = False,
        jsonfile_prefix: Optional[str] = None,
        eval_version: str = "detection_cvpr_2019",
        collect_device: str = "cpu",
        backend_args: Optional[dict] = None,
    ) -> None:
        if jsonfile_prefix is not None:
            jsonfile_prefix = str(Path(jsonfile_prefix).resolve())
        super().__init__(
            data_root,
            ann_file,
            metric,
            modality,
            prefix,
            format_only,
            jsonfile_prefix,
            eval_version,
            collect_device,
            backend_args,
        )
        self.render_tp_curves = render_tp_curves
        self.plot_examples = plot_examples

    def _evaluate_single(
        self,
        result_path: Union[str, Path],
        classes: Optional[List[str]] = None,
        result_name: str = "pred_instances_3d",
    ) -> Dict[str, float]:
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            classes (List[str], optional): A list of class name.
                Defaults to None.
            result_name (str): Result name in the metric prefix.
                Defaults to 'pred_instances_3d'.

        Returns:
            Dict[str, float]: Dictionary of evaluation details.
        """
        result_path = Path(result_path)
        output_dir = result_path.parent
        print(f"Output Directory: {output_dir}")

        nusc = NuScenes(version=self.version, dataroot=self.data_root, verbose=False)
        eval_set_map = {
            "v1.0-mini": "mini_val",
            "v1.0-trainval": "val",
        }
        nusc_eval = NuScenesEval(
            nusc,
            config=self.eval_detection_configs,
            result_path=result_path,
            eval_set=eval_set_map[self.version],
            output_dir=output_dir,
            verbose=False,
        )

        nusc_eval.main(
            plot_examples=self.plot_examples, render_curves=self.render_tp_curves
        )

        # record metrics
        metrics = mmengine.load(output_dir / "metrics_summary.json")
        detail = dict()
        metric_prefix = f"{result_name}_NuScenes"
        assert classes is not None
        for name in classes:
            for k, v in metrics["label_aps"][name].items():
                val = float(f"{v:.4f}")
                detail[f"{metric_prefix}/{name}_AP_dist_{k}"] = val
            for k, v in metrics["label_tp_errors"][name].items():
                val = float(f"{v:.4f}")
                detail[f"{metric_prefix}/{name}_{k}"] = val
            for k, v in metrics["tp_errors"].items():
                val = float(f"{v:.4f}")
                detail[f"{metric_prefix}/{self.ErrNameMapping[k]}"] = val

        detail[f"{metric_prefix}/NDS"] = metrics["nd_score"]
        detail[f"{metric_prefix}/mAP"] = metrics["mean_ap"]
        return detail
