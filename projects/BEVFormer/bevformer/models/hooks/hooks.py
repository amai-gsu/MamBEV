# from mmcv.runner.hooks.hook import HOOKS, Hook
from mmengine.registry import HOOKS
from mmengine.hooks import Hook

from typing import Optional, Union

DATA_BATCH = Optional[Union[dict, tuple, list]]
# from projects.mmdet3d_plugin.models.utils import run_time


@HOOKS.register_module()
class GradChecker(Hook):
    def after_train_iter(
        self,
        runner,
        batch_idx: int,
        data_batch: DATA_BATCH = None,
        outputs: Optional[dict] = None,
    ) -> None:
        for key, val in runner.model.named_parameters():
            if val.grad is None and val.requires_grad:
                print(
                    "WARNNING: {key}'s parameters are not be used!!!!".format(key=key)
                )
