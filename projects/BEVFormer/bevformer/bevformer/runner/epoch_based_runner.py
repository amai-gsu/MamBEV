# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
"""This seems very broken but may not be necessary... or may need to be heavily modified for deepspeed training anyways"""

import torch

from typing import Sequence

# from mmcv.runner.epoch_based_runner import EpochBasedRunner
from mmengine.runner import EpochBasedTrainLoop

# from mmengine.registry import RUNNERS
from mmengine.registry import LOOPS, RUNNERS
from mmengine.structures import BaseDataElement
from mmengine.runner import Runner


# from mmcv.runner.builder import RUNNERS
# from mmcv.parallel.data_container import BaseDataElement
@RUNNERS.register_module()
class EpochBasedRunner_video(Runner):
    """
    # basic logic

    input_sequence = [a, b, c] # given a sequence of samples

    prev_bev = None
    for each in input_sequcene[:-1]
        prev_bev = eval_model(each, prev_bev)) # inference only.

    model(input_sequcene[-1], prev_bev) # train the last sample.
    """

    def __init__(
        self,
        model,
        eval_model=None,
        batch_processor=None,
        optimizer=None,
        work_dir=None,
        logger=None,
        meta=None,
        keys=["gt_bboxes_3d", "gt_labels_3d", "img"],
        max_iters=None,
        max_epochs=None,
    ):
        super().__init__(
            model,
            batch_processor,
            optimizer,
            work_dir,
            logger,
            meta,
            max_iters,
            max_epochs,
        )
        keys.append("img_metas")
        self.keys = keys
        self.eval_model = eval_model
        self.eval_model.eval()

    def run_iter(self, data_batch, train_mode, **kwargs):
        if self.batch_processor is not None:
            assert False
            # outputs = self.batch_processor(
            #     self.model, data_batch, train_mode=train_mode, **kwargs)
        elif train_mode:
            num_samples = data_batch["img"].data[0].size(1)
            data_list = []
            prev_bev = None
            for i in range(num_samples):
                data = {}
                for key in self.keys:
                    if key not in ["img_metas", "img", "points"]:
                        data[key] = data_batch[key]
                    else:
                        if key == "img":
                            data["img"] = BaseDataElement(
                                data=[data_batch["img"].data[0][:, i]],
                                cpu_only=data_batch["img"].cpu_only,
                                stack=True,
                            )
                        elif key == "img_metas":
                            data["img_metas"] = BaseDataElement(
                                data=[
                                    [
                                        each[i]
                                        for each in data_batch["img_metas"].data[0]
                                    ]
                                ],
                                cpu_only=data_batch["img_metas"].cpu_only,
                            )
                        else:
                            assert False
                data_list.append(data)
            with torch.no_grad():
                for i in range(num_samples - 1):
                    if data_list[i]["img_metas"].data[0][0]["prev_bev_exists"]:
                        data_list[i]["prev_bev"] = BaseDataElement(
                            data=[prev_bev], cpu_only=False
                        )
                    prev_bev = self.eval_model.val_step(
                        data_list[i], self.optimizer, **kwargs
                    )
            if data_list[-1]["img_metas"].data[0][0]["prev_bev_exists"]:
                data_list[-1]["prev_bev"] = BaseDataElement(
                    data=[prev_bev], cpu_only=False
                )
            outputs = self.model.train_step(data_list[-1], self.optimizer, **kwargs)
        else:
            assert False
            # outputs = self.model.val_step(data_batch, self.optimizer, **kwargs)

        if not isinstance(outputs, dict):
            raise TypeError(
                '"batch_processor()" or "model.train_step()"'
                'and "model.val_step()" must return a dict'
            )
        if "log_vars" in outputs:
            self.log_buffer.update(outputs["log_vars"], outputs["num_samples"])
        self.outputs = outputs


# @LOOPS.register_module()
# class CustomEpochBasedTrainLoop(EpochBasedTrainLoop):
#     """
#     Training loop for BEVFormer
#     """
#     def __init__(self, runner,dataloader,max_epochs,val_begin, val_interval,dynamic_intervals):
#         super().__init__(runner,dataloader,max_epochs,val_begin, val_interval, dynamic_intervals)
#
#
#     def run_iter(self, idx:int, data_batch: Sequence[dict]):
#         # num_samples = data_batch["img"].data[0].size(1)
#         num_samples = data_batch[0]["img"].size(1)
#         data_list = []
#         prev_bev = None
#         for i in range(num_samples):
#             data = {}
#             for key in data_batch[0].keys():
#                 if key not in ["img_metas", "img", "points"]:
#                     data[key] = data_batch[key]
#                 else:
#                     if key == "img":
#                         data["img"] = BaseDataElement(
#                             data=[data_batch[i]["img"]],
#                             cpu_only=data_batch["img"].cpu_only,
#                             stack=True,
#                         )
#                     elif key == "img_metas":
#                         data["img_metas"] = BaseDataElement(
#                             data=[
#                                 [each[i] for each in data_batch[i]["img_metas"].data[0]]
#                             ],
#                             cpu_only=data_batch["img_metas"].cpu_only,
#                         )
#                     else:
#                         assert False
#             data_list.append(data)
#         with torch.no_grad():
#             for i in range(num_samples - 1):
#                 if data_list[i]["img_metas"].data[0][0]["prev_bev_exists"]:
#                     data_list[i]["prev_bev"] = BaseDataElement(
#                         data=[prev_bev], cpu_only=False
#                     )
#                 prev_bev = self.eval_model.val_step(data_list[i], self.optimizer)
#         if data_list[-1]["img_metas"].data[0][0]["prev_bev_exists"]:
#             data_list[-1]["prev_bev"] = BaseDataElement(data=[prev_bev], cpu_only=False)
#         outputs = self.model.train_step(data_list[-1], self.optimizer)
#         if not isinstance(outputs, dict):
#             raise TypeError(
#                 '"batch_processor()" or "model.train_step()"'
#                 'and "model.val_step()" must return a dict'
#             )
#         if "log_vars" in outputs:
#             self.log_buffer.update(outputs["log_vars"], outputs["num_samples"])
#         self.outputs = outputs
#         pass
#
#
#
#
#
#
#
#
#
#
# class EpochBasedRunner_video(EpochBasedTrainLoop):
#     """
#     # basic logic
#
#     input_sequence = [a, b, c] # given a sequence of samples
#
#     prev_bev = None
#     for each in input_sequcene[:-1]
#         prev_bev = eval_model(each, prev_bev)) # inference only.
#
#     model(input_sequcene[-1], prev_bev) # train the last sample.
#     """
#
#     def __init__(
#         self,
#         model,
#         eval_model=None,
#         batch_processor=None,
#         optimizer=None,
#         work_dir=None,
#         logger=None,
#         meta=None,
#         keys=["gt_bboxes_3d", "gt_labels_3d", "img"],
#         max_iters=None,
#         max_epochs=None,
#     ):
#         super().__init__(
#             model,
#             batch_processor,
#             optimizer,
#             work_dir,
#             logger,
#             meta,
#             max_iters,
#             max_epochs,
#         )
#         super().__init__(runner=,dataloader=,max_epochs=max_epochs,val_begin=1, val_interval=,dynamic_intervals=)
#         keys.append("img_metas")
#         self.keys = keys
#         self.eval_model = eval_model
#         self.eval_model.eval()
#
#     def run_iter(self, idx, data_batch):
#         if self.batch_processor is not None:
#             assert False
#             # outputs = self.batch_processor(
#             #     self.model, data_batch, train_mode=train_mode, **kwargs)
#
#         num_samples = data_batch["img"].data[0].size(1)
#         data_list = []
#         prev_bev = None
#         for i in range(num_samples):
#             data = {}
#             for key in self.keys:
#                 if key not in ["img_metas", "img", "points"]:
#                     data[key] = data_batch[key]
#                 else:
#                     if key == "img":
#                         data["img"] = BaseDataElement(
#                             data=[data_batch["img"].data[0][:, i]],
#                             cpu_only=data_batch["img"].cpu_only,
#                             stack=True,
#                         )
#                     elif key == "img_metas":
#                         data["img_metas"] = BaseDataElement(
#                             data=[
#                                 [each[i] for each in data_batch["img_metas"].data[0]]
#                             ],
#                             cpu_only=data_batch["img_metas"].cpu_only,
#                         )
#                     else:
#                         assert False
#             data_list.append(data)
#         with torch.no_grad():
#             for i in range(num_samples - 1):
#                 if data_list[i]["img_metas"].data[0][0]["prev_bev_exists"]:
#                     data_list[i]["prev_bev"] = BaseDataElement(
#                         data=[prev_bev], cpu_only=False
#                     )
#                 prev_bev = self.eval_model.val_step(data_list[i], self.optimizer)
#         if data_list[-1]["img_metas"].data[0][0]["prev_bev_exists"]:
#             data_list[-1]["prev_bev"] = BaseDataElement(data=[prev_bev], cpu_only=False)
#         outputs = self.model.train_step(data_list[-1], self.optimizer)
#         if not isinstance(outputs, dict):
#             raise TypeError(
#                 '"batch_processor()" or "model.train_step()"'
#                 'and "model.val_step()" must return a dict'
#             )
#         # if "log_vars" in outputs:
#         #     self.log_buffer.update(outputs["log_vars"], outputs["num_samples"])
#         self.outputs = outputs
