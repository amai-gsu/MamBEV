from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from typing import Optional, Union

DATA_BATCH = Optional[Union[dict, tuple, list]]


@HOOKS.register_module()
class TransferWeight(Hook):
    def __init__(self, every_n_inters=1):
        self.every_n_inters = every_n_inters

    def after_train_iter(
        self,
        runner,
        batch_idx: int,
        data_batch: DATA_BATCH = None,
        outputs: Optional[dict] = None,
    ) -> None:
        """

        Args:
            runner (Runner): The runner of the training process.
            batch_idx (int): The index of the current batch in the train loop.
            data_batch (dict tuple or list, optional): Data from dataloader.
            outputs (dict, optional): Outputs from model.
        """
        if self.every_n_inner_iters(runner, self.every_n_inters):
            runner.eval_model.load_state_dict(runner.model.state_dict())
