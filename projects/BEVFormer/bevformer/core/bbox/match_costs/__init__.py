from mmengine.registry import TASK_UTILS
from .match_cost import BBox3DL1Cost, SmoothL1Cost


def build_match_cost(cfg, default_args=None):
    return TASK_UTILS.build(cfg, default_args=default_args)


__all__ = ["build_match_cost", "BBox3DL1Cost", "SmoothL1Cost"]
