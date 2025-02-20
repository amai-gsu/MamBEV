import pytest
import torch
from projects.MAMBEV.mambev.modules.hydra_blocks import DeformableHydraBlock
from test_utils import config_test, data, device  # type:ignore
from typing import Dict


@pytest.fixture
def base_cfg(device):
    return dict(
        d_model=256,
        pillar_points=4,
        deform_heads=1,
        d_state=8,
        d_conv=7,
        expand=1,
        headdim=256,
        ngroups=1,
        bias=False,
        conv_bias=True,
        use_post_norm=True,
        layer_idx=0,
        attn_idx=0,
        chunk_size=32,
        sequence_parallel=True,
        device=device,
        dtype=torch.bfloat16,
        feature_levels=[0],
        traversal_methods=("tl0cross",),
        collect_method="tbc",
        debug=True,
    )


@pytest.mark.parametrize(
    "class_name",
    [DeformableHydraBlock],
)
def test_RefPtsVis(class_name, base_cfg: Dict, data: Dict, device: torch.device):
    config_test(class_name, base_cfg, device, data)
