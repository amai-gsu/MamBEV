import pytest
import torch
from projects.MAMBEV.mambev.modules.hydra_blocks.conv_gate_ablations import (
    PrePreHydraBlock,
    PrePostHydraBlock,
    PostPreHydraBlock,
    PostPostHydraBlock,
)
from test_utils import config_test, data, device  # type:ignore
from typing import Dict


@pytest.fixture
def base_cfg(device):
    return dict(
        d_model=256,
        d_state=8,
        d_conv=7,
        expand=1,
        headdim=256,
        ngroups=1,
        bias=False,
        conv_bias=True,
        use_post_norm=True,
        layer_idx=0,
        chunk_size=32,
        sequence_parallel=True,
        x_og_activation=False,
        average_slots=True,
        device=device,
        dtype=torch.bfloat16,
        feature_levels=[0],
        traversal_methods=("tl0cross",),
        collect_method="tbc",
    )


@pytest.mark.parametrize(
    "class_name",
    [PrePostHydraBlock, PrePreHydraBlock, PostPreHydraBlock, PostPostHydraBlock],
)
@pytest.mark.parametrize(
    "updates",
    [
        dict(),
        dict(
            traversal_methods=("tl0cross", "tl1cross"),
        ),
        dict(
            use_post_norm=False,
        ),
        dict(mem_eff_inference=True),
        dict(
            ref2int_convert_method="round",
            ref2int_offset=1,
        ),
        dict(
            x_og_activation=True,
        ),
        dict(
            v_zero_params={
                "z": True,
                "C": True,
            },
            q_zero_params={
                "B": True,
                "dt": True,
            },
        ),
    ],
)
def test_ConvGateBlocks(
    class_name, base_cfg: Dict, updates: Dict, data: Dict, device: torch.device
):
    base_cfg.update(updates)
    config_test(class_name, base_cfg, device, data)
