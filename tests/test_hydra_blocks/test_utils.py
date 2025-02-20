from copy import deepcopy
from pickle import Unpickler
from typing import Union, TypeAlias, Callable, Dict, Optional
import pytest
from pathlib import Path
from random import randint

from einops import repeat
import torch
import io


MAP_LOCATION: TypeAlias = Optional[
    Union[
        Callable[[torch.Tensor, str], torch.Tensor], torch.device, str, Dict[str, str]
    ]
]


class TorchUnpickler(Unpickler):
    def __init__(self, file, device: MAP_LOCATION = "cpu"):
        super(TorchUnpickler, self).__init__(file)
        self.device = device

    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location=self.device)
        return super().find_class(module, name)


device_str = "cuda"


@pytest.fixture
def device():
    return torch.device(device_str)


here_path = Path(__file__)
input_path = here_path.parent.parent / "data" / "spatialxa_input1lvl.pkl"

with open(input_path, "rb") as f:
    unpickler = TorchUnpickler(f, device_str)
    torch_data = unpickler.load()


@pytest.fixture
def data():
    return deepcopy(torch_data)


def config_test(class_name, cfg, device, data):
    model = class_name(**cfg)
    model.to(device)
    model.init_weights()
    assert model._is_init
    output = model(**data)
    assert output.size() == data["query"].size()
    return model


def config_test_both(class_name, cfg, device, data):
    model = class_name(**cfg)
    model.to(device)
    model.init_weights()
    assert model._is_init
    query, value = model(**data)
    assert query.size() == data["query"].size()
    assert value.size() == data["value"].size()
    return model


def create_inputs(
    model: torch.nn.Module,
    bs: int,
    nc: int,
    z: int,
    bev_h: int,
    bev_w: int,
    H: int,
    W: int,
):
    inp_res = torch.zeros((bs, bev_h * bev_w, 1))
    qxBCdt = repeat(
        torch.arange(-1, -(bev_h * bev_w + 1), -1),
        "hw -> bs hw d",
        bs=bs,
        d=model.d_x + 2 * (model.d_B + model.d_C + model.d_dt),
    ).clone()
    vxBCdt = repeat(
        torch.arange(H * W),
        "hw -> bs nc hw d",
        bs=bs,
        nc=nc,
        d=model.d_inner + (4 * model.ngroups * model.d_state) + 2 * model.nheads,
    ).clone()
    reference_points_cam = torch.normal(
        torch.zeros([bs, nc, bev_h * bev_w, z, 2]) + 0.5, 0.75
    )
    bev_mask = (
        (reference_points_cam[..., 1:2] > 0.0)
        & (reference_points_cam[..., 1:2] < 1.0)
        & (reference_points_cam[..., 0:1] < 1.0)
        & (reference_points_cam[..., 0:1] > 0.0)
    ).squeeze(-1)
    ref_int = model.ref2int(reference_points_cam[bev_mask], H, W, False)  # type:ignore

    spatial_shapes = torch.tensor(((H, W),))
    return (
        inp_res,
        qxBCdt,
        vxBCdt,
        reference_points_cam,
        bev_mask,
        ref_int,
        spatial_shapes,
    )


def traversal():
    tb = ["t", "b"]
    lr = ["l", "r"]
    rot = ["0", "1"]
    method = ["snake", "cross"]

    idx = torch.randint(0, 2, (4,))

    return tb[idx[0]] + lr[idx[1]] + rot[idx[2]] + method[idx[3]]


@pytest.fixture
def traversals():
    travs = []
    for _ in range(randint(1, 4)):
        temp = []
        for _ in range(randint(1, 2)):
            temp.append(traversal())
        if len(temp) == 1:
            temp = temp[0]
        else:
            temp = tuple(temp)
        travs.append(temp)
    return travs
