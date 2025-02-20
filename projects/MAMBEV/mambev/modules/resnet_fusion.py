from einops import pack, rearrange
from mmengine.runner import autocast
from torch import Tensor
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmdet.models.backbones.resnet import BasicBlock
from mmengine.model import BaseModule


class ResNetFusionV3(BaseModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        inter_channels: int,
        num_layer: int,
        norm_cfg: dict = dict(type="SyncBN"),
        with_cp: bool = False,
    ):
        super(ResNetFusionV3, self).__init__()
        layers: list[BasicBlock] = []
        self.inter_channels = inter_channels
        for i in range(num_layer):
            if i == 0:
                if inter_channels == in_channels:
                    layers.append(
                        BasicBlock(
                            in_channels, inter_channels, stride=1, norm_cfg=norm_cfg
                        )
                    )
                else:
                    downsample = nn.Sequential(
                        build_conv_layer(
                            None,
                            in_channels,
                            inter_channels,
                            3,
                            stride=1,
                            padding=1,
                            dilation=1,
                            bias=False,
                        ),
                        build_norm_layer(norm_cfg, inter_channels)[1],
                    )
                    layers.append(
                        BasicBlock(
                            in_channels,
                            inter_channels,
                            stride=1,
                            norm_cfg=norm_cfg,
                            downsample=downsample,
                        )
                    )
            else:
                layers.append(
                    BasicBlock(
                        inter_channels, inter_channels, stride=1, norm_cfg=norm_cfg
                    )
                )
        self.layers = nn.Sequential(*layers)
        self.layer_norm = nn.Sequential(
            nn.Linear(inter_channels, out_channels), nn.LayerNorm(out_channels)
        )
        self.with_cp = with_cp

    # @autocast(dtype=torch.float32)
    # @autocast(dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    @autocast()
    def forward(self, input_tup: tuple[Tensor, ...]):
        x, _ = pack(input_tup, "batch_size * bev_h bev_w")
        # x should be [1, in_channels, bev_h, bev_w]
        for lid, layer in enumerate(self.layers):
            assert x is not None, "Layer output null values in ResNet Fusion"
            if self.with_cp and x.requires_grad:
                x = checkpoint.checkpoint(layer, x)
            else:
                x = layer(x)
        x = rearrange(x, "n c h w -> n (h w) c")
        # x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # nchw -> n(hw)c
        x = self.layer_norm(x)
        return x
