import torch
from einops import rearrange, repeat
from torch import nn
from torch.nn import functional as F

from .typing import HasConvAttrs, HasDistributedAttrs, HasSizeAttrs, Hasdt


class VConvMixin(HasDistributedAttrs, HasConvAttrs, HasSizeAttrs, Hasdt):
    def _init_conv(self, conv_bias: bool, d_conv: int):
        # convolve over V_xB for all traversals
        conv_dim = int(self.v_conv_mask.sum().item())
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv // 2,
            **self.factory_kwargs,
        )

    def _vconv(self, vxBCdt: torch.Tensor):
        vxBCdt = rearrange(vxBCdt, "bs nc nv c -> bs (nc nv) c")
        vxBCdt[..., self.v_conv_mask] = self.act(
            self.conv1d(vxBCdt[..., self.v_conv_mask].transpose(1, 2)).transpose(1, 2)
        )  # (B, L, self.d_inner + 2 * (2 * ngroups * d_state))

        vxBCdt[..., -2 * self.d_dt :] = F.softplus(
            vxBCdt[..., -2 * self.d_dt :] + repeat(self.dt_bias, "nh -> (2 nh)")
        )  # (2 * B, L, nheads)
        return vxBCdt


class SplitConvMixin(HasDistributedAttrs, HasConvAttrs, HasSizeAttrs, Hasdt):
    def _init_conv(self, conv_bias: bool, d_conv: int):
        # convolve over V_xB for all traversals
        self.conv1d_x = nn.Conv1d(
            in_channels=self.d_x,
            out_channels=self.d_x,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_x,
            padding=d_conv // 2,
            **self.factory_kwargs,
        )
        self.conv1d_B = nn.Conv1d(
            in_channels=2 * self.d_B,
            out_channels=2 * self.d_B,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=2 * self.d_B,
            padding=d_conv // 2,
            **self.factory_kwargs,
        )

        # if self.conv_init is not None:
        #     nn.init.uniform_(self.conv1d_x.weight, -self.conv_init, self.conv_init)
        #     nn.init.uniform_(self.conv1d_B.weight, -self.conv_init, self.conv_init)

    def _split_conv(self, vx: torch.Tensor, vBCdt: torch.Tensor):
        vx = rearrange(vx, "bs nc nv c -> bs (nc nv) c")
        vBCdt = rearrange(vBCdt, "bs nc nv c -> bs (nc nv) c")
        vx = self.act(
            self.conv1d_x(vx.transpose(1, 2)).transpose(1, 2)
        )  # (B, L, self.d_inner + 2 * (2 * ngroups * d_state))

        vBCdt[..., self.v_conv_mask[..., self.d_x :]] = self.act(
            self.conv1d_B(
                vBCdt[..., self.v_conv_mask[..., self.d_x :]].transpose(1, 2)
            ).transpose(1, 2)
        )  # (B, L, self.d_inner + 2 * (2 * ngroups * d_state))

        vBCdt[..., -2 * self.d_dt :] = F.softplus(
            vBCdt[..., -2 * self.d_dt :] + repeat(self.dt_bias, "nh -> (2 nh)")
        )  # (2 * B, L, nheads)

        return vx, vBCdt


class ConvMixin(HasDistributedAttrs, HasConvAttrs, HasSizeAttrs):
    def _init_conv(self, conv_bias: bool, d_conv: int):
        conv_dim = self.d_x + 2 * (self.d_B + self.d_C)
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            # padding=d_conv - 1,
            padding=d_conv // 2,
            **self.factory_kwargs,
        )
        # if self.conv_init is not None:
        #     nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)

    def _conv(self, xBCdt: torch.Tensor):
        xBCdt[..., self.conv_mask] = self.act(
            self.conv1d(xBCdt[..., self.conv_mask].transpose(1, 2)).transpose(1, 2)
        )  # (B, L, self.d_inner + 2 * (2 * ngroups * d_state))
        return xBCdt
