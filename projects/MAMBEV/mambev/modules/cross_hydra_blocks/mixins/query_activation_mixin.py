import torch
import torch.nn.functional as F

from .typing import HasSizeAttrs


class QueryActivationMixin(HasSizeAttrs):
    def query_act(self, qxBCdt: torch.Tensor):
        out = qxBCdt.clone()
        out[..., : -2 * self.d_dt] = F.silu(qxBCdt[..., : -2 * self.d_dt])
        return qxBCdt[..., : self.d_x], out


class QueryActivationMixinV2(HasSizeAttrs):
    def query_act(self, qxBCdt: torch.Tensor):
        out = qxBCdt.clone()
        out[..., : -2 * self.d_dt] = F.silu(qxBCdt[..., : -2 * self.d_dt])
        x_og = out[..., : self.d_x].clone()
        out[..., : self.d_x] = 0
        return x_og, out


class QueryActivationMixinV3(HasSizeAttrs):
    def query_act(self, qxBCdt: torch.Tensor):
        return F.silu(qxBCdt[..., : self.d_x]), qxBCdt
