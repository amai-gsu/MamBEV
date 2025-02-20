from projects.MAMBEV.mambev.utils.debug_utils import RefPtsVis
from .typing import HasBEVAttrs, HasMiscAttrs


class DebugRefMixin(HasBEVAttrs, HasMiscAttrs):
    def _init_debug(self, debug_kwargs):
        default_debug_kwargs = dict(
            layer_idx=self.layer_idx,
            attn_idx=self.attn_idx,
            dest_dir=f"./ref_pts_vis/{self.__class__.__name__}/",
            ref2int=self.ref2int,
            active=self.debug,
        )
        default_debug_kwargs.update(debug_kwargs)

        self.debug_vis = RefPtsVis(**default_debug_kwargs)  # type:ignore
