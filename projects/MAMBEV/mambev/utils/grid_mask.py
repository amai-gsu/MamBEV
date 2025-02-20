import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms.functional as Fv


class GridMask(nn.Module):
    def __init__(
        self,
        use_h: bool,
        use_w: bool,
        rotate: int = 1,
        offset: bool = False,
        ratio: float = 0.5,
        mode: int = 0,
        prob: float = 1.0,
    ):
        super(GridMask, self).__init__()
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.prob = prob

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * epoch / max_epoch  # + 1.#0.5

    def forward(self, x):
        if torch.rand(1) > self.prob or not self.training:
            return x
        n, c, h, w = x.size()
        x = x.view(-1, h, w)
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        d = int(torch.randint(low=2, high=h, size=(1,)).item())
        l = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        mask = torch.ones((hh, ww), dtype=torch.float32, device=x.device)
        st_h = torch.randint(high=d, size=(1,)).item()
        st_w = torch.randint(high=d, size=(1,)).item()
        if self.use_h:
            for i in range(hh // d):
                s = d * i + st_h
                t = min(s + l, hh)
                mask[s:t, :] *= 0
        if self.use_w:
            for i in range(ww // d):
                s = d * i + st_w
                t = min(s + l, ww)
                mask[:, s:t] *= 0

        # this always returns 0 when rotate is one
        r = int(torch.randint(high=self.rotate, size=(1,)).item())
        if r != 0:
            mask = Fv.rotate(mask.unsqueeze(0), r).squeeze(0)

        mask = mask[
            (hh - h) // 2 : (hh - h) // 2 + h, (ww - w) // 2 : (ww - w) // 2 + w
        ]

        if self.mode == 1:
            mask = 1 - mask
        mask = mask.expand_as(x)
        if self.offset:
            offset = torch.from_numpy(2 * (np.random.rand(h, w) - 0.5)).to(
                device=x.device, dtype=x.dtype
            )
            x = x * mask + offset * (1 - mask)
        else:
            x = x * mask

        return x.view(n, c, h, w)
