from pathlib import Path
from mmengine.config import Config
from mmengine.registry import MODELS
from mmengine.runner import Runner
import torch
from projects.BEVFormer.bevformer.bevformer.modules import SpatialCrossAttention

# file_path = "./ckpts/bevformer_v4.pth"
# model = torch.load(file_path, map_location="cpu")
# all = 0
# for key in list(model["state_dict"].keys()):
#     all += model["state_dict"][key].nelement()
# print(all)

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
_points_in_pillar_ = 4
_img_feat_levels_ = 1
_dim_ = 256
cfg = dict(type="MultiheadAttention", embed_dims=_dim_, num_heads=8, attn_drop=0.1)
xcfg = dict(
    type="SpatialCrossAttention",
    pc_range=point_cloud_range,
    embed_dims=_dim_,
    batch_first=True,
    deformable_attention=dict(
        type="MSDeformableAttention3D",
        embed_dims=_dim_,
        num_points=_points_in_pillar_,
        num_levels=_img_feat_levels_,
        im2col_step=96,
    ),
)

model = MODELS.build(cfg)  # type:ignore
params = sum(p.numel() for p in model.parameters())
print(params)


def count_params(file_path):
    model = torch.load(file_path, map_location="cpu")
    print(file_path)
    _count_model_params(model)


def _count_model_params(model):
    all = 0
    for key in list(model["state_dict"].keys()):
        all += model["state_dict"][key].nelement()
    print(f"\t:{all:,}")


# smaller 63374123
# v4 69140395
if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("files", type=str, nargs="+")
    parser.add_argument("--filetype", type=str, nargs=1, default="weights")
    args = parser.parse_args()
    if args.filetype == "weights":
        for file in args.files:
            file = Path(file)
            count_params(file)
    else:
        for file in args.files:
            file = Path(file)
            print(file)
            cfg = Config.fromfile(file)
            cfg.train_dataset["indices"] = 1
            cfg.train_cfg["total_epochs"] = 1

            work_dir
            runner = Runner.from_cfg(cfg)
            runner.train()

            _count_model_params(runner.model)
