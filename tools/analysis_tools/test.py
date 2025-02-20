from mmengine.registry import MODELS
import sys
sys.path.append('/home/hongyu/ICML_BEV/projects')
from test_io import wrap_forward_methods
import mmdet3d_plugin.mambev
import mmdet3d

import pickle
import torch

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# import projects.mmdet3d_plugin.mambev

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]
_dim_ = 256
_pos_dim_ = _dim_ // 2
_ffn_dim_ = _dim_ * 2
_num_levels_ = 4
bev_h_ = 200
bev_w_ = 200
queue_length = 4
group_detr = 11

# custom_imports = dict(imports=["projects.mmdet3d_plugin.mambev"], allow_failed_imports=False)

cfg_encoder = dict(type="BEVFormerEncoderV3",
                num_layers=6,
                pc_range=point_cloud_range,
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type="BEVFormerLayerV3",
                    attn_cfgs=[
                        dict(
                            type="TemporalSelfAttention", embed_dims=_dim_, num_levels=1
                        ),
                        dict(
                            type="SpatialCrossAttention",
                            pc_range=point_cloud_range,
                            deformable_attention=dict(
                                type="MSDeformableAttention3D",
                                embed_dims=_dim_,
                                num_points=8,
                                num_levels=_num_levels_,
                            ),
                            embed_dims=_dim_,
                        ),
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=(
                        "self_attn",
                        "norm",
                        "cross_attn",
                        "norm",
                        "ffn",
                        "norm",
                    ),
                ),
)

cfg_decoder = dict(
                type="DetectionTransformerDecoderV3",
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type="mmdet.DetrTransformerDecoderLayer",
                    attn_cfgs=[
                        dict(
                            type="MultiheadAttention",
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1,
                        ),
                        dict(
                            type="CustomMSDeformableAttentionV1",
                            embed_dims=_dim_,
                            num_levels=1,
                        ),
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=(
                        "self_attn",
                        "norm",
                        "cross_attn",
                        "norm",
                        "ffn",
                        "norm",
                    ),
                ),
            )

cfg_transformer = dict(
            type="PerceptionTransformerV3",
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            embed_dims=_dim_,
            encoder=dict(type="BEVFormerEncoderV3",
                num_layers=6,
                pc_range=point_cloud_range,
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type="BEVFormerLayerV3",
                    attn_cfgs=[
                        dict(
                            type="TemporalSelfAttention", embed_dims=_dim_, num_levels=1
                        ),
                        dict(
                            type="SpatialCrossAttention",
                            pc_range=point_cloud_range,
                            deformable_attention=dict(
                                type="MSDeformableAttention3D",
                                embed_dims=_dim_,
                                num_points=8,
                                num_levels=_num_levels_,
                            ),
                            embed_dims=_dim_,
                        ),
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=(
                        "self_attn",
                        "norm",
                        "cross_attn",
                        "norm",
                        "ffn",
                        "norm",
                    ),
                ),
            ),
            decoder=dict(
                type="DetectionTransformerDecoderV3",
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type="mmdet.DetrTransformerDecoderLayer",
                    attn_cfgs=[
                        dict(
                            type="MultiheadAttention",
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1,
                        ),
                        dict(
                            type="CustomMSDeformableAttentionV1",
                            embed_dims=_dim_,
                            num_levels=1,
                        ),
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=(
                        "self_attn",
                        "norm",
                        "cross_attn",
                        "norm",
                        "ffn",
                        "norm",
                    ),
                ),
            ),
            # pre_norm=False,
        )

cfg_bbox_head=dict(
        type="BEVFormerHead_GroupDETRV3",
        group_detr=group_detr,
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_query=900,
        num_classes=10,
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        train_cfg=dict(
            pts=dict(
                grid_size=[512, 512, 1],
                voxel_size=voxel_size,
                point_cloud_range=point_cloud_range,
                out_size_factor=4,
                pc_range=point_cloud_range,
            ),
            assigner=dict(
                type="mmdet.HungarianAssigner3D",
                cls_cost=dict(type="mmdet.FocalLossCost", weight=2.0),
                reg_cost=dict(type="mmdet3d.BBox3DL1Cost", weight=0.25),
                # Fake cost. This is just to make it compatible with DETR head.
                iou_cost=dict(type="mmdet.IoUCost", weight=0.0),
            ),
        ),
        transformer=dict(
            type="PerceptionTransformerV3",
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            embed_dims=_dim_,
            encoder=dict(type="BEVFormerEncoderV3",
                num_layers=6,
                pc_range=point_cloud_range,
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type="BEVFormerLayerV3",
                    attn_cfgs=[
                        dict(
                            type="TemporalSelfAttention", embed_dims=_dim_, num_levels=1
                        ),
                        dict(
                            type="SpatialCrossAttention",
                            pc_range=point_cloud_range,
                            deformable_attention=dict(
                                type="MSDeformableAttention3D",
                                embed_dims=_dim_,
                                num_points=8,
                                num_levels=_num_levels_,
                            ),
                            embed_dims=_dim_,
                        ),
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=(
                        "self_attn",
                        "norm",
                        "cross_attn",
                        "norm",
                        "ffn",
                        "norm",
                    ),
                ),
            ),
            decoder=dict(
                type="DetectionTransformerDecoderV3",
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type="mmdet.DetrTransformerDecoderLayer",
                    attn_cfgs=[
                        dict(
                            type="MultiheadAttention",
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1,
                        ),
                        dict(
                            type="CustomMSDeformableAttentionV1",
                            embed_dims=_dim_,
                            num_levels=1,
                        ),
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=(
                        "self_attn",
                        "norm",
                        "cross_attn",
                        "norm",
                        "ffn",
                        "norm",
                    ),
                ),
            ),
            # pre_norm=False,
        ),
        bbox_coder=dict(
            type="mmdet3d.NMSFreeCoder",
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=10,
        ),
        positional_encoding=dict(
            type="mmdet.LearnedPositionalEncoding",
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
        ),
        loss_cls=dict(
            type="mmdet.FocalLoss",
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0,
        ),
        loss_bbox=dict(type="mmdet.L1Loss", loss_weight=0.25),
        loss_iou=dict(type="mmdet.GIoULoss", loss_weight=0.0),
    )
   

def run_test(cfg, input_path):
    model = MODELS.build(cfg)
    model.eval()
    # print(model)

    with open(input_path, 'rb') as file:
        data = pickle.load(file)
    if cfg['type'] == 'BEVFormerEncoderV3':
        bev_h = data[1]['bev_h']
        bev_w = data[1]['bev_w']
        bev_pos = data[1]['bev_pos']
        spatial_shapes = data[1]['spatial_shapes'].to(device)
        level_start_index = data[1]['level_start_index'].to(device)
        prev_bev = data[1]['prev_bev']
        shift = data[1]['shift'].to(device)
        img_metas=data[1]['img_metas']

        wrap_forward_methods(model, cfg, "/home/hongyu/ICML_BEV/data/io/bevformer_base", True, True, 1)
        model.forward(bev_query=data[0][0].to(device), key=data[0][1].to(device), value=data[0][2].to(device), bev_h=bev_h,bev_w=bev_w,bev_pos=bev_pos,spatial_shapes=spatial_shapes,level_start_index=level_start_index,prev_bev=prev_bev,shift=shift,img_metas=img_metas)

    if cfg['type'] == 'DetectionTransformerDecoderV3':
        # print(len(data[0]))
        # print(data[1])
        query = data[1]['query']
        key = data[1]['key']
        value = data[1]['value']
        query_pos = data[1]['query_pos']
        reference_points = data[1]['reference_points']
        reg_branches = data[1]['reg_branches']
        cls_branches = data[1]['cls_branches']
        spatial_shapes = data[1]['spatial_shapes']
        level_start_index = data[1]['level_start_index']
        img_metas = data[1]['img_metas']

        wrap_forward_methods(model, cfg, "/home/hongyu/ICML_BEV/data/io/bevformer_base", True, True, 1)
        model.forward(query=query, key=key,value=value,query_pos=query_pos,reference_points=reference_points,reg_branches=reg_branches,cls_branches=cls_branches,spatial_shapes=spatial_shapes,level_start_index=level_start_index,img_metas=img_metas)

    if cfg['type'] == 'PerceptionTransformerV3':
        img_metas = data[1]['img_metas']
        bev_pos=data[1]['bev_pos']
        reg_branches=data[1]['reg_branches']
        cls_branches=None
        prev_bev=None
        grid_length=[0.512, 0.512]

        wrap_forward_methods(model, cfg, "/home/hongyu/ICML_BEV/data/io/bevformerv2-r50-t1-24ep", False, True, 1)
        model.forward(mlvl_feats=data[0][0],bev_queries = data[0][1], object_query_embed=data[0][2],bev_h=data[0][3], bev_w=data[0][4], img_metas=img_metas, grid_length=grid_length,bev_pos=bev_pos,reg_branches=reg_branches,cls_branches=cls_branches,prev_bev=prev_bev)
        # model.forward(*data[0], **data[1])
        
    if cfg['type'] == 'BEVFormerHead_GroupDETRV3':
        wrap_forward_methods(model, cfg, "/home/hongyu/ICML_BEV/data/io/bevformerv2-r50-t1-24ep", False, True, 1)
        model.forward(*data[0], **data[1])

decoder_forward_path = "/home/hongyu/ICML_BEV/data/io/bevformer_base/DetectionTransformerDecoder_forward_input.pkl"
perception_transformer = '/home/hongyu/ICML_BEV/data/io/bevformer_base/PerceptionTransformer_forward_input.pkl'
perception_transformerV2 = '/home/hongyu/ICML_BEV/data/io/bevformerv2-r50-t1-24ep/PerceptionTransformerV2_forward_input.pkl'
BEVFormerHead_GroupDETRV3 = '/home/hongyu/ICML_BEV/data/io/bevformerv2-r50-t1-24ep/BEVFormerHead_GroupDETR_forward_input.pkl'
run_test(cfg_bbox_head, BEVFormerHead_GroupDETRV3)