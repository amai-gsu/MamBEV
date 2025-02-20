# _base_ = ["../datasets/custom_nus-3d.py", "../_base_/default_runtime.py"]


# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]


custom_imports = dict(
    imports=[
        "projects.BEVFormer.bevformer",
        "projects.DETR3D.detr3d",
        "projects.MAMBEV.mambev",
    ],
    allow_failed_imports=False,
)
img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
# For nuScenes we usually do 10-class detection
class_names = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]

input_modality = dict(
    use_lidar=False, use_camera=True, use_radar=False, use_map=False, use_external=True
)

_dim_ = 256
_pos_dim_ = _dim_ // 2
_ffn_dim_ = _dim_ * 2
_num_levels_ = 4
bev_h_ = 200
bev_w_ = 200
queue_length = 4  # each sequence contains `queue_length` frames.

model = dict(
    type="BEVFormer",
    use_grid_mask=True,
    video_test_mode=True,
    img_backbone=dict(
        type="mmdet.ResNet",
        depth=101,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type="BN2d", requires_grad=False),
        norm_eval=True,
        style="caffe",
        dcn=dict(
            type="DCNv2", deform_groups=1, fallback_on_stride=False
        ),  # original DCNv2 will print log when perform load_state_dict
        stage_with_dcn=(False, False, True, True),
    ),
    img_neck=dict(
        type="mmdet.FPN",
        in_channels=[512, 1024, 2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs="on_output",
        num_outs=4,
        relu_before_extra_convs=True,
    ),
    pts_bbox_head=dict(
        type="BEVFormerHead",
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
            type="PerceptionTransformer",
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            embed_dims=_dim_,
            encoder=dict(
                type="BEVFormerEncoder",
                num_layers=6,
                pc_range=point_cloud_range,
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type="BEVFormerLayer",
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
                type="mmdet.DetectionTransformerDecoder",
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
                            type="CustomMSDeformableAttention",
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
    ),
    # # model training and testing settings
)
total_epochs = 24
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=total_epochs, val_interval=2)
# train_cfg = dict(type="TrainLoop", max_epochs=100)
val_cfg = dict(type="ValLoop")

# dataset_type = "CustomNuScenesDataset"
dataset_type = "mmdet3d.NuScenesDataset"
data_root = "data/nuscenes/"
file_client_args = dict(backend="disk")


train_pipeline = [
    dict(type="mmdet3d.LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="mmdet.PhotoMetricDistortionMultiViewImage"),
    dict(
        type="mmdet3d.LoadAnnotations3D",
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False,
    ),
    dict(type="mmdet3d.ObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="mmdet3d.ObjectNameFilter", classes=class_names),
    dict(type="mmdet.NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="mmdet.PadMultiViewImage", size_divisor=32),
    dict(
        type="mmdet3d.Pack3DDetInputs",
        # class_names=class_names,
        keys=["gt_bboxes_3d", "gt_labels_3d", "img"],
    ),
    # dict(type="CustomCollect3D", keys=["gt_bboxes_3d", "gt_labels_3d", "img"]),
]

test_pipeline = [
    dict(type="mmdet3d.LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="mmdet.NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="mmdetPadMultiViewImage", size_divisor=32),
    dict(
        type="MultiScaleFlipAug3D",
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type="mmdet3d.Pack3DDetInputs",
                # class_names=class_names,
                # with_labels=False,
                keys=["img"],
            ),
        ],
    ),
]

metainfo = dict(classes=class_names)
backend_args = None
data_prefix = dict(
    pts="",
    CAM_FRONT="samples/CAM_FRONT",
    CAM_FRONT_LEFT="samples/CAM_FRONT_LEFT",
    CAM_FRONT_RIGHT="samples/CAM_FRONT_RIGHT",
    CAM_BACK="samples/CAM_BACK",
    CAM_BACK_RIGHT="samples/CAM_BACK_RIGHT",
    CAM_BACK_LEFT="samples/CAM_BACK_LEFT",
)
train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="nuscenes_infos_train.pkl",
        # ann_file="nuscenes_infos_temporal_train_v2.pkl",
        pipeline=train_pipeline,
        load_type="frame_based",
        metainfo=metainfo,
        modality=input_modality,
        test_mode=False,
        data_prefix=data_prefix,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d="LiDAR",
        backend_args=backend_args,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="nuscenes_infos_val.pkl",
        # ann_file="nuscenes_infos_temporal_val_v2.pkl",
        pipeline=train_pipeline,
        load_type="frame_based",
        metainfo=metainfo,
        modality=input_modality,
        test_mode=False,
        data_prefix=data_prefix,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d="LiDAR",
        backend_args=backend_args,
    ),
)
test_dataloader = val_dataloader
#     shuffler_sampler=dict(type="DistributedGroupSampler"),
#     nonshuffler_sampler=dict(type="DistributedSampler"),
val_evaluator = dict(
    type="mmdet3d.NuScenesMetric",
    data_root=data_root,
    ann_file="infos_val.pkl",
    metric="bbox",
    backend_args=None,
)
test_cfg = dict(type="TestLoop")
test_evaluator = val_evaluator
optimizer = dict(
    type="AdamW",
    lr=2e-4,
    weight_decay=0.01,
)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=optimizer,
    paramwise_cfg=dict(
        custom_keys={
            "img_backbone": dict(lr_mult=0.1),
        }
    ),
)
# learning policy
lr_config = dict(
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)
evaluation = dict(interval=1, pipeline=test_pipeline)

# runner = dict(
#     type="EpochBasedRunner",
#     max_epochs=total_epochs,
# )
runner = dict(type="EpochBasedRunner")
# TODO: Find out why this isnt working
load_from = "ckpts/r101_dcn_fcos3d_pretrain.pth"
log_config = dict(
    interval=50, hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")]
)

checkpoint_config = dict(interval=1)
