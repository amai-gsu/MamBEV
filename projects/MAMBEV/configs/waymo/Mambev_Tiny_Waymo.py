default_scope = "mmdet3d"


custom_imports = dict(
    imports=[
        "projects.MAMBEV.mambev",
        "projects.PETR.petr",
        "projects.mmcv_dep",
        "projects.DETR3D.detr3d",
        "projects.DETR_dep.detr",
        "projects.BEVFormer.bevformer",
    ],
    allow_failed_imports=False,
)
img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
# BEVFormerV1 Backbone setting
# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
#
# For nuScenes we usually do 10-class detection
data_root = "data/waymo/kitti_format/"
class_names = [
    "Pedestrian",
    "Cyclist",
    "Car",
]
num_classes = len(class_names)
data_prefix = dict(
    pts="training/velodyne",
    CAM_FRONT="training/image_0",
    CAM_FRONT_LEFT="training/image_1",
    CAM_FRONT_RIGHT="training/image_2",
    CAM_SIDE_LEFT="training/image_3",
    CAM_SIDE_RIGHT="training/image_4",
)


input_modality = dict(use_lidar=False, use_camera=True)

# Size of BEV Query Map
bev_h_ = 50
bev_w_ = 50
# dependent on size of bev map
voxel_size = [110 / bev_h_, 150 / bev_w_, 6]

group_detr = 4
# if point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-35.0, -75.0, -2, 75.0, 75.0, 4]
_dim_ = 256
_pos_dim_ = _dim_ // 2
_ffn_dim_ = _dim_ * 2
_points_in_pillar_ = 4
out_indices = [3]
in_channels = [2048]
assert (
    len(out_indices) == len(in_channels)
), "Number of outputs from Img Backbone do not match the number of Inputs expected by the Img Neck"
_img_feat_levels_ = 1
_mono_img_feat_levels_ = 0
_img_neck_outs_ = max(_img_feat_levels_, _mono_img_feat_levels_)
_encoder_layers_ = 2
_decoder_layers_ = 6
traversal_methods = ["tl0snake"]
ntraversals = len(traversal_methods)
frames = [-1, 0]
total_epochs = 12
batch_size = 4
val_batch_size = 1
num_gpus = 2
iter_mult = 32 / (batch_size * num_gpus)
grad_accum = int(iter_mult) if 32 % (batch_size * num_gpus) == 0 else int(iter_mult) + 1
total_steps = 28130 * total_epochs

# warmup for 10% of total steps
warmup_end = total_steps // (10 * batch_size * num_gpus)

hydra_cfg = dict(
    type="PostPreHydraBlock",
    d_model=_dim_,
    d_state=32,
    d_conv=7,
    conv_init=None,
    expand=1,
    headdim=16,
    d_ssm=None,  # If not None, we only apply SSM on this many dimensions, the rest uses gated MLP
    ngroups=1,
    A_init_range=(1, 16),
    D_has_hdim=False,
    rmsnorm=True,
    norm_before_gate=False,
    dt_min=0.001,
    dt_max=0.1,
    dt_init_floor=1e-4,
    dt_limit=(0.0, float("inf")),
    bias=False,
    conv_bias=True,
    # Fused kernel and sharding options
    chunk_size=512,
    sequence_parallel=True,
    # MY opts
    traversal_methods=traversal_methods,
    average_slots=True,
    x_og_activation=False,
    dt_bias=True,  # only for values
    use_post_norm=True,
    q_zero_params={
        "B": True,
        "dt": True,
    },
    v_zero_params={
        "z": True,
        "C": True,
    },
    feature_levels=[0],
    dropout=0.1,
)

self_traversals = ["tl0snake"]
expand = 4

hydra_self_attn = dict(
    type="TraversalSelfAttnBlock",
    embed_dims=_dim_,
    bev_h=bev_h_,
    bev_w=bev_w_,
    traversal_methods=self_traversals,
    # patch_size = (4,4),
    proj_drop=0.05,
    mamba_cfg=dict(
        type="Hydra",
        d_model=_dim_,
        d_state=128,
        d_conv=7,
        expand=expand,
        headdim=128,
        chunk_size=256,
        ngroups=1,
    ),
)
deform_xa_cfg = dict(
    type="SpatialCrossAttention",
    pc_range=point_cloud_range,
    embed_dims=_dim_,
    batch_first=True,
    num_cams=len(data_prefix) - 1,
    deformable_attention=dict(
        type="MSDeformableAttention3D",
        embed_dims=_dim_,
        num_points=_points_in_pillar_,
        num_levels=_img_feat_levels_,
        im2col_step=96,
    ),
)


################# Model Hyperparameters #####################
model_wrapper_cfg = dict(
    type="MMDistributedDataParallel",
    # detect_anomalous_params=True,
    find_unused_parameters=True,
    # device_ids=device_ids
)
model = dict(
    type="MamBEV",
    embed_dim=_dim_,
    use_grid_mask=True,
    with_bptt=False,
    num_mono_levels=_mono_img_feat_levels_,
    num_levels=_img_feat_levels_,
    frames=frames,
    img_backbone=dict(
        type="mmdet.ResNet",
        depth=50,
        num_stages=4,
        out_indices=out_indices,
        frozen_stages=1,
        norm_cfg=dict(type="SyncBN"),
        norm_eval=False,
        style="caffe",
        init_cfg=dict(
            type="Pretrained",
            checkpoint="ckpts/fcos_r50_coco_2mmdet.pth",
            prefix="img_backbone",
        ),
    ),
    img_neck=dict(
        type="CPFPN",
        in_channels=in_channels,
        out_channels=_dim_,
        start_level=0,
        add_extra_convs="on_output",
        num_outs=_img_neck_outs_,
        relu_before_extra_convs=True,
    ),
    pts_bbox_head=dict(
        type="BEVFormerHead_GroupDETRV3",
        depr_bbox_format=True,
        group_detr=group_detr,
        encoder_self_attn=True,
        code_weights=[
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ],
        code_size=8,
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_query=900,
        num_classes=num_classes,
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        transformer=dict(
            type="PerceptionTransformerV3",
            embed_dims=_dim_,
            num_cams=len(data_prefix) - 1,
            num_fusion=3,
            num_feature_levels=_img_feat_levels_,
            frames=frames,
            use_cams_embeds=True,
            encoder=dict(
                type="BEVFormerEncoderV3",
                num_layers=_encoder_layers_,
                pc_range=point_cloud_range,
                num_points_in_pillar=_points_in_pillar_,
                return_intermediate=False,
                transformerlayers=dict(
                    type="BEVFormerLayerV3",
                    attn_cfgs=[
                        hydra_cfg,
                        deform_xa_cfg,
                        hydra_self_attn,
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=[
                        "cross_attn",
                        "cross_attn",
                        "norm",
                        "mamba_self_attn",
                    ],
                    batch_first=True,
                ),
            ),
            decoder=dict(
                type="DetectionTransformerDecoder",
                num_layers=_decoder_layers_,
                return_intermediate=True,
                transformerlayers=dict(
                    type="GroupDeformableDetrTransformerDecoderLayer",
                    self_attn_cfg=dict(
                        group=group_detr,
                        embed_dims=_dim_,
                        num_heads=8,
                        dropout=0.1,
                        batch_first=True,
                    ),
                    cross_attn_cfg=dict(
                        embed_dims=_dim_,
                        num_heads=8,
                        num_levels=1,
                        num_points=4,
                        im2col_step=64,
                        dropout=0.1,
                        batch_first=True,
                        value_proj_ratio=1.0,
                    ),
                    ffn_cfg=dict(
                        embed_dims=_dim_,
                        feedforward_channels=_ffn_dim_,
                        num_fcs=2,
                        ffn_drop=0.1,
                        act_cfg=dict(type="ReLU", inplace=True),
                    ),
                ),
            ),
        ),
        bbox_coder=dict(
            type="mmdet3d.NMSFreeCoder",
            post_center_range=[-45.0, -85.0, -10.0, 85.0, 85.0, 10.0],
            # post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=num_classes,
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
        loss_bbox=dict(type="mmdet.SmoothL1Loss", loss_weight=0.75, beta=1.0),
        loss_iou=dict(type="mmdet.GIoULoss", loss_weight=0.0),
    ),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=4,
            assigner=dict(
                type="mmdet3d.HungarianAssigner3D",
                cls_cost=dict(type="mmdet.FocalLossCost", weight=2.0),
                reg_cost=dict(type="mmdet3d.SmoothL1Cost", weight=0.75),
                pc_range=point_cloud_range,
                # Fake cost. This is just to make it compatible with DETR head.
                iou_cost=dict(type="mmdet.IoUCost", weight=0.0),
            ),
        ),
    ),
)

#############################################################
##################### Data Settings #########################
randomness = dict(seed=0)
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=total_epochs, val_interval=12)
val_cfg = dict(type="ValLoop")


dataset_type = "WaymoDatasetMultiFrame"
file_client_args = dict(backend="local")

meta_keys = (
    # rot scale trans mats
    "ego2global",
    "lidar2ego",
    "lidar2img",
    "lidar2cam",
    "cam2img",
    "context_name",
    "axis_align_matrix",
    # mono data
    "mono_input_dict",
    # indices
    "mono_ann_idx",
    "sample_idx",
    "frame_idx",
    "scene_token",
    # augmentation
    "aug_param",
    "box_mode_3d",
    "box_type_3d",
    "pad_shape",
    "ori_shape",
    "img_shape",
    "crop_offset",
    "img_crop_offset",
    "img_norm_cfg",
    "resize_img_shape",
    "scale_factor",
    # other
    "filename",
    "num_pts_feats",
    "pts_filename",
    "timestamp",
)
train_pipeline = [
    dict(type="mmdet3d.LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(
        type="mmdet3d.LoadAnnotations3D",
        with_bbox=False,
        with_label=False,
        with_attr_label=False,
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox_depth=False,
    ),
    dict(type="mmdet3d.ObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="mmdet3d.ObjectNameFilter", classes=class_names),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="RandomScaleImageMultiViewImage", scales=[0.5]),
    dict(type="PadMultiViewImage", size_divisor=32),
    dict(
        type="mmdet3d.Pack3DDetInputs",
        keys=[
            "gt_bboxes_3d",
            "gt_labels_3d",
            "img",
        ],
        meta_keys=meta_keys,
    ),
]
test_pipeline = [
    dict(type="mmdet3d.LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="PadMultiViewImage", size_divisor=32),
    dict(
        type="mmdet3d.MultiScaleFlipAug3D",
        img_scale=(1238, 832),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type="RandomScaleImageMultiViewImage", scales=[0.5]),
            dict(type="PadMultiViewImage", size_divisor=32),
            dict(
                type="mmdet3d.Pack3DDetInputs",
                keys=[
                    "gt_bboxes_3d",
                    "gt_labels_3d",
                    "img",
                ],
                meta_keys=meta_keys,
            ),
        ],
    ),
]


metainfo = dict(classes=class_names)
backend_args = None


train_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    frames=frames,
    ann_file="waymo_infos_train.pkl",
    pipeline=train_pipeline,
    # debug_pipeline_keys={"lidar2img", "lidar2cam"},
    load_type="frame_based",
    metainfo=metainfo,
    modality=input_modality,
    test_mode=False,
    data_prefix=data_prefix,
    # serialize_data=False,
    box_type_3d="LiDAR",
    backend_args=None,
    load_interval=5,
    # indices=10,
)
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=10,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=train_dataset,
    collate_fn=dict(type="default_collate"),
)
val_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    frames=frames,
    ann_file="waymo_infos_val.pkl",
    pipeline=test_pipeline,
    load_type="frame_based",
    metainfo=metainfo,
    modality=input_modality,
    test_mode=True,
    data_prefix=data_prefix,
    # indices=500,
    box_type_3d="LiDAR",
    backend_args=None,
)
val_dataloader = dict(
    batch_size=val_batch_size,
    num_workers=10,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=val_dataset,
    collate_fn=dict(type="default_collate"),
)

test_dataloader = val_dataloader

test_cfg = dict(type="TestLoop")

############# Optimizer and LR settings #############
max_lr = 8e-4
optimizer = dict(
    type="AdamW",
    lr=max_lr,
    weight_decay=0.01,
)

optim_wrapper = dict(
    type="AmpOptimWrapper",
    loss_scale=512.0,
    # dtype="bfloat16",
    optimizer=optimizer,
    accumulative_counts=grad_accum,
    clip_grad=dict(type="norm", max_norm=35, norm_type=2, error_if_nonfinite=False),
    paramwise_cfg=dict(
        custom_keys={
            "img_backbone": dict(lr_mult=0.1),
            "A_log": dict(decay_mult=0.0),
            "D": dict(decay_mult=0.0),
            "dt_bias": dict(decay_mult=0.0),
            "sampling_offsets": dict(lr_mult=0.1),
        }
    ),
)
param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=1 / 3,
        by_epoch=False,
        end=warmup_end,
    ),
    dict(
        type="CosineAnnealingLR",
        begin=total_epochs // 10,
        by_epoch=True,
        eta_min=1e-1 * max_lr,
    ),
]
# compile = dict(backend="inductor")
######################################################
############## Callbacks and Hooks ###################
#
# val_evaluator = dict(
#     type="WaymoMetric",
#     waymo_bin_file="./data/waymo/waymo_format/cam_gt.bin",
#     metric="LET_mAP",
#     result_prefix="eval/waymo_tiny/tiny",
#     format_only=True,
#
# )
val_evaluator = dict(
    type="KittiMetric",
    ann_file="data/waymo/kitti_format/waymo_infos_val.pkl",
    metric="bbox",
    pcd_limit_range=[-45.0, -85.0, -10.0, 85.0, 85.0, 10.0],
    default_cam_key="CAM_FRONT",
    collect_device="gpu",
)
test_evaluator = val_evaluator

default_hooks = dict(
    logger=dict(type="LoggerHook", interval=int(iter_mult * 50)),
    checkpoint=dict(
        type="CheckpointHook", interval=1, max_keep_ckpts=5, save_last=True
    ),
)
visualizer = dict(
    type="Visualizer",
    vis_backends=[
        dict(
            type="WandbVisBackend",
            init_kwargs=dict(
                project="waymo",
                name="TinyWaymo",
            ),
        )
    ],
)
# log_level = "DEBUG"

custom_hooks = [
    dict(
        type="EMAHook",
        begin_iter=0,
    ),
]
