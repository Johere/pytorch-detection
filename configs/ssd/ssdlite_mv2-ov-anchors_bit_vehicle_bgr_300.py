checkpoint_config = dict(interval=5)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
custom_hooks = [
    dict(type='NumClassCheckHook'),
    dict(type='CheckInvalidLossHook', interval=50, priority='VERY_LOW')
]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
model = dict(
    type='SingleStageDetector',
    backbone=dict(
        type='MobileNetV2',
        widen_factor=1.0,
        out_indices=(4, 7),
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.03),
        init_cfg=dict(type='TruncNormal', layer='Conv2d', std=0.03),
        first_channel=16,
        last_channel=112,
        predefined_arch_settings=[[1, 16, 1, 1], [6, 16, 2, 2], [6, 16, 3, 2],
                                  [6, 24, 4, 2], [6, 32, 3, 1], [6, 56, 3, 2],
                                  [6, 112, 1, 1]]),
    neck=dict(
        type='SSDNeck',
        in_channels=(32, 112),
        out_channels=(32, 112, 192, 96, 96, 48),
        level_strides=(2, 2, 2, 2),
        level_paddings=(1, 1, 1, 1),
        l2_norm_scale=None,
        use_depthwise=False,
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.03),
        act_cfg=dict(type='ReLU6'),
        init_cfg=dict(type='TruncNormal', layer='Conv2d', std=0.03)),
    bbox_head=dict(
        type='SSDHead',
        in_channels=(32, 112, 192, 96, 96, 48),
        num_classes=1,
        use_depthwise=False,
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.03),
        act_cfg=dict(type='ReLU6'),
        init_cfg=dict(type='Normal', layer='Conv2d', std=0.001),
        anchor_generator=dict(
            type='SSDAnchorGenerator',
            scale_major=False,
            ov_anchors=True,
            strides=[16, 32, 64, 107, 160, 300],
            ratios=[[], [2], [2], [], [], [2]],
            min_sizes=[48, 100, 150, 202, 253, 284],
            max_sizes=[100, 150, 202, 253, 284, 300]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[0.1, 0.1, 0.2, 0.2])),
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.0,
            ignore_iof_thr=-1,
            gt_max_assign_all=False),
        smoothl1_beta=1.0,
        allowed_border=-1,
        pos_weight=-1,
        neg_pos_ratio=3,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        nms=dict(type='nms', iou_threshold=0.45),
        min_bbox_size=0,
        score_thr=0.02,
        max_per_img=200))
cudnn_benchmark = True
dataset_type = 'HybridDataset'
data_root = '/mnt/disk1/data_for_linjiaojiao/datasets/'
img_norm_cfg = dict(
    mean=[103.53, 116.28, 123.675], std=[57.375, 57.12, 58.395], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Expand',
        mean=[103.53, 116.28, 123.675],
        to_rgb=False,
        ratio_range=(1, 4)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=(300, 300), keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(
        type='Normalize',
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        to_rgb=False),
    dict(type='Pad', size_divisor=300),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(300, 300),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(
                type='Normalize',
                mean=[103.53, 116.28, 123.675],
                std=[57.375, 57.12, 58.395],
                to_rgb=False),
            dict(type='Pad', size_divisor=300),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=48,
    workers_per_gpu=0,
    train=[
        dict(
            _delete_=True,
            type='RepeatDataset',
            times=1,
            dataset=dict(
                type='HybridDataset',
                ann_file=
                '/mnt/disk1/data_for_linjiaojiao/datasets/UA_DETRAC_fps5/train_meta.list',
                img_prefix=
                '/mnt/disk1/data_for_linjiaojiao/datasets/UA_DETRAC_fps5/images/',
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations', with_bbox=True),
                    dict(
                        type='Expand',
                        mean=[103.53, 116.28, 123.675],
                        to_rgb=False,
                        ratio_range=(1, 4)),
                    dict(
                        type='MinIoURandomCrop',
                        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
                        min_crop_size=0.3),
                    dict(
                        type='Resize', img_scale=(300, 300), keep_ratio=False),
                    dict(type='RandomFlip', flip_ratio=0.5),
                    dict(
                        type='PhotoMetricDistortion',
                        brightness_delta=32,
                        contrast_range=(0.5, 1.5),
                        saturation_range=(0.5, 1.5),
                        hue_delta=18),
                    dict(
                        type='Normalize',
                        mean=[103.53, 116.28, 123.675],
                        std=[57.375, 57.12, 58.395],
                        to_rgb=False),
                    dict(type='Pad', size_divisor=300),
                    dict(type='DefaultFormatBundle'),
                    dict(
                        type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
                ]),
            separate_eval=False),
        dict(
            _delete_=True,
            type='RepeatDataset',
            times=1,
            dataset=dict(
                type='HybridDataset',
                ann_file=
                '/mnt/disk1/data_for_linjiaojiao/datasets/BITVehicle/train_meta.list',
                img_prefix=
                '/mnt/disk1/data_for_linjiaojiao/datasets/BITVehicle/images/',
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations', with_bbox=True),
                    dict(
                        type='Expand',
                        mean=[103.53, 116.28, 123.675],
                        to_rgb=False,
                        ratio_range=(1, 4)),
                    dict(
                        type='MinIoURandomCrop',
                        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
                        min_crop_size=0.3),
                    dict(
                        type='Resize', img_scale=(300, 300), keep_ratio=False),
                    dict(type='RandomFlip', flip_ratio=0.5),
                    dict(
                        type='PhotoMetricDistortion',
                        brightness_delta=32,
                        contrast_range=(0.5, 1.5),
                        saturation_range=(0.5, 1.5),
                        hue_delta=18),
                    dict(
                        type='Normalize',
                        mean=[103.53, 116.28, 123.675],
                        std=[57.375, 57.12, 58.395],
                        to_rgb=False),
                    dict(type='Pad', size_divisor=300),
                    dict(type='DefaultFormatBundle'),
                    dict(
                        type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
                ]),
            separate_eval=False)
    ],
    val=dict(
        type='HybridDataset',
        ann_file=
        '/mnt/disk1/data_for_linjiaojiao/datasets/UA_DETRAC_fps5/val_meta.list',
        img_prefix=
        '/mnt/disk1/data_for_linjiaojiao/datasets/UA_DETRAC_fps5/images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(300, 300),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=False),
                    dict(
                        type='Normalize',
                        mean=[103.53, 116.28, 123.675],
                        std=[57.375, 57.12, 58.395],
                        to_rgb=False),
                    dict(type='Pad', size_divisor=300),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='HybridDataset',
        ann_file=
        '/mnt/disk1/data_for_linjiaojiao/datasets/BITVehicle/test_meta.list',
        img_prefix=
        '/mnt/disk1/data_for_linjiaojiao/datasets/BITVehicle/images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(300, 300),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=False),
                    dict(
                        type='Normalize',
                        mean=[103.53, 116.28, 123.675],
                        std=[57.375, 57.12, 58.395],
                        to_rgb=False),
                    dict(type='Pad', size_divisor=300),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
optimizer = dict(type='SGD', lr=0.015, momentum=0.9, weight_decay=4e-05)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    min_lr=0)
runner = dict(type='EpochBasedRunner', max_epochs=120)
evaluation = dict(interval=5, metric='mAP')
work_dir = 'output/ssdlite_exp13.1_4gpus'
auto_resume = False
gpu_ids = range(0, 4)
