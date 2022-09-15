_base_ = [
    '../_base_/models/ssd300_vehicle.py',
    '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py'
]
# dataset settings
dataset_type = 'HybridDataset'
# data_root = 'data/coco/'
data_root = '/mnt/disk1/data_for_linjiaojiao/datasets/'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[1, 1, 1], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
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
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(300, 300),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=3,
    train=dict(
        _delete_=True,
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'UA_DETRAC_fps5/train_meta.list',
            img_prefix=data_root + 'UA_DETRAC_fps5/images/',
            pipeline=train_pipeline)),
    val=dict(
            type=dataset_type,
            ann_file=data_root + 'UA_DETRAC_fps5/val_meta.list',
            img_prefix=data_root + 'UA_DETRAC_fps5/images/',
            pipeline=test_pipeline),
    test=dict(
            type=dataset_type,
            ann_file=data_root + 'BITVehicle/bit_vehicle_annotation.list',
            img_prefix=data_root + 'BITVehicle/images/',
            pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=2e-3, momentum=0.9, weight_decay=5e-4)
optimizer_config = dict(_delete_=True)
custom_hooks = [
    dict(type='NumClassCheckHook'),
    dict(type='CheckInvalidLossHook', interval=50, priority='VERY_LOW')
]
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)

# evaluation
evaluation = dict(interval=2, metric='mAP')
