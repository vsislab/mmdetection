model = dict(
    type='TianchiDetector',
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels=1,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=[
        dict(
            type='FPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            num_outs=5),
        dict(
            type='TianchiTopBottom',
            in_channels=256,
            num_levels=5,
            out_size=(320, 320))
    ],
    head=dict(
        type='TianchiBaseHead',
        in_channels=256,
        loss_mask=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0)))
train_cfg = None
test_cfg = None
# data settings
dataset_type = 'TianchiDataset'
data_root = '/data/lumbar/'
train_pipeline = [
    dict(type='TianchiLoadImageFromDICOM'),
    dict(type='TianchiLoadAnnotation'),
    dict(type='TianchiResize', img_scale=(320, 320), keep_ratio=True),
    dict(type='TianchiPad', size=(320, 320)),
    dict(type='TianchiFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_points', 'gt_labels', 'valid'],
        meta_keys=[
            'ori_shapes', 'img_shapes', 'pad_shapes', 'scale_factors', 'gt_idx'
        ])
]
samples_per_gpu = 1
workers_per_gpu = 8
# make sure num_frame <= 7
num_frames = 4
data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        img_prefix=data_root + 'lumbar_train150',
        ann_file=data_root + 'annotations/lumbar_train150_annotation.json',
        num_frames=num_frames,
        workers_per_gpu=workers_per_gpu,
        pipeline=train_pipeline),
    test=dict(
        type=dataset_type,
        img_prefix=data_root + 'lumbar_test',
        num_frames=num_frames,
        workers_per_gpu=workers_per_gpu,
        test_mode=True))
# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 19])
total_epochs = 20
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
