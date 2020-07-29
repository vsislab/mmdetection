_base_ = [
    './_base_/tianchi_lumbar.py',
    './_base_/schedule_40e.py',
    './_base_/default_runtime.py',
]
model = dict(
    type='TianchiDetector',
    backbone=dict(
        type='ResNet',
        depth=50,
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
            out_size=(128, 128))
    ],
    bbox_head=dict(
        type='TianchiBaseHead',
        num_classes=7,
        in_channels=256,
        feat_channels=256,
        stacked_convs=4))
train_cfg = dict(assigner=dict(type='TianchiPointAssigner', pos_num=1))
test_cfg = None
