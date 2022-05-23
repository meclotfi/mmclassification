_base_ = [
    '../_base_/datasets/cifar10_tl.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/cifar10_bs128.py'
]
model = dict(
    type='ImageClassifier',
    backbone=dict(type='MobileViT'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=10,
        in_channels=80,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1,),
    ))