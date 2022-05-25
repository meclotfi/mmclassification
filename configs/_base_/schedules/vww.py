# optimizer

optimizer = dict(type='Adam', lr=0.002)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='Step',
    step=1,
    gamma=0.95,
    min_lr=0.00001)
runner = dict(type='EpochBasedRunner', max_epochs=100)