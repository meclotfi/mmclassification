_base_ = [
    '../_base_/datasets/vww.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/vww.py'
]
model = dict(
    type='ImageClassifier',
    backbone=dict(type='MobileViT',
                    out_indices=(2,3,4),
                    Layers_config={
                                                "layer1": {
                                                        "type":"mobilenet2",
                                                        "out_channels": 8,
                                                        "expand_ratio": 2,
                                                        "num_blocks": 1,
                                                        "stride": 1,
                                                        
                                                    },
                                                "layer2": {
                                                        "type":"mobilenet2",
                                                        "out_channels": 16,
                                                        "expand_ratio": 2,
                                                        "num_blocks": 3,
                                                        "stride": 2,
                                                        
                                                    },
                                                "layer3": {  # 28x28
                                                        "type":"mobilevit",
                                                        "out_channels": 24,
                                                        "head_dim": 12,
                                                        "ffn_dim": 96,
                                                        "n_transformer_blocks": 2,
                                                        "patch_h": 2,  # 8,
                                                        "patch_w": 2,  # 8,
                                                        "stride": 2,
                                                        "mv_expand_ratio": 2,
                                                        "num_heads": 4,
                                                        
                                                    },
                                                "layer4": {  # 14x14
                                                        "type":"mobilevit",
                                                        "out_channels": 48,
                                                        "head_dim": 16,
                                                        "ffn_dim": 128,
                                                        "n_transformer_blocks": 4,
                                                        "patch_h": 4,  # 4,
                                                        "patch_w": 4,  # 4,
                                                        "stride": 2,
                                                        "mv_expand_ratio": 2,
                                                        "num_heads": 4,
                                                        
                                                    },
                                                "layer5": {  # 7x7
                                                        "type":"mobilevit",
                                                        "out_channels": 64,
                                                        "head_dim": 20,
                                                        "ffn_dim": 160,
                                                        "n_transformer_blocks": 2,
                                                        "patch_h": 2,
                                                        "patch_w": 2,
                                                        "stride": 2,
                                                        "mv_expand_ratio": 2,
                                                        "num_heads": 4,
                                                        
                                                    },
                                            },
                    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=2,
        in_channels=64,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1,)
    ))
seed=0