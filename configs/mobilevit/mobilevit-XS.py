_base_ = [
    '../_base_/datasets/vww.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/vww.py'
]
model = dict(
    type='ImageClassifier',
    backbone=dict(type='MobileViT',
                    out_indices=(4,),
                    Layers_config={
                                                "layer1": {
                                                        "type":"mobilenet2",
                                                        "out_channels": 32,
                                                        "expand_ratio": 4,
                                                        "num_blocks": 1,
                                                        "stride": 1,
                                                        
                                                    },
                                                "layer2": {
                                                        "type":"mobilenet2",
                                                        "expand_ratio": 4,
                                                        "out_channels": 48,
                                                        "num_blocks": 3,
                                                        "stride": 2,
                                                        
                                                    },
                                                "layer3": {  # 28x28
                                                        "type":"mobilevit",
                                                        "out_channels": 64,
                                                        "head_dim": 24,
                                                        "ffn_dim": 192,
                                                        "n_transformer_blocks": 2,
                                                        "patch_h": 2,  # 8,
                                                        "patch_w": 2,  # 8,
                                                        "stride": 2,
                                                        "mv_expand_ratio": 4,
                                                        "num_heads": 4,
                                                        
                                                    },
                                                "layer4": {  # 14x14
                                                        "type":"mobilevit",
                                                        "out_channels": 80,
                                                        "head_dim": 30,
                                                        "ffn_dim": 240,
                                                        "n_transformer_blocks": 4,
                                                        "patch_h": 2,  # 4,
                                                        "patch_w": 2,  # 4,
                                                        "stride": 2,
                                                        "mv_expand_ratio": 4,
                                                        "num_heads": 4,
                                                        
                                                    },
                                                "layer5": {  # 7x7
                                                        "type":"mobilevit",
                                                        "out_channels": 96,
                                                        "head_dim": 36,
                                                        "ffn_dim": 288,
                                                        "n_transformer_blocks": 3,
                                                        "patch_h": 2,
                                                        "patch_w": 2,
                                                        "stride": 2,
                                                        "mv_expand_ratio": 4,
                                                        "num_heads": 4,
                                                        
                                                    },
                                            },
                    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=2,
        in_channels=128,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1,)
    ))