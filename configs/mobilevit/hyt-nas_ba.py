_base_ = [
    '../_base_/datasets/vww.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/vww.py'
]
model = dict(
    type='ImageClassifier',
    backbone=dict(type='HytNAS',
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
                                                        "out_channels": 32,
                                                        "num_blocks": 4,
                                                        "stride": 2,
                                                        
                                                    },
                                                "layer3": {  # 28x28
                                                        "type":"mobilevit",
                                                        "out_channels": 64,
                                                        "head_dim": 16,
                                                        "ffn_dim": 96,
                                                        "n_transformer_blocks": 1,
                                                        "patch_h": 8,  # 8,
                                                        "patch_w": 8,  # 8,
                                                        "stride": 2,
                                                        "mv_expand_ratio": 4,
                                                        "num_heads": 4,
                                                        
                                                    },
                                                "layer4": {  # 14x14
                                                        "type":"mobilevit",
                                                        "out_channels": 128,
                                                        "head_dim": 32,
                                                        "ffn_dim": 256,
                                                        "n_transformer_blocks": 1,
                                                        "patch_h": 8,  # 4,
                                                        "patch_w": 8,  # 4,
                                                        "stride": 2,
                                                        "mv_expand_ratio": 1,
                                                        "num_heads": 4,
                                                        
                                                    },
                                                "layer5": {  # 7x7
                                                        "type":"mobilevit",
                                                        "out_channels": 256,
                                                        "head_dim": 256,
                                                        "ffn_dim": 512,
                                                        "n_transformer_blocks": 1,
                                                        "patch_h": 4,
                                                        "patch_w": 4,
                                                        "stride": 2,
                                                        "mv_expand_ratio": 2,
                                                        "num_heads": 1,
                                                        
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