# model settings
norm_cfg = dict(type='BN', requires_grad=True)  # SyncBN
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='L2ViT',
        depths=[2, 2, 6, 2],
        mlp_ratio=[4., 4., 4., 4.],
        dims=[96, 192, 384, 768],
        block_type='mix_local_enhanced_vit',
        conv_stem=True,
        drop_path_rate=0.3,
    ),
    decode_head=dict(
        type='UPerHead',
        in_channels=[96, 192, 384, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=384,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))