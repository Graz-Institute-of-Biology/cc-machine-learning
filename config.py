# _base_ = [
#     '/usr/people/EDVZ/faulhamm/cc-machine-learning/configs/_base_/models/segformer.py',
#     '/usr/people/EDVZ/faulhamm/cc-machine-learning/configs_base_/default_runtime.py',
#     '/usr/people/EDVZ/faulhamm/cc-machine-learning/configs_base_/schedules/schedule_160k_adamw.py'
# ]

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
find_unused_parameters = True
model = dict(
    type='EncoderDecoder',
    pretrained='pretrained/mit_b0.pth',
    backbone=dict(
        type='mit_b0',
        style='pytorch'),
    decode_head=dict(
        type='SegFormerHead',
        in_channels=[32, 64, 160, 256],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(embed_dim=256),
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))