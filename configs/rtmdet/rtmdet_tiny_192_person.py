# coding: UTF-8
_base_ = './rtmdet_s_8xb32-300e_coco.py'

# custom_imports = dict(imports=['models', 'datasets'], allow_failed_imports=False)

model = dict(
    backbone=dict(
        deepen_factor=0.167,
        widen_factor=0.375,
        init_cfg=dict(
            type='Pretrained', prefix='backbone.')),
    neck=dict(in_channels=[96, 192, 384], out_channels=96, num_csp_blocks=1),
    bbox_head=dict(num_classes=1, in_channels=96, feat_channels=96, exp_on_reg=False))


dataset_type = 'mmdet.CocoDataset'
classes = ('person',)


file_client_args = dict(backend='disk')
train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(128, 128), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='mmdet.PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='Resize', scale=(128, 128), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

data_root = 'D:/gitlab/node2-person'
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train/annotations/train.json',
        data_prefix=dict(img='train/images/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='valid/annotations/vaild.json',
        data_prefix=dict(img='valid/images/'),
        test_mode=True,
        pipeline=test_pipeline
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='valid/annotations/vaild.json',
        data_prefix=dict(img='valid/images/'),
        test_mode=True,
        pipeline=test_pipeline
    )
)
train_dataloader = dict(
    batch_size=32,
    num_workers=6,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler', _delete_ = True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train/annotations/train.json',
        data_prefix=dict(img='train/images/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='valid/annotations/vaild.json',
        data_prefix=dict(img='valid/images/'),
        test_mode=True,
        pipeline=test_pipeline))
test_dataloader = val_dataloader

evaluation = dict(
    interval=1,
    metric=['bbox'])

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + '/valid/annotations/vaild.json',
    metric='bbox',
    proposal_nums=[1,10,50],
    format_only=False)
test_evaluator = val_evaluator

env_cfg=dict(dist_cfg=dict(backend='nccl'))

