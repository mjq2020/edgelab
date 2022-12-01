checkpoint_config = dict(interval=5)
log_config = dict(interval=150, hooks=[dict(type='TextLoggerHook')])
runner = dict(type='EpochBasedRunner', max_epochs=1500)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs'
load_from = None
resume_from = None
workflow = [('train', 1)]
custom_imports = dict(
    imports=['models', 'datasets', 'core'], allow_failed_imports=False)
model = dict(
    type='Audio_classify',
    backbone=dict(
        type='SoundNetRaw',
        nf=2,
        clip_length=64,
        factors=[4, 4, 4],
        out_channel=36),
    head=dict(type='Audio_head', in_channels=36, n_classes=4, drop=0.2),
    loss_cls=dict(
        type='LabelSmoothCrossEntropyLoss', reduction='sum', smoothing=0.1))
dataset_type = 'Speechcommand'
transforms = [
    'amp', 'neg', 'tshift', 'tmask', 'ampsegment', 'cycshift', 'awgn', 'abgn',
    'apgn', 'argn', 'avgn', 'aun', 'phn', 'sine'
]
data_root = '/home/dq/github/datasets/speech_commands_v0.02'
train_pipeline = dict(
    type='AudioAugs',
    k_augs=[
        'amp', 'neg', 'tshift', 'tmask', 'ampsegment', 'cycshift', 'awgn',
        'abgn', 'apgn', 'argn', 'avgn', 'aun', 'phn', 'sine'
    ])
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='CenterCrop',
        crop_size=224,
        efficientnet_style=True,
        interpolation='bicubic',
        backend='pillow'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=4,
    train=dict(
        type='Speechcommand',
        root='/home/dq/github/datasets/speech_commands_v0.02',
        sampling_rate=8000,
        segment_length=8192,
        pipeline=dict(
            type='AudioAugs',
            k_augs=[
                'amp', 'neg', 'tshift', 'tmask', 'ampsegment', 'cycshift',
                'awgn', 'abgn', 'apgn', 'argn', 'avgn', 'aun', 'phn', 'sine'
            ]),
        mode='train',
        use_background=False),
    val=dict(
        type='Speechcommand',
        root='/home/dq/github/datasets/speech_commands_v0.02',
        sampling_rate=8000,
        segment_length=8192,
        mode='val',
        use_background=False),
    test=dict(
        type='Speechcommand',
        root='/home/dq/github/datasets/speech_commands_v0.02',
        sampling_rate=8000,
        segment_length=8192,
        mode='test',
        use_background=False))
custom_hooks = dict(
    type='Audio_hooks',
    n_cls=4,
    multilabel=False,
    loss=dict(
        type='LabelSmoothCrossEntropyLoss', reduction='sum', smoothing=0.1),
    seq_len=8192,
    sampling_rate=8000,
    device='0',
    augs_mix=['mixup', 'timemix', 'freqmix', 'phmix'],
    mix_ratio=1,
    local_rank=0,
    epoch_mix=12,
    mix_loss='bce',
    priority=0)
evaluation = dict(
    save_best='acc',
    interval=1,
    metric='accuracy',
    metric_options=dict(topk=(1, )))
optimizer = dict(
    type='AdamW', lr=0.0003, betas=[0.9, 0.99], weight_decay=0, eps=1e-08)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='OneCycle', max_lr=0.0003, pct_start=0.1)
auto_resume = False
gpu_ids = [0]
