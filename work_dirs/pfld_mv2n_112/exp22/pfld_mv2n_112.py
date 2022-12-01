checkpoint_config = dict(interval=1)
log_config = dict(interval=5, hooks=[dict(type='TextLoggerHook')])
log_level = 'INFO'
load_from = None
resume_from = None
dist_params = dict(backend='nccl')
workflow = [('train', 1)]
opencv_num_threads = 1
mp_start_method = 'fork'
custom_imports = dict(
    imports=['models', 'datasets'], allow_failed_imports=False)
model = dict(
    type='PFLD',
    backbone=dict(type='PFLDInference'),
    loss_cfg=dict(type='PFLDLoss'))
train_pipeline = [
    dict(type='Resize', height=112, width=112),
    dict(type='ColorJitter', brightness=0.3, p=0.5),
    dict(type='MedianBlur', blur_limit=3, p=0.3),
    dict(type='HorizontalFlip'),
    dict(type='VerticalFlip'),
    dict(type='Rotate'),
    dict(type='Affine', translate_percent=[0.05, 0.1], p=0.6)
]
val_pipeline = [dict(type='Resize', height=112, width=112)]
dataset_type = 'MeterData'
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=dict(
        type='MeterData',
        data_root='/home/dq/gitlab/datasets/table',
        index_file='train_data/annotations.txt',
        pipeline=[
            dict(type='Resize', height=112, width=112),
            dict(type='ColorJitter', brightness=0.3, p=0.5),
            dict(type='MedianBlur', blur_limit=3, p=0.3),
            dict(type='HorizontalFlip'),
            dict(type='VerticalFlip'),
            dict(type='Rotate'),
            dict(type='Affine', translate_percent=[0.05, 0.1], p=0.6)
        ],
        transform=True),
    val=dict(
        type='MeterData',
        data_root='/home/dq/gitlab/datasets/table',
        index_file='test_data/annotations.txt',
        pipeline=[dict(type='Resize', height=112, width=112)],
        transform=False),
    test=dict(
        type='MeterData',
        data_root='/home/dq/gitlab/datasets/table',
        index_file='test_data/annotations.txt',
        pipeline=[dict(type='Resize', height=112, width=112)],
        transform=False))
evaluation = dict(save_best='loss')
optimizer = dict(type='Adam', lr=0.0001, betas=(0.9, 0.99), weight_decay=1e-06)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(policy='OneCycle', max_lr=0.0001, pct_start=0.1)
total_epochs = 500
find_unused_parameters = True
work_dir = '/home/dq/github/edgelab/work_dirs/pfld_mv2n_112/exp22'
auto_resume = False
gpu_ids = [0]
