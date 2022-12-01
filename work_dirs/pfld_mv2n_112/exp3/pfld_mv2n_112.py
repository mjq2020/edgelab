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
dataset_type = 'MeterData'
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=dict(
        type='MeterData',
        index_file='/home/dq/gitlab/datasets/table/train_data/list_d.txt',
        transform=True),
    val=dict(
        type='MeterData',
        index_file='/home/dq/gitlab/datasets/table/test_data/list_d.txt',
        transform=False),
    test=dict(
        type='MeterData',
        index_file='/home/dq/gitlab/datasets/table/test_data/list_d.txt',
        transform=False))
evaluation = dict(save_best='loss')
optimizer = dict(type='Adam', lr=0.0001, betas=(0.9, 0.99), weight_decay=1e-06)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=400,
    warmup_ratio=0.0001,
    step=[440, 490])
total_epochs = 500
find_unused_parameters = True
runner = dict(max_epochs=1500)
work_dir = '/home/dq/github/edgelab/work_dirs/pfld_mv2n_112/exp3'
auto_resume = False
gpu_ids = [0]
