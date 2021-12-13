_base_ = [
    '/content/mmaction2/configs/_base_/models/bsn_tem.py', '/content/mmaction2/configs/_base_/default_runtime.py'
]

# dataset settings
dataset_type = 'ActivityNetDataset'
data_root = '/content/automation-suites/sa_dataset/mmaction_feat/'
data_root_val = '/content/automation-suites/sa_dataset/mmaction_feat/'
ann_file_train = '/content/automation-suites/sa_dataset/sa_dataset_train.json'
ann_file_val = '/content/automation-suites/sa_dataset/sa_dataset_test.json'
ann_file_test = '/content/automation-suites/sa_dataset/sa_dataset_test.json'

test_pipeline = [
    dict(type='LoadLocalizationFeature'),
    dict(
        type='Collect',
        keys=['raw_feature'],
        meta_name='video_meta',
        meta_keys=['video_name']),
    dict(type='ToTensor', keys=['raw_feature'])
]
train_pipeline = [
    dict(type='LoadLocalizationFeature'),
    dict(type='GenerateLocalizationLabels'),
    dict(
        type='Collect',
        keys=['raw_feature', 'gt_bbox'],
        meta_name='video_meta',
        meta_keys=['video_name']),
    dict(type='ToTensor', keys=['raw_feature', 'gt_bbox']),
    dict(type='ToDataContainer', fields=[dict(key='gt_bbox', stack=False)])
]
val_pipeline = [
    dict(type='LoadLocalizationFeature'),
    dict(type='GenerateLocalizationLabels'),
    dict(
        type='Collect',
        keys=['raw_feature', 'gt_bbox'],
        meta_name='video_meta',
        meta_keys=['video_name']),
    dict(type='ToTensor', keys=['raw_feature', 'gt_bbox']),
    dict(type='ToDataContainer', fields=[dict(key='gt_bbox', stack=False)])
]

data = dict(
    videos_per_gpu=16,
    workers_per_gpu=8,
    train_dataloader=dict(drop_last=True),
    val_dataloader=dict(videos_per_gpu=1),
    test_dataloader=dict(videos_per_gpu=1),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        pipeline=test_pipeline,
        data_prefix=data_root_val),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        pipeline=val_pipeline,
        data_prefix=data_root_val),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        pipeline=train_pipeline,
        data_prefix=data_root))

# optimizer
optimizer = dict(
    type='Adam', lr=0.001, weight_decay=0.0001)  # this lr is used for 1 gpus
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=7)
total_epochs = 20

# runtime settings
checkpoint_config = dict(interval=1, filename_tmpl='tem_epoch_{}.pth')
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
workflow = [('train', 1), ('val', 1)]
work_dir = 'work_dirs/bsn_400x100_20e_1x16_activitynet_feature/'
tem_results_dir = f'{work_dir}/tem_results/'
output_config = dict(out=tem_results_dir, output_format='csv')