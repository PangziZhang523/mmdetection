dataset_type = 'CocoDataset'
data_root = '/data/zhangguanghao/train_my_data/tile_round2_train_20210204/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=[(2200, 2000), (2200, 2200)], keep_ratio=True), #single-scale
    # dict(type='Resize', img_scale=[(2000, 1000),(2000,1800)], keep_ratio=True), #multi-scale
    # dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='RandomFlip', flip_ratio=[0.3, 0.5], direction=['horizontal', 'vertical']),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
    #dict(type='MixUp',p=0.5, lambd=0.5),
    #dict(type='BBoxJitter', min=0.9, max=1.1),
    #dict(type='BBoxJitter', min=0.95, max=1.05),
    #dict(type='Grid', use_w=True, use_h=True), #Gridmask
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        # img_scale=(1333, 800),
        img_scale=[(2200, 2000), (2200, 2100), (2200, 2200)],  #test multi-scale
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            # dict(type='BBoxJitter', min=0.95, max=1.05),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'tile_coco_train.json',
        img_prefix=data_root + 'train_imgs/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'val/tile_coco_val.json',
        img_prefix=data_root + '/val/images',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
