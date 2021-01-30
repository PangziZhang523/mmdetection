'''
Author: your name
Date: 2021-01-27 08:45:44
LastEditTime: 2021-01-28 15:01:33
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /mmdetection/configs/_base_/default_runtime.py
'''
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
# load_from = None
load_from = '/data/train_my/mmdetection/weights/num7/cascade_rcnn_r50_fpn_1x_coco_coco_pretrained_weights_classes_7.pth'
resume_from = None
workflow = [('train', 1)]
# work_dir = '/data_raid5_21T/zgh/ZGh/work_dirs/cascade_r2_1'
# gpu_ids = range(4)
