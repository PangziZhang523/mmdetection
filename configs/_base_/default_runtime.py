'''
Author: your name
Date: 2021-01-27 08:45:44
LastEditTime: 2021-02-08 12:02:54
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
load_from = '/data_raid5_21T/zgh/ZGh/mmdetection/weights/num9/cascade_rcnn_x101_64x4d_fpn_20e_coco_20200509_coco_pretrained_weights_classes_9.pth'
resume_from = None
workflow = [('train', 1)]
# work_dir = '/data_raid5_21T/zgh/ZGh/work_dirs/cascade_r2_1'
# gpu_ids = range(4)
