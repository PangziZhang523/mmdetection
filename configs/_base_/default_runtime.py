'''
Author: your name
Date: 2021-01-27 08:45:44
LastEditTime: 2021-01-27 15:53:36
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
load_from = '/data_raid5_21T/zgh/ZGh/mmdetection/weights/cascade_rcnn_r2_101_fpn_20e_coco-f4b7b7db.pth'
resume_from = None
workflow = [('train', 1)]
# work_dir = '/data_raid5_21T/zgh/ZGh/work_dirs/cascade_r2_1'
# gpu_ids = range(4)