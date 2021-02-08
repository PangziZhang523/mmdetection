'''
Author: your name
Date: 2021-02-05 04:16:55
LastEditTime: 2021-02-08 12:04:13
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /mmdetection/configs/_base_/schedules/schedule_1x.py
'''
# optimizer
# optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001) # lr=0.00125*num_GPUs*imgs_per_gpu
# optimizer_config = dict(grad_clip=None)
# # learning policy
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=0.001,
#     step=[8, 11])
# total_epochs = 12
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 19])
total_epochs = 20