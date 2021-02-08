###
 # @Author: your name
 # @Date: 2021-01-28 09:05:53
 # @LastEditTime: 2021-02-07 10:11:52
 # @LastEditors: Please set LastEditors
 # @Description: In User Settings Edit
 # @FilePath: /mmdetection/tools/train.sh
### 
set -ex
CUDA_VISIBLE_DEVICES=0,1,2,3 ./dist_train.sh ../config/cascade_rcnn_x101_64x4d_fpn_20e_coco.py 4 --no-validate
