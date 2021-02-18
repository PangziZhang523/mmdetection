###
 # @Author: your name
 # @Date: 2021-01-28 09:05:53
 # @LastEditTime: 2021-02-18 05:59:55
 # @LastEditors: Please set LastEditors
 # @Description: In User Settings Edit
 # @FilePath: /mmdetection/tools/train.sh
### 
set -ex
CUDA_VISIBLE_DEVICES=0,1,2,3 ./dist_train.sh ../config/cascade_rcnn_x101_64x4d_fpn_20e_dcn.py 4 --no-validate
#CUDA_VISIBLE_DEVICES=0,1,2,3 ./dist_train.sh ../configs/dcn/cascade_rcnn_r101_fpn_dconv_c3-c5_1x_coco.py 4 --no-validate