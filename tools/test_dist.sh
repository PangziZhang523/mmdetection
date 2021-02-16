###
 # @Author: your name
 # @Date: 2021-01-28 09:05:53
 # @LastEditTime: 2021-01-28 14:16:17
 # @LastEditors: Please set LastEditors
 # @Description: In User Settings Edit
 # @FilePath: /mmdetection/tools/train.sh
### 
set -ex
export CUDA_VISIBLE_DEVICES=4,5,6,7
./dist_test.sh \
../configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py \
../../work_dirs/cascade_rcnn_r50_fpn_1x_/latest.pth \
4 \
--format-only \
--options "jsonfile_prefix=./r50_coco"
--cfg-options data.test.samples_per_gpu=1
