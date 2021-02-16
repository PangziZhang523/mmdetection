###
 # @Author: your name
 # @Date: 2021-01-28 09:05:53
 # @LastEditTime: 2021-01-28 14:16:17
 # @LastEditors: Please set LastEditors
 # @Description: In User Settings Edit
 # @FilePath: /mmdetection/tools/train.sh
### 
set -ex
python test.py ../configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py ../../work_dirs/cascade_rcnn_r50_fpn_1x_/latest.pth --format-only \
--options "jsonfile_prefix=./r50_coco_test"
#results.pkl
