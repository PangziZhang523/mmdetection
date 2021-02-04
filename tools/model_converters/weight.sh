###
 # @Author: your name
 # @Date: 2021-01-28 09:05:53
 # @LastEditTime: 2021-01-28 14:16:17
 # @LastEditors: Please set LastEditors
 # @Description: In User Settings Edit
 # @FilePath: /mmdetection/tools/train.sh
### 
set -ex
python /data/train_my/mmdetection/tools/model_converters/weight.py \
--org_path /data/train_my/mmdetection/weights/cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth \
--num_classes 9
