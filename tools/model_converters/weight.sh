###
 # @Author: your name
 # @Date: 2021-01-28 09:05:53
 # @LastEditTime: 2021-02-07 11:00:39
 # @LastEditors: Please set LastEditors
 # @Description: In User Settings Edit
 # @FilePath: /mmdetection/tools/train.sh
### 
set -ex
python /data_raid5_21T/zgh/ZGh/mmdetection/tools/model_converters/weight.py \
--org_path /data_raid5_21T/zgh/ZGh/mmdetection/weights/cascade_rcnn_x101_64x4d_fpn_20e_coco_20200509_224357-051557b1.pth \
--num_classes 9
