# 修改预训练权重类别数
import os
import torch
import argparse

'''
python3 tools/data_process/weight.py --org_path \
 /data/train_my/mmdetection/weights/cascade_rcnn_r2_101_fpn_20e_coco-f4b7b7db.pth \
--num_classes 7
'''


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--org_path", type=str, help="the path of pretrained model")
    parser.add_argument("--num_classes", type=int, default=26, help="number of classes")
    return parser.parse_args()


def mkdir(path):
    '''make dir'''

    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)


def modify_other_rcnn(model_coco, num_classes):


    model_coco["state_dict"]["bbox_head.atss_cls.weight"].resize_(num_classes, 256, 3, 3)
    model_coco["state_dict"]["bbox_head.atss_reg.bias"].resize_(num_classes*4, 1024)
    # bias
    model_coco["state_dict"]["bbox_head.atss_cls.bias"].resize_(num_classes)
    model_coco["state_dict"]["bbox_head.atss_reg.bias"].resize_(4)


def modify_faster_rcnn(model_coco, num_classes):
    print(model_coco["state_dict"])
    model_coco["state_dict"]["roi_head.bbox_head.fc_cls.weight"].resize_(num_classes, 1024)
    model_coco["state_dict"]["roi_head.bbox_head.fc_reg.weight"].resize_(num_classes*4, 1024)
    # bias
    model_coco["state_dict"]["roi_head.bbox_head.fc_cls.bias"].resize_(num_classes)
    model_coco["state_dict"]["roi_head.bbox_head.fc_reg.bias"].resize_(num_classes * 4)


def modify_cascade_rcnn(model_coco, num_classes):

    # print(model_coco["state_dict"])
    model_coco["state_dict"]["roi_head.bbox_head.0.fc_cls.weight"].resize_(num_classes, 1024)
    model_coco["state_dict"]["roi_head.bbox_head.1.fc_cls.weight"].resize_(num_classes, 1024)
    model_coco["state_dict"]["roi_head.bbox_head.2.fc_cls.weight"].resize_(num_classes, 1024)

    # bias
    model_coco["state_dict"]["roi_head.bbox_head.0.fc_cls.bias"].resize_(num_classes)
    model_coco["state_dict"]["roi_head.bbox_head.1.fc_cls.bias"].resize_(num_classes)
    model_coco["state_dict"]["roi_head.bbox_head.2.fc_cls.bias"].resize_(num_classes)


def main(args):
    save_dir = "/data/train_my/mmdetection/weights/num%s/" % args.num_classes
    mkdir(save_dir)
    pth_dir = args.org_path
    model_coco = torch.load(pth_dir)
    base_name = os.path.basename(pth_dir)
    mode = base_name.split('_')
    if (mode[0] == 'faster') and (mode[1] == 'rcnn'):
        print('Current model is faster rcnn.')
        print('Converting ...')
        modify_faster_rcnn(model_coco, args.num_classes)
    elif (mode[0] == 'cascade') and (mode[1] == 'rcnn') or (mode[0] == 'epoch') or (mode[0] == 'detectors'):
        print('Current model is cascade rcnn.')
        print('Converting ...')
        modify_cascade_rcnn(model_coco, args.num_classes)
    else:
        print('Current model is cascade rcnn.')
        print('Converting ...')
        modify_other_rcnn(model_coco, args.num_classes)
    # save new model
    model_name = save_dir + base_name.replace(base_name.split('_')[-1], 'coco_pretrained_weights_classes_') + str(args.num_classes) + ".pth"

    print(model_name)
    torch.save(model_coco, model_name)
    print("Convert successful.")


if __name__ == '__main__':
    args = init_args()
    main(args)

