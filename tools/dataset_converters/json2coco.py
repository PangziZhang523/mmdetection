import os.path as osp

import mmcv

from glob import glob
from tqdm import tqdm
from PIL import Image

import json


def tile_classes():
    # return ['边异常', '角异常', '白色点瑕疵', '浅色块瑕疵', '深色点块瑕疵', '光圈瑕疵', '记号笔', '划伤']
    return ['1', '2', '3', '4', '5', '6', '7', '8']


label_ids = {name: i + 1 for i, name in enumerate(tile_classes())}


def get_segmentation(points):

    return [points[0], points[1], points[2] + points[0], points[1],
             points[2] + points[0], points[3] + points[1], points[0], points[3] + points[1]]


def parse_json(train_anno, img_id, anno_id):
    annotation = []

    category_id = train_anno["category"]

    xmin = round(float(train_anno["bbox"][0]), 4)
    ymin = round(float(train_anno["bbox"][1]), 4)
    xmax = round(float(train_anno["bbox"][2]), 4)
    ymax = round(float(train_anno["bbox"][3]), 4)

    w = xmax - xmin + 1
    h = ymax - ymin + 1

    area = w*h
    segmentation = get_segmentation([xmin, ymin, w, h])
    annotation.append({
                    "segmentation": segmentation,
                    "area": area,
                    "iscrowd": 0,
                    "image_id": img_id,
                    "bbox": [xmin, ymin, w, h],
                    "category_id": category_id,
                    "id": anno_id,
                    "ignore": 0})
    anno_id += 1
    return annotation, anno_id


def cvt_annotations(img_path, json_path, out_file):
    images = []
    annotations = []

    img_id = 1
    anno_id = 1

    f = open("{}".format(json_path), "r")
    train_annos = json.load(f)

    for img_path in tqdm(glob(img_path + '/*.jpg')):
        w, h = Image.open(img_path).size
        img_name = osp.basename(img_path)
        img = {"file_name": img_name, "height": int(h), "width": int(w), "id": img_id}

        images.append(img)

        for train_anno in train_annos:
            if img_name == train_anno["name"] and train_anno["category"] != 0:

                annos, anno_id = parse_json(train_anno, img_id, anno_id)
                annotations.extend(annos)


        img_id += 1


    categories = []
    for k, v in label_ids.items():
        categories.append({"name": k, "id": v})
    final_result = {"images": images, "annotations": annotations, "categories": categories}
    mmcv.dump(final_result, out_file)
    return annotations


def main():
    img_path = '/data_raid5_21T/zgh/ZGh/round2_data/tile_round2_train_20210204/train_imgs/'
    json_path = '/data_raid5_21T/zgh/ZGh/round2_data/tile_round2_train_20210204/train_annos.json'

    print('processing {} ...'.format("xml format annotations"))
    cvt_annotations(img_path, json_path, '/data_raid5_21T/zgh/ZGh/round2_data/tile_round2_train_20210204/tile_coco_train_1.json')
    print('Done!')


if __name__ == '__main__':
    main()

