import os
import os.path as osp
import xml.etree.ElementTree as ET

import mmcv

from glob import glob
from tqdm import tqdm
from PIL import Image
import numpy as np

def mars_classes():
    return ['l_sleeve', 'trousers', 's_sleeve', 'shorts', 'unsure']


label_ids = {name: i + 1 for i, name in enumerate(mars_classes())}


def get_segmentation(points):

    return [points[0], points[1], points[2] + points[0], points[1],
             points[2] + points[0], points[3] + points[1], points[0], points[3] + points[1]]


def parse_xml(xml_path, img_id, anno_id):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    annotation = []

    for obj in root.findall('object'):
        for na in obj.findall('class'):
            print(na.text)             
            category_id = label_ids[na.text]
            bnd_box = obj.find('bndbox')

            xmin = float(bnd_box.find('xmin').text)
            ymin = float(bnd_box.find('ymin').text)
            xmax = float(bnd_box.find('xmax').text)
            ymax = float(bnd_box.find('ymax').text)

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
#         name = obj.findall('class')
#         print(name)
#         print(len(name))
#         name = obj.find('class').text
#         print(name)
#         for i in range(len(name)):
            
#         print(name)
#         if name =="person":
#             continue

#         print(obj)

    return annotation, anno_id


def cvt_annotations(img_path, xml_path, out_file):
    images = []
    annotations = []

    img_id = 1
    anno_id = 1

    for img_path in tqdm(glob(img_path + '/*.jpg')):
        w, h = Image.open(img_path).size
        img_name = osp.basename(img_path)
        img = {"file_name": img_name, "height": int(h), "width": int(w), "id": img_id}
        print("file_name",img_name)
        images.append(img)

        xml_file_name = img_name.split('.')[0] + '.xml'
        xml_file_path = osp.join(xml_path, xml_file_name)

        if os.path.exists(xml_file_path):
            annos, anno_id = parse_xml(xml_file_path, img_id, anno_id)
            annotations.extend(annos)

        img_id += 1

    categories = []
    for k, v in label_ids.items():
        categories.append({"name": k, "id": v})
    final_result = {"images": images, "annotations": annotations, "categories": categories}
    mmcv.dump(final_result, out_file)
    return annotations


def main():
    img_path = '/home/data/50/'
    xml_path = '/home/data/50/'
    wrong_img = './wrong_format_file_list.txt'
    data1 = np.loadtxt(wrong_img,dtype=str)
    for i in range(len(data1)):
        if data1[i] in os.listdir(img_path):
            print("!!!!!!!!")
            print(os.path.join(img_path,data[i]))
            os.remove(os.path.join(img_path,data[i]))
    
    print('processing {} ...'.format("xml format annotations"))
    cvt_annotations(img_path, xml_path, '/project/train/src_repo/temp_data/instances_train2020_coco.json')
    print('Done!')


if __name__ == '__main__':
    main()


