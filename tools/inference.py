'''
server中使用GPU做infer
'''
from ai_hub import inferServer
# import json
import torch

import os
import cv2
import numpy as np

from mmdet.apis import init_detector
from mmdet.apis import inference_detector


class myserver(inferServer):

    def __init__(self, model):
        super().__init__(model)
        print("init_myserver")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.model= model.to(device)

    def pre_process(self, request):

        preprocessed_data = {}

        # file example
        file = request.files['img']
        # file_t = request.files['img_t']

        file_data = file.read()
        # file_t_data = file_t.read()

        img = cv2.imdecode(np.frombuffer(file_data, np.uint8), cv2.IMREAD_COLOR)

        preprocessed_data[file.filename] = img

        return preprocessed_data

    # def pre_process(self, data):
    #     preprocessed_data = {}
    #     for k, v in data.items():
    #         for file_name, file_content in v.items():
    #             if isinstance(file_content, str):
    #                 img = cv2.imread(file_content)
    #             else:
    #                 img = cv2.imdecode(np.frombuffer(file_content.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    #
    #
    #             preprocessed_data[file_content] = np.concatenate([img, img, img], axis=2)
    #         return preprocessed_data

    # pridict default run as follow：
    def pridect(self, data):

        category = [
                    {'name': '1', 'id': 1}, {'name': '2', 'id': 2},
                    {'name': '3', 'id': 3}, {'name': '4', 'id': 4},
                    {'name': '5', 'id': 5}, {'name': '6', 'id': 6},
                    {'name': '7', 'id': 7}, {'name': '8', 'id': 8}
                   ]

        # data = self.pre_process(data)

        for k, v in data.items():
            image_name = k
            data = v

        predict_rslt = []

        with torch.no_grad():
            results = inference_detector(self.model, data)

            for j, bboxes in enumerate(results, 1):
                if len(bboxes) > 0:
                    for bbox in bboxes:
                        x1, y1, x2, y2, score = bbox.tolist()

                        if score < 0.001:
                            continue

                        dict_instance = dict()
                        dict_instance['name'] = image_name
                        dict_instance['category'] = category[j - 1]["id"]
                        dict_instance["score"] = round(float(score), 6)
                        dict_instance["bbox"] = [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]
                        predict_rslt.append(dict_instance)

        return predict_rslt

    def post_process(self, data):
        return data


# class mymodel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc = nn.Linear(1, 1)
#         self.model = lambda x: torch.mul(x , 2)
#     def forward(self, x):
#         y = self.model(x)
#         # y = self.fc(y)
#         return y


if __name__ == '__main__':

    model_path = os.path.join(os.path.dirname(__file__), 'epoch_20.pth')
    config_file = os.path.join(os.path.dirname(__file__), 'config/cascade_rcnn_x101_64x4d_fpn_20e_coco.py')
    model = init_detector(config_file, model_path)

    myserver = myserver(model)
    # run your server, defult ip=localhost port=8080 debuge=false
    myserver.run(debuge=True)  # myserver.run("127.0.0.1", 1234)

    # m = myserver(model=model)
    # data = {'images': {'0': '/data_raid5_21T/lizhe/tile_detection/data/tile_round2_train_20210204/train_imgs/198_19_t20201119103227654_CAM3_3.jpg'}}
    # m.pridect(data)

