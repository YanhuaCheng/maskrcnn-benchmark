#!/usr/bin/env python
#encoding=utf8
#########################################################################
# Author:
# Created Time: Thu Jul  4 17:27:38 2019
# File Name: visualize.py
# Description:
#########################################################################
import os
import random

import numpy as np

import cv2
from maskrcnn_benchmark.structures.bounding_box import BoxList


def visual_transforms(image, target):
    image = np.array(image)
    image = image[:, :, ::-1].copy()
    print(image.shape)
    bboxs = target.convert('xyxy').bbox
    labels = target.get_field('labels')
    for bbox_id, (bbox, label) in enumerate(zip(bboxs, labels)):
        color = [0, 0, 255]
        x1, y1, x3, y3 = bbox[:4].tolist()
        image = cv2.rectangle(
            image, (int(x1), int(y1)), (int(x3), int(y3)), tuple(color), 1
        )
        cv2.putText(
            image, 'cat_{:0<2d}'.format(label), (int(x1), int(y1+15)), cv2.FONT_HERSHEY_SIMPLEX, .6, tuple(color), 1
        )
    save_path = '/data/user/data/breezecheng/pytorch_project/maskrcnn-benchmark/interface/visual/'
    if not os.path.isdir(save_path):
       os.makedirs(save_path)
    cv2.imwrite('{}/img_{:0<2d}.jpg'.format(save_path, random.randint(0, 100000)), image)
