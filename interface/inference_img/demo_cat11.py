#!/usr/bin/env python
# encoding=utf8
#########################################################################
# Author: breezecheng
# Created Time: Fri Dec 14 18:41:25 2018
# File Name: demo_1.0.py
# Description:
#########################################################################

import glob
import os
import pdb
import time

import numpy as np

import cv2
from maskrcnn_benchmark.config import cfg
from predictor_cat11 import COCODemo

# update the config options with the config file
config_file = "../../configs/inference_full_product39_faster_rcnn_R_50_FPN_1x.yaml"
cfg.merge_from_file(config_file)
# manual override some options
# cfg.merge_from_list(["MODEL.DEVICE", "cpu"])
cfg.merge_from_list(["MODEL.ROI_BOX_HEAD.NUM_CLASSES", 12])
cfg.merge_from_list(["MODEL.IGNORE_CLASS_WEIGHT_BIAS", False])
cfg.merge_from_list(["MODEL.ROI_HEADS.NMS", 0.3])
cfg.merge_from_list(["MODEL.WEIGHT", '../../pretrained_models/.torch/models/full_product_det11_384-512_v4_classs_weight_0200000.pth'])

visualize_flag = True
data_source = 1 #0: query, 1: product
if data_source == 0:
   #####params for query_img
   min_image_size=448
   score_weight=0.60
   pos_weight=0.30
   area_weight=0.10
   min_bbox_w=0.1
   min_bbox_h=0.1
   min_bbox_area=0.02
   topn=1
   score_thresh=0.2
   score_thresh_high=0.25 # bbox which is score < score_high, will be relabeled to other
   cfg.merge_from_list(["MODEL.ROI_HEADS.SCORE_THRESH", score_thresh])
   cfg.merge_from_list(["MODEL.ROI_HEADS.SCORE_THRESH_HIGH", score_thresh_high])
else:
   ######params for product_img
   min_image_size=448
   score_weight=0.60
   pos_weight=0.30
   area_weight=0.10
   min_bbox_w=0.05
   min_bbox_h=0.05
   min_bbox_area=0.01
   topn=6
   score_thresh=0.25
   score_thresh_high=0.4 # bbox which is score < score_high, will be relabeled to other
   cfg.merge_from_list(["MODEL.ROI_HEADS.SCORE_THRESH", score_thresh])
   cfg.merge_from_list(["MODEL.ROI_HEADS.SCORE_THRESH_HIGH", score_thresh_high])
   cfg.merge_from_list(["MODEL.ROI_HEADS.USE_NMS_INTER_CLASS", True])
   cfg.merge_from_list(["MODEL.ROI_HEADS.NMS_INTER_CLASS", 0.6])
   cfg.merge_from_list(["MODEL.ROI_HEADS.USE_NMS_AREA", True])
   cfg.merge_from_list(["MODEL.ROI_HEADS.NMS_AREA", 0.8])

coco_demo = COCODemo(cfg, data_source=data_source, min_image_size=min_image_size, score_weight=score_weight, pos_weight=pos_weight, area_weight=area_weight, min_bbox_w=min_bbox_w, min_bbox_h=min_bbox_h, min_bbox_area=min_bbox_area, topn=topn, visualize_flag=visualize_flag)
# load image and then run prediction
img_names = glob.glob("../../demo/badcase_imgs/*/*.*")
#img_names = glob.glob("badcase_imgs/*.jpg")
start_time = time.time()
end_time = time.time()
for img_idx, img_name in enumerate(img_names):
    #if not img_name.endswith('UNADJUSTEDNONRAW_thumb_16e.jpg'):
    #  continue
    print("{}/{}:{}".format(img_idx + 1, len(img_names), img_name))
    image = cv2.imread(img_name)
    if image is None:
        continue
    top_predictions = coco_demo.run_on_opencv_image(image, topn=5)
    if visualize_flag:
       save_path = "./result/{}".format(os.path.split(cfg.MODEL.WEIGHT)[1])
       if not os.path.isdir(save_path):
          os.makedirs(save_path)
       cv2.imwrite("{}/{}.jpg".format(save_path, os.path.split(img_name[:-4])[1]), top_predictions)
    else:
       print(top_predictions)
    print("using time: {:.4f}s".format(time.time() - end_time))
    end_time = time.time()
print("Average time {:.4f}s ({:.4f}s/{})".format((end_time - start_time) / len(img_names), (end_time - start_time), len(img_names)))
