#!/usr/bin/env python
# encoding=utf8
#########################################################################
# Author: breezecheng
# Created Time: Fri Dec 14 18:41:25 2018
# File Name: demo.py
# Description:
#########################################################################
import argparse
import os
import time

import cv2
from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
from transforms import build_transforms

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file", type=str,)
    parser.add_argument("--root-dir", default="", dest="root_dir", help="path to images", type=str,)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER,)
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    assert(cfg.TEST.IMS_PER_BATCH == 1)
    img_names = []
    if not args.root_dir.endswith('/'):
        args.root_dir += '/'
    for root, parent, img_files in os.walk(args.root_dir):
        for img_file in img_files:
            if img_file.lower().endswith(('.jpg', '.png', '.bmp', '.jpeg')):
                img_names.append(os.path.join(root, img_file)[len(args.root_dir):])
    transformer = build_transforms(cfg)
    coco_demo = COCODemo(cfg)
    start_time = time.time()
    end_time = time.time()
    for img_idx, img_name in enumerate(img_names):
        print('{}/{}: img_name={}'.format(img_idx, len(img_names), img_name))
        img_ori = cv2.imread(os.path.join(args.root_dir, img_name))
        img = transformer(img_ori[:, :, ::-1])
        top_predictions = coco_demo.get_top_detections(img, img_ori.shape[0], img_ori.shape[1], img_name, topn=None)
        print("using time: {:.4f}s".format(time.time() - end_time))
        end_time = time.time()
    print("Average time {:.4f}s ({:.4f}s/{})".format((end_time - start_time) / len(img_names), (end_time - start_time), len(img_names)))
