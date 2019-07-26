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

from data_loader import data_loader
from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file", type=str,)
    parser.add_argument("--root-dir", default="", dest="root_dir", help="path to images", type=str,)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER,)
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    #assert(cfg.TEST.IMS_PER_BATCH == 1)
    img_list = []
    if not args.root_dir.endswith('/'):
        args.root_dir += '/'
    for root, parent, img_files in os.walk(args.root_dir):
        for img_file in img_files:
            if img_file.lower().endswith(('.jpg', '.png', '.bmp', '.jpeg')):
                img_list.append(os.path.join(root, img_file)[len(args.root_dir):])
    test_loader = data_loader(cfg, img_list, args.root_dir)
    coco_demo = COCODemo(cfg, args.root_dir)
    start_time = time.time()
    end_time = time.time()
    for batch_idx, imgs_data in enumerate(test_loader):
        imgs_name, imgs, imgs_h, imgs_w = imgs_data
        print('batch_idx={}, num_imgs={}'.format(batch_idx, len(imgs_name)))
        top_predictions = coco_demo.get_top_detections(imgs, imgs_h, imgs_w, imgs_name, topn=None)
        print("using time: {:.4f}s".format(time.time() - end_time))
        end_time = time.time()
    print("Average time {:.4f}s ({:.4f}s/{})".format((end_time - start_time) / len(img_list), (end_time - start_time), len(img_list)))
