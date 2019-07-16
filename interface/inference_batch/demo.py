#!/usr/bin/env python
# encoding=utf8
#########################################################################
# Author: breezecheng
# Created Time: Fri Dec 14 18:41:25 2018
# File Name: demo_1.0.py
# Description:
#########################################################################
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.config import cfg

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file", type=str,)
    parser.add_argument("--root-dir", default="", dest="root_dir", help="path to images", type=str,) 
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER,)
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
   img_names = []
   for root, parent, img_files in os.walk(cfg.root_dir):
       for img_file in img_files:
           if img_file.lower().endswith(('.jpg', '.png', '.bmp', '.jpeg')):
              img_names.append(os.path.join(root, img_file)[len(cfg.root_dir):])
   test_loader = data_loader(cfg, img_names, root_dir)
   coco_demo = COCODemo(cfg)
   for batch_idx, batch_data in enumerate(test_loader):
       imgs_name, imgs, imgs_h, imgs_w = batch_data
       top_predictions = coco_demo.get_top_detections(imgs, imgs_h, imgs_w, imgs_name, topn=None)
       
