#!/bin/bash
#########################################################################
# Author: 
# Created Time: Wed Dec 26 15:43:07 2018
# File Name: train_multi_gpu.sh
# Description: 
#########################################################################
CUDA_VISIBLE_DEVICES=0 python demo.py --config-file "./configs/inference_full_product11_faster_rcnn_retinanet_R-50-FPN_1x.yaml" --root-dir "../../demo/badcase_imgs" DEPLOY.TOPN 3 OUTPUT_DIR "save_result" MODEL.WEIGHT ../../pretrained_models/.torch/models/full_product_det11_448-576_retinanet_00200000.pth

#CUDA_VISIBLE_DEVICES=0 python demo.py --config-file "./configs/inference_full_product11_faster_rcnn_R-50-FPN_1x.yaml" --root-dir "../../demo/badcase_imgs" DEPLOY.TOPN 3 OUTPUT_DIR "save_result" MODEL.WEIGHT ../../pretrained_models/.torch/models/full_product_det11_384-512_v4_classs_weight_0200000.pth
