#!/bin/bash
#########################################################################
# Author: 
# Created Time: Wed Dec 26 15:43:07 2018
# File Name: train_single_gpu.sh
# Description: 
#########################################################################
TIME=$(date "+%Y%m%d-%H%M%S")

#############Single GPU#######################
CUDA_VISIBLE_DEVICES=2 python tools/train_net.py --config-file "configs/e2e_mask_rcnn_R_50_FPN_1x.yaml" SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 SOLVER.MAX_ITER 720000 SOLVER.STEPS "(480000, 640000)" TEST.IMS_PER_BATCH 1 OUTPUT_DIR log/$TIME


#CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --config-file "configs/e2e_mask_rcnn_R_50_FPN_1x.yaml" OUTPUT_DIR log/$TIME

#############Mulitple GPUs#######################
#NGPUS=4
#python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/test_net.py --config-file "configs/e2e_mask_rcnn_R_50_FPN_1x.yaml" OUTPUT_DIR log/$TIME
