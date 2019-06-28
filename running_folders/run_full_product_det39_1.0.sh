#!/bin/bash
#########################################################################
# Author: 
# Created Time: Wed Dec 26 15:43:07 2018
# File Name: train_multi_gpu.sh
# Description: 
#########################################################################
cd /data/user/data/breezecheng/pytorch_project/maskrcnn-benchmark/
NGPUS=2
python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py --config-file "configs/full_product39_faster_rcnn_R_50_FPN_1x.yaml"  OUTPUT_DIR log/log_full_product_det39_1.0 DTYPE "float16"
