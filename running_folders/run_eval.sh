#!/bin/bash
cd /data/user/data/breezecheng/pytorch_project/maskrcnn-benchmark/ 

DATASET_PREFIX="full_product_det11"
TAG="384-512_v4_class_weight_0125000"
NGPUS=2
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/test_net.py --config-file "configs/full_product11_faster_rcnn_R_50_FPN_1x.yaml"  DATASETS.DATASET_PREFIX ${DATASET_PREFIX} OUTPUT_DIR log/log_${DATASET_PREFIX}_${TAG} TEST.IMS_PER_BATCH 6 MODEL.WEIGHT pretrained_models/.torch/models/full_product_det11_384-512_v4_classs_weight_0125000.pt
