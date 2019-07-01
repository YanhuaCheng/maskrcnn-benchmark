#!/bin/bash
#########################################################################
# Author: 
# Created Time: Wed Dec 26 15:43:07 2018
# File Name: train_multi_gpu.sh
# Description: 
#########################################################################
source /mnt/yardcephfs/mmyard/g_wxg_ob_dc/breezecheng/pytorch_project/maskrcnn-benchmark/ver4/maskrcnn-benchmark/setting.env
LOCAL_PATH="/data/yard/workspace/breeze_dataset"
mkdir -p $LOCAL_PATH
cd $LOCAL_PATH
START_TIME=$(date +%s)
END_TIME=$(date +%s)
function GetUseTime()
{
        END_TIME=$(date +%s)
        (( USE_TIME=$END_TIME - $START_TIME ))
        START_TIME=$END_TIME
        echo $USE_TIME
        return $USE_TIME
}
#---------------prepare training data-------------#
cd $LOCAL_PATH
echo 'qspace' > qspace.secrets
chmod 600 qspace.secrets
rsync -avR qspace@10.62.81.79::breezecheng/object_detect_catall_20190625 --port=8090 ./ --password-file=qspace.secrets

cd /mnt/yardcephfs/mmyard/g_wxg_ob_dc/breezecheng/pytorch_project/maskrcnn-benchmark/ver4/maskrcnn-benchmark/

DATASET_PREFIX="full_product_det39"
TAG=1.0
NGPUS=4
python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py --config-file "configs/full_product39_faster_rcnn_R_50_FPN_1x.yaml"  DATASETS.DATASET_PREFIX ${DATASET_PREFIX} OUTPUT_DIR log/log_${DATASET_PREFIX}_${TAG} DTYPE "float16"
