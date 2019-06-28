#!/bin/bash
#########################################################################
# Author: 
# Created Time: Wed Dec 26 15:43:07 2018
# File Name: train_multi_gpu.sh
# Description: 
#########################################################################

#######################step2: download datasets#########################
#LOCAL_PATH="/data/user/breeze_dataset"
#mkdir -p $LOCAL_PATH
#cd $LOCAL_PATH
#START_TIME=$(date +%s)
#END_TIME=$(date +%s)
#function GetUseTime()
#{
#        END_TIME=$(date +%s)
#        (( USE_TIME=$END_TIME - $START_TIME ))
#        START_TIME=$END_TIME
#        echo $USE_TIME
#        return $USE_TIME
#}
###general_ocr_detect_pigz
#CEPH_PATH="/mnt/yardcephfs/mmyard/g_wxg_ob_dc/breezecheng/dataset/ocr_data/sohu/text_detection/general_ocr_detect_pigz"
#for part in $(find $CEPH_PATH -type f )
#do
#        echo $part
#        cp $part $LOCAL_PATH &
#done
#echo "start to download file, waiting----------"
#wait
#USE_TIME=$(GetUseTime)
#echo "down file time eslapse "$USE_TIME
#cat general_ocr_detect.tar.gz.* | tar -Ipigz -x
#USE_TIME=$(GetUseTime)
#echo "un tar file time eslapse "$USE_TIME
#ls $LOCAL_PATH"/general_ocr_detect"
#echo "total file number :"
#find $LOCAL_PATH"/general_ocr_detect" -type f |wc -l
#rm general_ocr_detect_pigz.tar.gz.* -rf 
#
########################step3: training maskrcnn#########################
cd /mnt/yardcephfs/mmyard/g_wxg_ob_dc/breezecheng/pytorch_project/maskrcnn-benchmark
python setup.py build develop
TIME=$(date "+%Y%m%d-%H%M%S")
NGPUS=4
python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py --config-file "configs/e2e_mask_rcnn_R_50_FPN_1x.yaml" SOLVER.IMS_PER_BATCH 8 OUTPUT_DIR log/$TIME
