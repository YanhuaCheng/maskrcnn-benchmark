#!/bin/bash
#########################################################################
# Author: 
# Created Time: Wed Dec 26 15:43:07 2018
# File Name: train_multi_gpu.sh
# Description: 
#########################################################################

#######################step1: get the work_hosts#########################
. /mnt/yardcephfs/mmyard/g_wxg_ob_dc/breezecheng/all-lib/shflags/shflags
FLAGS "$@" || exit $?
DEFINE_string 'ps_hosts' 'none' 'ps_hosts' 'ph'
DEFINE_string 'worker_hosts' 'none' 'woker_hosts' 'wh'
DEFINE_string 'job_name' 'none' 'job_name' 'jn'
DEFINE_string 'task_index' 'none' 'task_index' 'ti'
eval set -- "${FLAGS_ARGV}"
echo "${FLAGS_worker_hosts}"
#######################step2: download datasets#########################
source activate maskrcnn_benchmark
LOCAL_PATH="/data/user/breeze_dataset"
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
##general_ocr_detect_pigz
CEPH_PATH="/mnt/yardcephfs/mmyard/g_wxg_ob_dc/breezecheng/dataset/ocr_data/sohu/text_detection/general_ocr_detect_pigz"
for part in $(find $CEPH_PATH -type f )
do
        echo $part
        cp $part $LOCAL_PATH &
done
echo "start to download file, waiting----------"
wait
USE_TIME=$(GetUseTime)
echo "down file time eslapse "$USE_TIME
cat general_ocr_detect.tar.gz.* | tar -Ipigz -x
USE_TIME=$(GetUseTime)
echo "un tar file time eslapse "$USE_TIME
ls $LOCAL_PATH"/general_ocr_detect"
echo "total file number :"
find $LOCAL_PATH"/general_ocr_detect" -type f |wc -l
rm general_ocr_detect_pigz.tar.gz.* -rf 

#######################step3: training maskrcnn#########################
cd /mnt/yardcephfs/mmyard/g_wxg_ob_dc/breezecheng/pytorch_project/maskrcnn-benchmark
python setup.py build develop

TIME=$(date "+%Y%m%d-%H%M%S")
NGPUS=8
NNODES=4
NPROC_PER_NODE=2
worker_hosts=(${FLAGS_worker_hosts//,/ })
for(( i=0;i<${#worker_hosts[@]};i++)) do
   worker_host=(${worker_hosts[i]//:/ })
   worker=${worker_host[0]}
   port=${worker_host[1]}
   python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE --nnodes=$NNODES --node_rank=$i --master_addr=$worker --master_port=$port tools/train_net.py --config-file "configs/e2e_mask_rcnn_R_50_FPN_1x.yaml" OUTPUT_DIR log/$TIME
done;
