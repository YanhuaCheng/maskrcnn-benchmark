#!/bin/bash
#########################################################################
# Author: 
# Created Time: Fri Jan  4 14:34:41 2019
# File Name: xx.sh
# Description: 
#########################################################################
python setup.py build develop

. /mnt/yardcephfs/mmyard/g_wxg_ob_dc/breezecheng/all-lib/shflags/shflags

DEFINE_string 'ps_hosts' 'none' 'ps_hosts' 'ph'
DEFINE_string 'worker_hosts' 'none' 'woker_hosts' 'wh'
DEFINE_string 'job_name' 'none' 'job_name' 'jn'
DEFINE_string 'task_index' 'none' 'task_index' 'ti'
# parse the command-line
FLAGS "$@" || exit $?
eval set -- "${FLAGS_ARGV}"

echo "worker_hosts, ${FLAGS_worker_hosts}!"
worker_hosts=(${FLAGS_worker_hosts//,/ })
for(( i=0;i<${#worker_hosts[@]};i++)) do
  worker_host=(${worker_hosts[i]//:/ })
  worker=${worker_host[0]}
  port=${worker_host[1]}
  echo "worker_$i"
  echo ${worker}
  echo ${port}
done;
#for worker_host in ${worker_hosts[@]}
#do
#  worker_host=(${worker_host//:/ })
#  worker=${worker_host[0]}
#  port=${worker_host[1]}
#  echo ${worker}
#  echo ${port}
#done
