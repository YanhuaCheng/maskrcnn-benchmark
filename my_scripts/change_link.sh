#!/bin/bash
#########################################################################
# Author: 
# Created Time: Mon Mar 25 15:05:32 2019
# File Name: change_link.sh
# Description: 
#########################################################################
for link in $(find ./product_catall -type l)
do
    echo "Link is:"
    echo $link
    dir=$(readlink $link)
    echo "dir is:"
    echo $dir
    new_dir=`echo $dir | sed 's#/data/user/data/breezecheng/dataset/image_retrieval/dataset_det/#/data/yard/workspace/breeze_dataset/#g'`
    ln -s $new_dir $link -f 
    echo "new_dir is:"
    echo $new_dir
done
