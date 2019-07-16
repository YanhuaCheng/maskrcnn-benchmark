#!/bin/bash
#########################################################################
# Author: 
# Created Time: Wed Dec 26 15:43:07 2018
# File Name: train_multi_gpu.sh
# Description: 
#########################################################################
CUDA_VISIBLE_DEVICES=2 python predictor.py --idx_start 0 --idx_end 200
