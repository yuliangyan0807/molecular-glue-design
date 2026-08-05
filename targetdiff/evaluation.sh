#!/bin/bash

python evaluation.py \
    ./configs/sampling.yml \
    --mgd_test_dir /home/yuliangyan/Code/Trust-App-AI-Lab/molecular_glue_design/data/TernaryDB/MGD_test \
    --gpu_ids 0,1,2,3,4,5,6,7 \
    --pocket_radius 10 \
    --batch_size 500 \
    --result_path /home/yuliangyan/Code/Trust-App-AI-Lab/molecular_glue_design/targetdiff/outputs_mgd_test