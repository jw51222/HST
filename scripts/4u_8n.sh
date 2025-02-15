#!/bin/bash
filename=$(basename "$0");exp_id="${filename%.*}"
CUDA_VISIBLE_DEVICES=1 \
nohup python main.py \
--exp_id "$exp_id" --about "有监督实验测试" \
--mode train \
--train_manner supervised \
--train_path ./dataset/data_4u_8n/train \
--val_path ./dataset/data_4u_8n/val \
--test_path  ./dataset/data_4u_8n/test \
--batch_size 16 \
--user_num 4 \
--antenna_num 4 \
--save_model true --progress_bar false \
--lr 1e-3 --patience 1 --weight_decay 0 & \

