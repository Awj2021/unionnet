#!/bin/bash

set -e

# Running UnionNet-B
# CUDA_VISIBLE_DEVICES=1 python main.py --wandb
CUDA_VISIBLE_DEVICES=0 python main.py --expert_num 3 --num_classes 100 --network resnet18 --dataset cifar100 --data_path /home/wenjie/projects/DivideMix/cifar-100-python --batch_size 64 --sched cosine --lr 0.0004 --wandb --epochs 150
# Running UnionNet-A

