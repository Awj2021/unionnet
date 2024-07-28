#!/bin/bash

set -e
export HOME=$(pwd)
# Running UnionNet-B
# only using the (0.5, 0.5, 0.5) normalization. Without the Flip and Padding.
# CUDA_VISIBLE_DEVICES=1 python main.py --config ./configs/cifar10n.yml --aug_data_dir ./cifar-10-batches-py/gen_samples_and_lab_agree_x4_w_1.5_2.0.pt --p_name agree_x4_w_1.5_2.0 &
# CUDA_VISIBLE_DEVICES=0 python main.py --config ./configs/cifar10n.yml --aug_data_dir ./cifar-10-batches-py/gen_samples_and_lab_disagree_x4_w_1.5_2.0.pt --p_name disagree_x4_w_1.5_2.0 &

# CUDA_VISIBLE_DEVICES=1 python main.py --config ./configs/cifar10n.yml --aug_data_dir ./cifar-10-batches-py/gen_samples_and_lab_agree_x1.pt --p_name agree_x1 &
# CUDA_VISIBLE_DEVICES=0 python main.py --config ./configs/cifar10n.yml --aug_data_dir ./cifar-10-batches-py/gen_samples_and_lab_disagree_x1.pt --p_name disagree_x1 &

# CUDA_VISIBLE_DEVICES=1 python main.py --config ./configs/cifar10n.yml --aug_data_dir ./cifar-10-batches-py/gen_samples_and_lab_agree_x2.pt --p_name agree_x2 &
# CUDA_VISIBLE_DEVICES=0 python main.py --config ./configs/cifar10n.yml --aug_data_dir ./cifar-10-batches-py/gen_samples_and_lab_disagree_x2.pt --p_name disagree_x2 &

# CUDA_VISIBLE_DEVICES=1 python main.py --config ./configs/cifar10n.yml --aug_data_dir ./cifar-10-batches-py/gen_samples_and_lab_agree_x4.pt --p_name agree_x4 &
# CUDA_VISIBLE_DEVICES=0 python main.py --config ./configs/cifar10n.yml --aug_data_dir ./cifar-10-batches-py/gen_samples_and_lab_disagree_x4.pt --p_name disagree_x4 &


# I just find that maybe the cifar10n is not suitable for (0.5, 0.5, 0.5) normalization. 
# So I change to (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261) normalization.
# CUDA_VISIBLE_DEVICES=1 python main.py --config ./configs/cifar10n.yml --aug_data_dir ./cifar-10-batches-py/gen_samples_and_lab_agree_x4_w_1.5_2.0.pt --project unionb_new_norm --p_name agree_x4_w_1.5_2.0 &
# CUDA_VISIBLE_DEVICES=0 python main.py --config ./configs/cifar10n.yml --aug_data_dir ./cifar-10-batches-py/gen_samples_and_lab_disagree_x4_w_1.5_2.0.pt --project unionb_new_norm --p_name disagree_x4_w_1.5_2.0 &

# CUDA_VISIBLE_DEVICES=1 python main.py --config ./configs/cifar10n.yml --aug_data_dir ./cifar-10-batches-py/gen_samples_and_lab_agree_x1.pt --project unionb_new_norm --p_name agree_x1 &
# CUDA_VISIBLE_DEVICES=0 python main.py --config ./configs/cifar10n.yml --aug_data_dir ./cifar-10-batches-py/gen_samples_and_lab_disagree_x1.pt --project unionb_new_norm --p_name disagree_x1 &

# CUDA_VISIBLE_DEVICES=1 python main.py --config ./configs/cifar10n.yml --aug_data_dir ./cifar-10-batches-py/gen_samples_and_lab_agree_x2.pt --project unionb_new_norm --p_name agree_x2 &
# CUDA_VISIBLE_DEVICES=0 python main.py --config ./configs/cifar10n.yml --aug_data_dir ./cifar-10-batches-py/gen_samples_and_lab_disagree_x2.pt --project unionb_new_norm --p_name disagree_x2 &

# CUDA_VISIBLE_DEVICES=1 python main.py --config ./configs/cifar10n.yml --aug_data_dir ./cifar-10-batches-py/gen_samples_and_lab_agree_x4.pt --project unionb_new_norm --p_name agree_x4 &
# CUDA_VISIBLE_DEVICES=0 python main.py --config ./configs/cifar10n.yml --aug_data_dir ./cifar-10-batches-py/gen_samples_and_lab_disagree_x4.pt --project unionb_new_norm --p_name disagree_x4 &

# Using the (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261) normalization. With the Flip and Padding.
CUDA_VISIBLE_DEVICES=1 python main.py --config ./configs/cifar10n.yml --aug_data_dir ./cifar-10-batches-py/gen_samples_and_lab_agree_x4_w_1.5_2.0.pt --project unionb_w_trans --p_name agree_x4_w_1.5_2.0 &
CUDA_VISIBLE_DEVICES=0 python main.py --config ./configs/cifar10n.yml --aug_data_dir ./cifar-10-batches-py/gen_samples_and_lab_disagree_x4_w_1.5_2.0.pt --project unionb_w_trans --p_name disagree_x4_w_1.5_2.0 &

CUDA_VISIBLE_DEVICES=1 python main.py --config ./configs/cifar10n.yml --aug_data_dir ./cifar-10-batches-py/gen_samples_and_lab_agree_x1.pt --project unionb_w_trans --p_name agree_x1 &
CUDA_VISIBLE_DEVICES=0 python main.py --config ./configs/cifar10n.yml --aug_data_dir ./cifar-10-batches-py/gen_samples_and_lab_disagree_x1.pt --project unionb_w_trans --p_name disagree_x1 &

CUDA_VISIBLE_DEVICES=1 python main.py --config ./configs/cifar10n.yml --aug_data_dir ./cifar-10-batches-py/gen_samples_and_lab_agree_x2.pt --project unionb_w_trans --p_name agree_x2 &
CUDA_VISIBLE_DEVICES=0 python main.py --config ./configs/cifar10n.yml --aug_data_dir ./cifar-10-batches-py/gen_samples_and_lab_disagree_x2.pt --project unionb_w_trans --p_name disagree_x2 &

CUDA_VISIBLE_DEVICES=1 python main.py --config ./configs/cifar10n.yml --aug_data_dir ./cifar-10-batches-py/gen_samples_and_lab_agree_x4.pt --project unionb_w_trans --p_name agree_x4 &
CUDA_VISIBLE_DEVICES=0 python main.py --config ./configs/cifar10n.yml --aug_data_dir ./cifar-10-batches-py/gen_samples_and_lab_disagree_x4.pt --project unionb_w_trans --p_name disagree_x4 &