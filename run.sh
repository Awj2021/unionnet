#!/bin/bash

set -e
export HOME=$(pwd)
# Running UnionNet-B
CUDA_VISIBLE_DEVICES=0 python main.py --config ./configs/cifar10n.yml
# Running UnionNet-A

