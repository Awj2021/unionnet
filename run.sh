#!/bin/bash

set -e

# Running UnionNet-B
CUDA_VISIBLE_DEVICES=1 python main.py --wandb

# Running UnionNet-A

