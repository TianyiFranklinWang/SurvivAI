#!/bin/bash
config_file="$1"
fold_nb="$2"

OMP_NUM_THREADS=16 torchrun \
                    --nnodes=1 \
                    --nproc_per_node=1 \
                    train_fusion.py --config "$config_file" --k "$fold_nb"
