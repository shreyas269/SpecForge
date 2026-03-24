#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels
export SPECFORGE_DATA_NUM_PROC=32

NUM_GPUS=${1:-8}
NUM_NODES=${2:-2}
NODE_RANK=${3:-0}
MASTER_ADDR=${4:-"node0"}
MASTER_PORT=${5:-29500}

ATTENTION_BACKEND=${ATTENTION_BACKEND:-flex_attention}

torchrun \
    --nnodes $NUM_NODES \
    --node-rank $NODE_RANK \
    --nproc_per_node $NUM_GPUS \
    --master-addr $MASTER_ADDR \
    --master-port $MASTER_PORT \
    $ROOT_DIR/scripts/train_dflash.py \
    --target-model-path meta-llama/Llama-3.1-8B-Instruct \
    --draft-config-path $ROOT_DIR/configs/llama3-8b-dflash.json \
    --train-data-path $ROOT_DIR/cache/dataset/perfectblend_llama3.1-8b_regen.jsonl \
    --output-dir $ROOT_DIR/outputs/llama3.1-8b-dflash-perfectblend \
    --num-epochs 6 \
    --batch-size 4 \
    --learning-rate 6e-4 \
    --warmup-ratio 0.04 \
    --max-grad-norm 1.0 \
    --max-length 3072 \
    --chat-template llama3 \
    --attention-backend $ATTENTION_BACKEND \
    --num-anchors 512 \
    --loss-decay-gamma 7.0 \
    --log-interval 50 \
    --save-interval 1000 \
    --report-to wandb \
    --wandb-project specforge-llama3.1-8b-dflash \
    --target-model-backend sglang \
    --block-size 16 \
    --num-anchors 512 \
    --mask-token-id 128002 \
    --wandb-name llama3.1-8b-dflash-perfectblend
