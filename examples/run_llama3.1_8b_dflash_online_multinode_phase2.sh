#!/bin/bash
# Phase 2: Agent-specific fine-tuning with pretraining data replay
#
# Usage: bash run_llama3.1_8b_dflash_online_multinode_phase2.sh [NUM_GPUS] [NUM_NODES] [NODE_RANK] [MASTER_ADDR] [MASTER_PORT]
#
# Before running, set PHASE1_CKPT_DIR to your best Phase 1 checkpoint, e.g.:
#   export PHASE1_CKPT_DIR=/path/to/outputs/llama3.1-8b-dflash-perfectblend/epoch_6_step_XXXX

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

# Phase 1 checkpoint directory (set this before running)
PHASE1_CKPT_DIR=${PHASE1_CKPT_DIR:?ERROR: Set PHASE1_CKPT_DIR to your Phase 1 checkpoint path}

torchrun \
    --nnodes $NUM_NODES \
    --node-rank $NODE_RANK \
    --nproc_per_node $NUM_GPUS \
    --master-addr $MASTER_ADDR \
    --master-port $MASTER_PORT \
    $ROOT_DIR/scripts/train_dflash.py \
    --target-model-path meta-llama/Llama-3.1-8B-Instruct \
    --draft-config-path $ROOT_DIR/configs/llama3-8b-dflash.json \
    --train-data-path \
        "$ROOT_DIR/cache/dataset/perfectblend_llama3.1-8b_regen.jsonl::0.1" \
        "$ROOT_DIR/cache/dataset/agent_synthetic_data.jsonl" \
    --output-dir $ROOT_DIR/outputs/llama3.1-8b-dflash-agent-phase2 \
    --ckpt-dir $PHASE1_CKPT_DIR \
    --reset-scheduler \
    --num-epochs 3 \
    --batch-size 4 \
    --learning-rate 1e-4 \
    --warmup-ratio 0.06 \
    --max-grad-norm 1.0 \
    --max-length 3072 \
    --chat-template llama3 \
    --attention-backend $ATTENTION_BACKEND \
    --num-anchors 512 \
    --loss-decay-gamma 7.0 \
    --log-interval 25 \
    --save-interval 500 \
    --report-to wandb \
    --wandb-project specforge-llama3.1-8b-dflash \
    --target-model-backend sglang \
    --block-size 16 \
    --mask-token-id 128002 \
    --wandb-name llama3.1-8b-dflash-agent-phase2
