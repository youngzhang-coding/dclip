#!/bin/bash
set -e

SCRIPT_DIR=$(cd $(dirname $0); pwd)

PARENT_DIR=$(dirname $SCRIPT_DIR)

DATA_DIR=$HOME/Database/yfcc15m

echo "Running training script..."

cd $PARENT_DIR

source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch # Change to your conda environment name

export PYTHONPATH=$PARENT_DIR
export CUDA_VISIBLE_DEVICES=0,1

torchrun --nproc_per_node=1 \
    $PARENT_DIR/train/main.py \
    --train_tar  $DATA_DIR/shard-000001.tar \
    --batch_size 8 \
    --devices 1 \
    --num_workers 1 \
    --embed_dim 768 \
    --image_resolution 224 \
    --vision_layers 12 \
    --vision_width 768 \
    --vision_patch_size 16 \
    --context_length 77 \
    --vocab_size 49408 \
    --transformer_width 512 \
    --transformer_heads 8 \
    --transformer_layers 12 \
    --model_id $PARENT_DIR/checkpoint/models--stable-diffusion-v1-5--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14 \
    --contrastive_loss_weight 1.0 \
    --diffusion_loss_weight 1.0 \
    --region_iou_threshold 0.5 \
    --lr 5e-4 \
    --lr_start 1e-6 \
    --lr_end 1e-5 \
    --weight_decay 0.5 \
    --beta1 0.9 \
    --beta2 0.98 \
    --eps 1e-8 \
    --warmup_epochs 1 \
    --epochs 50 \
    --update_freq 1 \
    --no_ema \
    --accelerator gpu \
    --seed 42 \
    --output_dir $PARENT_DIR/outputs \
    --project dclip \
    --name pretrain \
    --enable_gradient_checkpoint

