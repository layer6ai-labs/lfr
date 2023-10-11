#!/usr/bin/env bash
epochs=600 
bs=4096 
lr=1e-3 
warmup_lr=1e-5 
warmup_epochs=$((epochs/10)) 
ckpt_path=ckpt
pred_lr_scale=5
eval_lr=1e-4 
eval_epochs=300
pred_epochs=5

dim=64
pred_dim=256

lambd=1e-5
num_of_targets=10
target_sample_ratio=5

python main.py --method lfr --dataset mimic3-los -a tcn \
         --epochs $epochs --train_with_steps -b $bs --optimizer-type adam \
        --lr $lr --warmup_lr $warmup_lr --warmup_epochs $warmup_epochs \
        --num_targets $num_of_targets --fix_pred_lr  \
        --train-predictor-individually \
        --eval_epochs $eval_epochs --eval_lr $eval_lr --fix_eval_lr \
        --target_layers 2  --pred_epochs $pred_epochs --num_of_classes 10 \
        --dim $dim --loss barlow-batch --pred_layers 2 --eval_bs $bs \
        --pred_dim $pred_dim --num-of-runs 1--target_sample_ratio $target_sample_ratio \
        --lambd $lambd
        
