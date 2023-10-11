#!/usr/bin/env bash
epochs=100 
epochs=100
lr=0.001
num_targets=6

dim=16
pred_dim=16

python main.py --method lfr --dataset uci-income -a mlp \
         --epochs $epochs -b 128 --lr $lr --eval_bs 256\
        --num_targets $num_targets --fix_pred_lr \
        --train-predictor-individually  --optimizer-type adam --wd 0\
        --target_layers 2 --pred_epochs 1 --fix_lr\
        --dim $dim --loss barlow-batch  --pred_layers 1\
        --pred_dim $pred_dim --run_eval 1 --num_of_classes 2 --num-of-runs 1 \
        --eval_freq 1 --target_sample_ratio 10

