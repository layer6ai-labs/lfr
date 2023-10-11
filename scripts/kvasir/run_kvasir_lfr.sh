#!/usr/bin/env bash
epochs=400
eval_epochs=100

dim=2048
pred_dim=256

python main.py --method lfr --dataset kvasir -a resnet18\
        --num_targets 6  --target_layers 2\
        --init-beta --random-dropout\
        --target_sample_ratio 1 --num_of_classes 8\
        --epochs $epochs -b 256 --lr 0.0001\
        --optimizer-type sgd --momentum 0.9 --wd 5e-4\
        --fix_pred_lr --train-predictor-individually\
        --pred_epochs 5 --pred_layers 2\
        --eval_epochs $eval_epochs --eval_lr 0.001 --eval_bs 256\
        --dim $dim --pred_dim $pred_dim --loss barlow-batch\
        --num-of-runs 1 