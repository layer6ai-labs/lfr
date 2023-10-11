#!/usr/bin/env bash
epochs=400
eval_epochs=100

dim=2048
pred_dim=256

python main.py --method diet --dataset kvasir -a resnet18\
        --num_targets 0  --pred_layers 2 --target_layers 1\
        --num_of_classes 8 --warmup_epochs 10\
        --dim $dim --pred_dim $pred_dim --loss ce-smooth\
        --epochs $epochs -b 256 --lr 0.0001\
        --optimizer-type sgd --momentum 0.9 --wd 5e-4\
        --eval_epochs $eval_epochs --eval_lr 0.001 --eval_bs 256\
        --num-of-runs 1