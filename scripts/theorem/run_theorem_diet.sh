#!/usr/bin/env bash
epochs=100 
lr=0.001
dim=256
pred_dim=256


python main.py --method diet --dataset theorem -a mlp\
        --num_targets 0  --target_layers 1 --num_of_classes 6\
        --epochs $epochs -b 128 --lr $lr\
        --wd 0 --eval_lr 3e-4 --eval_bs 128\
        --optimizer-type adam --fix_lr --fix_eval_lr \
        --dim $dim --pred_dim $pred_dim --loss ce-smooth\
        --num-of-runs 1 
        